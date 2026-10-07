#!/usr/bin/env python3
"""
Milvus Server Vector Database Integration

Handles vector database operations using Milvus server with proper connection management.
"""

import os
import re
import time
from urllib.parse import unquote as _url_unquote
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Milvus
try:
    from pymilvus import (
        connections, Collection, FieldSchema, CollectionSchema, DataType,
        utility, Index, MilvusException
    )
    try:
        from pymilvus.client.types import LoadState
    except ImportError:
        LoadState = None  # Older pymilvus; fallback to load() always
    MILVUS_AVAILABLE = True
except ImportError as e:
    LoadState = None
    MILVUS_AVAILABLE = False
    logger.warning(f"Milvus not available: {e}, falling back to mock implementation")

from src.semantic.semantic_components import TextChunk
from src.embeddings import EmbeddingResult

# Safe temporal reasoning for numeric year handling (no filename years)
try:
    from src.semantic.temporal_engine import TemporalReasoningEngine
except ImportError:  # Fallback for relative imports
    try:
        from .semantic.temporal_engine import TemporalReasoningEngine  # type: ignore
    except Exception:  # pragma: no cover - extremely defensive
        TemporalReasoningEngine = None  # type: ignore


@dataclass
class SearchResult:
    """Represents a search result from the vector database."""
    
    def __init__(self, chunk_id: str, text: str, score: float, metadata: Dict[str, Any], object_id: str = None, user_id: str = None):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata
        self.object_id = object_id
        self.user_id = user_id
        # Create a mock chunk object for compatibility
        self.chunk = type('Chunk', (), {
            'chunk_id': chunk_id,
            'text': text,
            'chunk_index': metadata.get('chunk_index', 0),
            'metadata': metadata
        })()


class MilvusServerVectorDatabase:
    """Handles vector database operations using Milvus server."""
    def __init__(
        self,
        collection_name: str = "document_chunks",
        vector_size: int = 768,
        distance_metric: str = "COSINE",
        host: str = None,
        port: str = None,
        is_image_collection: bool = False
    ):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            distance_metric: Distance metric for similarity search
            host: Milvus server host
            port: Milvus server port
            is_image_collection: If True, creates schema for image vectors (CLIP embeddings)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.is_image_collection = is_image_collection
        self.use_milvus = False
        self.collection = None
        self.temporal = TemporalReasoningEngine() if TemporalReasoningEngine else None
        # Control whether we eagerly load the collection at init time.
        # HARD DEFAULT: keep this False so API startup is fast and health checks succeed; collections load lazily on first search.
        # We intentionally ignore MILVUS_AUTO_LOAD_ON_INIT env here to avoid long blocking loads + retries on every container start.
        self.auto_load_on_init = False
        
        # Try to connect to Milvus server
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus not available - install pymilvus")
        
        self._connect()
        self._initialize_collection()
        self.use_milvus = True
        logger.info(f"Vector database initialized with Milvus server: {collection_name}")
    
    def _connect(self, wait_for_milvus: bool = True):
        """Connect to Milvus server with robust retry and exponential back-off.

        During heavy insert load, Milvus's embedded etcd can lose leader
        election for 30-60 seconds, causing short connection outages.
        This method waits through those outages rather than failing fast,
        so the Celery worker stays alive instead of crashing and restarting.

        For host-side tests set MILVUS_HOST=127.0.0.1 (not 'localhost') to
        force IPv4 and avoid gRPC connecting to the IPv6 loopback ::1.
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus not available")

        # First disconnect any stale connection to avoid "already connected" errors
        try:
            connections.disconnect("default")
        except Exception:
            pass

        # How long to keep trying before giving up (seconds)
        # wait_for_milvus=True: used at task time — wait up to 5 min for Milvus to recover
        # wait_for_milvus=False: used at worker prewarm — fail fast (30s) to not block startup
        max_wait = 300 if wait_for_milvus else 30
        grpc_timeout_s = float(os.getenv("MILVUS_CONNECT_TIMEOUT", "15"))

        attempt = 0
        start = time.time()
        last_exc: Optional[Exception] = None

        while (time.time() - start) < max_wait:
            attempt += 1
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=grpc_timeout_s,
                )
                elapsed = time.time() - start
                logger.info(
                    f"[MILVUS] Connected to {self.host}:{self.port} "
                    f"(attempt {attempt}, elapsed={elapsed:.1f}s)"
                )
                return
            except Exception as exc:
                last_exc = exc
                elapsed = time.time() - start
                remaining = max_wait - elapsed
                if remaining <= 0:
                    break
                # Exponential back-off: 1s, 2s, 4s, 8s, 16s, 30s (cap)
                wait = min(2 ** (attempt - 1), 30)
                wait = min(wait, remaining)
                logger.warning(
                    f"[MILVUS CONNECT] Attempt {attempt} failed ({elapsed:.0f}s elapsed, "
                    f"{remaining:.0f}s remaining): {exc}. Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Failed to connect to Milvus at {self.host}:{self.port} "
            f"after {attempt} attempts ({time.time()-start:.0f}s): {last_exc}"
        ) from last_exc

    def _reconnect_if_needed(self) -> bool:
        """Check if Milvus connection is alive; reconnect if not.

        Returns True if connected (or reconnected OK), False if failed.
        Called before every insert/search to ensure the connection is live
        after a Milvus restart.
        """
        try:
            # Cheapest way to test liveness: check connection alias exists
            if connections.get_connection_addr("default"):
                return True
        except Exception:
            pass
        try:
            logger.warning("[MILVUS] Connection dropped — attempting reconnect...")
            self._connect(wait_for_milvus=True)
            # Re-grab the collection object after reconnect
            if self.collection_name and MILVUS_AVAILABLE:
                from pymilvus import Collection, utility
                if utility.has_collection(self.collection_name):
                    self.collection = Collection(self.collection_name)
            return True
        except Exception as exc:
            logger.error(f"[MILVUS] Reconnect failed: {exc}")
            return False

    
    
    def _initialize_collection(self):
        """Initialize or create the collection in Milvus server."""
        if not MILVUS_AVAILABLE:
            return
        
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                # Skip load() here - will be loaded lazily if auto_load_on_init is True
                
                # Check if existing collection has correct dimensions
                for field in self.collection.schema.fields:
                    if field.name == "embedding":
                        existing_dim = field.params.get("dim")
                        if existing_dim != self.vector_size:
                            logger.warning(
                                f"Collection {self.collection_name} has wrong dimension: {existing_dim} "
                                f"(expected {self.vector_size}). Dropping and recreating..."
                            )
                            utility.drop_collection(self.collection_name)
                            self.collection = None
                            logger.info(f"Creating collection {self.collection_name} with correct dimension {self.vector_size}")
                            self._create_collection()
                            self._create_index()
                            # Skip load() here - will be loaded lazily if auto_load_on_init is True
                            return
                        else:
                            logger.info(f"Collection {self.collection_name} has correct dimension: {existing_dim}")
                        break
                
                # Collection exists with correct dimensions, just load it
                if not self.collection.has_index():
                    self._create_index()
                
                # Check if collection has user_id field
                try:
                    schema = self.collection.schema
                    field_names = [field.name for field in schema.fields]
                    logger.info(f"Collection fields: {field_names}")
                    has_user_id = any(field.name == "user_id" for field in schema.fields)
                    logger.info(f"Collection has user_id field: {has_user_id}")
                    
                    if not has_user_id:
                        logger.warning("Collection does not have user_id field, recreating with new schema...")
                        utility.drop_collection(self.collection_name)
                        self.collection = None
                        logger.info(f"Creating collection {self.collection_name} with user_id field")
                        self._create_collection()
                        # Create index for new collection
                        self._create_index()
                    else:
                        logger.info("Collection has user_id field, using existing collection")
                except Exception as e:
                    logger.error(f"Error checking collection schema: {e}")
                    logger.warning("Assuming collection needs recreation due to schema check error")
                    utility.drop_collection(self.collection_name)
                    self.collection = None
                    logger.info(f"Creating collection {self.collection_name} with user_id field")
                    self._create_collection()
                    # Create index for new collection
                    self._create_index()
                    # Check if index exists, create if not
                    indexes = self.collection.indexes
                    if not indexes:
                        logger.info("No index found, creating index...")
                        self._create_index()
                    else:
                        logger.info(f"Index already exists: {indexes}")
            else:
                logger.info(f"Creating collection {self.collection_name}")
                self._create_collection()
                # Create index for new collection
                self._create_index()
            
            # Load collection into memory (optional, controlled by env)
            if self.auto_load_on_init:
                logger.info(f"Auto-loading collection {self.collection_name} into memory at init")
                self.collection.load()
            else:
                logger.info(
                    f"Skipping collection.load() at init for {self.collection_name} "
                    "(MILVUS_AUTO_LOAD_ON_INIT=false). Collection will be loaded lazily on first search."
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _create_collection(self):
        """Create a new collection with proper schema."""
        if not MILVUS_AVAILABLE:
            return
        
        # Define field schemas based on collection type
        if self.is_image_collection:
            # Image collection schema for CLIP embeddings
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="objects", dtype=DataType.VARCHAR, max_length=10000),  # JSON string
                FieldSchema(name="scene", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="dominant_colors", dtype=DataType.VARCHAR, max_length=2000),  # JSON string
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=500)
            ]
            description = "Image vectors with CLIP embeddings"
        else:
            # Document collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="element_type", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="bucket_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="connection_id", dtype=DataType.VARCHAR, max_length=500),
            ]
            description = "Document chunks with embeddings"
        
        # Create collection schema
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=True
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )
        
        # Create index
        self._create_index()
    
    def _create_index(self):
        """Create index for vector field."""
        if not MILVUS_AVAILABLE or not self.collection:
            return
        
        try:
            # Check if index already exists
            indexes = self.collection.indexes
            if indexes:
                logger.info(f"Index already exists for collection: {self.collection_name}")
                return
            
            # REPLACE your current index_params block with:
            index_params = {
                "metric_type": self.distance_metric,   # COSINE
                "index_type": "HNSW",
                "params": {
                    "M": 32,               # graph degree
                    "efConstruction": 200  # build-time ef
                }
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info("Index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            # Try with simpler FLAT index as fallback
            try:
                logger.info("Trying FLAT index as fallback...")
                flat_index_params = {
                    "metric_type": self.distance_metric,
                    "index_type": "FLAT"
                }
                
                self.collection.create_index(
                    field_name="embedding",
                    index_params=flat_index_params
                )
                
                logger.info("FLAT index created successfully")
                
            except Exception as e2:
                logger.error(f"Failed to create FLAT index: {e2}")
                raise
    
    def insert_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray, user_id: str) -> bool:
        """
        Insert chunks with their embeddings into the vector database.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Numpy array of embeddings
            user_id: User ID for tenant isolation
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Prepare data for insertion
            data = []
            for i, chunk in enumerate(chunks):
                # Generate a shorter ID for the primary key (max 100 chars)
                short_id = chunk.chunk_id[:95] if len(chunk.chunk_id) > 95 else chunk.chunk_id
                chunk_meta = chunk.metadata or {}
                data.append({
                    "id": short_id,
                    "embedding": embeddings[i].tolist(),
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_number": chunk_meta.get("page_number", 0),
                    "element_type": chunk_meta.get("element_type", "unknown"),
                    # Canonical identity & filename metadata
                    "source_file": chunk_meta.get("file_id") or chunk_meta.get("source_file", "unknown"),
                    "file_id": chunk_meta.get("file_id") or chunk_meta.get("source_file", "unknown"),
                    "filename": chunk_meta.get("filename", ""),
                    "original_filename": chunk_meta.get("original_filename", chunk_meta.get("filename", "")),
                    "created_at": str(int(time.time())),
                    "object_id": chunk.object_id or chunk_meta.get("file_id") or chunk_meta.get("source_file", ""),
                    "user_id": str(user_id) if user_id else "default_user",  # Ensure user_id is a string
                    # Bucket / connection scoping (same values as ingest API)
                    "bucket_id": chunk_meta.get("bucket_id") or "",
                    "path": chunk_meta.get("path") or "",
                    "connection_id": chunk_meta.get("connection_id") or "",
                })

            # ---------------------------------------------------------------------------
            # MILVUS INSERT WITH AUTO-RECONNECT + EXPONENTIAL BACK-OFF RETRY
            # Milvus's embedded etcd can lose leadership for 30-60s under load,
            # causing gRPC connection errors. We reconnect transparently so the
            # Celery task completes instead of crashing and causing a worker restart.
            # ---------------------------------------------------------------------------
            import random
            _max_attempts = 8
            _base_delay = 2.0   # seconds
            last_exc: Optional[Exception] = None
            for _attempt in range(_max_attempts):
                try:
                    # Ensure connection is alive before each attempt
                    if not self._reconnect_if_needed():
                        raise ConnectionError("Could not establish Milvus connection")
                    self.collection.insert(data)
                    break  # success
                except Exception as _ins_exc:
                    last_exc = _ins_exc
                    err_str = str(_ins_exc).lower()
                    if _attempt == _max_attempts - 1:
                        raise  # re-raise on final attempt
                    # Longer wait for connection errors (Milvus is restarting)
                    if any(kw in err_str for kw in ("connection", "unavailable", "refused", "timeout", "rpc", "etcd")):
                        _sleep = min(30, _base_delay * (2 ** _attempt)) + random.uniform(0, 2.0)
                    else:
                        _sleep = _base_delay * (2 ** min(_attempt, 3)) + random.uniform(0, 1.0)
                    logger.warning(
                        f"[MILVUS] Insert attempt {_attempt + 1}/{_max_attempts} failed: {_ins_exc}. "
                        f"Retrying in {_sleep:.1f}s..."
                    )
                    time.sleep(_sleep)


            logger.info(f"Inserted {len(chunks)} chunks into Milvus server (flush deferred)")
            return True

        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            return False
    
    def _ensure_collection_loaded(self) -> bool:
        """
        Ensure collection is loaded before search/query operations.

        We keep init fast by skipping eager load, but search paths must call
        load() explicitly; otherwise Milvus returns "collection not loaded".
        Checks load state first to avoid redundant load() on already-loaded or
        empty collections (which can fail in some Milvus versions).
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return False
        try:
            # Check if already loaded (avoids redundant load; for empty
            # collections like ultimate_image_vectors where load can fail)
            if LoadState is not None:
                state = utility.load_state(
                    collection_name=self.collection_name,
                    timeout=5.0,
                )
                if state == LoadState.Loaded:
                    logger.debug(f"Collection {self.collection_name} already loaded")
                    return True
                if state == LoadState.Loading:
                    # Wait for loading to complete
                    logger.info(f"Collection {self.collection_name} loading, waiting...")
                    utility.wait_for_loading_complete(
                        self.collection_name, timeout=60.0
                    )
                    return True
                if state == LoadState.NotExist:
                    logger.error(f"Collection {self.collection_name} does not exist")
                    return False

            logger.info(f"Loading collection {self.collection_name} into memory (timeout=60s)...")
            self.collection.load(timeout=60.0)
            return True
        except Exception as e:
            err_str = str(e).lower()
            # Empty collections may fail load; return False so search returns []
            if any(kw in err_str for kw in ("no segment", "empty", "no data")):
                logger.info(
                    f"Collection {self.collection_name} appears empty (load failed: {e}). "
                    f"Image search will return no results."
                )
            else:
                logger.error(f"Failed to load collection {self.collection_name}: {e}")
            return False

    def _delete_by_primary_keys(self, ids: List[str], batch_size: Optional[int] = None) -> int:
        """
        Milvus 2.3 delete() accepts only PK plans (e.g. varchar id in [\"a\",\"b\"]).
        Non-PK filters (user_id == ..., id != '') raise invalid plan node type.
        """
        if not ids or not self.collection:
            return 0
        bs = batch_size if batch_size is not None else max(
            1, int(os.getenv("MILVUS_DELETE_PK_BATCH", "128"))
        )
        deleted = 0
        for i in range(0, len(ids), bs):
            batch = ids[i : i + bs]
            escaped = [str(x).replace("\\", "\\\\").replace('"', '\\"') for x in batch]
            pk_expr = "id in [" + ", ".join(f'"{x}"' for x in escaped) + "]"
            self.collection.delete(expr=pk_expr)
            deleted += len(batch)
        return deleted

    def insert_image_vector(
        self,
        file_id: str,
        embedding: np.ndarray,
        analysis: Dict[str, Any],
        user_id: str = None,
        bucket_id: Optional[str] = None,
        path: Optional[str] = None,
        connection_id: Optional[str] = None,
    ) -> bool:
        """
        Insert image vector with analysis metadata.
        
        Args:
            file_id: Unique file identifier
            embedding: CLIP image embedding vector
            analysis: Image analysis results
            user_id: User ID for tenant isolation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_image_collection:
            logger.error("This method is only for image collections")
            return False
        
        try:
            import json
            
            # Prepare data for insertion
            row: Dict[str, Any] = {
                "id": file_id,
                "embedding": embedding.tolist(),
                "source_file": file_id,
                "objects": json.dumps(analysis.get("objects", [])),
                "scene": analysis.get("scene", "general"),
                "dominant_colors": json.dumps(analysis.get("dominant_colors", [])),
                "description": analysis.get("description", ""),
                "created_at": str(int(time.time())),
                "user_id": str(user_id) if user_id else "default_user",
            }
            if bucket_id:
                row["bucket_id"] = str(bucket_id)
            if path:
                row["path"] = str(path)
            if connection_id:
                row["connection_id"] = str(connection_id)
            data = [row]
            
            # Bound retry duration to avoid long blocking loops on Milvus blips.
            self.collection.insert(
                data,
                timeout=8,
                retry_times=2,
            )
            logger.info(f"Inserted image vector for {file_id} into Milvus (flush deferred)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert image vector: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, limit: int = 10, score_threshold: float = 0.0, filter_conditions: Dict = None, user_id: str = None) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Additional filter conditions
            user_id: User ID for tenant isolation
            
        Returns:
            List of SearchResult objects
        """
        try:
            if not self._ensure_collection_loaded():
                return []
            # Milvus index has ef=64, so limit (k) must be < 64
            # Cap limit at 50 to be safe (leaves room for ef=64)
            if limit > 50:
                logger.warning(
                    f"Limit {limit} exceeds maximum (50), capping to 50 to avoid Milvus ef/k constraint"
                )
                limit = 50
            
            # Search parameters - ensure HNSW ef is always >= limit (Milvus requires ef >= k)
            # MILVUS_SEARCH_EF: override for prod (lower=faster, 48-64 typical; default from limit*2)
            ef_override = os.getenv("MILVUS_SEARCH_EF", "")
            ef_value = int(ef_override) if ef_override.isdigit() else max(64, min(512, max(1, limit) * 2))
            ef_value = max(ef_value, limit)  # Ef must be >= limit or Milvus fails
            search_params = {
                "metric_type": self.distance_metric,
                "params": {"ef": ef_value}
            }
            
            filter_expr = None
            # Always filter by user_id if provided to ensure strict tenant isolation
            if user_id:
                # Ensure user_id is a string and properly escaped
                user_id_str = str(user_id).strip()
                if user_id_str:
                    filter_expr = f'user_id == "{user_id_str}"'
            # Add additional filter conditions (bucket_id, path)
            # Production: escape strings for Milvus expr; use lowercase like (Milvus 2.x)
            # Path uses (exact or prefix) with proper parentheses for AND/OR precedence
            if filter_conditions:
                def _escape(s):
                    """Escape double-quotes for Milvus expr string literals."""
                    if not isinstance(s, str):
                        return str(s)
                    # Escape backslash first, then double-quote
                    return str(s).replace("\\", "\\\\").replace('"', '\\"')

                conditions = []
                if filter_expr:
                    conditions.append(filter_expr)
                for key, value in filter_conditions.items():
                    if value is None or (isinstance(value, str) and not value.strip()):
                        continue
                    val_str = str(value).strip()
                    escaped = _escape(val_str)
                    if key == "path":
                        # Exact match OR prefix match; parentheses required for correct AND/OR precedence
                        conditions.append(f'(path == "{escaped}" or path like "{escaped}/%")')
                    elif isinstance(value, str):
                        conditions.append(f'{key} == "{escaped}"')
                    else:
                        conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions)
            
            # Determine output fields based on collection type
            if self.is_image_collection:
                output_fields = ["source_file", "objects", "scene", "dominant_colors", "description", "created_at", "user_id"]
            else:
                output_fields = [
                    "chunk_id",
                    "text",
                    "page_number",
                    "element_type",
                    "source_file",
                    "file_id",
                    "filename",
                    "original_filename",
                    "created_at",
                    "object_id",
                    "user_id",
                    "bucket_id",
                    "path",
                    "connection_id",
                ]
            
            # Perform search (with fallback on expr errors for bucket/path filters)
            use_app_side_filter = False
            try:
                results = self.collection.search(
                    data=[query_embedding.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    expr=filter_expr if filter_expr else "",
                    output_fields=output_fields
                )
            except Exception as expr_err:
                if filter_conditions and (
                    filter_conditions.get("bucket_id")
                    or filter_conditions.get("path")
                    or filter_conditions.get("connection_id")
                ):
                    logger.warning(
                        f"Milvus expr failed ({expr_err}), falling back to app-side bucket/path/connection filter"
                    )
                    user_id_str = str(user_id).strip() if user_id else ""
                    filter_expr_fallback = f'user_id == "{user_id_str}"' if user_id_str else ""
                    results = self.collection.search(
                        data=[query_embedding.tolist()],
                        anns_field="embedding",
                        param=search_params,
                        limit=min(limit * 3, 150),
                        expr=filter_expr_fallback,
                        output_fields=output_fields
                    )
                    use_app_side_filter = True
                else:
                    raise

            # Process results
            search_results = []
            for hit in results[0]:
                if hit.score >= score_threshold:
                    # Access entity fields - handle both dict and object access
                    entity_dict = {}
                    try:
                        if hasattr(hit, 'entity'):
                            if hasattr(hit.entity, 'get'):
                                entity_dict = {field: hit.entity.get(field) for field in output_fields}
                            elif isinstance(hit.entity, dict):
                                entity_dict = {field: hit.entity.get(field) for field in output_fields}
                            else:
                                for field in output_fields:
                                    try:
                                        entity_dict[field] = getattr(hit.entity, field, None)
                                    except:
                                        entity_dict[field] = None
                        
                        if not any(entity_dict.values()):
                            for field in output_fields:
                                try:
                                    value = getattr(hit, field, None)
                                    if value is not None:
                                        entity_dict[field] = value
                                except:
                                    pass
                    except Exception as e:
                        logger.warning(f"Error accessing entity fields: {e}")
                        entity_dict = {field: None for field in output_fields}
                    
                    if self.is_image_collection:
                        # For image collections, use source_file as chunk_id and description as text
                        search_results.append(SearchResult(
                            chunk_id=entity_dict.get("source_file"),
                            text=entity_dict.get("description", ""),
                            score=float(hit.score),
                            metadata={
                                "objects": entity_dict.get("objects", "[]"),
                                "scene": entity_dict.get("scene", "general"),
                                "dominant_colors": entity_dict.get("dominant_colors", "[]"),
                                "source_file": entity_dict.get("source_file"),
                                "created_at": entity_dict.get("created_at", "")
                            },
                            object_id=entity_dict.get("source_file"),
                            user_id=entity_dict.get("user_id")
                        ))
                    else:
                        # For document collections
                        search_results.append(SearchResult(
                            chunk_id=entity_dict.get("chunk_id"),
                            text=entity_dict.get("text", ""),
                            score=float(hit.score),
                            metadata={
                                "page_number": entity_dict.get("page_number", 0),
                                "element_type": entity_dict.get("element_type", ""),
                                "source_file": entity_dict.get("source_file", ""),
                                "file_id": entity_dict.get("file_id", "") or entity_dict.get("source_file", ""),
                                "filename": entity_dict.get("filename", ""),
                                "original_filename": entity_dict.get("original_filename", "") or entity_dict.get("filename", ""),
                                "created_at": entity_dict.get("created_at", ""),
                                "bucket_id": entity_dict.get("bucket_id") or None,
                                "path": entity_dict.get("path") or None,
                                "connection_id": entity_dict.get("connection_id") or None,
                            },
                            object_id=entity_dict.get("object_id") or entity_dict.get("file_id") or entity_dict.get("source_file"),
                            user_id=entity_dict.get("user_id")
                        ))

            if use_app_side_filter and filter_conditions:
                search_results = [
                    r for r in search_results
                    if (not filter_conditions.get("bucket_id") or (r.metadata or {}).get("bucket_id") == filter_conditions["bucket_id"])
                    and (not filter_conditions.get("connection_id") or (r.metadata or {}).get("connection_id") == filter_conditions["connection_id"])
                    and (not filter_conditions.get("path") or (
                        (r.metadata or {}).get("path", "").rstrip("/") == filter_conditions["path"].rstrip("/")
                        or (r.metadata or {}).get("path", "").startswith(filter_conditions["path"].rstrip("/") + "/")
                    ))
                ]

            logger.info(f"Found {len(search_results)} similar chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def query_all_chunks(self, user_id: str = None, limit: int = 10000, filter_conditions: Dict = None, expr: str = None, include_text: bool = True) -> List[SearchResult]:
        """
        Query ALL chunks for a user directly from Milvus (bypasses similarity search).
        This ensures 100% recall for exact text matching.

        IMPORTANT:
            Milvus enforces a hard window on (offset + limit) of 16384.
            To avoid `invalid max query result window` errors, we always clamp
            the requested limit into [1, 16384] on the server side, regardless
            of what the caller passes.

        Args:
            user_id: User ID for tenant isolation
            limit: Maximum number of chunks to retrieve (caller hint, unclamped)
            filter_conditions: Additional filter conditions (bucket_id is filtered at Milvus level,
                              path must be filtered in Python as it's not a Milvus field)
            expr: Raw Milvus query expression (if provided, overrides user_id and filters)
            include_text: Whether to include the large 'text' field. False speeds up metadata scanning.

        Returns:
            List of SearchResult objects
        """
        if not MILVUS_AVAILABLE or self.collection is None:
            return []

        # Clamp limit to [1, 16384] to prevent Milvus errors
        clamped_limit = max(1, min(limit, 16384))
        
        try:
            # 1) Build expression
            if expr:
                filter_expr = expr
            else:
                conditions = []
                if user_id:
                    conditions.append(f'user_id == "{user_id}"')
                
                if filter_conditions:
                    for key, value in filter_conditions.items():
                        if key == "path":
                            continue # skip path as it is not indexed in milvus
                        if isinstance(value, str):
                            conditions.append(f'{key} == "{value}"')
                        else:
                            conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions)

            # 2) Determine output fields
            if self.is_image_collection:
                output_fields = ["source_file", "objects", "scene", "dominant_colors", "description", "created_at", "user_id"]
            else:
                output_fields = [
                    "chunk_id", "page_number", "element_type", "source_file",
                    "file_id", "filename", "original_filename", "created_at", "object_id",
                    "user_id", "bucket_id", "path", "connection_id",
                ]
                if include_text:
                    output_fields.insert(1, "text")
            
            # 3) Query
            logger.info(f"Querying all chunks with filter: '{filter_expr}' (limit: {clamped_limit})")
            results = self.collection.query(
                expr=filter_expr,
                output_fields=output_fields,
                limit=clamped_limit
            )
            
            if not results:
                return []
            
            # 4) Convert to SearchResult
            search_results = []
            for entity in results:
                try:
                    # Handle both dict and object access
                    if isinstance(entity, dict):
                        entity_dict = entity
                    else:
                        entity_dict = {field: getattr(entity, field, None) for field in output_fields}
                    
                    if self.is_image_collection:
                        search_results.append(SearchResult(
                            chunk_id=entity_dict.get("source_file"),
                            text=entity_dict.get("description", ""),
                            score=1.0,
                            metadata=entity_dict,
                            object_id=entity_dict.get("source_file"),
                            user_id=entity_dict.get("user_id")
                        ))
                    else:
                        meta = {
                            "page_number": entity_dict.get("page_number", 0),
                            "element_type": entity_dict.get("element_type", ""),
                            "source_file": entity_dict.get("source_file", ""),
                            "file_id": entity_dict.get("file_id", "") or entity_dict.get("source_file", ""),
                            "filename": entity_dict.get("filename", ""),
                            "original_filename": entity_dict.get("original_filename", "") or entity_dict.get("filename", ""),
                            "created_at": entity_dict.get("created_at", ""),
                            "bucket_id": entity_dict.get("bucket_id"),
                            "path": entity_dict.get("path"),
                            "connection_id": entity_dict.get("connection_id"),
                        }
                        search_results.append(SearchResult(
                            chunk_id=entity_dict.get("chunk_id"),
                            text=entity_dict.get("text", ""),
                            score=1.0,
                            metadata=meta,
                            object_id=entity_dict.get("object_id") or entity_dict.get("file_id") or entity_dict.get("source_file"),
                            user_id=entity_dict.get("user_id")
                        ))
                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    continue
            
            logger.info(f"Queried {len(search_results)} chunks for user_id: {user_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to query all chunks: {e}")
            return []
    
    
    def query_by_metadata_years(self, years: List[int], user_id: str = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Query collection by year using ONLY metadata + safe-context text.
        IMPORTANT: No filename-based year extraction (to avoid false positives).

        Returns a list of dict-like records: {'chunk_id','text','score','metadata'}.
        """
        results: List[Dict[str, Any]] = []
        if not self.collection:
            return results
        
        # Fallback-only implementation: scan chunks and use safe temporal extraction from text
        try:
            all_chunks = self.query_all_chunks(user_id=user_id, limit=limit * 5)
            if not all_chunks:
                return results

            # Normalize target years to ints
            target_years: list[int] = []
            for y in years:
                try:
                    target_years.append(int(y))
                except (TypeError, ValueError):
                    continue
            
            for chunk in all_chunks:
                meta = chunk.metadata or {}
                text = chunk.text or ""
                
                matched = False
                
                # 1) Metadata-based years (safe)
                meta_years_raw = meta.get("years") or []
                meta_years: list[int] = []
                for y in meta_years_raw:
                    try:
                        meta_years.append(int(y))
                    except (TypeError, ValueError):
                        continue

                if meta_years and any(y in meta_years for y in target_years):
                    matched = True
                elif self.temporal:
                    # 2) Safe text-based year extraction
                    try:
                        safe_years = self.temporal.safe_extract_years_from_text(text)
                    except Exception:
                        safe_years = []
                    if safe_years and any(y in safe_years for y in target_years):
                        matched = True
                if not matched:
                    # 3) Simple year-in-text fallback for 100% recall (e.g. filename "2019.pdf")
                    text_years = set(int(m.group(0)) for m in re.finditer(r"\b(19|20)\d{2}\b", text))
                    if text_years and any(y in text_years for y in target_years):
                        matched = True
                
                if matched:
                    results.append({
                        "chunk_id": chunk.chunk_id or "",
                        "text": chunk.text or "",
                        "score": 1.0,
                        "metadata": meta or {},
                    })
                    
                    if len(results) >= limit:
                        break
            
            logger.debug(f"Metadata year query found {len(results)} results via safe text/metadata scan")
            return results
            
        except Exception as exc:
            logger.warning(f"Safe metadata year scan failed: {exc}", exc_info=True)
        return results
    
    def query_by_metadata_single_year(self, year: int, user_id: str = None, limit: int = 200) -> List[Dict[str, Any]]:
        """Query by a single year."""
        return self.query_by_metadata_years([year], user_id=user_id, limit=limit)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        if not self.collection:
            return {
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "distance_metric": self.distance_metric,
                "status": "not_connected"
            }
        
        try:
            # Get collection stats
            stats = self.collection.get_stats()
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": stats.get("row_count", 0),
                "vector_size": self.vector_size,
                "distance_metric": self.distance_metric,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "vector_size": self.vector_size,
                "distance_metric": self.distance_metric,
                "status": "error"
            }

    def aggregate_storage_by_user(
        self, max_rows: Optional[int] = None, filter_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Bounded scan of the collection: per user_id, chunk_count and distinct file identifiers
        (file_id / source_file / object_id).

        If filter_user_id is set, only rows for that tenant are scanned (accurate for that user
        up to max_rows). Otherwise the iterator walks the whole collection up to max_rows
        (other users may be missing if the cap is exceeded).
        """
        out_base = {
            "collection": self.collection_name,
            "is_image_collection": self.is_image_collection,
            "scanned_chunks": 0,
            "max_rows_cap": 0,
            "users": {},
        }
        if not self.use_milvus or not self.collection:
            out_base["error"] = "no_collection"
            return out_base
        if not self._ensure_collection_loaded():
            out_base["error"] = "not_loaded"
            return out_base

        cap = max_rows if max_rows is not None else int(os.getenv("VECTOR_STATS_MAX_ROWS", "400000"))
        cap = max(1, min(int(cap), 2_000_000))

        def _esc_stats(value: str) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"')

        scan_expr = 'id != ""'
        if filter_user_id and str(filter_user_id).strip():
            scan_expr = f'user_id == "{_esc_stats(str(filter_user_id).strip())}"'

        output_fields = ["user_id", "source_file"]
        if not self.is_image_collection:
            output_fields.extend(["file_id", "object_id"])

        user_chunks: Dict[str, int] = defaultdict(int)
        user_files: Dict[str, Set[str]] = defaultdict(set)
        scanned = 0

        try:
            iterator = self.collection.query_iterator(
                expr=scan_expr,
                batch_size=2048,
                limit=cap,
                output_fields=output_fields,
            )
            try:
                while scanned < cap:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        uid = str(row.get("user_id") or "").strip() or "_unknown"
                        user_chunks[uid] += 1
                        fk = (
                            str(row.get("file_id") or "").strip()
                            or str(row.get("source_file") or "").strip()
                            or str(row.get("object_id") or "").strip()
                        )
                        if fk:
                            user_files[uid].add(fk)
                    scanned += len(batch)
            finally:
                try:
                    iterator.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error("aggregate_storage_by_user failed: %s", e, exc_info=True)
            out_base["error"] = str(e)
            return out_base

        users_out: Dict[str, Any] = {}
        for uid, nch in user_chunks.items():
            users_out[uid] = {
                "chunk_count": nch,
                "file_count": len(user_files.get(uid, set())),
            }
        out_base["scanned_chunks"] = scanned
        out_base["max_rows_cap"] = cap
        out_base["filter_user_id"] = str(filter_user_id).strip() if filter_user_id else None
        out_base["users"] = users_out
        return out_base

    def delete_by_file_id(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        bucket_id: Optional[str] = None,
        path: Optional[str] = None,
        connection_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Delete all chunks for a specific file_id from Milvus.

        Args:
            file_id: The source_file ID to delete
            user_id: Optional user_id filter for tenant isolation
            bucket_id: Optional bucket_id filter
            path: Optional path filter
            connection_id: Optional app-level S3 connection id (same as ingest)

        Returns:
            {"chunks_deleted": N, "files_deleted": M} with M = distinct logical file ids in those rows
        """
        if not self.use_milvus or not self.collection:
            logger.warning("No active collection to delete from")
            return {"chunks_deleted": 0, "files_deleted": 0}
        
        try:
            # Must load collection before query/delete (Milvus returns "collection not loaded" otherwise)
            if not self._ensure_collection_loaded():
                logger.warning(f"Cannot delete: collection {self.collection_name} not loaded")
                return {"chunks_deleted": 0, "files_deleted": 0}

            def _esc(value: str) -> str:
                return str(value).replace("\\", "\\\\").replace('"', '\\"')

            base_filters: List[str] = []
            if user_id:
                base_filters.append(f'user_id == "{_esc(user_id)}"')
            if bucket_id:
                base_filters.append(f'bucket_id == "{_esc(bucket_id)}"')
            if path:
                base_filters.append(f'path == "{_esc(path)}"')
            if connection_id:
                base_filters.append(f'connection_id == "{_esc(connection_id)}"')

            # Exact-match attempts across all canonical file identifier fields used at ingest.
            target = str(file_id or "")
            target_esc = _esc(target)

            def _like_esc(s: str) -> str:
                """Escape Milvus LIKE wildcards inside user-supplied file_id."""
                return (
                    str(s)
                    .replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )

            # Image collection: only scalar fields in schema (no chunk_id / object_id).
            if self.is_image_collection:
                candidate_exprs = [
                    f'source_file == "{target_esc}"',
                ]
            else:
                # Object_id + source_file are fixed schema; file_id/filename may be dynamic fields.
                # Chunk_id prefix catches upsert format: "{file_id}::chunk::{idx}::{ts}" and caption rows.
                _like_pat = _like_esc(target) + "%"
                candidate_exprs = [
                    f'source_file == "{target_esc}"',
                    f'object_id == "{target_esc}"',
                    f'file_id == "{target_esc}"',
                    f'filename == "{target_esc}"',
                    f'original_filename == "{target_esc}"',
                    f'chunk_id like "{_like_pat}"',
                ]

            ids: List[str] = []
            seen_ids = set()
            # Each probe must paginate: a single Milvus query() is capped (default 16384) and would
            # leave chunks behind for very large single-file ingestions.
            max_probe = int(os.getenv("DELETE_PROBE_MAX_ROWS", "2000000"))

            def _collect_ids_for_probe(field_expr: str) -> None:
                expr_parts = [field_expr] + base_filters
                expr_full = " && ".join(expr_parts)
                try:
                    it = self.collection.query_iterator(
                        expr=expr_full,
                        batch_size=2048,
                        limit=max_probe,
                        output_fields=["id"],
                    )
                    try:
                        pulled = 0
                        while pulled < max_probe:
                            batch = it.next()
                            if not batch:
                                break
                            for row in batch:
                                row_id = row.get("id")
                                if row_id and row_id not in seen_ids:
                                    seen_ids.add(row_id)
                                    ids.append(str(row_id))
                            pulled += len(batch)
                    finally:
                        try:
                            it.close()
                        except Exception:
                            pass
                except Exception as q_err:
                    logger.debug(
                        "delete_by_file_id: iterator probe failed (%s): %s",
                        field_expr[:72],
                        q_err,
                    )
                    try:
                        rows = self.collection.query(
                            expr=expr_full, output_fields=["id"], limit=16384
                        )
                        for row in rows or []:
                            row_id = row.get("id")
                            if row_id and row_id not in seen_ids:
                                seen_ids.add(row_id)
                                ids.append(str(row_id))
                    except Exception as q2:
                        logger.debug(
                            "delete_by_file_id: fallback query probe skipped (%s): %s",
                            field_expr[:72],
                            q2,
                        )

            for field_expr in candidate_exprs:
                _collect_ids_for_probe(field_expr)

            # Fallback: if exact match found nothing, do bounded user-scoped scan and
            # match common filename/file_id variations in Python.
            # This handles truncated UI values like "upload_XYZ..." or source/filename drift.
            if not ids and user_id:
                scope_expr_parts = [f'user_id == "{_esc(user_id)}"']
                if bucket_id:
                    scope_expr_parts.append(f'bucket_id == "{_esc(bucket_id)}"')
                if path:
                    scope_expr_parts.append(f'path == "{_esc(path)}"')
                if connection_id:
                    scope_expr_parts.append(f'connection_id == "{_esc(connection_id)}"')
                scope_expr = " && ".join(scope_expr_parts)

                norm_target = target.strip().lower()
                norm_target_dec = _url_unquote(target.strip()).strip().lower()
                bn_target = os.path.basename(norm_target_dec.split("?")[0]).strip().lower()
                max_scan = int(os.getenv("DELETE_SCAN_MAX_ROWS", "120000"))
                scan_fields = ["id", "source_file"]
                if not self.is_image_collection:
                    scan_fields.extend(
                        [
                            "file_id",
                            "filename",
                            "original_filename",
                            "chunk_id",
                            "object_id",
                        ]
                    )

                def _basename_key(s: str) -> str:
                    t = str(s or "").strip().lower()
                    if not t:
                        return ""
                    t = _url_unquote(t.split("?")[0])
                    return os.path.basename(t).strip().lower()

                def _consume_scan_rows(rows: List[Dict[str, Any]]) -> None:
                    for row in rows or []:
                        row_id = row.get("id")
                        if not row_id:
                            continue
                        candidates = [
                            str(row.get("source_file") or "").strip().lower(),
                            str(row.get("file_id") or "").strip().lower(),
                            str(row.get("filename") or "").strip().lower(),
                            str(row.get("original_filename") or "").strip().lower(),
                        ]
                        if not self.is_image_collection:
                            ck_full = str(row.get("chunk_id") or "").strip().lower()
                            if ck_full:
                                candidates.append(ck_full)
                                # Logical document key in upsert format: "{file_id}::chunk::..."
                                candidates.append(ck_full.split("::")[0])
                            candidates.append(str(row.get("object_id") or "").strip().lower())
                        exact_hit = any(
                            c == norm_target or c == norm_target_dec for c in candidates if c
                        )
                        basename_hit = bool(bn_target) and any(
                            _basename_key(c) == bn_target for c in candidates if c
                        )
                        prefix_hit = (
                            len(norm_target) >= 20
                            and any(
                                (c.startswith(norm_target) or norm_target.startswith(c))
                                for c in candidates
                                if c
                            )
                        )
                        chunk_prefix = (
                            not self.is_image_collection
                            and norm_target
                            and any(
                                c.startswith(norm_target + "::") or c.startswith(norm_target + "_")
                                for c in candidates
                                if c
                            )
                        )
                        if exact_hit or basename_hit or prefix_hit or chunk_prefix:
                            if row_id not in seen_ids:
                                seen_ids.add(row_id)
                                ids.append(str(row_id))

                scanned = 0
                try:
                    iterator = self.collection.query_iterator(
                        expr=scope_expr,
                        batch_size=2048,
                        limit=max_scan,
                        output_fields=scan_fields,
                    )
                    try:
                        while scanned < max_scan:
                            rows = iterator.next()
                            if not rows:
                                break
                            _consume_scan_rows(rows)
                            scanned += len(rows)
                    finally:
                        try:
                            iterator.close()
                        except Exception:
                            pass
                except Exception as iter_err:
                    logger.warning(
                        "delete_by_file_id: query_iterator fallback to single batch: %s",
                        iter_err,
                    )
                    rows = self.collection.query(
                        expr=scope_expr,
                        output_fields=scan_fields,
                        limit=min(8192, max_scan),
                    )
                    _consume_scan_rows(rows)

            count = len(ids)
            distinct_files: Set[str] = set()

            if count > 0:
                # Resolve distinct logical file ids before delete (for API reporting).
                meta_fields = ["source_file"]
                if not self.is_image_collection:
                    meta_fields.extend(["file_id", "object_id"])
                batch_size = 500
                for i in range(0, count, batch_size):
                    batch_ids = ids[i : i + batch_size]
                    escaped = [x.replace("\\", "\\\\").replace('"', '\\"') for x in batch_ids]
                    pk_expr = "id in [" + ", ".join(f'"{x}"' for x in escaped) + "]"
                    try:
                        meta_rows = self.collection.query(
                            expr=pk_expr,
                            output_fields=["id"] + [f for f in meta_fields if f != "id"],
                            limit=len(batch_ids) + 50,
                        )
                    except Exception as meta_err:
                        logger.debug("delete_by_file_id: meta query skipped: %s", meta_err)
                        meta_rows = []
                    for r in meta_rows or []:
                        fk = (
                            str(r.get("file_id") or "").strip()
                            or str(r.get("source_file") or "").strip()
                            or str(r.get("object_id") or "").strip()
                        )
                        if fk:
                            distinct_files.add(fk)
                if not distinct_files:
                    t = str(file_id or "").strip()
                    if t:
                        distinct_files.add(t)
                    elif count > 0:
                        distinct_files.add("_unknown")

                self._delete_by_primary_keys(ids, batch_size=batch_size)
                self.collection.flush()

                logger.info(
                    "Deleted file_id=%s user_id=%s: chunks=%s distinct_files=%s (flush ok)",
                    file_id,
                    user_id,
                    count,
                    len(distinct_files),
                )

            return {
                "chunks_deleted": count,
                "files_deleted": len(distinct_files),
                "distinct_file_ids": sorted(distinct_files)[:50],
            }

        except Exception as e:
            logger.error(f"Failed to delete by file_id: {e}", exc_info=True)
            return {"chunks_deleted": 0, "files_deleted": 0, "distinct_file_ids": []}

    def delete_all_for_user(
        self,
        user_id: str,
        bucket_id: Optional[str] = None,
        connection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete every row matching user_id, optionally scoped to bucket_id and/or connection_id.

        Pass bucket_id and/or connection_id to drop one S3 connection's vectors; omit both to remove
        all rows for the user in this collection. If both are set, both must match (AND).
        """
        uid = str(user_id or "").strip()
        if not uid:
            return {"chunks_deleted": 0, "error": "empty user_id"}
        if not self.use_milvus or not self.collection:
            return {"chunks_deleted": 0, "error": "no_collection"}
        if not self._ensure_collection_loaded():
            return {"chunks_deleted": 0, "error": "not_loaded"}

        def _esc(value: str) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"')

        parts = [f'user_id == "{_esc(uid)}"']
        bkt = str(bucket_id).strip() if bucket_id is not None else ""
        if bkt:
            parts.append(f'bucket_id == "{_esc(bkt)}"')
        conn = str(connection_id).strip() if connection_id is not None else ""
        if conn:
            parts.append(f'connection_id == "{_esc(conn)}"')
        expr = " && ".join(parts)
        ids: List[str] = []
        try:
            iterator = self.collection.query_iterator(
                expr=expr,
                batch_size=2048,
                limit=2_000_000,
                output_fields=["id"],
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        rid = row.get("id")
                        if rid is not None and str(rid).strip():
                            ids.append(str(rid))
            finally:
                try:
                    iterator.close()
                except Exception:
                    pass
        except Exception as cnt_err:
            logger.error("delete_all_for_user: collect PKs failed: %s", cnt_err, exc_info=True)
            return {"chunks_deleted": 0, "error": str(cnt_err), "expression": expr}

        if not ids:
            logger.info("delete_all_for_user: no rows matched expr=%s", expr[:120])
            return {"chunks_deleted": 0, "expression": expr}

        try:
            n = self._delete_by_primary_keys(ids)
            self.collection.flush()
            logger.info(
                "delete_all_for_user: deleted %s chunks (PK batches) expr=%s",
                n,
                expr[:120],
            )
            return {"chunks_deleted": n, "expression": expr}
        except Exception as e:
            logger.error("delete_all_for_user failed: %s", e, exc_info=True)
            return {"chunks_deleted": 0, "error": str(e)}

    def clear_collection(self) -> bool:
        if not self.use_milvus or not self.collection:
            logger.info("No active collection to clear")
            return True
        try:
            if not self._ensure_collection_loaded():
                logger.warning("clear_collection: collection not loaded")
                return False
            ids: List[str] = []
            iterator = self.collection.query_iterator(
                expr='id != ""',
                batch_size=2048,
                limit=10_000_000,
                output_fields=["id"],
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        rid = row.get("id")
                        if rid is not None and str(rid).strip():
                            ids.append(str(rid))
            finally:
                try:
                    iterator.close()
                except Exception:
                    pass
            if not ids:
                logger.info("clear_collection: collection already empty")
                return True
            self._delete_by_primary_keys(ids)
            self.collection.flush()
            logger.info("clear_collection: removed %s entities", len(ids))
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    
    def drop_collection(self) -> bool:
        """Drop the entire collection."""
        if not self.use_milvus:
            logger.info("Mock collection dropped")
            return True
        
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} dropped successfully")
                self.collection = None
                return True
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.use_milvus and MILVUS_AVAILABLE:
            try:
                connections.disconnect("default")
                logger.info("Disconnected from Milvus server")
            except Exception as e:
                logger.error(f"Failed to disconnect: {e}")

    def insert_vector(self, file_id: str, text: str, embedding: np.ndarray, user_id: str = None) -> bool:
        """
        Insert a single semantic text entry (used for both document text + image captions).
        Ensures that all searchable content is stored consistently in the document_chunks collection.
        """
        if self.is_image_collection:
            logger.error("insert_vector called on image collection — use insert_image_vector instead.")
            return False

        try:
            data = [{
                "id": file_id,
                "embedding": embedding.tolist(),
                "chunk_id": file_id,
                "text": text,
                "page_number": 0,
                "element_type": "semantic",
                "source_file": file_id,
                "created_at": str(int(time.time())),
                "object_id": file_id,
                "user_id": str(user_id) if user_id else "default_user"
            }]

            self.collection.insert(
                data,
                timeout=8,
                retry_times=2,
            )
            logger.info(f"[SEMANTIC] Inserted vector for {file_id} (flush deferred)")
            return True

        except Exception as e:
            logger.error(f"[SEMANTIC] Failed to insert semantic vector: {e}")
            return False
