#!/usr/bin/env python3
"""
M4 - Embeddings (CPU Only)

This module handles text embedding generation using SentenceTransformers
with CPU optimization and ONNX runtime for better performance.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import time
import hashlib
from pathlib import Path
import os

# Heavy import moved inside __init__ to save RAM when using remote API
# from sentence_transformers import SentenceTransformer
# onnxruntime is optional - not required for basic embedding functionality
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("onnxruntime not available - ONNX optimization will be disabled")
from src.semantic.semantic_components import TextChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Represents the result of embedding generation."""
    embeddings: np.ndarray
    chunk_ids: List[str]
    model_name: str
    embedding_dimension: int
    total_chunks: int
    processing_time: float
    batch_size: int
    
    @property
    def embedding_shape(self) -> tuple:
        """Get the shape of the embeddings array."""
        return self.embeddings.shape


class EmbeddingGenerator:
    """Handles text embedding generation with CPU optimization."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions
        batch_size: int = 64,
        use_onnx: bool = True,
        cache_dir: Optional[str] = None,
        embedder_url: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the SentenceTransformers model
            batch_size: Batch size for embedding generation
            use_onnx: Whether to use ONNX runtime for optimization
            cache_dir: Directory to cache models and embeddings
            embedder_url: Override for remote embedder (default: EMBEDDER_URL).
                Use EMBEDDER_SEARCH_URL for search to avoid ingestion queue.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_onnx = use_onnx
        self.cache_dir = cache_dir or "cache"
        self._embedder_url_override = embedder_url
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.embedding_dimension = None
        self._initialize_model()
        
        logger.info(f"Initialized embedding generator: {model_name}")
        logger.info(f"Batch size: {batch_size}, ONNX: {use_onnx}")
    
    def _initialize_model(self):
        """Initialize the SentenceTransformers model."""
        import os
        self.embedder_url = self._embedder_url_override or os.environ.get("EMBEDDER_URL")
        
        # Hardcode the dimension for all-mpnet-base-v2 since we are offloading
        self.embedding_dimension = 768

        if self.embedder_url:
            logger.info(f"Using remote embedder API at {self.embedder_url}")
            return
            
        try:
            # Load model with CPU optimization
            self.model = SentenceTransformer(
                self.model_name,
                device='cpu',  # Force CPU usage
                cache_folder=self.cache_dir,
            )
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # Optimize with ONNX if requested
            if self.use_onnx:
                try:
                    # Convert to ONNX for better CPU performance
                    onnx_path = Path(self.cache_dir) / f"{self.model_name.replace('/', '_')}_onnx"
                    if not onnx_path.exists():
                        logger.info("Converting model to ONNX format...")
                        self.model.save(str(onnx_path))
                        logger.info(f"ONNX model saved to: {onnx_path}")
                    
                    # Load ONNX model
                    self.model = SentenceTransformer(str(onnx_path), device='cpu')
                    logger.info("Using ONNX optimized model")
                    
                except Exception as e:
                    logger.warning(f"ONNX optimization failed, using standard model: {e}")
            
            logger.info(f"Model loaded: {self.embedding_dimension} dimensions")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, np.linalg.norm(embeddings, axis=1, keepdims=True))
        return embeddings / norms

    def _generate_embedding_hash(self, text: str) -> str:
        """Generate SHA256 hash for text to use as cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_embedding_cache(self, cache_file: str) -> Dict[str, np.ndarray]:
        """Load embedding cache from file."""
        try:
            if Path(cache_file).exists():
                with open(cache_file, "rb") as f:
                    cache_data = np.load(f, allow_pickle=True)
                    return dict(cache_data.item())
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
        return {}

    def _save_embedding_cache(self, cache: Dict[str, np.ndarray], cache_file: str):
        """Save embedding cache to file."""
        try:
            with open(cache_file, "wb") as f:
                np.save(f, cache)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        if not texts:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        start_time = time.time()

        cache: Dict[str, np.ndarray] = {}
        cache_file = Path(self.cache_dir) / "embeddings_cache.npy"
        if use_cache:
            cache = self._load_embedding_cache(str(cache_file))

        all_embeddings: List[np.ndarray] = []
        texts_to_embed: List[str] = []
        text_indices: List[int] = []

        for i, text in enumerate(texts):
            if use_cache:
                text_hash = self._generate_embedding_hash(text)
                if text_hash in cache:
                    all_embeddings.append(cache[text_hash])
                    continue

            texts_to_embed.append(text)
            text_indices.append(i)

        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts")
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch_texts = texts_to_embed[i : i + self.batch_size]

                if hasattr(self, "embedder_url") and self.embedder_url:
                    import requests
                    max_retries = int(os.getenv("EMBEDDER_MAX_RETRIES", "5"))
                    last_err = None
                    for attempt in range(max_retries):
                        try:
                            resp = requests.post(
                                self.embedder_url,
                                json={"texts": batch_texts},
                                timeout=int(os.getenv("EMBEDDER_TIMEOUT", "180")),
                            )
                            resp.raise_for_status()
                            batch_embeddings = np.array(resp.json()["embeddings"], dtype=np.float32)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            if attempt < max_retries - 1:
                                backoff = min(2 ** attempt, 30)
                                logger.warning(
                                    f"Embedder API attempt {attempt + 1}/{max_retries} failed: {e}. "
                                    f"Retrying in {backoff}s..."
                                )
                                time.sleep(backoff)
                            else:
                                logger.error(f"Failed to generate embeddings via remote API after {max_retries} attempts: {e}")
                                raise last_err
                else:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=min(self.batch_size, len(batch_texts)),
                    )
                    batch_embeddings = self._normalize_embeddings(batch_embeddings)

                if use_cache:
                    for j, text in enumerate(batch_texts):
                        text_hash = self._generate_embedding_hash(text)
                        cache[text_hash] = batch_embeddings[j]

                all_embeddings.extend(batch_embeddings)

        if use_cache and cache:
            self._save_embedding_cache(cache, str(cache_file))

        embeddings = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
        for i, emb in enumerate(all_embeddings):
            embeddings[i] = emb

        processing_time = time.time() - start_time
        logger.info(f"Generated {len(texts)} embeddings in {processing_time:.2f}s")
        return embeddings

    def embed_chunks(
        self,
        chunks: List[TextChunk],
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Generate embeddings for a list of TextChunk objects.

        Args:
            chunks: List of TextChunk objects to embed.
            use_cache: Whether to use the on-disk embedding cache.

        Returns:
            EmbeddingResult containing the embedding matrix and metadata.
        """
        if not chunks:
            return EmbeddingResult(
                embeddings=np.zeros((0, self.embedding_dimension), dtype=np.float32),
                chunk_ids=[],
                model_name=self.model_name,
                embedding_dimension=self.embedding_dimension,
                total_chunks=0,
                processing_time=0.0,
                batch_size=self.batch_size,
            )

        start_time = time.time()
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = self.embed_texts(texts, use_cache=use_cache)
        processing_time = time.time() - start_time

        result = EmbeddingResult(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            total_chunks=len(chunks),
            processing_time=processing_time,
            batch_size=self.batch_size,
        )
        logger.info(f"Embedded {len(chunks)} chunks: {embeddings.shape}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding for a single query string (no cache)."""
        embeddings = self.embed_texts([query], use_cache=False)
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.embedding_dimension, dtype=np.float32)


# ---------------------------------------------------------------------------
# Process-wide singleton accessors so all components share the same embedder
# within a given process (API worker or Celery worker). This guarantees that
# SentenceTransformer is only loaded once per process.
# ---------------------------------------------------------------------------

_GLOBAL_EMBEDDERS: Dict[str, EmbeddingGenerator] = {}


def get_global_embedding_generator(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    use_onnx: bool = True,
) -> EmbeddingGenerator:
    """Return a process-wide EmbeddingGenerator instance for the given model.

    Ensures all callers in a given process share the same underlying
    SentenceTransformer instance instead of each module loading its own copy.
    """
    # Prefer SENTENCE_TRANSFORMERS_HOME when set (inside containers),
    # otherwise fall back to a local writable directory for host-side tests.
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or "cache"
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "24"))
    key = f"{model_name}|{cache_dir}|onnx={use_onnx}|bs={batch_size}"
    if key not in _GLOBAL_EMBEDDERS:
        logger.info(f"[EMBEDDINGS] Creating global EmbeddingGenerator for {model_name} (cache={cache_dir}, batch_size={batch_size})")
        _GLOBAL_EMBEDDERS[key] = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size,
            use_onnx=use_onnx,
            cache_dir=cache_dir,
        )
    return _GLOBAL_EMBEDDERS[key]


def get_search_embedding_generator(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    use_onnx: bool = True,
) -> EmbeddingGenerator:
    """Embedder for search queries. Uses EMBEDDER_SEARCH_URL when set to avoid
    queueing behind ingestion. Falls back to default embedder otherwise."""
    search_url = os.getenv("EMBEDDER_SEARCH_URL", "").strip()
    if not search_url:
        return get_global_embedding_generator(model_name=model_name, use_onnx=use_onnx)
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or "cache"
    batch_size = 1  # Search typically embeds 1 query at a time
    key = f"search|{model_name}|{search_url}|{cache_dir}"
    if key not in _GLOBAL_EMBEDDERS:
        logger.info(f"[EMBEDDINGS] Creating search embedder at {search_url}")
        _GLOBAL_EMBEDDERS[key] = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size,
            use_onnx=use_onnx,
            cache_dir=cache_dir,
            embedder_url=search_url,
        )
    return _GLOBAL_EMBEDDERS[key]




