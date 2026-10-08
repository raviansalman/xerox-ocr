"""Vector index. Milvus in production; an in-memory store for tests; ``disabled`` for deployments without
semantic search (exact, lexical and structured retrieval do not depend on it).

Only identifiers and the embedding live here (chunk text and metadata are in PostgreSQL). Every search is
filtered by tenant inside the index, and all filter values are validated and quoted, never interpolated raw.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from docintel.config import get_settings
from docintel.model_registry import ModelSpec, model_spec
from docintel.security import is_valid_tenant_id

logger = logging.getLogger(__name__)
_UUID = re.compile(r"^[0-9a-f-]{36}$")
_LABEL = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass
class VectorHit:
    chunk_id: str
    document_id: str
    score: float


class VectorStore(Protocol):
    def upsert(self, tenant_id: str, document_id: str, doc_type: str, chunk_ids: list[str], vectors: np.ndarray) -> None: ...
    def delete_document(self, tenant_id: str, document_id: str) -> None: ...
    def search(self, tenant_id: str, vector: np.ndarray, k: int, document_ids: list[str] | None = None,
               doc_types: list[str] | None = None) -> list[VectorHit]: ...
    def ping(self) -> None: ...


def _check(tenant_id: str, document_ids: list[str] | None = None, doc_types: list[str] | None = None) -> None:
    if not is_valid_tenant_id(tenant_id):
        raise ValueError("invalid tenant id")
    for d in document_ids or []:
        if not _UUID.match(d):
            raise ValueError("invalid document id")
    for t in doc_types or []:
        if not _LABEL.match(t):
            raise ValueError("invalid document type")


class MilvusVectorStore:
    def __init__(self, uri: str, spec: ModelSpec, prefix: str, token: str | None = None, consistency: str = "Bounded"):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token or "")
        self.spec = spec
        slug = re.sub(r"[^a-z0-9]+", "_", spec.key.lower()).strip("_")
        self.collection = f"{prefix}_{slug}_{spec.dimension}"
        self.consistency = consistency
        self._ready = False
        self._lock = threading.Lock()

    def ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            from pymilvus import DataType

            if not self.client.has_collection(self.collection):
                schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
                schema.add_field("tenant_id", DataType.VARCHAR, max_length=128, is_partition_key=True)
                schema.add_field("document_id", DataType.VARCHAR, max_length=64)
                schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.spec.dimension)
                idx = self.client.prepare_index_params()
                idx.add_index("embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
                idx.add_index("document_id", index_type="INVERTED")
                self.client.create_collection(self.collection, schema=schema, index_params=idx,
                                              consistency_level=self.consistency)
                logger.info("created vector collection", extra={"collection": self.collection})
            self.client.load_collection(self.collection)
            self._ready = True

    def upsert(self, tenant_id, document_id, doc_type, chunk_ids, vectors):
        _check(tenant_id, [document_id], [doc_type])
        self.ensure()
        rows = [{"id": cid, "tenant_id": tenant_id, "document_id": document_id, "doc_type": doc_type,
                 "embedding": v.tolist()} for cid, v in zip(chunk_ids, vectors, strict=True)]
        for i in range(0, len(rows), 1000):                # upsert by unit id: concurrent writers converge
            self.client.upsert(self.collection, rows[i:i + 1000])
        stale = f"tenant_id == {json.dumps(tenant_id)} and document_id == {json.dumps(document_id)}"
        if chunk_ids:
            stale += f" and id not in {json.dumps(list(chunk_ids))}"
        self.client.delete(self.collection, filter=stale)    # units of an earlier version

    def delete_document(self, tenant_id, document_id):
        _check(tenant_id, [document_id])
        self.ensure()
        self.client.delete(self.collection,
                           filter=f"tenant_id == {json.dumps(tenant_id)} and document_id == {json.dumps(document_id)}")

    def search(self, tenant_id, vector, k, document_ids=None, doc_types=None):
        _check(tenant_id, document_ids, doc_types)
        self.ensure()
        expr = f"tenant_id == {json.dumps(tenant_id)}"
        if document_ids is not None:
            if not document_ids:
                return []
            expr += f" and document_id in {json.dumps(document_ids)}"
        if doc_types:
            expr += f" and doc_type in {json.dumps(doc_types)}"
        res = self.client.search(self.collection, data=[vector.tolist()], anns_field="embedding", limit=k, filter=expr,
                                 output_fields=["document_id"], search_params={"metric_type": "COSINE", "params": {"ef": max(k, 128)}},
                                 consistency_level=self.consistency)
        return [VectorHit(h["id"], h["entity"]["document_id"], float(h["distance"])) for h in res[0]]

    def ping(self) -> None:
        self.client.list_collections()

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self._ready = False


class MemoryVectorStore:
    """Exact cosine search in process memory. For tests and demos only (not persistent, single process)."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.rows: dict[str, tuple[str, str, str, np.ndarray]] = {}
        self._lock = threading.Lock()

    def upsert(self, tenant_id, document_id, doc_type, chunk_ids, vectors):
        _check(tenant_id, [document_id], [doc_type])
        with self._lock:
            self.delete_document(tenant_id, document_id)
            for cid, v in zip(chunk_ids, vectors, strict=False):
                self.rows[cid] = (tenant_id, document_id, doc_type, np.asarray(v, dtype=np.float32))

    def delete_document(self, tenant_id, document_id):
        for cid in [c for c, r in self.rows.items() if r[0] == tenant_id and r[1] == document_id]:
            del self.rows[cid]

    def search(self, tenant_id, vector, k, document_ids=None, doc_types=None):
        _check(tenant_id, document_ids, doc_types)
        allowed = set(document_ids) if document_ids is not None else None
        hits = []
        for cid, (t, d, dt, v) in list(self.rows.items()):
            if t != tenant_id or (allowed is not None and d not in allowed) or (doc_types and dt not in doc_types):
                continue
            hits.append(VectorHit(cid, d, float(np.dot(v, vector))))
        return sorted(hits, key=lambda h: -h.score)[:k]

    def ping(self) -> None:
        return None


class DisabledVectorStore:
    """Semantic search switched off (DOCINTEL_VECTOR_BACKEND=disabled): writes are skipped, searches find nothing."""

    enabled = False

    def upsert(self, tenant_id, document_id, doc_type, chunk_ids, vectors):
        return None

    def delete_document(self, tenant_id, document_id):
        return None

    def search(self, tenant_id, vector, k, document_ids=None, doc_types=None):
        return []

    def ping(self) -> None:
        return None


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        s = get_settings()
        spec = model_spec(s.embedding_model)
        if s.vector_backend == "disabled":
            _store = DisabledVectorStore()
        elif s.vector_backend == "memory":
            _store = MemoryVectorStore(spec.dimension)
        else:
            _store = MilvusVectorStore(s.require("milvus_uri"), spec, s.milvus_collection_prefix, s.milvus_token,
                                       s.milvus_consistency)
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    global _store
    _store = store
