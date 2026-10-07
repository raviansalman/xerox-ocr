"""
Semantic Search Module

Production-grade semantic search with:
- all-mpnet-base-v2 embeddings for documents
- Cross-encoder re-ranking
- File-level aggregation
- Query routing

NOTE:
This package's __init__ intentionally avoids importing heavy modules like
SemanticPipeline at import time to prevent circular imports with
vector_db_milvus_server. Import SemanticPipeline directly from
`src.semantic.semantic_pipeline` where needed.
"""

# Reranker is optional in lightweight environments (e.g., unit tests without
# torch/transformers). Make the import resilient so code can still run even
# when CrossEncoderReranker cannot be constructed.
try:  # pragma: no cover - exercised indirectly in tests
    from .semantic_components import CrossEncoderReranker
except Exception:  # noqa: BLE001 - broad on purpose
    CrossEncoderReranker = None

__all__ = [
    "CrossEncoderReranker",
]
