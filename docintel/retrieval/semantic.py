"""Semantic retrieval: the question's embedding against the vector index (passages and document units).

Candidates must clear the embedding model's calibrated similarity floor and stay within ``SEMANTIC_MARGIN`` of the
best match, so an unrelated question returns nothing instead of the "closest" documents. When semantic search is
disabled (DOCINTEL_VECTOR_BACKEND=disabled) the retriever returns nothing and says so; no other retriever depends on
it.
"""
from __future__ import annotations

from docintel.config import get_settings
from docintel.indexing.embeddings import get_embedder
from docintel.indexing.vectors import get_vector_store
from docintel.model_registry import model_spec
from docintel.retrieval.contracts import Budget, Candidate, RetrievalContext, Scope
from docintel.search.plan import QueryPlan

SEMANTIC_MARGIN = 0.22


class SemanticUnavailable(RuntimeError):
    """Semantic search is switched off for this deployment."""


class SemanticRetriever:
    name = "semantic"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        s = get_settings()
        if not s.semantic_enabled:
            raise SemanticUnavailable("semantic search is disabled")
        text = plan.semantic_text if (plan.text or plan.phrases) else ""
        if not text.strip() or (scope.document_ids is not None and not scope.document_ids):
            return []
        vec = get_embedder().embed_query(text)
        ids = list(scope.document_ids) if scope.document_ids is not None and len(scope.document_ids) <= s.structured_id_limit else None
        hits = get_vector_store().search(rc.auth.tenant_id, vec, s.vector_top_k, document_ids=ids)
        if scope.document_ids is not None and ids is None:
            hits = [h for h in hits if h.document_id in scope.document_ids]
        if not hits:
            return []
        floor = max(model_spec(s.embedding_model).semantic_floor, max(h.score for h in hits) - SEMANTIC_MARGIN)
        return [Candidate(h.document_id, h.chunk_id, self.name, "semantic", h.score, [], (text,))
                for h in hits if h.score >= floor]
