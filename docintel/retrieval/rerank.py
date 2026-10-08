"""Optional cross-encoder reranking of the fused results (off unless DOCINTEL_RERANKER_MODEL is set).

A cross-encoder reads the question and a document's best passage together, which judges relevance better than
comparing two embeddings, at the cost of one model call per result. It only reorders: documents are never added or
removed, the top ``rerank_top_n`` are considered, and exact evidence keeps its precedence because reordering happens
only among documents of the same fusion group outside the exact tier. When the reranker fails the fused order
stands and the query reports the degradation.
"""
from __future__ import annotations

import threading

import httpx
import psycopg

from docintel.config import get_settings
from docintel.retrieval.fusion import _PRIORITY, DocumentResult, _tier

MAX_TEXT_CHARS = 1500


class RerankError(RuntimeError):
    pass


class RerankClient:
    def __init__(self, base_url: str, key: str, timeout: float = 10.0):
        self.base_url, self.key = base_url.rstrip("/"), key
        self._http = httpx.Client(timeout=timeout)
        self._checked = False

    def verify(self) -> None:
        if self._checked:
            return
        info = self._http.get(f"{self.base_url}/info").raise_for_status().json()
        if info.get("reranker") != self.key:
            raise RerankError(f"embedding service serves reranker {info.get('reranker')!r}, configured {self.key!r}")
        self._checked = True

    def scores(self, query: str, texts: list[str]) -> list[float]:
        self.verify()
        try:
            r = self._http.post(f"{self.base_url}/rerank", json={"query": query, "texts": texts})
            r.raise_for_status()
            out = [float(x) for x in r.json()["scores"]]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise RerankError(f"reranker unavailable: {e}") from e
        if len(out) != len(texts):
            raise RerankError("reranker returned the wrong number of scores")
        return out


_client: RerankClient | None = None
_lock = threading.Lock()


def get_reranker() -> RerankClient | None:
    global _client
    s = get_settings()
    if not s.reranker_model or not s.embedder_url:
        return None
    with _lock:
        if _client is None or _client.key != s.reranker_model:
            _client = RerankClient(s.embedder_url, s.reranker_model)
        return _client


def reset_reranker() -> None:
    global _client
    _client = None


def _best_unit(d: DocumentResult) -> str | None:
    units = [c for c in d.candidates if c.unit_id]
    units.sort(key=lambda c: (_tier(c.match_type), _PRIORITY.get(c.match_type, 5), -c.raw_score))
    return units[0].unit_id if units else None


def rerank(conn: psycopg.Connection, question: str, ranked: list[DocumentResult], client: RerankClient,
           top_n: int) -> list[DocumentResult]:
    """Reorder the head of ``ranked`` within each fusion group (never across groups, never the exact tier)."""
    head, tail = ranked[:top_n], ranked[top_n:]
    best = {d.document_id: _best_unit(d) for d in head if d.rank_key > 1}
    unit_ids = sorted({u for u in best.values() if u})
    if not unit_ids:
        return ranked
    rows = conn.execute("SELECT id, text, context FROM chunks WHERE id = ANY(%s)", (unit_ids,)).fetchall()
    texts = {r["id"]: (f"{r['context']}\n{r['text']}" if r["context"] else r["text"])[:MAX_TEXT_CHARS] for r in rows}
    scored = [d for d in head if texts.get(best.get(d.document_id) or "")]
    if len(scored) < 2:
        return ranked
    values = client.scores(question, [texts[best[d.document_id]] for d in scored])
    score = {d.document_id: v for d, v in zip(scored, values, strict=True)}
    for d in scored:
        d.rerank_score = score[d.document_id]
    out = list(head)
    for key in sorted({d.rank_key for d in scored}):
        slots = [i for i, d in enumerate(out) if d.rank_key == key and d.document_id in score]
        group = sorted((out[i] for i in slots), key=lambda d: -score[d.document_id])
        for i, d in zip(slots, group, strict=True):
            out[i] = d
    return out + tail


__all__ = ["RerankClient", "RerankError", "get_reranker", "rerank", "reset_reranker"]
