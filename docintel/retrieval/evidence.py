"""Evidence: turn fused document results into results that prove themselves.

For each result, up to three pieces of evidence are chosen (best match types first). Every piece is re-read from the
stored canonical text, checked against what the retriever claimed (an identifier match must contain the identifier,
a phrase match the phrase), and reported with page, character offsets in the page text, the window shown, the match
type and the retriever. Evidence that fails the check is dropped; a result left without evidence keeps only its
structured or semantic justification and is labelled accordingly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg

from docintel import text as T
from docintel.retrieval.contracts import Candidate
from docintel.retrieval.fusion import _PRIORITY, DocumentResult, _tier

logger = logging.getLogger(__name__)
WINDOW = 360


@dataclass
class _Unit:
    id: str
    document_id: str
    page: int
    char_start: int | None
    char_end: int | None
    text: str
    context: str
    unit_type: str


def _window(text: str, terms: list[str], size: int = WINDOW) -> tuple[int, int]:
    """(start, end) of a window of ``text`` around the first occurrence of any term."""
    low = text.lower()
    pos = min((low.find(t.lower()) for t in terms if t and low.find(t.lower()) >= 0), default=0)
    start = max(0, pos - size // 3)
    while 0 < start < len(text) and not text[start - 1].isspace():
        start += 1
    return start, min(len(text), start + size)


def _verified(c: Candidate, unit_text: str) -> bool:
    """Lexical evidence must contain what matched (checked on normalized forms, as indexed)."""
    if c.match_type == "exact_identifier":
        found = T.identifiers(unit_text)
        return any(i in found for i in c.matched)
    if c.match_type == "exact_phrase":
        st = f" {T.search_text(unit_text)} "
        return all(f" {T.search_text(m)} " in st for m in c.matched)
    if c.match_type == "exact_term":
        return all(f" {T.search_text(m)} " in f" {T.search_text(unit_text)} " for m in c.matched)
    return True


def build(conn: psycopg.Connection, results: list[DocumentResult], terms: list[str], max_evidence: int = 3
          ) -> dict[str, list[dict]]:
    """document_id -> evidence items, verified against the stored text."""
    unit_ids = sorted({c.unit_id for r in results for c in r.candidates if c.unit_id})
    units: dict[str, _Unit] = {}
    if unit_ids:
        for row in conn.execute("SELECT id, document_id, page_start, char_start, char_end, text, context, unit_type "
                                "FROM chunks WHERE id = ANY(%s)", (unit_ids,)).fetchall():
            units[row["id"]] = _Unit(row["id"], str(row["document_id"]), row["page_start"], row["char_start"],
                                     row["char_end"], row["text"], row["context"] or "", row["unit_type"])
    # a document-unit match (title, type, file name, key fields) is shown with the document's first passage, so
    # evidence is always source text
    summary_docs = sorted({u.document_id for u in units.values() if u.unit_type == "document"})
    first_passage: dict[str, _Unit] = {}
    if summary_docs:
        for row in conn.execute("SELECT DISTINCT ON (document_id) id, document_id, page_start, char_start, char_end, text, "
                                "context, unit_type FROM chunks WHERE document_id = ANY(%s::uuid[]) AND unit_type = 'passage' "
                                "ORDER BY document_id, ordinal", (summary_docs,)).fetchall():
            first_passage[str(row["document_id"])] = _Unit(row["id"], str(row["document_id"]), row["page_start"],
                                                           row["char_start"], row["char_end"], row["text"],
                                                           row["context"] or "", row["unit_type"])
    page_keys = {(r.document_id, s.page) for r in results for c in r.candidates for s in c.spans}
    pages: dict[tuple[str, int], str] = {}
    if page_keys:                                     # only the pages that evidence points at
        keys = sorted(page_keys)
        for row in conn.execute("SELECT p.document_id, p.page_number, p.text FROM pages p JOIN unnest(%s::uuid[], %s::int[]) "
                                "AS k(d, n) ON p.document_id = k.d AND p.page_number = k.n",
                                ([d for d, _ in keys], [n for _, n in keys])).fetchall():
            pages[(str(row["document_id"]), row["page_number"])] = row["text"]
    out: dict[str, list[dict]] = {}
    for r in results:
        items: list[dict] = []
        used: set[tuple] = set()
        def order(c: Candidate) -> tuple:
            summary = c.unit_id in units and units[c.unit_id].unit_type == "document"
            return (summary, _tier(c.match_type), _PRIORITY.get(c.match_type, 5), -c.raw_score)

        for c in sorted(r.candidates, key=order):
            if len(items) >= max_evidence:
                break
            if c.spans:
                for s in c.spans:
                    text = pages.get((r.document_id, s.page))
                    if text is None or s.char_start is None or s.char_end is None or s.char_end > len(text):
                        continue
                    key = (s.page, s.char_start)
                    if key in used:
                        continue
                    used.add(key)
                    a = max(0, s.char_start - 60)
                    while 0 < a < s.char_start and not text[a - 1].isspace():
                        a += 1
                    b = min(len(text), max(s.char_end, a + WINDOW))
                    items.append({"page": s.page, "char_start": a, "char_end": b, "match_start": s.char_start,
                                  "match_end": s.char_end, "text": text[a:b], "match_type": c.match_type,
                                  "retriever": c.retriever})
                    break
                continue
            u = units.get(c.unit_id or "")
            if u is None or u.document_id != r.document_id:
                continue
            if not _verified(c, f"{u.context} {u.text}"):
                logger.warning("evidence failed verification", extra={"match_type": c.match_type, "unit": u.id})
                continue
            if u.unit_type == "document":                # show a passage; a document without one shows its summary
                u = first_passage.get(r.document_id) or u
            if u.id in used:
                continue
            used.add(u.id)
            ws, we = _window(u.text, terms)
            item = {"page": u.page, "text": u.text[ws:we], "match_type": c.match_type, "retriever": c.retriever,
                    "chunk_id": u.id, "unit_type": u.unit_type}
            if u.char_start is not None:
                item.update(char_start=u.char_start + ws, char_end=u.char_start + we)
            items.append(item)
        out[r.document_id] = items
    return out
