"""Candidate fusion: candidates from every retriever are grouped by document and ranked.

Ranking keys, in order:
1. Evidence tier. Tier 1 (exact phrase, identifier, file name, structured match) always comes first. For questions
   that look like exact lookups (identifiers, quotes, short keyword queries) the remaining tiers are kept strictly
   apart; for natural-language questions, strong evidence (all words, entities, clauses, relations) and strong
   semantic similarity compete on fused rank, followed by tolerant matches and weak similarity.
2. Reciprocal rank fusion of the lexical and the semantic rank, weighted by query type, plus a small bonus per
   additional independent retriever that agrees and a preference for the question's soft document types.

Rules that remove documents:
* a query made only of identifiers returns only documents that contain the identifier (exactly, in part, OCR-tolerant,
  in the file name or in an extracted field): word overlap ("INV" and "2024" of another invoice number), typo
  correction and embeddings do not count, since a code that differs by one digit is a different document;
* a quoted phrase must appear (exactly, OCR-tolerant or as a file name);
* typo-corrected matches are used only when nothing else matched lexically;
* semantic-only documents below the calibrated floor never reach this module (SemanticRetriever).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from docintel.retrieval.contracts import Candidate
from docintel.search.plan import QueryPlan

RRF_K = 60
AGREEMENT_BONUS = 0.0015
SOFT_TYPE_BONUS = 0.006
_PHRASE_EVIDENCE = {"exact_phrase", "exact_identifier", "ocr_tolerant", "fuzzy", "filename"}
_TOLERANT = {"ocr_tolerant", "fuzzy", "most_terms"}
_IDENTIFIER_EVIDENCE = {"exact_identifier", "exact_phrase", "exact_term", "joined_form", "partial_identifier", "filename",
                        "ocr_tolerant", "structured", "entity"}
# order of lexical evidence inside a tier (lower first)
_PRIORITY = {"exact_phrase": 0, "exact_identifier": 0, "filename": 0, "structured": 0, "exact_term": 1, "joined_form": 1,
             "metadata": 1, "entity": 1, "relation": 1, "concept_clause": 1, "partial_identifier": 2, "all_terms": 2,
             "concept": 2, "filename_terms": 2, "ocr_tolerant": 3, "most_terms": 3, "fuzzy": 4}


@dataclass
class DocumentResult:
    document_id: str
    candidates: list[Candidate] = field(default_factory=list)
    lex_rank: int | None = None
    vec_rank: int | None = None
    vec_score: float = 0.0
    score: float = 0.0
    rank_key: int = 0                # fusion group (1 exact, 2 strong, 3 tolerant or weak); set by fuse()
    rerank_score: float | None = None

    @property
    def tier(self) -> int:
        return min((c.tier for c in self.candidates), default=4)

    @property
    def match_types(self) -> list[str]:
        mts = {c.match_type for c in self.candidates}
        return sorted(mts, key=lambda m: (_tier(m), _PRIORITY.get(m, 5), m))

    @property
    def retrievers(self) -> list[str]:
        return sorted({c.retriever for c in self.candidates})


def _tier(match_type: str) -> int:
    from docintel.retrieval.contracts import TIERS
    return TIERS.get(match_type, 4)


def identifier_only(plan: QueryPlan) -> bool:
    """True when every word of the query is an identifier or a code-like number ("INV-2026-1", "00481")."""
    words = plan.text.split()
    if not words or plan.phrases:
        return False
    return all(re.sub(r"[^0-9a-z]", "", w.casefold()) in plan.identifiers or
               (re.fullmatch(r"\d{4,}", w) and not re.fullmatch(r"(?:19|20)\d{2}", w)) for w in words)


_CONCEPTUAL = {"concept", "concept_clause", "relation", "semantic"}


def fuse(plan: QueryPlan, candidates: list[Candidate], doc_types: dict[str, str | None] | None = None,
         strong_semantic: float = 1.0, unmatched_terms: bool = False) -> list[DocumentResult]:
    """Rank documents. ``strong_semantic`` is the similarity at which a semantic-only match counts as strong evidence
    (the model's floor plus a margin); weaker semantic matches rank after strong lexical, entity and contextual
    evidence. ``unmatched_terms``: the question uses words that occur in none of the tenant's documents (other than
    the words that named a concept), so a document matched only by concept or by weak similarity is about something
    else (a question about another company's merger is not answered by a confidentiality clause) and is dropped."""
    lexical = [c for c in candidates if c.match_type != "semantic"]
    if any(c.match_type != "fuzzy" for c in lexical):
        candidates = [c for c in candidates if c.match_type != "fuzzy"]
        lexical = [c for c in lexical if c.match_type != "fuzzy"]
    docs: dict[str, DocumentResult] = {}
    for c in candidates:
        docs.setdefault(c.document_id, DocumentResult(c.document_id)).candidates.append(c)

    order = list(dict.fromkeys(c.document_id for c in sorted(
        lexical, key=lambda c: (c.tier, _PRIORITY.get(c.match_type, 5), -c.raw_score))))
    for i, d in enumerate(order):
        docs[d].lex_rank = i
    seen: set[str] = set()
    for c in sorted((c for c in candidates if c.match_type == "semantic"), key=lambda c: -c.raw_score):
        if c.document_id not in seen:
            docs[c.document_id].vec_rank = len(seen)
            docs[c.document_id].vec_score = c.raw_score
            seen.add(c.document_id)

    if identifier_only(plan):
        docs = {k: d for k, d in docs.items() if {c.match_type for c in d.candidates} & _IDENTIFIER_EVIDENCE}
    elif plan.phrases:
        docs = {k: d for k, d in docs.items() if {c.match_type for c in d.candidates} & _PHRASE_EVIDENCE}

    if unmatched_terms:
        docs = {k: d for k, d in docs.items()
                if not {c.match_type for c in d.candidates} <= _CONCEPTUAL or d.vec_score >= strong_semantic}

    exact_query = plan.exact_intent
    w_lex, w_vec = (1.0, 0.6) if exact_query else (0.7, 1.0)
    soft = set(plan.filters.doc_types) if plan.filters.doc_types and not plan.filters.doc_types_hard else set()
    ranked = []
    for d in docs.values():
        s = 0.0
        if d.lex_rank is not None:
            s += w_lex / (RRF_K + d.lex_rank)
        if d.vec_rank is not None:
            s += w_vec / (RRF_K + d.vec_rank)
        s += AGREEMENT_BONUS * (len(d.retrievers) - 1)
        if soft and (doc_types or {}).get(d.document_id) in soft:
            s += SOFT_TYPE_BONUS
        d.score = s
        tier = d.tier
        if exact_query or tier == 1:
            key = tier
        elif tier == 2 or d.vec_score >= strong_semantic:
            key = 2                              # strong lexical/contextual/entity evidence and strong similarity compete
        else:
            key = 3                              # tolerant matches and weak similarity follow
        d.rank_key = key
        ranked.append((key, -s, d))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [r[2] for r in ranked]
