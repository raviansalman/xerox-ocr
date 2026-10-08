"""The retrieval contract (docs/design/RETRIEVAL_CONTRACTS.md).

Every retrieval mechanism implements ``Retriever.retrieve`` and returns ``Candidate`` objects: a unit of one document
with the evidence that made it match. Retrievers run independently and never call each other; the tenant comes only
from the request's ``RetrievalContext`` (a tenant-scoped connection and the authenticated identity).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import psycopg

from docintel.packs import Domain
from docintel.search.plan import QueryPlan
from docintel.security import AuthContext

# Evidence quality, best first. Tier 1 can never be outranked by similarity; see docintel/retrieval/fusion.py.
TIERS: dict[str, int] = {
    "exact_phrase": 1, "exact_identifier": 1, "filename": 1, "structured": 1,
    "exact_term": 2, "joined_form": 2, "all_terms": 2, "concept": 2, "concept_clause": 2, "relation": 2,
    "entity": 2, "metadata": 2, "filename_terms": 2,
    "partial_identifier": 2, "ocr_tolerant": 3, "fuzzy": 3, "most_terms": 3,
    "semantic": 4,
}


@dataclass(frozen=True)
class EvidenceSpan:
    page: int
    char_start: int | None
    char_end: int | None
    text: str | None = None            # filled by evidence verification from the canonical page text


@dataclass
class Candidate:
    document_id: str
    unit_id: str | None
    retriever: str
    match_type: str
    raw_score: float                   # retriever-specific; never compared across retrievers
    spans: list[EvidenceSpan] = field(default_factory=list)
    matched: tuple[str, ...] = ()      # query parts this candidate satisfies

    @property
    def tier(self) -> int:
        return TIERS.get(self.match_type, 4)


@dataclass(frozen=True)
class Scope:
    """Documents a retriever may return: None means all of the tenant's indexed documents."""
    document_ids: tuple[str, ...] | None = None

    def allows(self, document_id: str) -> bool:
        return self.document_ids is None or document_id in self.document_ids


@dataclass(frozen=True)
class Budget:
    max_candidates: int = 200
    deadline: float = float("inf")     # time.monotonic() value after which retrievers should return what they have

    def expired(self) -> bool:
        return time.monotonic() > self.deadline


@dataclass
class RetrievalContext:
    conn: psycopg.Connection           # tenant-scoped (row-level security applies to every statement)
    auth: AuthContext
    domain: Domain


class Retriever(Protocol):
    name: str

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]: ...
