"""Typed query plan: what the planner understood and what the executor runs. Serialized in ``explain``."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

Intent = Literal["search", "fact", "count", "percentage", "sum", "average", "min", "max", "group", "lookup"]
INTENTS: tuple[str, ...] = ("search", "fact", "count", "percentage", "sum", "average", "min", "max", "group", "lookup")
GROUP_KEYS: tuple[str, ...] = ("type", "month", "year", "jurisdiction", "language", "status", "party", "currency")
STRATEGIES: tuple[str, ...] = ("exact", "lexical", "fuzzy", "semantic", "entity", "contextual", "metadata", "structured")


@dataclass
class DateFilter:
    field: str            # expiry_date | issue_date | effective_date | due_date | signature_date | any
    start: date | None
    end: date | None      # inclusive
    source: str


@dataclass
class AmountFilter:
    op: Literal[">", ">=", "<", "<=", "between"]
    value: float
    value2: float | None
    currency: str | None
    source: str


@dataclass
class Filters:
    doc_types: list[str] = field(default_factory=list)
    doc_types_hard: bool = False
    jurisdictions: list[str] = field(default_factory=list)       # canonical names
    signer: str | None = None
    has_signature: bool | None = None
    parties: list[str] = field(default_factory=list)
    dates: list[DateFilter] = field(default_factory=list)
    amounts: list[AmountFilter] = field(default_factory=list)
    clause_types: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.doc_types or self.jurisdictions or self.signer or self.has_signature is not None or
                    self.parties or self.dates or self.amounts or self.clause_types or self.kinds or self.extensions)


@dataclass
class QueryPlan:
    question: str
    intent: Intent
    text: str                         # residual text for lexical retrieval
    semantic_text: str                # text to embed
    phrases: list[str] = field(default_factory=list)      # quoted phrases (must appear)
    identifiers: list[str] = field(default_factory=list)  # canonical identifiers in the question
    exact_intent: bool = False
    filters: Filters = field(default_factory=Filters)
    base: Filters | None = None       # percentage: denominator filters ("of contracts")
    aggregate_field: str | None = None
    group_by: str | None = None
    lookup_field: str | None = None
    notes: list[str] = field(default_factory=list)
    packs: tuple[str, ...] = ()       # domain packs the question was interpreted with
    compare: list[DateFilter] = field(default_factory=list)     # periods compared ("2025 and 2026")
    concepts: list[str] = field(default_factory=list)           # pack concepts the question refers to
    entities: list[str] = field(default_factory=list)           # known entity names found in the question
    strategies: list[str] = field(default_factory=list)         # retrievers the executor runs
    version: int = 2

    def to_dict(self) -> dict[str, Any]:
        def conv(v):
            if isinstance(v, date):
                return v.isoformat()
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items() if x not in (None, [], False) or k in ("has_signature",)}
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v
        return conv(asdict(self))
