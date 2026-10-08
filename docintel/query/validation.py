"""Query plan validation.

Every plan is validated before it is executed, whoever produced it. A plan built by the deterministic planner passes
by construction; a plan proposed by a language model (``plan_from_untrusted``) is untrusted input: it is parsed with a
strict schema (unknown keys rejected), every vocabulary item must exist in the tenant's enabled packs, every value is
bounded, and anything it cannot express is impossible to express:

* no tenant, user or document identifiers (the tenant always comes from the authenticated caller);
* no SQL, expressions or code: filters are typed values combined by fixed operators;
* no numbers that become answers: computed values only ever come from extracted data.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from docintel.packs import Domain
from docintel.search.plan import GROUP_KEYS, INTENTS, STRATEGIES, AmountFilter, DateFilter, Filters, QueryPlan

DATE_FIELDS = ("issue_date", "effective_date", "expiry_date", "due_date", "signature_date", "any")
CURRENCY = r"^[A-Z]{3}$"


class PlanError(ValueError):
    """The plan refers to something that does not exist or is not allowed."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DateIn(_Strict):
    field: Literal["issue_date", "effective_date", "expiry_date", "due_date", "signature_date", "any"] = "any"
    start: date | None = None
    end: date | None = None


class AmountIn(_Strict):
    op: Literal[">", ">=", "<", "<=", "between"]
    value: float = Field(ge=0, le=1e15)
    value2: float | None = Field(default=None, ge=0, le=1e15)
    currency: str | None = Field(default=None, pattern=CURRENCY)


class FiltersIn(_Strict):
    doc_types: list[str] = Field(default_factory=list, max_length=20)
    jurisdictions: list[str] = Field(default_factory=list, max_length=10)
    signer: str | None = Field(default=None, max_length=120)
    has_signature: bool | None = None
    parties: list[str] = Field(default_factory=list, max_length=10)
    dates: list[DateIn] = Field(default_factory=list, max_length=5)
    amounts: list[AmountIn] = Field(default_factory=list, max_length=5)
    clause_types: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("parties", "jurisdictions")
    @classmethod
    def _short(cls, v):
        if any(len(x) > 120 or not x.strip() for x in v):
            raise ValueError("names must be 1 to 120 characters")
        return v


class PlanIn(_Strict):
    """What an untrusted planner may propose."""
    intent: Literal["search", "fact", "count", "percentage", "sum", "average", "min", "max", "group", "lookup"]
    text: str = Field(default="", max_length=500)
    phrases: list[str] = Field(default_factory=list, max_length=5)
    filters: FiltersIn = Field(default_factory=FiltersIn)
    base: FiltersIn | None = None
    group_by: Literal["type", "month", "year", "jurisdiction", "language", "status", "party", "currency"] | None = None
    lookup_field: str | None = Field(default=None, max_length=60)
    aggregate_field: Literal["total_amount", "periodic_amount", "amount", "tax_amount", "salary"] | None = None
    concepts: list[str] = Field(default_factory=list, max_length=5)


def _known_fields(domain: Domain) -> set[str]:
    """Field names the tenant's packs define (lookups may only ask for these)."""
    return (set(domain.lookup_terms) | set(domain.identifier_fields) | set(domain.date_roles) | set(domain.amount_roles) |
            {"jurisdiction", "signer", "party"})


def validate(plan: QueryPlan, domain: Domain) -> QueryPlan:
    """Check a plan against the tenant's packs and the engine's limits; raise PlanError on anything unknown."""
    problems = []
    if plan.intent not in INTENTS:
        problems.append(f"unknown intent {plan.intent!r}")
    for f in [plan.filters, *([plan.base] if plan.base else [])]:
        unknown = [t for t in f.doc_types if t not in domain.types]
        if unknown:
            problems.append(f"unknown document types {unknown}")
        unknown = [c for c in f.clause_types if c not in domain.clause_types]
        if unknown:
            problems.append(f"unknown clause types {unknown}")
        for d in f.dates:
            if d.field not in DATE_FIELDS:
                problems.append(f"unknown date field {d.field!r}")
            if d.start and d.end and d.start > d.end:
                problems.append("date range ends before it starts")
        for a in f.amounts:
            if a.op not in (">", ">=", "<", "<=", "between"):
                problems.append(f"unknown amount operator {a.op!r}")
    if plan.group_by and plan.group_by not in GROUP_KEYS:
        problems.append(f"unknown grouping {plan.group_by!r}")
    if plan.lookup_field and plan.lookup_field not in _known_fields(domain):
        problems.append(f"unknown field {plan.lookup_field!r}")
    unknown = [c for c in plan.concepts if c not in domain.concepts]
    if unknown:
        problems.append(f"unknown concepts {unknown}")
    unknown = [s for s in plan.strategies if s not in STRATEGIES]
    if unknown:
        problems.append(f"unknown retrieval strategies {unknown}")
    if len(plan.question) > 1000 or len(plan.text) > 1000:
        problems.append("question too long")
    if problems:
        raise PlanError("; ".join(problems))
    return plan


def _filters(f: FiltersIn) -> Filters:
    return Filters(doc_types=list(f.doc_types), jurisdictions=list(f.jurisdictions), signer=f.signer,
                   has_signature=f.has_signature, parties=list(f.parties),
                   dates=[DateFilter(d.field, d.start, d.end, "planner") for d in f.dates],
                   amounts=[AmountFilter(a.op, a.value, a.value2, a.currency, "planner") for a in f.amounts],
                   clause_types=list(f.clause_types))


def plan_from_untrusted(data: object, question: str, domain: Domain) -> QueryPlan:
    """Build a plan from untrusted structured output (for example a language model's), or raise PlanError."""
    try:
        p = PlanIn.model_validate(data)
    except ValidationError as e:
        raise PlanError(f"invalid plan: {e.error_count()} problem(s): " +
                        "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])) from e
    from docintel import text as T
    filters = _filters(p.filters)
    if p.intent in ("count", "percentage", "sum", "average", "min", "max", "group"):
        filters.doc_types_hard = True
    plan = QueryPlan(question=question, intent=p.intent, text=p.text, semantic_text=question, phrases=list(p.phrases),
                     identifiers=sorted(T.identifiers(" ".join([*p.phrases, p.text]))),
                     exact_intent=bool(p.phrases) or 0 < len(p.text.split()) <= 6, filters=filters,
                     base=_filters(p.base) if p.base else None, aggregate_field=p.aggregate_field or (
                         "total_amount" if p.intent in ("sum", "average", "min", "max") else None),
                     group_by=p.group_by, lookup_field=p.lookup_field, packs=domain.packs, concepts=list(p.concepts),
                     notes=["plan proposed by a language model and validated"])
    return validate(plan, domain)
