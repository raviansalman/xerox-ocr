"""Plans are validated against the tenant's packs; plans from a language model are untrusted input."""
from datetime import date

import pytest

from docintel.packs import get_domain
from docintel.query.validation import PlanError, plan_from_untrusted, validate
from docintel.search.planner import plan_query

D = get_domain(("legal", "finance"))


def test_planner_output_is_always_valid():
    for q in ["How many NDAs do we have?", "What is the total value of all invoices?", "INV-1024-77",
              "Which contracts have termination clauses?", "invoices per vendor", "When does the lease expire?"]:
        validate(plan_query(q, today=date(2026, 10, 7), domain=D), D)


def test_untrusted_plan_is_parsed_strictly():
    plan = plan_from_untrusted({"intent": "count", "filters": {"doc_types": ["invoice"], "dates": [
        {"field": "issue_date", "start": "2025-01-01", "end": "2025-12-31"}]}}, "how many invoices in 2025", D)
    assert plan.intent == "count" and plan.filters.doc_types_hard and plan.filters.dates[0].end == date(2025, 12, 31)


@pytest.mark.parametrize("data,why", [
    ({"intent": "count", "tenant_id": "other"}, "tenant"),                                   # cannot name a tenant
    ({"intent": "count", "filters": {"doc_types": ["invoice"], "sql": "1=1"}}, "sql"),      # no SQL or expressions
    ({"intent": "drop_table"}, "intent"),
    ({"intent": "count", "filters": {"doc_types": ["no_such_type"]}}, "unknown document types"),
    ({"intent": "search", "filters": {"clause_types": ["backdoor"]}}, "unknown clause types"),
    ({"intent": "lookup", "lookup_field": "password"}, "unknown field"),
    ({"intent": "sum", "filters": {"amounts": [{"op": "!=", "value": 1}]}}, "op"),
    ({"intent": "sum", "filters": {"amounts": [{"op": ">", "value": 1, "currency": "dollars"}]}}, "currency"),
    ({"intent": "group", "group_by": "tenant_id"}, "group_by"),
    ({"intent": "count", "answer": 42}, "answer"),                                           # no numbers that become answers
    ({"intent": "search", "concepts": ["exfiltrate"]}, "unknown concepts"),
    ("count everything", "dictionary"),
])
def test_untrusted_plans_are_rejected(data, why):
    with pytest.raises(PlanError):
        plan_from_untrusted(data, "question", D)


def test_concepts_need_the_pack_that_defines_them():
    core_only = get_domain(())
    with pytest.raises(PlanError):
        plan_from_untrusted({"intent": "search", "concepts": ["termination"]}, "q", core_only)
