"""Question → plan. The planner is deterministic, so every query type is pinned here."""
from datetime import date

import pytest

from docintel.packs import default_domain
from docintel.search.planner import plan_query

TODAY = date(2026, 10, 7)
CONTRACTS = set(default_domain().families["contract"])


def p(q):
    return plan_query(q, today=TODAY)


@pytest.mark.parametrize("q,text,exact", [
    ("FOR IMMEDIATE RELEASE", "FOR IMMEDIATE RELEASE", True),
    ("Nadia Hartwell", "Nadia Hartwell", True),
    ("INV 2026 00481", "INV 2026 00481", True),
    ("how do I change the toner", "how I change toner", False),
])
def test_search_text_and_exact_intent(q, text, exact):
    plan = p(q)
    assert plan.intent == "search" and plan.text == text and plan.exact_intent is exact and not plan.filters.dates


def test_identifier_and_quoted_phrase():
    plan = p('find "blue heron protocol 7781" in INV-2026-00481')
    assert plan.phrases == ["blue heron protocol 7781"] and "inv202600481" in plan.identifiers and plan.exact_intent


def test_count_by_type():
    plan = p("How many NDAs do we have?")
    assert plan.intent == "count" and plan.filters.doc_types == ["nda"] and plan.filters.doc_types_hard and plan.text == ""


def test_percentage_has_a_denominator():
    plan = p("What percentage of our contracts are NDAs?")
    assert plan.intent == "percentage" and plan.filters.doc_types == ["nda"]
    assert set(plan.base.doc_types) == CONTRACTS and plan.base.doc_types_hard


def test_count_with_jurisdiction():
    plan = p("How many contracts are governed by California law?")
    assert plan.intent == "count" and plan.filters.jurisdictions == ["California"] and set(plan.filters.doc_types) == CONTRACTS


def test_date_field_inference():
    plan = p("How many invoices were issued in 2026?")
    (d,) = plan.filters.dates
    assert (d.field, d.start, d.end) == ("issue_date", date(2026, 1, 1), date(2026, 12, 31))
    (d,) = p("Which agreements expire in 2027?").filters.dates
    assert d.field == "expiry_date"


@pytest.mark.parametrize("q,start,end", [
    ("contracts expiring in Q1 2026", date(2026, 1, 1), date(2026, 3, 31)),
    ("contracts expiring in March 2027", date(2027, 3, 1), date(2027, 3, 31)),
    ("contracts expiring between 2025 and 2027", date(2025, 1, 1), date(2027, 12, 31)),
    ("contracts expiring next year", date(2027, 1, 1), date(2027, 12, 31)),
    ("contracts expiring in the next 90 days", date(2026, 10, 7), date(2027, 1, 5)),
    ("invoices since 2025", date(2025, 1, 1), None),
    ("invoices before 2025", None, date(2024, 12, 31)),
])
def test_date_expressions(q, start, end):
    (d,) = p(q).filters.dates
    assert (d.start, d.end) == (start, end)


def test_year_inside_identifier_is_not_a_date():
    assert p("INV 2026 00481").filters.dates == []


def test_amount_filters():
    (a,) = p("Show contracts where the payment amount is greater than $100,000").filters.amounts
    assert (a.op, a.value, a.currency) == (">", 100000.0, "USD")
    (a,) = p("invoices above SAR 400,000").filters.amounts
    assert (a.op, a.value, a.currency) == (">", 400000.0, "SAR")
    (a,) = p("invoices between 1,000 and 5,000 USD").filters.amounts
    assert (a.op, a.value, a.value2) == ("between", 1000.0, 5000.0)
    assert p("contracts over 5 years").filters.amounts == []


def test_signature_filters():
    assert p("documents signed by John Smith").filters.signer == "John Smith"
    assert p("Which contracts are unsigned?").filters.has_signature is False
    assert p("How many documents contain a signature?").filters.has_signature is True


def test_clause_and_kind_filters():
    assert p("Which contracts have termination clauses?").filters.clause_types == ["termination"]
    assert p("scanned documents from 2025").filters.kinds == ["scanned", "image", "mixed"]
    assert p("pdf invoices").filters.extensions == [".pdf"]


def test_aggregates():
    plan = p("What is the total value of contracts with Acme?")
    assert plan.intent == "sum" and plan.filters.parties == ["Acme"] and plan.aggregate_field == "total_amount"
    assert p("What is the average invoice amount?").intent == "average"
    assert p("What is the largest invoice amount?").intent == "max"
    plan = p("How many documents per type?")
    assert plan.intent == "group" and plan.group_by == "type" and plan.text == ""


def test_lookups():
    plan = p("What is the invoice number of the Saudi Aramco invoice?")
    assert plan.intent == "lookup" and plan.lookup_field == "invoice_number" and plan.text == "Saudi Aramco invoice"
    assert p("When does the service contract expire?").lookup_field == "expiry_date"
    assert p("who signed the Acme NDA").lookup_field == "signer"


def test_fact_questions_are_not_document_counts():
    assert p("How many employees were paid in 2024?").intent == "fact"
    assert p("How many days notice is required to terminate the service contract?").intent == "fact"
    assert p("How many documents mention Jeddah?").intent == "count"


def test_type_words_stay_in_text_when_they_carry_meaning():
    plan = p("maintenance contract")
    assert plan.text == "maintenance contract" and not plan.filters.doc_types_hard
    plan = p("press release")
    assert plan.text == "" and plan.filters.doc_types == ["press_release"] and plan.filters.doc_types_hard


def test_plan_is_serializable():
    import json
    json.dumps(p("How many contracts expire in 2027 governed by California law above USD 1,000?").to_dict())


@pytest.mark.parametrize("q,intent,group_by", [
    ("invoices per vendor", "group", "party"),
    ("How many invoices per year?", "group", "year"),
    ("total invoice value by currency", "sum", "currency"),
    ("What is the total value of invoices by customer?", "sum", "party"),
])
def test_groupings(q, intent, group_by):
    plan = p(q)
    assert (plan.intent, plan.group_by) == (intent, group_by)


def test_comparisons_compute_per_period_over_one_range():
    plan = p("Compare the total value of invoices in 2025 and 2026")
    assert plan.intent == "sum" and [d.start.year for d in plan.compare] == [2025, 2026]
    (d,) = plan.filters.dates
    assert (d.start, d.end) == (date(2025, 1, 1), date(2026, 12, 31))
    assert p("How many invoices were issued in 2025 and 2026?").compare


def test_spend_questions_are_sums():
    plan = p("How much did Northwind spend in 2025?")
    assert plan.intent == "sum" and plan.text == "Northwind"


def test_concepts_and_strategies():
    plan = p("Can either party cancel the contract early?")
    assert plan.concepts == ["termination"] and {"exact", "contextual", "semantic"} <= set(plan.strategies)
    plan = p("INV-1024-77")
    assert "semantic" not in plan.strategies and plan.strategies[0] == "exact"
    assert p("How many NDAs do we have?").strategies == ["structured"]


def test_fully_quoted_questions_use_only_phrase_retrievers():
    from docintel.search.planner import plan_query, strategies_for
    assert strategies_for(plan_query('"either party may terminate"')) == ["exact", "fuzzy"]
    assert "lexical" in strategies_for(plan_query('"either party may terminate" in service contracts'))
