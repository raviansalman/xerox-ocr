"""Ranking rules that do not need a database."""
from datetime import date

import pytest

from docintel.search.engine import _identifier_only, _terms
from docintel.search.planner import plan_query


@pytest.mark.parametrize("q,expected", [
    ("WB-7731", True), ("PO-2025-0193 INV-2026-00481", True), ("inv202600481", True),
    ("Contract No. 17/2024", False), ("toner WB-7731", False), ("Nadia Hartwell", False), ("2027", False),
])
def test_identifier_only(q, expected):
    assert _identifier_only(plan_query(q, today=date(2026, 10, 7))) is expected


def test_highlight_terms_keep_the_identifier_as_written():
    terms = _terms(plan_query("INV-2026-00481", today=date(2026, 10, 7)))
    assert "INV-2026-00481" in terms


def test_amounts_without_a_currency_are_never_summed():
    from docintel.query import compute
    from docintel.search.plan import QueryPlan

    class Conn:
        def execute(self, sql, args):
            rows = [{"document_id": "a", "unit": "USD", "value_num": 10.0, "value_text": "USD 10", "name": "total_amount",
                     "page": 1, "char_start": 0, "char_end": 6, "snippet": ""},
                    {"document_id": "b", "unit": None, "value_num": 99.0, "value_text": "99", "name": "total_amount",
                     "page": 1, "char_start": 0, "char_end": 2, "snippet": ""}]
            return type("R", (), {"fetchall": lambda _s: rows})()

    plan = QueryPlan(question="total", intent="sum", text="", semantic_text="", aggregate_field="total_amount")
    answer, valued = compute.amounts(Conn(), plan, ["a", "b", "c"])
    assert [(r["currency"], r["value"]) for r in answer["rows"]] == [("USD", 10.0)] and valued == ["a"]
    calc = answer["calculation"]
    assert calc["documents_with_amounts_of_unknown_currency"] == 1 and calc["documents_without_values"] == 1
    assert "without a currency" in answer["note"]
