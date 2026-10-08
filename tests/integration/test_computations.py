"""Computed answers are deterministic, carry their calculation and supporting records, and quote stated figures."""
import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration


def ask(client, q, key="acme_reader"):
    r = client.post("/api/v1/query", headers=headers(key), json={"q": q, "limit": 20})
    assert r.status_code == 200, r.text
    return r.json()


def test_sum_has_calculation_and_records_with_spans(client, ingested):
    a = ask(client, "What is the total value of all invoices?")["answer"]
    calc = a["calculation"]
    assert calc["operation"] == "sum" and calc["field"] == "total_amount" and calc["documents_with_values"] == 2
    assert calc["filters"]["doc_types"] == ["invoice"]
    rec = {r["currency"]: r for r in a["records"]}
    assert rec["SAR"]["value"] == 418750.0 and rec["SAR"]["char_start"] is not None
    page = client.get(f"/api/v1/documents/{rec['SAR']['document_id']}/pages/{rec['SAR']['page']}",
                      headers=headers("acme_reader")).json()
    assert page["text"][rec["SAR"]["char_start"]:rec["SAR"]["char_end"]].replace(" ", "").startswith("SAR418,750")


def test_counts_are_deterministic_and_explained(client, ingested):
    answers = [ask(client, "How many contracts are governed by California law?")["answer"] for _ in range(3)]
    assert {a["value"] for a in answers} == {3}
    calc = answers[0]["calculation"]
    assert calc["operation"] == "count" and calc["filters"]["jurisdictions"] == ["California"]


def test_comparison_by_period(client, ingested):
    a = ask(client, "Compare the total value of invoices in 2025 and 2026")["answer"]
    assert {(r["period"], r["currency"], r["value"]) for r in a["rows"]} == {("2025", "USD", 12500.0), ("2026", "SAR", 418750.0)}
    assert a["calculation"]["periods"] == ["2025", "2026"]
    c = ask(client, "How many invoices were issued in 2025 and 2026?")["answer"]
    assert {(r["period"], r["count"]) for r in c["rows"]} == {("2025", 1), ("2026", 1)}


def test_group_by_party_and_currency(client, ingested):
    a = ask(client, "What is the total value of invoices by customer?")["answer"]
    by = {r["party"]: (r["currency"], r["value"]) for r in a["rows"]}
    assert by["Saudi Aramco"] == ("SAR", 418750.0) and by["Gulf Trading LLC"][1] == 12500.0
    g = ask(client, "total invoice value by currency")["answer"]
    assert {(r["currency"], r["value"]) for r in g["rows"]} == {("SAR", 418750.0), ("USD", 12500.0)}


def test_party_named_without_cues_is_resolved_from_extracted_entities(client, ingested):
    out = ask(client, "How much did Northwind spend?")
    a = out["answer"]
    assert a["kind"] == "table" and a["rows"][0]["value"] == 432000.0
    assert [r["filename"] for r in out["results"]] == ["Equipment_Lease_Northwind.pdf"]


def test_stated_figure_comes_from_a_relation_with_its_span(client, ingested):
    a = ask(client, "How many days notice is required to terminate the service contract?")["answer"]
    assert a["kind"] == "figure" and a["value"] == "60" and a["source"] == "relation:may_terminate"
    assert a["char_start"] is not None and a["filename"] == "Service_Contract_17-2024.pdf"


def test_field_lookup_points_at_the_value(client, ingested):
    a = ask(client, "What is the invoice number of the Saudi Aramco invoice?")["answer"]
    row = a["rows"][0]
    page = client.get(f"/api/v1/documents/{row['document_id']}/pages/{row['page']}", headers=headers("acme_reader")).json()
    assert page["text"][row["char_start"]:row["char_end"]] == "INV-2026-00481"


def test_amounts_missing_from_matching_documents_are_reported(client, ingested):
    a = ask(client, "What is the total value of all NDAs?")["answer"]
    assert a["kind"] == "none" and a["calculation"]["documents_without_values"] == 2


def test_grouped_counts_apply_the_questions_text(client, ingested):
    a = ask(client, "How many invoices mentioning Aramco per year?")["answer"]
    assert {(r["key"], r["count"]) for r in a["rows"]} == {("2026", 1)}, a      # not every invoice per year
    assert a["calculation"]["text_condition"] == "Aramco"


def test_sums_grouped_by_type_and_currency(client, ingested):
    a = ask(client, "What is the total value of invoices by type?")["answer"]
    assert {(r["type"], r["currency"], r["value"]) for r in a["rows"]} == {("invoice", "SAR", 418750.0), ("invoice", "USD", 12500.0)}
    assert "invoice: SAR" in a["text"] and ": :" not in a["text"]
    c = ask(client, "total invoice value by currency")["answer"]
    assert "Total: SAR 418,750.00" in c["text"] and ": :" not in c["text"], c["text"]


def test_a_reversed_date_range_is_read_in_order(client, ingested):
    r = client.post("/api/v1/query", headers=headers("acme_reader"), json={"q": "invoices from 2026 to 2025", "explain": True})
    assert r.status_code == 200, r.text
    d = r.json()["plan"]["filters"]["dates"][0]
    assert (d["start"], d["end"]) == ("2025-01-01", "2026-12-31")
    assert {x["filename"] for x in r.json()["results"]} == {"Tax_Invoice_INV-2026-00481.pdf", "Invoice_Gulf_Trading.docx"}
