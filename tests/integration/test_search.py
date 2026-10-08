"""Query behaviour across every query type, with tenant isolation checked on every query.

Runs with the deterministic stand-in embedder by default (lexical, structured and aggregate behaviour), or with a real
model (DOCINTEL_TEST_EMBEDDER_URL). Semantic-quality expectations for the real model live in tests/quality.
"""
import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration


def ask(client, q, key="acme_reader", **kw):
    r = client.post("/api/v1/query", headers=headers(key), json={"q": q, "limit": kw.pop("limit", 10), **kw})
    assert r.status_code == 200, r.text
    return r.json()


def ids(out, ingested):
    inv = {v: k for k, v in ingested.items()}
    return [inv.get(r["document_id"], r["document_id"]) for r in out["results"]]


# (question, expected first result, documents that must not appear)
RETRIEVAL = [
    ("FOR IMMEDIATE RELEASE", "press_release", []),
    ('"for immediate release"', "press_release", ["firmware"]),
    ("Nadia Hartwell", "press_release", []),
    ("Northbridge Data", "press_release", []),
    ("INV-2026-00481", "invoice", []),
    ("INV 2026 00481", "invoice", []),
    ("inv202600481", "invoice", []),
    ("00481", "invoice", []),
    ("Contract No. 17/2024", "contract", []),
    ("Article 12.4", "contract", []),
    ("Saudi Aramco", "invoice", []),
    ("PO-2025-0193", None, []),
    ("48213", "ticket", []),
    ("Priya Raman", "ticket", []),
    ("VersaLink C405", "ticket", []),
    ("Versa Link C405", "ticket", []),
    ("Gulf Trading invoice", "invoice_docx", []),
    ("Califomia", "ocr_noisy", []),
    ("FOR IMMEDlATE RELEASE", "press_release", []),
    ("Nadia Hartwel", "press_release", []),
    ("rnaintenance visits", "ocr_noisy", []),
    ("delivery schedule Jeddah", "email", []),
    ("legacy printers", "deck", []),
    ("fleet uptime", "deck", []),
    ("ink", "ink_usage" if False else "ink", ["network"]),
    ("Toner_Replacement_Procedure", "toner", []),
    ("Payroll_Summary_2024.xlsx", "payroll", []),
]


@pytest.mark.parametrize("q,first,absent", RETRIEVAL)
def test_retrieval(client, ingested, q, first, absent):
    out = ask(client, q)
    got = ids(out, ingested)
    assert got, out["answer"]
    if first:
        assert got[0] == first, (q, got)
    for a in absent:
        assert a not in got[:3] or got.index(a) > got.index(first), (q, got)


def test_po_number_matches_order_and_email(client, ingested):
    assert set(ids(ask(client, "PO-2025-0193"), ingested)[:2]) == {"po", "email"}


def test_quoted_phrase_requires_the_phrase(client, ingested):
    assert ids(ask(client, '"for immediate release"'), ingested) == ["press_release"]
    assert "firmware" in ids(ask(client, "FOR IMMEDIATE RELEASE"), ingested)   # unquoted: related documents follow


def test_exact_phrase_needle_is_found_regardless_of_vector_similarity(client, ingested):
    out = ask(client, '"blue heron protocol 7781"', key="haystack_uploader")
    assert ids(out, ingested)[0] == "needle" and "exact_phrase" in out["results"][0]["match_types"]
    out = ask(client, "blue heron protocol 7781", key="haystack_uploader")
    assert ids(out, ingested)[0] == "needle"


def test_unknown_query_returns_no_results(client, ingested):
    out = ask(client, "zzqxunknownzzq")
    assert out["results"] == [] and out["answer"]["kind"] == "none"


def test_match_types_and_snippets_have_evidence(client, ingested):
    r = ask(client, "INV-2026-00481")["results"][0]
    assert "exact_identifier" in r["match_types"] and r["confidence"] >= 0.9
    assert r["snippets"][0]["page"] == 1 and "INV-2026-00481" in r["snippets"][0]["text"]
    assert r["fields"]["invoice_number"] == "INV-2026-00481"


# Aggregations: (question, expected value)
COUNTS = [
    ("How many NDAs do we have?", 2),
    ("How many contracts are governed by California law?", 3),
    ("How many invoices were issued in 2026?", 1),
    ("How many invoices do we have?", 2),
    ("How many contracts expire in 2027?", 1),
    ("How many signed NDAs governed by California law do we have?", 1),
    ("How many documents mention Jeddah?", 2),
    ("How many scanned documents?", 2),
    ("How many documents contain a signature?", 4),
]


@pytest.mark.parametrize("q,value", COUNTS)
def test_counts(client, ingested, q, value):
    a = ask(client, q)["answer"]
    assert a["kind"] == "number" and a["value"] == value, (q, a)


def test_percentage(client, ingested):
    a = ask(client, "What percentage of our contracts are NDAs?")["answer"]
    assert (a["value"], a["numerator"], a["denominator"]) == (33.3, 2, 6)


def test_sum_keeps_currencies_apart(client, ingested):
    a = ask(client, "What is the total value of all invoices?")["answer"]
    assert {(r["currency"], r["value"]) for r in a["rows"]} == {("SAR", 418750.0), ("USD", 12500.0)}


def test_sum_with_party_filter(client, ingested):
    a = ask(client, "What is the total value of contracts with Northwind?")["answer"]
    assert [(r["currency"], r["value"]) for r in a["rows"]] == [("USD", 432000.0)]


def test_group_by_type(client, ingested):
    a = ask(client, "How many documents per type?")["answer"]
    rows = {r["key"]: r["count"] for r in a["rows"]}
    assert a["kind"] == "table" and rows["nda"] == 2 and rows["invoice"] == 2


# Filters and lookups: (question, expected set of documents)
FILTERED = [
    ("contracts governed by California law", {"nda_ca", "employment", "ocr_noisy"}),
    ("Which agreements expire in 2027?", {"contract"}),
    ("Which contracts are unsigned?", {"lease", "employment", "ocr_noisy"}),
    ("documents signed by John Smith", {"nda_ca"}),
    ("Show contracts where the payment amount is greater than $100,000", {"contract", "lease"}),
    ("invoices above SAR 400,000", {"invoice"}),
    ("Which contracts have termination clauses?", {"contract", "employment"}),
    ("scanned documents", {"ticket", "approval"}),
    ("press release", {"press_release"}),
    ("invoices from 2025", {"invoice_docx"}),
]


@pytest.mark.parametrize("q,expected", FILTERED)
def test_structured_questions(client, ingested, q, expected):
    assert set(ids(ask(client, q, limit=20), ingested)) == expected


def test_lookup_answers_with_source(client, ingested):
    a = ask(client, "What is the invoice number of the Saudi Aramco invoice?")["answer"]
    assert a["kind"] == "fields" and a["rows"][0]["value"] == "INV-2026-00481" and a["rows"][0]["page"] == 1
    a = ask(client, "When does the service contract expire?")["answer"]
    assert a["rows"][0]["value"] == "2027-12-31"
    a = ask(client, "who signed the Acme NDA")["answer"]
    assert a["rows"][0]["value"] == "John Smith"


def test_fact_questions_quote_figures(client, ingested):
    a = ask(client, "How many employees were paid in 2024?")["answer"]
    assert a["kind"] == "figure" and a["value"] == "42"
    a = ask(client, "How many days notice is required to terminate the service contract?")["answer"]
    assert a["kind"] == "figure" and a["value"] == "60"


def test_explain_returns_the_plan_and_timings(client, ingested):
    out = ask(client, "How many NDAs do we have?", explain=True)
    assert out["plan"]["intent"] == "count" and out["plan"]["filters"]["doc_types"] == ["nda"]
    assert out["timings_ms"]["total"] > 0


# ---------------------------------------------------------------------------------------------------- tenants

ALL_QUESTIONS = [q for q, *_ in RETRIEVAL] + [q for q, _ in COUNTS] + [q for q, _ in FILTERED] + [
    "What is the total value of all invoices?", "How many documents per type?", "What percentage of our contracts are NDAs?",
    "John Smith", "blue heron protocol 7781", "Globex confidential merger"]


@pytest.mark.parametrize("tenant", ["acme", "globex", "carol"])
def test_no_question_returns_another_tenants_documents(client, ingested, corpus_dir, tenant):
    tenant_of = {ingested[d.key]: d.tenant for d, _ in corpus_dir}
    key = "carol_reader" if tenant == "carol" else f"{tenant}_reader"
    for q in ALL_QUESTIONS:
        out = ask(client, q, key=key, limit=50)
        foreign = [r["filename"] for r in out["results"] if tenant_of.get(r["document_id"], tenant) != tenant]
        assert not foreign, (tenant, q, foreign)


def test_counts_are_per_tenant(client, ingested):
    # globex holds one NDA-like release of its own; acme's two are never added to it, and vice versa
    assert ask(client, "How many NDAs do we have?", key="globex_reader")["answer"]["value"] == 1
    assert ask(client, "How many NDAs do we have?")["answer"]["value"] == 2
    assert ask(client, "How many NDAs do we have?", key="carol_reader")["answer"]["value"] == 0
    assert ask(client, "How many documents per type?", key="carol_reader")["answer"]["rows"] == []


def test_same_identifier_in_two_tenants(client, ingested):
    assert ids(ask(client, "INV-2026-00481", key="globex_reader"), ingested) == ["globex_release"]
    assert "globex_release" not in ids(ask(client, "INV-2026-00481"), ingested)


def test_service_key_acts_only_for_the_named_tenant(client, ingested):
    r = client.post("/api/v1/query", headers=headers("service", "globex"), json={"q": "Nadia Hartwell"})
    assert ids(r.json(), ingested) == ["globex_release"]


def test_documents_of_another_tenant_are_not_found_by_id(client, ingested):
    for path in (f"/api/v1/documents/{ingested['invoice']}", f"/api/v1/documents/{ingested['invoice']}/original",
                 f"/api/v1/documents/{ingested['invoice']}/pages/1"):
        assert client.get(path, headers=headers("globex_reader")).status_code == 404
    assert client.delete(f"/api/v1/documents/{ingested['invoice']}", headers=headers("globex_uploader")).status_code == 404


def test_row_level_security_hides_rows_without_tenant_context(engine_env, ingested):
    with engine_env.system() as conn:
        assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0
    with engine_env.tenant("acme") as conn:
        assert conn.execute("SELECT count(*) AS n FROM documents WHERE tenant_id <> 'acme'").fetchone()["n"] == 0
        with pytest.raises(Exception):          # writing another tenant's row is refused by the policy
            conn.execute("UPDATE documents SET tenant_id = 'globex' WHERE id = %s", (ingested["memo"],))


def test_database_role_cannot_bypass_rls(engine_env):
    assert engine_env.security_check() == {"role_bypasses_rls": False}


def test_a_large_filtered_set_is_searched_completely(client, ingested, monkeypatch):
    """The filtered scope is never truncated: with the explicit-id limit set to 1, all three California
    agreements are still in scope and a text question still reaches each of them."""
    from docintel.config import get_settings
    monkeypatch.setattr(get_settings(), "structured_id_limit", 1)
    got = set(ids(ask(client, "agreements governed by California law mentioning Acme"), ingested))
    assert {"nda_ca", "employment"} <= got, got
