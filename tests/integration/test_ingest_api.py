"""Ingestion through the API: every format, extraction results, lifecycle operations and API contract."""

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration


def doc(client, ingested, key, tenant="acme"):
    r = client.get(f"/api/v1/documents/{ingested[key]}", headers=headers(f"{tenant}_reader" if tenant != "haystack" else "haystack_uploader"))
    assert r.status_code == 200
    return r.json()


def fields(d):
    return {(f["name"], f["value_text"]) for f in d["fields"]}


@pytest.mark.parametrize("key,doc_type,kind", [
    ("press_release", "press_release", "native"), ("invoice", "invoice", "native"), ("invoice_docx", "invoice", "office"),
    ("contract", "service_agreement", "native"), ("nda_ca", "nda", "native"), ("employment", "employment_agreement", "office"),
    ("ocr_noisy", "supply_agreement", "text"), ("payroll", "payroll", "spreadsheet"), ("po", "purchase_order", "spreadsheet"),
    ("lease", "lease_agreement", "native"), ("toner", "policy", "text"), ("remote", "policy", "office"),
    ("ticket", None, "scanned"), ("approval", "letter", "image"), ("deck", None, "presentation"),
    ("email", "email", "email"), ("memo", "memo", "text"),
])
def test_every_format_is_indexed_and_classified(client, ingested, key, doc_type, kind):
    d = doc(client, ingested, key)
    assert d["status"] == "indexed" and d["kind"] == kind and d["page_count"] >= 1 and d["word_count"] > 0
    if doc_type:
        assert d["doc_type"] == doc_type, (key, d["doc_type"], d["doc_type_method"])


def test_invoice_fields(client, ingested):
    f = fields(doc(client, ingested, "invoice"))
    assert {("invoice_number", "INV-2026-00481"), ("issue_date", "2026-01-14"), ("due_date", "2026-02-13"),
            ("total_amount", "SAR 418,750.00"), ("party", "Saudi Aramco")} <= f


def test_contract_fields_clauses_and_signer(client, ingested):
    d = doc(client, ingested, "contract")
    f = fields(d)
    assert {("contract_number", "17/2024"), ("expiry_date", "2027-12-31"), ("jurisdiction", "Saudi Arabia"),
            ("total_amount", "USD 250,000.00"), ("signer", "Omar Haddad")} <= f
    assert d["has_signature"] is True
    assert any(c["clause_type"] == "termination" and c["ref"] == "12.4" for c in d["clauses"])


def _page(client, ingested, key, n=1):
    r = client.get(f"/api/v1/documents/{ingested[key]}/pages/{n}", headers=headers("acme_reader"))
    assert r.status_code == 200
    return r.json()


def test_every_annotation_points_at_its_source_text(client, ingested):
    """Provenance: page, block and character span of every field, entity and clause, checked against the page."""
    checked = 0
    for key in ("invoice", "contract", "nda_ca", "lease", "ticket", "email"):
        d = doc(client, ingested, key)
        pages = {}
        for item in [*d["fields"], *d["entities"], *d["clauses"]]:
            if item.get("char_start") is None:
                continue
            page = pages.setdefault(item["page"], _page(client, ingested, key, item["page"]))
            span = page["text"][item["char_start"]:item["char_end"]]
            block = next(b for b in page["blocks"] if b["ordinal"] == item["block"])
            assert block["char_start"] <= item["char_start"] < block["char_end"] + 1, (key, item)
            if item.get("type") in ("person", "organization", "identifier"):
                assert span == item["value"], (key, item, span)
            checked += 1
    assert checked > 20


def test_relations_are_extracted_with_qualifiers(client, ingested):
    rels = doc(client, ingested, "contract")["relations"]
    term = next(r for r in rels if r["predicate"] == "may_terminate")
    assert term["subject"].lower() == "either party" and "60" in term["qualifiers"]["notice"]
    assert any(r["predicate"] == "signed" and r["subject"] == "Omar Haddad" for r in rels)
    assert any(r["predicate"] == "governed_by" and r["object"] == "Saudi Arabia" for r in rels)
    ticket = doc(client, ingested, "ticket")["relations"]
    assert any(r["predicate"] == "attribute" and r["subject"] == "technician" for r in ticket)


def test_pages_have_blocks_and_tables_with_cells(client, ingested):
    page = _page(client, ingested, "payroll")
    assert page["tables"] and page["tables"][0]["rows"][1][:2] == ["Employees paid", "42"]
    assert [b["type"] for b in page["blocks"]] == ["heading", "table"]
    pdf = _page(client, ingested, "invoice")
    assert all(b["bbox"] for b in pdf["blocks"]) and pdf["text"][pdf["blocks"][0]["char_start"]:pdf["blocks"][0]["char_end"]]


def test_processing_versions_are_recorded(client, ingested):
    d = doc(client, ingested, "memo")
    assert d["versions"][0]["status"] == "current" and d["versions"][0]["components"]["pipeline"]
    assert d["stale"] == []


def test_ocr_tolerant_jurisdiction(client, ingested):
    assert ("jurisdiction", "California") in fields(doc(client, ingested, "ocr_noisy"))


def test_scanned_ticket_is_ocrd_with_confidence(client, ingested):
    d = doc(client, ingested, "ticket")
    assert d["pages"][0]["kind"] == "ocr" and d["pages"][0]["ocr_confidence"] > 0.5
    r = client.get(f"/api/v1/documents/{ingested['ticket']}/pages/1", headers=headers("acme_reader"))
    assert "Priya Raman" in r.json()["text"] and r.json()["words"]


def test_visual_signature_mark_on_scanned_letter(client, ingested):
    d = doc(client, ingested, "approval")
    assert d["has_signature"] is True
    marks = [f for f in d["fields"] if f["name"] == "signature_mark"]
    assert marks and marks[0]["method"] == "visual" and marks[0]["confidence"] >= 0.6


def test_email_attachment_is_part_of_the_document(client, ingested):
    d = doc(client, ingested, "email")
    assert d["page_count"] == 2
    r = client.get(f"/api/v1/documents/{ingested['email']}/pages/2", headers=headers("acme_reader"))
    assert "Jeddah" in r.json()["text"]


def test_original_download_and_page_image(client, ingested):
    r = client.get(f"/api/v1/documents/{ingested['invoice']}/original", headers=headers("acme_reader"))
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    r = client.get(f"/api/v1/documents/{ingested['invoice']}/pages/1/image", headers=headers("acme_reader"))
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    r = client.get(f"/api/v1/documents/{ingested['memo']}/pages/1/image", headers=headers("acme_reader"))
    assert r.status_code == 404


def test_duplicate_upload_is_detected(client, ingested, corpus_dir):
    path = next(p for d, p in corpus_dir if d.key == "invoice")
    r = client.post("/api/v1/documents", headers=headers("acme_uploader"), files=[("files", (path.name, path.read_bytes()))])
    out = r.json()["documents"][0]
    assert out["duplicate"] is True and out["id"] == ingested["invoice"]


def test_same_file_in_another_tenant_is_a_separate_document(client, ingested, corpus_dir):
    path = next(p for d, p in corpus_dir if d.key == "memo")
    r = client.post("/api/v1/documents", headers=headers("globex_uploader"), files=[("files", ("memo.txt", path.read_bytes()))])
    out = r.json()["documents"][0]
    assert out["duplicate"] is False and out["id"] != ingested["memo"]
    client.delete(f"/api/v1/documents/{out['id']}", headers=headers("globex_uploader"))


def test_unsupported_and_empty_files_are_rejected(client, engine_env):
    r = client.post("/api/v1/documents", headers=headers("acme_uploader"),
                    files=[("files", ("x.bin", b"\x00\x01\x02\x00binary")), ("files", ("empty.txt", b""))])
    assert [d["status"] for d in r.json()["documents"]] == ["rejected", "rejected"]


def test_corrupted_pdf_is_marked_failed_with_reason(client, engine_env):
    r = client.post("/api/v1/documents", headers=headers("acme_uploader"), files=[("files", ("broken.pdf", b"%PDF-1.4 garbage"))])
    d = r.json()["documents"][0]
    got = client.get(f"/api/v1/documents/{d['id']}", headers=headers("acme_reader")).json()
    assert got["status"] == "failed" and "PDF" in got["error"]
    client.delete(f"/api/v1/documents/{d['id']}", headers=headers("acme_uploader"))


def test_upload_delete_and_reprocess_lifecycle(client, engine_env):
    body = b"Quarterly toner audit for the Westbrook account. Reference WB-7731."
    r = client.post("/api/v1/documents", headers=headers("acme_uploader"), files=[("files", ("audit.txt", body))])
    doc_id = r.json()["documents"][0]["id"]
    def q():
        return client.post("/api/v1/query", headers=headers("acme_reader"), json={"q": "WB-7731"}).json()["results"]
    assert [x["document_id"] for x in q()] == [doc_id]
    r = client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=headers("acme_uploader"))
    assert r.status_code == 200
    detail = client.get(f"/api/v1/documents/{doc_id}", headers=headers("acme_reader")).json()
    assert detail["status"] == "indexed"
    assert [(v["number"], v["status"]) for v in detail["versions"]] == [(2, "current"), (1, "superseded")]
    assert [x["document_id"] for x in q()] == [doc_id]
    assert client.delete(f"/api/v1/documents/{doc_id}", headers=headers("acme_uploader")).status_code == 200
    assert q() == []
    assert client.get(f"/api/v1/documents/{doc_id}", headers=headers("acme_reader")).status_code == 404


def test_list_filter_paginate_and_stats(client, ingested):
    r = client.get("/api/v1/documents?doc_type=nda&limit=1", headers=headers("acme_reader")).json()
    assert r["total"] == 2 and len(r["documents"]) == 1
    r = client.get("/api/v1/documents?q=invoice", headers=headers("acme_reader")).json()
    assert {d["filename"] for d in r["documents"]} == {"Tax_Invoice_INV-2026-00481.pdf", "Invoice_Gulf_Trading.docx"}
    s = client.get("/api/v1/stats", headers=headers("acme_reader")).json()
    assert s["indexed_documents"] >= 21 and s["chunks"] >= 21


def test_roles_and_authentication(client, ingested):
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    r = client.post("/api/v1/documents", headers=headers("acme_reader"), files=[("files", ("a.txt", b"hello"))])
    assert r.status_code == 403
    assert client.delete(f"/api/v1/documents/{ingested['memo']}", headers=headers("acme_reader")).status_code == 403
    assert client.get("/api/v1/me", headers=headers("service")).status_code == 400
    me = client.get("/api/v1/me", headers=headers("service", "globex")).json()
    assert me["tenant_id"] == "globex"


def test_url_ingest_refuses_private_addresses(client, engine_env):
    r = client.post("/api/v1/documents/url", headers=headers("acme_uploader"), json={"url": "http://127.0.0.1:9091/healthz"})
    assert r.status_code == 400 and "non-public" in r.json()["detail"]


def test_url_ingest_is_refused_when_disabled(client, engine_env, monkeypatch):
    from docintel.config import get_settings
    monkeypatch.setattr(get_settings(), "url_fetch_enabled", False)
    r = client.post("/api/v1/documents/url", headers=headers("acme_uploader"), json={"url": "https://example.org/a.pdf"})
    assert r.status_code == 403


def test_health_endpoints(client, engine_env):
    assert client.get("/health/live").json() == {"status": "ok"}
    r = client.get("/health/ready")
    assert r.status_code == 200 and all(c["ok"] for c in r.json()["checks"].values())


def test_ui_is_served_with_a_strict_csp(client, engine_env):
    r = client.get("/")
    assert r.status_code == 200 and "Document Intelligence" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


def test_readiness_reports_an_unwritable_object_store(client, monkeypatch):
    from docintel.storage import objects

    class ReadOnly(objects.ObjectStore):
        def check_writable(self):
            raise PermissionError("read-only volume")

    assert client.get("/health/ready").json()["checks"]["object_store"]["ok"] is True
    monkeypatch.setattr(objects, "_store", ReadOnly(objects.get_object_store().root))
    r = client.get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["object_store"] == {"ok": False, "error": "PermissionError"}
