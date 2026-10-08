"""Adversarial tests of the API boundary: injection through every user-controlled string, tenant header and path
manipulation, hostile file names, cross-tenant probes of every index (postings, vocabulary, entities, structured
fields, vectors, settings, caches) and the metrics and request-id surfaces."""
import json
import uuid

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration

INJECTIONS = [
    "INV' OR 1=1 --", "'; DROP TABLE documents; --", "%' OR '%'='", "\\\" OR \"\"=\"", "acme')) UNION SELECT * FROM pages --",
    "$(rm -rf /)", "{{7*7}}", "${jndi:ldap://x/a}", "<script>alert(1)</script>", "tenant_id = 'globex'",
    "\u0000‮﻿", "a" * 1000, "* & | ! : ( ) <-> ''", "w:merger", "SET docintel.tenant = 'globex'",
]


def ask(client, q, key="acme_reader", **kw):
    r = client.post("/api/v1/query", headers=headers(key), json={"q": q, "limit": 50, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _doc_count(client, key="acme_reader"):
    return client.get("/api/v1/documents", headers=headers(key), params={"limit": 1}).json()["total"]


def _tenant_ids(client, key):
    return {d["id"] for d in client.get("/api/v1/documents", headers=headers(key), params={"limit": 500}).json()["documents"]}


@pytest.mark.parametrize("q", INJECTIONS)
def test_injection_in_questions_is_inert(client, ingested, q):
    before = _doc_count(client)
    out = ask(client, q, explain=True, answer=True)
    own = _tenant_ids(client, "acme_reader")
    assert all(r["document_id"] in own for r in out["results"])
    assert _doc_count(client) == before


@pytest.mark.parametrize("value", ["invoice' OR '1'='1", "x%' OR '1'='1", "nda); DELETE FROM documents; --"])
def test_injection_in_list_filters_is_inert(client, ingested, value):
    for param in ("doc_type", "status", "q", "collection"):
        r = client.get("/api/v1/documents", headers=headers("acme_reader"), params={param: value})
        assert r.status_code in (200, 422) and (r.status_code == 422 or r.json()["total"] == 0), (param, r.text)
    assert _doc_count(client) > 0


@pytest.mark.parametrize("header", ["globex", "acme", "../globex", "acme' OR 1=1", "*", ""])
def test_tenant_bound_keys_cannot_switch_tenant(client, ingested, header):
    r = client.post("/api/v1/query", headers={**headers("acme_reader"), "X-Tenant-Id": header}, json={"q": "Globex confidential merger"})
    if header in ("acme", ""):
        assert r.status_code == 200 and all("Globex" not in x["filename"] for x in r.json()["results"])
    else:
        assert r.status_code in (400, 403), (header, r.status_code)


@pytest.mark.parametrize("tenant", ["../acme", "acme;globex", "ACME' --", "a" * 200, "acme\nglobex"])
def test_service_keys_reject_malformed_tenants(client, ingested, tenant):
    r = client.post("/api/v1/query", headers=headers("service", tenant), json={"q": "INV-2026-00481"})
    assert r.status_code in (400, 403)


@pytest.mark.parametrize("path", ["../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "1 OR 1=1", "%00", str(uuid.uuid4())])
def test_document_paths_cannot_escape(client, ingested, path):
    for suffix in ("", "/original", "/pages/1", "/pages/1/image"):
        r = client.get(f"/api/v1/documents/{path}{suffix}", headers=headers("acme_reader"))
        assert r.status_code == 404, (path, suffix, r.status_code)


def test_other_tenants_documents_are_404_on_every_route(client, ingested):
    globex_id = ingested["globex_release"]
    for method, suffix in [("get", ""), ("get", "/original"), ("get", "/pages/1"), ("get", "/pages/1/image"),
                           ("post", "/reprocess"), ("delete", "")]:
        r = getattr(client, method)(f"/api/v1/documents/{globex_id}{suffix}", headers=headers("acme_admin"))
        assert r.status_code == 404, (method, suffix, r.status_code)
    assert globex_id in _tenant_ids(client, "globex_reader")          # and it still exists


def test_hostile_file_names_are_neutralized(client, engine_env):
    names = ["../../../../etc/passwd", "..\\..\\windows\\system.ini", 'evil".html', "line\r\nbreak.txt", "..", ".",
             "‮txt.exe", "x" * 400 + ".txt"]
    files = [("files", (n, f"content {i} for hostile name test".encode())) for i, n in enumerate(names)]
    r = client.post("/api/v1/documents", headers=headers("vault_uploader"), files=files)
    assert r.status_code == 201, r.text
    stored = [d["filename"] for d in r.json()["documents"]]
    for name in stored:
        assert "/" not in name and "\\" not in name and '"' not in name and "\r" not in name and "\n" not in name
        assert name not in (".", "..") and len(name) <= 255 and name.isprintable()
    doc = r.json()["documents"][2]
    got = client.get(f"/api/v1/documents/{doc['id']}/original", headers=headers("vault_uploader"))
    assert got.status_code == 200 and got.headers["content-disposition"].startswith("attachment")
    assert "sandbox" in got.headers["content-security-policy"] and got.headers["x-content-type-options"] == "nosniff"


def test_uploaded_html_is_never_served_inline(client, engine_env):
    html = b"<html><body><script>fetch('/api/v1/documents')</script>Quarterly memo</body></html>"
    r = client.post("/api/v1/documents", headers=headers("vault_uploader"), files=[("files", ("page.html", html))])
    doc_id = r.json()["documents"][0]["id"]
    got = client.get(f"/api/v1/documents/{doc_id}/original", headers=headers("vault_uploader"))
    assert got.headers["content-disposition"].startswith("attachment")
    assert "default-src 'none'" in got.headers["content-security-policy"]
    page = client.get(f"/api/v1/documents/{doc_id}/pages/1", headers=headers("vault_uploader")).json()
    assert "<script>" not in page["text"] and "fetch(" not in page["text"]


@pytest.mark.parametrize("q,foreign", [
    ("merger", "globex"), ("mergr", "globex"), ("Globex", "globex"),            # postings, vocabulary typo, entities
    ("heron", "haystack"), ("hern protocl", "haystack"), ("7781", "haystack"),
    ("How many documents mention Globex?", "globex"), ("documents signed by John Smith", "globex"),
])
def test_no_index_answers_from_another_tenant(client, ingested, corpus_dir, q, foreign):
    foreign_ids = {ingested[d.key] for d, _ in corpus_dir if d.tenant == foreign}
    for key in ("acme_reader", "carol_reader"):
        out = ask(client, q, key=key, explain=True, answer=True)
        assert not {r["document_id"] for r in out["results"]} & foreign_ids, (q, key)
        text = json.dumps(out)
        assert not any(i in text for i in foreign_ids), (q, key)
        if out["answer"].get("kind") == "number":
            assert out["answer"]["value"] == 0 or key == "acme_reader"


def test_typo_correction_never_suggests_another_tenants_words(client, ingested):
    # "mergr" is one edit from a word that exists only in another tenant's documents
    out = ask(client, "mergr", explain=True)
    assert out["results"] == [] and "merger" not in json.dumps(out).lower()


def test_tenant_caches_are_separate(client, ingested):
    """Per-tenant caches (unit counts for scoring, packs, settings) never carry one tenant's state to another."""
    a = ask(client, "INV-2026-00481", explain=True)
    g = ask(client, "INV-2026-00481", key="globex_reader", explain=True)
    assert a["results"][0]["filename"] != g["results"][0]["filename"]
    r = client.put("/api/v1/settings", headers=headers("vault_admin"),
                   json={"packs": ["business"], "examples": [{"label": "Vault", "q": "Vault example question"}]})
    assert r.status_code == 200, r.text
    assert "Vault example" not in client.get("/api/v1/settings", headers=headers("acme_reader")).text
    assert client.get("/api/v1/settings", headers=headers("vault_uploader")).json()["packs"] == ["business"]


def test_query_body_rejects_unknown_or_oversized_fields(client, ingested):
    for body in ({"q": "x", "tenant": "globex"}, {"q": "x" * 1001}, {"q": ""}, {"q": "x", "limit": 10_000},
                 {"q": ["x"]}, {"q": "x", "explain": "yes please"}):
        r = client.post("/api/v1/query", headers=headers("acme_reader"), json=body)
        assert r.status_code == 422, (body, r.status_code)


@pytest.mark.parametrize("rid", ["abc\r\nX-Injected: 1", "<script>", "a" * 500, "ok-id.123"])
def test_request_ids_are_validated(client, rid):
    r = client.get("/health/live", headers={"X-Request-ID": rid})
    got = r.headers["x-request-id"]
    assert got == rid if rid == "ok-id.123" else got != rid and got.isalnum()
    assert "x-injected" not in {k.lower() for k in r.headers}


def test_metrics_require_the_token_and_carry_no_tenant_data(client, ingested, monkeypatch):
    from docintel.config import get_settings
    monkeypatch.setattr(get_settings(), "metrics_token", "scrape-secret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"})
    assert r.status_code == 200 and "docintel_queries_total" in r.text and "docintel_uploads_total" in r.text
    for secret in ("acme", "globex", "haystack", "INV-2026", "test-key"):
        assert secret not in r.text
    monkeypatch.setattr(get_settings(), "metrics_enabled", False)
    assert client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"}).status_code == 404


def test_roles_are_enforced(client, ingested):
    assert client.post("/api/v1/documents", headers=headers("acme_reader"),
                       files=[("files", ("x.txt", b"hello"))]).status_code == 403
    assert client.put("/api/v1/settings", headers=headers("acme_uploader"), json={"packs": []}).status_code == 403
    assert client.delete(f"/api/v1/documents/{ingested['memo']}", headers=headers("acme_reader")).status_code == 403
    assert client.post("/api/v1/query", headers={"X-API-Key": "test-key-nobody"}, json={"q": "x"}).status_code == 401
    assert client.post("/api/v1/query", json={"q": "x"}).status_code == 401


def test_oversized_request_bodies_are_refused_before_spooling(client, monkeypatch):
    from docintel.config import get_settings
    monkeypatch.setattr(get_settings(), "max_request_bytes", 50_000)
    big = b"x" * 60_000
    r = client.post("/api/v1/documents", headers=headers("vault_uploader"), files=[("files", ("big.txt", big))])
    assert r.status_code == 413, ("declared", r.status_code, r.text)
    def stream():                                                  # valid multipart, sent without Content-Length
        yield b'--b\r\nContent-Disposition: form-data; name="files"; filename="x.txt"\r\nContent-Type: text/plain\r\n\r\n'
        for _ in range(8):
            yield b"x" * 10_000
        yield b"\r\n--b--\r\n"
    chunked = client.post("/api/v1/documents", headers={**headers("vault_uploader"), "content-type": "multipart/form-data; boundary=b"},
                          content=stream())
    assert chunked.status_code == 413, ("chunked", chunked.status_code, chunked.text)
    r = client.post("/api/v1/query", headers=headers("acme_reader"), content=b'{"q": "' + b"a" * 2_000_000 + b'"}')
    assert r.status_code == 413, ("query", r.status_code, r.text)
    ok = client.post("/api/v1/documents", headers=headers("vault_uploader"), files=[("files", ("small.txt", b"small note"))])
    assert ok.status_code == 201
