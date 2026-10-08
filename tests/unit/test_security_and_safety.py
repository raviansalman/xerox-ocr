import hashlib
import json

import numpy as np
import pytest

from docintel import config
from docintel import security as S
from docintel.indexing.vectors import MemoryVectorStore
from docintel.net_safety import UnsafeURLError, check_url, redact_url


def _key(name, tenant, roles):
    return {"name": name, "key_sha256": hashlib.sha256(name.encode()).hexdigest(), **({"tenant": tenant} if tenant else {}), "roles": roles}


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Tests here change the environment; never leave their settings or keys cached for later tests."""
    yield
    config.reset_settings()
    S.reset_auth_cache()


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("DOCINTEL_API_KEYS", json.dumps([_key("r", "acme", ["reader"]), _key("svc", None, ["service"])]))
    config.reset_settings()
    S.reset_auth_cache()
    yield
    config.reset_settings()
    S.reset_auth_cache()


def test_tenant_comes_from_the_key(keys):
    ctx = S.authenticate({"x-api-key": "r"})
    assert ctx.tenant_id == "acme" and ctx.has("reader") and not ctx.has("uploader")
    assert S.authenticate({"authorization": "Bearer r"}).tenant_id == "acme"


def test_tenant_key_cannot_switch_tenant(keys):
    with pytest.raises(S.AuthError) as e:
        S.authenticate({"x-api-key": "r", "x-tenant-id": "globex"})
    assert e.value.status_code == 403


def test_service_key_requires_valid_tenant_header(keys):
    with pytest.raises(S.AuthError) as e:
        S.authenticate({"x-api-key": "svc"})
    assert e.value.status_code == 400
    with pytest.raises(S.AuthError):
        S.authenticate({"x-api-key": "svc", "x-tenant-id": "bad tenant'"})
    assert S.authenticate({"x-api-key": "svc", "x-tenant-id": "globex"}).tenant_id == "globex"


def test_missing_and_wrong_keys(keys):
    for h, code in (({}, 401), ({"x-api-key": "nope"}, 401)):
        with pytest.raises(S.AuthError) as e:
            S.authenticate(h)
        assert e.value.status_code == code


def test_role_requirement(keys):
    with pytest.raises(S.AuthError) as e:
        S.authenticate({"x-api-key": "r"}).require("uploader")
    assert e.value.status_code == 403


def test_no_keys_configured_fails_closed(monkeypatch):
    monkeypatch.setenv("DOCINTEL_API_KEYS", "")
    config.reset_settings()
    S.reset_auth_cache()
    with pytest.raises(S.AuthError) as e:
        S.authenticate({"x-api-key": "anything"})
    assert e.value.status_code == 503


def test_dev_mode_is_refused_in_production(monkeypatch):
    monkeypatch.setenv("DOCINTEL_AUTH_MODE", "dev")
    monkeypatch.setenv("DOCINTEL_ENVIRONMENT", "production")
    config.reset_settings()
    with pytest.raises(S.AuthError):
        S.authenticate({})
    monkeypatch.setenv("DOCINTEL_ENVIRONMENT", "development")
    config.reset_settings()
    assert S.authenticate({}).tenant_id == "default"
    config.reset_settings()


@pytest.mark.parametrize("entry", [
    {"name": "x", "key_sha256": "abc", "tenant": "t", "roles": ["reader"]},
    {"name": "x", "key_sha256": "a" * 64, "roles": ["reader"]},
    {"name": "x", "key_sha256": "a" * 64, "tenant": "t", "roles": ["root"]},
])
def test_invalid_key_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        S.build_index([entry])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/x", "http://169.254.169.254/latest/meta-data",
                                 "http://user:pw@example.com/x", "ftp://example.com/a", "http://10.0.0.5/"])
def test_unsafe_urls_are_rejected(url):
    with pytest.raises(UnsafeURLError):
        check_url(url)


@pytest.mark.parametrize("address, public", [
    ("8.8.8.8", True), ("2606:4700::1111", True),
    ("::ffff:10.0.0.1", False), ("::ffff:8.8.8.8", True),           # IPv4-mapped
    ("64:ff9b::a9fe:a9fe", False), ("64:ff9b::808:808", True),       # NAT64 to 169.254.169.254 / 8.8.8.8
    ("2002:0a00:0001::1", False), ("2002:0808:0808::1", False),      # 6to4 is refused altogether
    ("fd00::1", False), ("ff02::1", False)])
def test_ipv6_translation_forms_are_checked_by_their_ipv4_address(address, public):
    import ipaddress

    from docintel.net_safety import public_address
    assert public_address(ipaddress.ip_address(address)) is public


def _serve(handler_body):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            handler_body(self)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)  # hardcoded-ok(ip-address): local test server
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_downloads_are_capped_in_size_and_time_and_ignore_proxy_settings(monkeypatch):
    import time

    from docintel.config import get_settings
    from docintel.net_safety import fetch
    monkeypatch.setattr(get_settings(), "url_allow_private", True)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:1")          # must not be used
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")

    def endless(h):                                                     # no Content-Length, more than the limit
        h.send_response(200)
        h.end_headers()
        for _ in range(64):
            h.wfile.write(b"x" * 65536)

    def trickle(h):                                                     # a byte at a time, never finishing
        h.send_response(200)
        h.end_headers()
        for _ in range(600):
            h.wfile.write(b"x")
            h.wfile.flush()
            time.sleep(0.05)

    srv = _serve(lambda h: endless(h) if h.path == "/big" else trickle(h))
    base = f"http://127.0.0.1:{srv.server_address[1]}"  # hardcoded-ok(ip-address,url): local test server
    try:
        chunks, _ = fetch(f"{base}/big", max_bytes=1_000_000)
        with pytest.raises(UnsafeURLError, match="byte limit"):
            b"".join(chunks)
        chunks, _ = fetch(f"{base}/slow", max_bytes=1_000_000, deadline_sec=0.5)
        started = time.monotonic()
        with pytest.raises(UnsafeURLError, match="in time"):
            b"".join(chunks)
        assert time.monotonic() - started < 5                           # not when the server finally stops
    finally:
        srv.shutdown()


def test_redact_url():
    assert redact_url("https://files.example.com/a/b.pdf?sig=SECRET") == "https://files.example.com/a/b.pdf?…"


def test_memory_vector_store_is_tenant_scoped_and_validates_input():
    store = MemoryVectorStore(3)
    doc_a, doc_b = "00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"
    store.upsert("acme", doc_a, "nda", ["a:0"], np.array([[1, 0, 0]], dtype=np.float32))
    store.upsert("globex", doc_b, "nda", ["b:0"], np.array([[1, 0, 0]], dtype=np.float32))
    hits = store.search("acme", np.array([1, 0, 0], dtype=np.float32), 10)
    assert [h.document_id for h in hits] == [doc_a]
    assert store.search("acme", np.array([1, 0, 0], dtype=np.float32), 10, document_ids=[doc_b]) == []
    with pytest.raises(ValueError):
        store.search('acme" or tenant_id != "x', np.array([1, 0, 0], dtype=np.float32), 10)
    with pytest.raises(ValueError):
        store.search("acme", np.array([1, 0, 0], dtype=np.float32), 10, document_ids=["1 or 1=1"])
    store.delete_document("acme", doc_a)
    assert store.search("acme", np.array([1, 0, 0], dtype=np.float32), 10) == []
