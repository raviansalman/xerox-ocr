"""SSRF guard, download/upload caps, and no destructive Milvus auto-migration (no services needed)."""
import socket

import pytest
from fastapi.testclient import TestClient
from pymilvus import DataType, FieldSchema

import src.net_safety as ns
import src.vector_db_milvus_server as vdb
from tests.support.auth import headers

PUBLIC = "93.184.216.34"
DNS = {"public.example": PUBLIC, "localhost": "127.0.0.1", "internal.corp": "10.1.2.3",
       "rebind.example": "169.254.169.254", "v6local.example": "::1"}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    def getaddrinfo(host, port, *a, **k):
        try:
            ip = DNS.get(host) or str(__import__("ipaddress").ip_address(host))
        except ValueError:
            raise socket.gaierror("no such host")
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(ns.socket, "getaddrinfo", getaddrinfo)
    for v in ("URL_FETCH_ALLOWED_HOSTS", "URL_FETCH_ALLOW_PRIVATE", "MAX_DOWNLOAD_BYTES"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://localhost:8000/admin", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/a.pdf", "http://internal.corp/a.pdf", "http://rebind.example/a", "http://[::1]/a",
    "http://v6local.example/a", "file:///etc/passwd", "ftp://public.example/a", "gopher://public.example/",
    "http://user:pw@public.example/a", "http:///nohost", "http://nonexistent.invalid/a",
])
def test_unsafe_urls_are_refused(url):
    with pytest.raises(ns.UnsafeURLError):
        ns.check_url(url)


def test_public_https_url_is_allowed():
    ns.check_url("https://public.example/bucket/a.pdf?X-Amz-Signature=abc")


def test_allowlist(monkeypatch):
    monkeypatch.setenv("URL_FETCH_ALLOWED_HOSTS", ".amazonaws.com,files.example.org")
    DNS.update({"bucket.s3.amazonaws.com": PUBLIC, "files.example.org": PUBLIC})
    ns.check_url("https://bucket.s3.amazonaws.com/a.pdf")
    ns.check_url("https://files.example.org/a.pdf")
    with pytest.raises(ns.UnsafeURLError, match="not in URL_FETCH_ALLOWED_HOSTS"):
        ns.check_url("https://public.example/a.pdf")


def test_private_targets_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("URL_FETCH_ALLOW_PRIVATE", "true")
    ns.check_url("http://10.0.0.5/a.pdf")


def test_redact_url_drops_signatures():
    assert ns.redact_url("https://b.s3.amazonaws.com/k/a.pdf?X-Amz-Signature=secret") == \
        "https://b.s3.amazonaws.com/k/a.pdf?…"
    assert "secret" not in ns.redact_url("https://u:secret@h/p")


class _Resp:
    def __init__(self, status=200, location=None, body=b"", headers=None):
        self.status_code = status
        self.headers = dict(headers or {})
        if location:
            self.headers["Location"] = location
        self._body = body
        self.is_redirect = location is not None

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(response=self)

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


def test_redirect_to_private_address_is_refused(monkeypatch):
    calls = []

    def fake_request(method, url, **kw):
        calls.append(url)
        assert kw["allow_redirects"] is False
        return _Resp(302, location="http://169.254.169.254/latest/meta-data/")
    monkeypatch.setattr(ns.requests, "request", fake_request)
    with pytest.raises(ns.UnsafeURLError):
        ns.safe_get("https://public.example/a.pdf")
    assert calls == ["https://public.example/a.pdf"]


def _download(monkeypatch, resp, **env):
    from src import ultimate_tasks
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ultimate_tasks, "safe_get", lambda url, **kw: resp)
    return ultimate_tasks.download_file("https://public.example/a.pdf", "f1")


def test_download_size_cap_streaming(monkeypatch):
    with pytest.raises(ValueError, match="MAX_DOWNLOAD_BYTES"):
        _download(monkeypatch, _Resp(body=b"x" * 5000), MAX_DOWNLOAD_BYTES="1000")


def test_download_size_cap_declared(monkeypatch):
    with pytest.raises(ValueError, match="MAX_DOWNLOAD_BYTES"):
        _download(monkeypatch, _Resp(body=b"x", headers={"Content-Length": "999999"}), MAX_DOWNLOAD_BYTES="1000")


def test_http_error_pages_are_not_indexed(monkeypatch):
    import requests
    with pytest.raises(requests.exceptions.HTTPError):
        _download(monkeypatch, _Resp(status=404, body=b"<html>Not Found</html>"))


def test_download_within_limits(monkeypatch):
    path, _ = _download(monkeypatch, _Resp(body=b"%PDF-1.4 hello"))
    assert open(path, "rb").read() == b"%PDF-1.4 hello"


# ---- API-level ---------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    import ultimate_ui

    return TestClient(ultimate_ui.app)


def test_process_rejects_internal_urls(client):
    r = client.post("/process", headers=headers("service"),
                    json={"userId": "alice", "fileId": "f1", "fileUrl": "http://169.254.169.254/latest/meta-data/"})
    assert r.status_code == 400 and "not allowed" in r.json()["detail"]


def test_upload_size_cap(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    r = client.post("/process-file", headers=headers("alice_uploader"),
                    files={"file": ("big.txt", b"x" * 5000, "text/plain")}, data={"fileId": "f-big"})
    assert r.status_code == 413
    assert list(tmp_path.iterdir()) == []


# ---- Milvus: no destructive auto-migration -----------------------------------

class _Schema:
    def __init__(self, dim=768, with_user=True):
        self.fields = [FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
                       FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)]
        if with_user:
            self.fields.append(FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=500))


@pytest.mark.parametrize("schema,match", [(_Schema(dim=384), "embedding dim 384"),
                                          (_Schema(with_user=False), "no user_id")])
def test_mismatched_existing_collection_fails_fast_without_dropping(monkeypatch, schema, match):
    dropped = []
    monkeypatch.setattr(vdb.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(vdb.utility, "drop_collection", lambda name: dropped.append(name))

    class C:
        def __init__(self, name):
            self.schema = schema

        def has_index(self):
            return True
    monkeypatch.setattr(vdb, "Collection", C)
    db = object.__new__(vdb.MilvusServerVectorDatabase)
    db.collection_name, db.vector_size, db.auto_load_on_init = "prod_docs", 768, False
    with pytest.raises(RuntimeError, match=match):
        db._initialize_collection()
    assert dropped == []
