"""Test configuration.

Tiers:
  unit         no services (default)
  integration  PostgreSQL + vector index + embedder: DOCINTEL_TEST_INTEGRATION=1 and DOCINTEL_TEST_DATABASE_URL
               (an application role WITHOUT superuser/BYPASSRLS, so row-level security is really enforced).
               Vector index: Milvus when DOCINTEL_TEST_MILVUS_URI is set, else the in-memory store.
               Embedder: the deterministic stand-in, or a real model with DOCINTEL_TEST_EMBEDDER_URL +
               DOCINTEL_TEST_EMBEDDING_MODEL.
  e2e          browser tests of the UI (DOCINTEL_TEST_E2E=1, implies integration)
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = os.environ.get("DOCINTEL_TEST_INTEGRATION") == "1" or os.environ.get("DOCINTEL_TEST_E2E") == "1"
E2E = os.environ.get("DOCINTEL_TEST_E2E") == "1"
# DOCINTEL_TEST_VECTORS=disabled runs the suite with semantic search switched off: exact, lexical, structured and
# contextual retrieval must still pass (tests marked "semantic" are skipped)
VECTORS_DISABLED = os.environ.get("DOCINTEL_TEST_VECTORS") == "disabled"

KEYS = {
    "acme_reader": ("acme", ["reader"]), "acme_uploader": ("acme", ["uploader"]), "acme_admin": ("acme", ["admin"]),
    "globex_reader": ("globex", ["reader"]), "globex_uploader": ("globex", ["uploader"]),
    "haystack_uploader": ("haystack", ["uploader"]), "carol_reader": ("carol", ["reader"]),
    "quill_uploader": ("tenantquill", ["uploader", "reader"]),
    "vault_uploader": ("tenantvault", ["uploader", "reader"]), "vault_admin": ("tenantvault", ["admin"]),
    "arabic_uploader": ("tenantarabic", ["uploader", "reader"]),
    "service": (None, ["service"]),
}


def _ocr_languages() -> str:
    """ara+eng when the Arabic Tesseract model is installed (as in the image), so the English suite also runs with
    the deployed OCR setting; eng otherwise."""
    try:
        import pytesseract
        return "ara+eng" if "ara" in pytesseract.get_languages(config="") else "eng"
    except Exception:
        return "eng"


def raw_key(name: str) -> str:
    return f"test-key-{name}"


def headers(name: str, tenant: str | None = None) -> dict:
    h = {"X-API-Key": raw_key(name)}
    if tenant:
        h["X-Tenant-Id"] = tenant
    return h


def _keys_json() -> str:
    return json.dumps([{"name": n, "key_sha256": hashlib.sha256(raw_key(n).encode()).hexdigest(),
                        **({"tenant": t} if t else {}), "roles": r} for n, (t, r) in KEYS.items()])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port: int) -> None:
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError("test server did not start")


def pytest_configure(config):
    os.environ["DOCINTEL_ENVIRONMENT"] = "test"
    os.environ["DOCINTEL_API_KEYS"] = _keys_json()
    os.environ["DOCINTEL_AUTH_MODE"] = "keys"
    os.environ["DOCINTEL_LOG_JSON"] = "false"
    os.environ.setdefault("DOCINTEL_TASK_MODE", "inline")
    os.environ.setdefault("DOCINTEL_OCR_LANGUAGES", _ocr_languages())
    os.environ["DOCINTEL_URL_FETCH_ENABLED"] = "true"           # off by default; the SSRF guard is tested here
    if INTEGRATION:
        url = os.environ.get("DOCINTEL_TEST_DATABASE_URL")
        if not url:
            raise pytest.UsageError("DOCINTEL_TEST_DATABASE_URL is required for integration tests")
        os.environ["DOCINTEL_DATABASE_URL"] = url
        os.environ["DOCINTEL_STORAGE_DIR"] = str(ROOT / ".pytest_cache" / f"objects_{os.getpid()}")
        if VECTORS_DISABLED:
            os.environ["DOCINTEL_VECTOR_BACKEND"] = "disabled"     # no vector index and no embedder at all
            os.environ.pop("DOCINTEL_EMBEDDER_URL", None)
        elif os.environ.get("DOCINTEL_TEST_MILVUS_URI"):
            os.environ["DOCINTEL_VECTOR_BACKEND"] = "milvus"
            os.environ["DOCINTEL_MILVUS_URI"] = os.environ["DOCINTEL_TEST_MILVUS_URI"]
            os.environ["DOCINTEL_MILVUS_COLLECTION_PREFIX"] = f"docintel_test_{os.getpid()}"
            os.environ["DOCINTEL_MILVUS_CONSISTENCY"] = "Strong"
        else:
            os.environ["DOCINTEL_VECTOR_BACKEND"] = "memory"
        if VECTORS_DISABLED:
            pass
        elif os.environ.get("DOCINTEL_TEST_EMBEDDER_URL"):
            os.environ["DOCINTEL_EMBEDDER_URL"] = os.environ["DOCINTEL_TEST_EMBEDDER_URL"]
            os.environ["DOCINTEL_EMBEDDING_MODEL"] = os.environ.get("DOCINTEL_TEST_EMBEDDING_MODEL", "all-mpnet-base-v2")
        else:
            from tests.support.fake_embedder import app as fake
            port = _free_port()
            _serve(fake, port)
            os.environ["DOCINTEL_EMBEDDER_URL"] = f"http://127.0.0.1:{port}"
            os.environ["DOCINTEL_EMBEDDING_MODEL"] = "test-hash-768"
    from docintel.config import reset_settings
    from docintel.security import reset_auth_cache
    reset_settings()
    reset_auth_cache()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" in item.keywords and not INTEGRATION:
            item.add_marker(pytest.mark.skip(reason="set DOCINTEL_TEST_INTEGRATION=1 with services running"))
        if "semantic" in item.keywords and VECTORS_DISABLED:
            item.add_marker(pytest.mark.skip(reason="semantic search is disabled in this run"))
        if "e2e" in item.keywords and not E2E:
            item.add_marker(pytest.mark.skip(reason="set DOCINTEL_TEST_E2E=1"))


@pytest.fixture(scope="session")
def engine_env():
    """Fresh schema, vector collection and object store for the session."""
    from docintel.indexing import vectors
    from docintel.storage.db import get_db

    db = get_db()
    with db.system() as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public; CREATE EXTENSION IF NOT EXISTS pg_trgm")
    db.migrate()
    store = vectors.get_vector_store()
    if hasattr(store, "drop"):
        store.drop()
    yield db
    if hasattr(store, "drop"):
        store.drop()


@pytest.fixture(scope="session")
def client(engine_env):
    from fastapi.testclient import TestClient

    from docintel.api.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    from tests.fixtures.corpus import build_all
    d = tmp_path_factory.mktemp("corpus")
    return build_all(d)


@pytest.fixture(scope="session")
def ingested(client, corpus_dir):
    """Upload the whole corpus through the API (inline processing) and return {key: document_id}."""
    ids, by_tenant = {}, {}
    for doc, path in corpus_dir:
        by_tenant.setdefault(doc.tenant, []).append((doc, path))
    for tenant, items in by_tenant.items():
        key = f"{tenant}_uploader"
        for i in range(0, len(items), 25):
            files = [("files", (p.name, p.read_bytes())) for _, p in items[i:i + 25]]
            r = client.post("/api/v1/documents", headers=headers(key), files=files)
            assert r.status_code == 201, r.text
            for (doc, _), out in zip(items[i:i + 25], r.json()["documents"]):
                assert out.get("status") != "rejected", out
                ids[doc.key] = out["id"]
    for key, doc_id in ids.items():
        tenant = next(d.tenant for d, _ in corpus_dir if d.key == key)
        r = client.get(f"/api/v1/documents/{doc_id}", headers=headers(f"{tenant}_uploader"))
        assert r.json()["status"] == "indexed", (key, r.json().get("error"))
    return ids
