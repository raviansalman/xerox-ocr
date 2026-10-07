"""Health checks: liveness vs readiness vs dependency failure (no services needed)."""
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import health

OK = {n: (lambda n=n: health.CheckResult(n, True, "ok")) for n in ("redis", "milvus", "embedder", "workers")}


def down(name, status="unreachable"):
    return lambda: health.CheckResult(name, False, status, "simulated")


@pytest.fixture(scope="module")
def client():
    import ultimate_ui

    return TestClient(ultimate_ui.app)


@pytest.fixture
def checks(monkeypatch):
    def _set(**overrides):
        for name in ("redis", "milvus", "embedder", "workers"):
            monkeypatch.setitem(health.DEFAULT_CHECKS, name, overrides.get(name, OK[name]))
    return _set


# ---- endpoint scenarios --------------------------------------------------------

@pytest.mark.parametrize("path", ["/health/ready", "/health"])
def test_all_dependencies_healthy(client, checks, path):
    checks()
    r = client.get(path)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy" and body["ready"] is True and body["failing"] == []
    assert set(body["checks"]) == {"redis", "milvus", "embedder", "workers"}


@pytest.mark.parametrize("dep", ["redis", "milvus", "embedder"])
@pytest.mark.parametrize("path", ["/health/ready", "/health"])
def test_critical_dependency_down_is_not_ready(client, checks, dep, path):
    checks(**{dep: down(dep)})
    r = client.get(path)
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy" and body["ready"] is False and body["failing"] == [dep]
    assert body["services"][dep] == "unreachable"


def test_worker_down_is_degraded_but_ready(client, checks):
    checks(workers=down("workers", "no_workers"))
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["failing"] == ["workers"]


def test_partial_failure_reports_every_failing_dependency(client, checks):
    checks(milvus=down("milvus", "collection_loading"), workers=down("workers", "no_workers"))
    body = client.get("/health/ready").json()
    assert body["status"] == "unhealthy"
    assert body["failing"] == ["milvus", "workers"]
    assert body["services"]["milvus"] == "collection_loading" and body["services"]["redis"] == "healthy"


def test_hung_dependency_times_out(client, checks, monkeypatch):
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SEC", "0.2")
    checks(redis=lambda: (time.sleep(3), health.CheckResult("redis", True, "ok"))[1])
    t = time.time()
    r = client.get("/health/ready")
    assert time.time() - t < 2.5
    assert r.status_code == 503 and r.json()["services"]["redis"] == "timeout"


def test_liveness_makes_no_dependency_calls(client, checks):
    def boom():
        raise AssertionError("liveness must not call dependencies")
    checks(redis=boom, milvus=boom, embedder=boom, workers=boom)
    r = client.get("/health/live")
    assert r.status_code == 200 and r.json()["status"] == "alive"


def test_health_no_longer_claims_unchecked_dependencies(client, checks):
    # Formerly KD-OPS-02: celery/redis were hardcoded "healthy".
    checks(redis=down("redis"), workers=down("workers", "no_workers"))
    services = client.get("/health").json()["services"]
    assert services["redis"] != "healthy" and services["workers"] != "healthy"


# ---- individual checks ---------------------------------------------------------

def test_redis_check(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert health.check_redis().status == "not_configured"
    monkeypatch.setenv("REDIS_URL", "redis://:secretpw@127.0.0.1:1/0")
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT_SEC", "0.5")
    r = health.check_redis()
    assert not r.ok and r.status == "unreachable" and "secretpw" not in r.detail


def _fake_milvus(monkeypatch, *, exists=True, state="Loaded", fail=None):
    import pymilvus

    def maybe_fail(*a, **k):
        if fail:
            raise fail
    monkeypatch.setattr(pymilvus.connections, "has_connection", lambda alias: True)
    monkeypatch.setattr(pymilvus.connections, "disconnect", lambda alias: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda *a, **k: (maybe_fail(), exists)[1])
    monkeypatch.setattr(pymilvus.utility, "load_state", lambda *a, **k: f"LoadState.{state}")


@pytest.mark.parametrize("exists,state,ok,status", [
    (True, "Loaded", True, "ok"),
    (True, "NotLoad", True, "ok"),
    (False, "Loaded", True, "ok"),
    (True, "Loading", False, "collection_loading"),
])
def test_milvus_check_states(monkeypatch, exists, state, ok, status):
    _fake_milvus(monkeypatch, exists=exists, state=state)
    r = health.check_milvus()
    assert (r.ok, r.status) == (ok, status)


def test_milvus_check_unreachable(monkeypatch):
    _fake_milvus(monkeypatch, fail=ConnectionError("refused"))
    r = health.check_milvus()
    assert not r.ok and r.status == "unreachable"


class _Resp:
    def __init__(self, dim):
        self.dim = dim

    def raise_for_status(self):
        pass

    def json(self):
        return {"embeddings": [[0.0] * self.dim]}


@pytest.mark.parametrize("dim,ok,status", [(768, True, "ok"), (384, False, "wrong_dimension")])
def test_embedder_check_verifies_model_dimension(monkeypatch, dim, ok, status):
    import requests
    monkeypatch.setenv("EMBEDDER_URL", "http://embedder:8080/embed")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(dim))
    r = health.check_embedder()
    assert (r.ok, r.status) == (ok, status)


def test_embedder_check_failures(monkeypatch):
    import requests
    monkeypatch.delenv("EMBEDDER_URL", raising=False)
    monkeypatch.delenv("EMBEDDER_SEARCH_URL", raising=False)
    assert health.check_embedder().status == "not_configured"
    monkeypatch.setenv("EMBEDDER_URL", "http://embedder:8080/embed")

    def refuse(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", refuse)
    assert health.check_embedder().status == "unreachable"


def test_workers_check(monkeypatch):
    from src.ultimate_celery_app import celery_app

    monkeypatch.setattr(celery_app.control, "ping", lambda timeout: [])
    r = health.check_workers()
    assert not r.ok and r.status == "no_workers"

    monkeypatch.setattr(celery_app.control, "ping", lambda timeout: [{"celery@pdf1": {"ok": "pong"}}])
    monkeypatch.setattr(celery_app.control, "inspect", lambda timeout: SimpleNamespace(
        active_queues=lambda: {"celery@pdf1": [{"name": "ultimate_pdf"}, {"name": "ultimate_pdf_large"}]}))
    r = health.check_workers()
    assert r.ok and r.info["workers"] == ["celery@pdf1"]
    assert "ultimate_pdf" not in r.info["queues_without_workers"]
    assert "ultimate_ocr" in r.info["queues_without_workers"]
