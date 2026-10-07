#!/usr/bin/env python3
"""
Dependency health checks for the API.

  liveness   GET /health/live   the process serves HTTP; no dependency calls (container healthcheck)
  readiness  GET /health/ready  every critical dependency answers; 503 if one does not (load balancer)
  report     GET /health        same checks and status code as readiness, with legacy fields for the UI

Critical (503 when failing): redis, milvus, embedder. Non-critical (status "degraded", still 200):
workers, because search keeps working while ingestion is paused.

Every check is time-boxed (HEALTH_CHECK_TIMEOUT_SEC, default 3) and the checks run concurrently.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Tuple

CRITICAL = ("redis", "milvus", "embedder")
EXPECTED_QUEUES = [f"ultimate_{t}{s}" for t in ("pdf", "word", "powerpoint", "spreadsheet", "image", "ocr")
                   for s in ("", "_large")]
_MILVUS_ALIAS = "xocr-health"


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: str
    detail: str = ""
    latency_ms: float = 0.0
    critical: bool = False
    info: Dict = field(default_factory=dict)


def _timeout() -> float:
    try:
        return float(os.getenv("HEALTH_CHECK_TIMEOUT_SEC", "3"))
    except ValueError:
        return 3.0


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


# ---- individual checks ------------------------------------------------------

def check_redis() -> CheckResult:
    import redis

    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return CheckResult("redis", False, "not_configured", "REDIS_URL is not set")
    t = _timeout()
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=t, socket_timeout=t)
        client.ping()
        return CheckResult("redis", True, "ok")
    except Exception as e:  # never echo the URL: it carries the password
        return CheckResult("redis", False, "unreachable", _short(e))


def check_milvus() -> CheckResult:
    from pymilvus import connections, utility

    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    collection = os.getenv("DOC_COLLECTION", "ultimate_document_chunks")
    t = _timeout()
    try:
        if not connections.has_connection(_MILVUS_ALIAS):
            connections.connect(alias=_MILVUS_ALIAS, host=host, port=port, timeout=t)
        if not utility.has_collection(collection, using=_MILVUS_ALIAS, timeout=t):
            # Fresh deployment: the collection is created on first ingest/search.
            return CheckResult("milvus", True, "ok", info={"collection": collection, "load_state": "absent"})
        state = str(utility.load_state(collection, using=_MILVUS_ALIAS, timeout=t)).split(".")[-1]
        info = {"collection": collection, "load_state": state}
        if state == "Loading":
            # After a Milvus restart the server reports healthy while collections are still loading;
            # queries fail until loading completes, so this instance is not ready yet.
            return CheckResult("milvus", False, "collection_loading", "collection is still loading", info=info)
        return CheckResult("milvus", True, "ok", info=info)  # NotLoad is fine: loaded on first use
    except Exception as e:
        try:
            connections.disconnect(_MILVUS_ALIAS)  # reconnect cleanly next time
        except Exception:
            pass
        return CheckResult("milvus", False, "unreachable", _short(e))


def check_embedder() -> CheckResult:
    import requests

    url = (os.getenv("EMBEDDER_SEARCH_URL") or os.getenv("EMBEDDER_URL") or "").strip()
    if not url:
        return CheckResult("embedder", False, "not_configured",
                           "EMBEDDER_URL / EMBEDDER_SEARCH_URL not set (the in-process model path does not work)")
    expected = int(os.getenv("EMBED_MODEL_DIM", "768"))
    try:
        r = requests.post(url, json={"texts": ["health check"]}, timeout=_timeout())
        r.raise_for_status()
        vecs = r.json().get("embeddings") or []
        dim = len(vecs[0]) if vecs else 0
        if dim != expected:
            return CheckResult("embedder", False, "wrong_dimension", f"got {dim}, expected {expected}",
                               info={"dimension": dim})
        return CheckResult("embedder", True, "ok", info={"dimension": dim})
    except Exception as e:
        return CheckResult("embedder", False, "unreachable", _short(e))


def check_workers() -> CheckResult:
    from src.ultimate_celery_app import celery_app

    t = min(_timeout(), 1.5)
    try:
        replies = celery_app.control.ping(timeout=t) or []
        workers = sorted(name for reply in replies for name in reply)
        if not workers:
            return CheckResult("workers", False, "no_workers", "no Celery worker answered; uploads will queue",
                               info={"workers": [], "queues_without_workers": EXPECTED_QUEUES})
        active = celery_app.control.inspect(timeout=t).active_queues() or {}
        served = {q.get("name") for qs in active.values() for q in (qs or [])}
        missing = [q for q in EXPECTED_QUEUES if q not in served]
        return CheckResult("workers", True, "ok",
                           info={"workers": workers, "queues_without_workers": missing})
    except Exception as e:
        return CheckResult("workers", False, "unreachable", _short(e))


DEFAULT_CHECKS: Dict[str, Callable[[], CheckResult]] = {
    "redis": check_redis,
    "milvus": check_milvus,
    "embedder": check_embedder,
    "workers": check_workers,
}

# Small dedicated pool so health checks never compete with search or request threads.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="health")


def run_checks(checks: Dict[str, Callable[[], CheckResult]] = None) -> List[CheckResult]:
    checks = checks or DEFAULT_CHECKS
    deadline = _timeout() + 1.0
    started = {name: (time.time(), _POOL.submit(fn)) for name, fn in checks.items()}
    out = []
    for name, (t0, fut) in started.items():
        try:
            res = fut.result(timeout=max(0.1, deadline - (time.time() - t0)))
        except FutureTimeout:
            res = CheckResult(name, False, "timeout", f"no answer within {deadline:.1f}s")
        except Exception as e:
            res = CheckResult(name, False, "error", _short(e))
        res.name = name
        res.critical = name in CRITICAL
        res.latency_ms = round((time.time() - t0) * 1000, 1)
        out.append(res)
    return out


def summarize(results: List[CheckResult]) -> Tuple[int, Dict]:
    critical_failed = [r.name for r in results if r.critical and not r.ok]
    degraded = [r.name for r in results if not r.critical and not r.ok]
    if critical_failed:
        status, code = "unhealthy", 503
    elif degraded:
        status, code = "degraded", 200
    else:
        status, code = "healthy", 200
    return code, {
        "status": status,
        "ready": code == 200,
        "failing": critical_failed + degraded,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {r.name: asdict(r) for r in results},
        "services": {r.name: ("healthy" if r.ok else r.status) for r in results},
    }
