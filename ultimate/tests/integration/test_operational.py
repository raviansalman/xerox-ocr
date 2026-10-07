"""Operational behaviour against real services: readiness and Milvus restarts.

The restart test needs control of the Milvus container: set XOCR_MILVUS_CONTAINER (CI sets it).
"""
import os
import subprocess
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

CONTAINER = os.getenv("XOCR_MILVUS_CONTAINER", "")


def test_readiness_with_real_dependencies():
    import ultimate_ui

    r = TestClient(ultimate_ui.app).get("/health/ready")
    body = r.json()
    checks = body["checks"]
    assert checks["redis"]["ok"] and checks["embedder"]["ok"] and checks["embedder"]["info"]["dimension"] == 768
    assert checks["milvus"]["ok"], checks["milvus"]
    # No Celery worker runs in the test environment: degraded, but still ready for search.
    assert checks["workers"]["status"] == "no_workers"
    assert r.status_code == 200 and body["status"] == "degraded"


def _wait_milvus_healthz(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:9091/healthz", timeout=2)
            return time.time() - t0
        except Exception:
            time.sleep(1)
    raise TimeoutError("Milvus did not become healthy")


@pytest.mark.skipif(not CONTAINER, reason="set XOCR_MILVUS_CONTAINER to the Milvus container name")
def test_full_scan_right_after_milvus_restart():
    from src import health
    from src.ultimate_tasks import get_worker_vector_integration

    vi = get_worker_vector_integration()
    vi.upsert_document(file_id="restart_doc", text_content="FOR IMMEDIATE RELEASE quarterly archive",
                       metadata={"filename": "Restart_Press_Release.pdf"}, user_id="restart_tenant")
    vi.vector_db.collection.flush()
    vi.vector_db._ensure_collection_loaded()

    # 1. restart Milvus  2. wait for readiness
    subprocess.run(["docker", "restart", CONTAINER], check=True, capture_output=True)
    _wait_milvus_healthz()

    # 3. full-scan queries from the same long-lived process, immediately (collection may still be loading)
    rows = vi.vector_db.query_all_chunks(user_id="restart_tenant", limit=100)
    lexical = vi.search_lexical("restart press release", user_id="restart_tenant", limit=10)

    # 4. correct results
    assert [r.metadata["file_id"] for r in rows] == ["restart_doc"]
    assert [h.file_id for h in lexical] == ["restart_doc"]

    # readiness converges to ok once loading has finished
    for _ in range(60):
        if health.check_milvus().ok:
            break
        time.sleep(2)
    assert health.check_milvus().ok
