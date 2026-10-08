"""Multi-node mode: uploads are queued in Redis and processed by a separate Celery worker process. A job submitted
while no worker runs waits in the broker and is processed once a worker starts (needs DOCINTEL_TEST_REDIS_URL)."""
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration
REDIS = os.environ.get("DOCINTEL_TEST_REDIS_URL")
KEY = "vault_uploader"


def _status(client, doc_id):
    return client.get(f"/api/v1/documents/{doc_id}", headers=headers(KEY)).json()["status"]


def _wait(client, doc_id, want="indexed", timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _status(client, doc_id) == want:
            return True
        time.sleep(0.5)
    return False


@pytest.fixture
def celery_mode(client, monkeypatch):
    from docintel.config import get_settings
    if not REDIS:
        pytest.skip("set DOCINTEL_TEST_REDIS_URL to run the Celery test")
    s = get_settings()
    if s.vector_backend == "memory":
        pytest.skip("the in-memory vector index cannot be shared with a worker process")
    queue = f"docintel.test.{uuid.uuid4().hex[:8]}"
    for k, v in dict(task_mode="celery", redis_url=REDIS, ingest_queue=queue).items():
        monkeypatch.setattr(s, k, v)
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        metrics_port = sock.getsockname()[1]
    env = {**os.environ, "DOCINTEL_TASK_MODE": "celery", "DOCINTEL_REDIS_URL": REDIS, "DOCINTEL_INGEST_QUEUE": queue,
           "DOCINTEL_WORKER_METRICS_PORT": str(metrics_port),
           "PROMETHEUS_MULTIPROC_DIR": str(Path(tempfile.mkdtemp(prefix="docintel-metrics-")))}
    workers = []

    def start():
        p = subprocess.Popen([sys.executable, "-m", "celery", "-A", "docintel.worker", "worker", "-Q", queue,
                              "--pool", "solo", "--concurrency", "1", "--loglevel", "WARNING", "--without-heartbeat",
                              "--without-gossip", "--without-mingle"], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
        workers.append(p)
        return p

    start.metrics_url = f"http://127.0.0.1:{metrics_port}/metrics"
    yield start
    for p in workers:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def test_jobs_wait_in_the_broker_until_a_worker_starts(client, celery_mode):
    r = client.post("/api/v1/documents", headers=headers(KEY),
                    files=[("files", ("celery_note.txt", b"Celery queued note: reactor coolant pump RCP-3302 inspection."))])
    doc_id = r.json()["documents"][0]["id"]
    time.sleep(1.0)
    assert _status(client, doc_id) == "queued"                      # nothing processes it in this process
    worker = celery_mode()
    assert _wait(client, doc_id), worker.stderr.read1(4000) if worker.poll() is not None else "timed out"
    out = client.post("/api/v1/query", headers=headers(KEY), json={"q": "RCP-3302"}).json()
    assert [x["document_id"] for x in out["results"]] == [doc_id]
    import httpx
    metrics = httpx.get(celery_mode.metrics_url, timeout=10).text          # processing metrics from the worker
    assert 'docintel_documents_processed_total{kind="text",outcome="indexed"}' in metrics


def test_a_worker_restart_does_not_lose_queued_jobs(client, celery_mode):
    first = celery_mode()
    first.terminate()
    first.wait(10)
    files = [("files", (f"batch_{i}.txt", f"Restart batch note {i}: valve VX-{7100 + i} serviced.".encode())) for i in range(4)]
    ids = [d["id"] for d in client.post("/api/v1/documents", headers=headers(KEY), files=files).json()["documents"]]
    celery_mode()
    assert all(_wait(client, d) for d in ids)
