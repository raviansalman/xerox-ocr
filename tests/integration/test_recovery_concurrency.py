"""Restart, recovery and concurrency: interrupted jobs are recovered, processing is idempotent, concurrent
processing of one document cannot duplicate or lose its content, concurrent identical uploads register one
document, and queries keep answering while documents are being written."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration
KEY = "vault_uploader"
TENANT = "tenantvault"
TEXT = ("Maintenance Contract MC-5521\n\nThe vendor services the cooling towers every month. Either party may "
        "terminate with ninety (90) days notice. Contract value: USD 48,000.\n")


def _counts(engine_env, doc_id):
    with engine_env.tenant(TENANT) as conn:
        return {t: conn.execute(f"SELECT count(*) AS n FROM {t} WHERE document_id = %s", (doc_id,)).fetchone()["n"]
                for t in ("pages", "blocks", "chunks", "fields", "entities", "clauses", "relations", "unit_terms")}


def _upload(client, name, body):
    r = client.post("/api/v1/documents", headers=headers(KEY), files=[("files", (name, body))])
    assert r.status_code == 201, r.text
    return r.json()["documents"][0]


@pytest.fixture(scope="module")
def doc(client, engine_env):
    d = _upload(client, "maintenance_contract.txt", TEXT.encode())
    assert client.get(f"/api/v1/documents/{d['id']}", headers=headers(KEY)).json()["status"] == "indexed"
    return d["id"]


def test_reprocessing_is_idempotent(client, engine_env, doc):
    from docintel.ingest import pipeline
    before = _counts(engine_env, doc)
    assert before["chunks"] > 0 and before["unit_terms"] > 0
    for _ in range(2):
        pipeline.process(TENANT, doc)
    assert _counts(engine_env, doc) == before
    versions = client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()["versions"]
    assert [v["status"] for v in versions].count("current") == 1 and len(versions) >= 3


def test_concurrent_processing_of_one_document_neither_duplicates_nor_fails(client, engine_env, doc):
    from docintel.ingest import pipeline
    before = _counts(engine_env, doc)
    barrier = threading.Barrier(4)

    def run():
        barrier.wait()
        return pipeline.process(TENANT, doc)

    with ThreadPoolExecutor(4) as pool:
        results = [f.result() for f in [pool.submit(run) for _ in range(4)]]
    assert all(r["status"] == "indexed" for r in results), results
    assert _counts(engine_env, doc) == before
    detail = client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()
    assert detail["status"] == "indexed" and [v["status"] for v in detail["versions"]].count("current") == 1
    out = client.post("/api/v1/query", headers=headers(KEY), json={"q": "MC-5521"}).json()
    assert [r["document_id"] for r in out["results"]] == [doc]


def test_interrupted_jobs_are_recovered_after_a_restart(client, engine_env, doc):
    """A process that stopped mid-job leaves the job row and a 'processing' document; recovery finishes it."""
    from docintel.ingest import dispatch
    with engine_env.tenant(TENANT) as conn:
        conn.execute("UPDATE documents SET status = 'processing' WHERE id = %s", (doc,))
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc,))         # half-written state
    with engine_env.system() as conn:
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state) VALUES (%s, %s, 'running') "
                     "ON CONFLICT (document_id) DO UPDATE SET state = 'running'", (doc, TENANT))
    assert dispatch.recover() >= 1
    assert client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()["status"] == "indexed"
    assert _counts(engine_env, doc)["chunks"] > 0 and not dispatch.queue_depth()


def test_concurrent_identical_uploads_register_one_document(client, engine_env):
    body = b"Identical upload race test. Purchase order PO-77120 for 12 chillers."
    barrier = threading.Barrier(6)

    def up(i):
        barrier.wait()
        return client.post("/api/v1/documents", headers=headers(KEY), files=[("files", (f"race_{i}.txt", body))]).json()

    with ThreadPoolExecutor(6) as pool:
        outs = list(pool.map(up, range(6)))
    ids = {o["documents"][0]["id"] for o in outs}
    assert len(ids) == 1 and sum(not o["documents"][0]["duplicate"] for o in outs) == 1


def test_queries_keep_answering_during_ingestion(client, engine_env, doc):
    errors, stop = [], threading.Event()

    def ask():
        while not stop.is_set():
            r = client.post("/api/v1/query", headers=headers(KEY), json={"q": "cooling towers"})
            if r.status_code != 200 or doc not in [x["document_id"] for x in r.json()["results"]]:
                errors.append(r.status_code)

    readers = [threading.Thread(target=ask) for _ in range(3)]
    for t in readers:
        t.start()
    for i in range(5):
        _upload(client, f"extra_{i}.txt", f"Extra note {i} about chiller maintenance windows.".encode())
    stop.set()
    for t in readers:
        t.join()
    assert errors == []


def test_stalled_jobs_are_requeued_and_finally_failed(client, engine_env, doc):
    """A job left 'running' (its worker died, or its failure could not be recorded) is retried by the reaper; one
    that keeps stalling is failed with the reason instead of staying 'processing' for ever."""
    from docintel.ingest import dispatch
    with engine_env.tenant(TENANT) as conn:
        conn.execute("UPDATE documents SET status = 'processing' WHERE id = %s", (doc,))
    with engine_env.system() as conn:
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state, attempts, started_at) "
                     "VALUES (%s, %s, 'running', 1, now() - interval '2 hours') ON CONFLICT (document_id) DO UPDATE "
                     "SET state = 'running', attempts = 1, started_at = now() - interval '2 hours'", (doc, TENANT))
    assert dispatch.stalled_count() == 1
    assert dispatch.reap_stalled() == {"requeued": 1, "failed": 0}
    assert client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()["status"] == "indexed"
    with engine_env.system() as conn:
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state, attempts, started_at) "
                     "VALUES (%s, %s, 'running', %s, now() - interval '2 hours')", (doc, TENANT, dispatch.MAX_ATTEMPTS))
    assert dispatch.reap_stalled() == {"requeued": 0, "failed": 1}
    d = client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()
    assert d["status"] == "failed" and "did not complete" in d["error"] and not dispatch.queue_depth()
    from docintel.ingest import pipeline
    pipeline.process(TENANT, doc)                                  # restore for later tests


def test_a_vector_index_outage_is_retried_not_a_permanent_failure(client, engine_env, doc, monkeypatch):
    """Vectors that cannot be written leave the job queued for retry; the document recovers once the index is back."""
    from docintel.config import get_settings
    from docintel.ingest import dispatch, pipeline
    if not get_settings().semantic_enabled:
        pytest.skip("no vector index in this mode")

    class Down:
        def upsert(self, *a, **k):
            raise ConnectionError("vector index unreachable")

        def delete_document(self, *a, **k):
            raise ConnectionError("vector index unreachable")

    monkeypatch.setattr(pipeline, "get_vector_store", lambda: Down())
    with engine_env.system() as conn:                     # as a submitted job has
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state) VALUES (%s, %s, 'running') "
                     "ON CONFLICT (document_id) DO UPDATE SET state = 'running'", (doc, TENANT))
    with pytest.raises(pipeline.TransientError):
        pipeline.process(TENANT, doc)
    detail = client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()
    assert detail["status"] == "failed" and "vector index unavailable" in detail["error"]
    with engine_env.system() as conn:
        assert conn.execute("SELECT count(*) AS n FROM ingest_jobs WHERE document_id = %s", (doc,)).fetchone()["n"] == 1
    monkeypatch.undo()
    assert dispatch.recover() >= 1
    assert client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()["status"] == "indexed"
    out = client.post("/api/v1/query", headers=headers(KEY), json={"q": "MC-5521"}).json()
    assert [r["document_id"] for r in out["results"]] == [doc]


def test_processing_and_queries_survive_a_database_restart(client, engine_env, doc):
    """Every pooled connection dies (as when PostgreSQL restarts); the next job and query get fresh ones."""
    import os
    import time

    import psycopg

    from docintel.ingest import pipeline
    held = [engine_env.pool.getconn() for _ in range(6)]                # a busy pool: several idle connections
    for c in held:
        engine_env.pool.putconn(c)
    with psycopg.connect(os.environ["DOCINTEL_TEST_DATABASE_URL"], autocommit=True) as admin:
        killed = admin.execute("SELECT count(*) FROM (SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                               "WHERE usename = current_user AND datname = current_database() "
                               "AND pid <> pg_backend_pid()) t").fetchone()[0]
    assert killed >= 6
    started = time.monotonic()
    assert pipeline.process(TENANT, doc)["status"] == "indexed"
    assert time.monotonic() - started < 10       # the pool's own check took 32 s for 6 dead connections
    out = client.post("/api/v1/query", headers=headers(KEY), json={"q": "MC-5521"}).json()
    assert [r["document_id"] for r in out["results"]] == [doc]


def _job_rows(engine_env, doc):
    with engine_env.system() as conn:
        return conn.execute("SELECT count(*) AS n FROM ingest_jobs WHERE document_id = %s", (doc,)).fetchone()["n"]


def _submitted(engine_env, doc):
    with engine_env.system() as conn:
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state) VALUES (%s, %s, 'running') "
                     "ON CONFLICT (document_id) DO UPDATE SET state = 'running'", (doc, TENANT))


def test_a_database_failure_mid_run_is_retried_not_permanent(client, engine_env, doc, monkeypatch):
    import psycopg

    from docintel.ingest import dispatch, pipeline

    def lost(*a, **k):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    _submitted(engine_env, doc)
    monkeypatch.setattr(pipeline.repo, "replace_content", lost)
    with pytest.raises(pipeline.TransientError):
        pipeline.process(TENANT, doc)
    assert _job_rows(engine_env, doc) == 1
    monkeypatch.undo()
    assert dispatch.recover() >= 1
    assert client.get(f"/api/v1/documents/{doc}", headers=headers(KEY)).json()["status"] == "indexed"


def test_an_invalid_vector_write_fails_the_document_instead_of_retrying_forever(client, engine_env, doc, monkeypatch):
    from docintel.config import get_settings
    from docintel.ingest import pipeline
    if not get_settings().semantic_enabled:
        pytest.skip("no vector index in this mode")

    class Refuses:
        def upsert(self, *a, **k):
            raise ValueError("invalid document type")

    _submitted(engine_env, doc)
    monkeypatch.setattr(pipeline, "get_vector_store", lambda: Refuses())
    assert pipeline.process(TENANT, doc)["status"] == "failed"
    assert _job_rows(engine_env, doc) == 0
    monkeypatch.undo()
    pipeline.process(TENANT, doc)


def test_a_job_enqueued_again_during_its_run_keeps_its_row(engine_env, doc):
    from docintel.ingest import pipeline
    with engine_env.system() as conn:
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id, state, started_at, enqueued_at) "
                     "VALUES (%s, %s, 'queued', now() - interval '5 seconds', now()) ON CONFLICT (document_id) DO UPDATE "
                     "SET state = 'queued', started_at = now() - interval '5 seconds', enqueued_at = now()", (doc, TENANT))
    pipeline._finish_job(doc)                       # the run that started before the reprocess request ends
    assert _job_rows(engine_env, doc) == 1
    pipeline.process(TENANT, doc)                   # the rerun finishes the job
    assert _job_rows(engine_env, doc) == 0


def test_only_one_process_may_run_thread_mode_jobs(engine_env):
    from docintel.ingest import dispatch
    first = dispatch.claim_thread_mode()
    try:
        with pytest.raises(RuntimeError, match="already processing"):
            dispatch.claim_thread_mode()
    finally:
        first.close()
    dispatch.claim_thread_mode().close()


def test_a_document_without_text_is_found_by_its_file_name(client, engine_env, tmp_path):
    from PIL import Image
    Image.new("L", (400, 300), 255).save(tmp_path / "Blank_Scan_Q3_Archive.png")
    d = _upload(client, "Blank_Scan_Q3_Archive.png", (tmp_path / "Blank_Scan_Q3_Archive.png").read_bytes())
    assert client.get(f"/api/v1/documents/{d['id']}", headers=headers(KEY)).json()["status"] == "indexed"
    out = client.post("/api/v1/query", headers=headers(KEY), json={"q": "Blank_Scan_Q3_Archive"}).json()
    assert d["id"] in [r["document_id"] for r in out["results"]]


def test_deleting_a_document_prunes_only_its_own_words_from_the_vocabulary(client, engine_env):
    """Typo correction must never suggest a word that no remaining document contains."""
    def vocab(words):
        with engine_env.tenant(TENANT) as conn:
            return {r["term"] for r in conn.execute("SELECT term FROM vocabulary WHERE term = ANY(%s)", (words,)).fetchall()}

    a = _upload(client, "vocab_a.txt", b"Quarterly zephyrquill inspection of the cooling towers.\n")
    b = _upload(client, "vocab_b.txt", b"Annual inspection of the cooling towers by the vendor.\n")
    assert vocab(["zephyrquill", "inspection"]) == {"zephyrquill", "inspection"}
    assert client.delete(f"/api/v1/documents/{a['id']}", headers=headers(KEY)).status_code == 200
    assert vocab(["zephyrquill", "inspection"]) == {"inspection"}       # still in vocab_b
    client.delete(f"/api/v1/documents/{b['id']}", headers=headers(KEY))
