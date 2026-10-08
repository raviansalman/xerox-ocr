"""Job dispatch. ``celery`` for multi-node deployments, ``thread`` for a single node, ``inline`` for tests.

Every job is recorded in ``ingest_jobs`` when it is registered, so pending work survives restarts:
``recover()`` re-submits jobs that were queued or running when the process stopped (thread mode), and Celery
redelivers unacknowledged tasks (``acks_late``).
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg

from docintel.config import get_settings
from docintel.ingest import pipeline
from docintel.storage.db import get_db

logger = logging.getLogger(__name__)
_pool: ThreadPoolExecutor | None = None
_lock = threading.Lock()
_inflight: set[str] = set()
_again: set[str] = set()              # submitted again while running: run once more when the current run ends


def _thread_pool() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=get_settings().worker_threads, thread_name_prefix="ingest")
        return _pool


def _run(tenant_id: str, document_id: str) -> None:
    try:
        for attempt in range(3):
            try:
                pipeline.process(tenant_id, document_id)
                return
            except (pipeline.TransientError, psycopg.OperationalError):   # a dependency is down
                if attempt == 2:
                    logger.error("giving up after transient failures", extra={"document_id": document_id})
                    return
                time.sleep(5 * (attempt + 1))
    except Exception:
        # the job row stays; reap_stalled() retries it later (for example after a full disk was cleared)
        logger.exception("ingest job interrupted", extra={"document_id": document_id})
    finally:
        with _lock:
            rerun = document_id in _again
            _again.discard(document_id)
            if not rerun:
                _inflight.discard(document_id)
        if rerun:                     # a reprocess requested mid-run must see the newest request, not be dropped
            _thread_pool().submit(_run, tenant_id, document_id)


def submit(tenant_id: str, document_id: str) -> None:
    mode = get_settings().task_mode
    document_id = str(document_id)
    if mode == "inline":
        _run(tenant_id, document_id)
    elif mode == "thread":
        with _lock:
            if document_id in _inflight:
                _again.add(document_id)
                return
            _inflight.add(document_id)
        _thread_pool().submit(_run, tenant_id, document_id)
    else:
        from docintel.worker import process_document
        process_document.apply_async(args=[tenant_id, document_id], queue=get_settings().ingest_queue)


THREAD_MODE_LOCK = 4244


def claim_thread_mode() -> psycopg.Connection:
    """Thread mode keeps its job queue in process memory, so exactly one process may process jobs: two would each
    see the other's waiting jobs as stalled and process them again. The returned connection holds a session lock
    for as long as it stays open; a second process (another API worker, ``docintel ingest``) is refused."""
    conn = psycopg.connect(get_settings().require("database_url"), autocommit=True)
    if not conn.execute("SELECT pg_try_advisory_lock(%s)", (THREAD_MODE_LOCK,)).fetchone()[0]:
        conn.close()
        raise RuntimeError("another process is already processing jobs in thread mode (DOCINTEL_TASK_MODE=thread "
                           "runs in one process; use Celery mode for several)")
    return conn


def recover() -> int:
    """Re-submit jobs left queued or running by a previous process (thread mode)."""
    with get_db().system() as conn:
        rows = conn.execute("SELECT tenant_id, document_id FROM ingest_jobs ORDER BY enqueued_at").fetchall()
    for r in rows:
        submit(r["tenant_id"], str(r["document_id"]))
    if rows:
        logger.info("recovered pending ingest jobs", extra={"count": len(rows)})
    return len(rows)


MAX_ATTEMPTS = 5


def reap_stalled(stall_after_sec: float | None = None) -> dict:
    """Requeue jobs that stopped making progress (a worker died, or the job failed while its failure could not be
    recorded) and fail the ones that keep stalling. One process reaps at a time."""
    s = get_settings()
    stall = s.task_time_limit_sec + 60 if stall_after_sec is None else stall_after_sec
    states = ["running", "queued"] if s.task_mode == "thread" else ["running"]   # Celery keeps queued work in Redis
    requeued, failed = [], []
    with get_db().system() as conn:
        if not conn.execute("SELECT pg_try_advisory_xact_lock(4243) AS ok").fetchone()["ok"]:
            return {"requeued": 0, "failed": 0}
        rows = conn.execute(
            "SELECT document_id, tenant_id, attempts FROM ingest_jobs WHERE state = ANY(%s) "
            "AND coalesce(started_at, enqueued_at) < now() - make_interval(secs => %s) FOR UPDATE SKIP LOCKED",
            (states, stall)).fetchall()
        with _lock:
            rows = [r for r in rows if str(r["document_id"]) not in _inflight]
        for r in rows:
            if r["attempts"] >= MAX_ATTEMPTS:
                conn.execute("DELETE FROM ingest_jobs WHERE document_id = %s", (r["document_id"],))
                failed.append(r)
            else:
                conn.execute("UPDATE ingest_jobs SET state = 'queued', enqueued_at = now(), started_at = NULL "
                             "WHERE document_id = %s", (r["document_id"],))
                requeued.append(r)
    for r in failed:
        with get_db().tenant(r["tenant_id"]) as conn:
            from docintel.storage import repo
            repo.set_status(conn, str(r["document_id"]), "failed",
                            error=f"processing did not complete after {r['attempts']} attempts")
    for r in requeued:
        submit(r["tenant_id"], str(r["document_id"]))
    if requeued or failed:
        logger.warning("stalled ingest jobs", extra={"requeued": len(requeued), "failed": len(failed)})
    return {"requeued": len(requeued), "failed": len(failed)}


def start_reaper(interval_sec: float = 60.0) -> threading.Event:
    """Run reap_stalled() periodically in a daemon thread; set the returned event to stop it."""
    stop = threading.Event()

    def loop():
        while not stop.wait(interval_sec):
            try:
                reap_stalled()
            except Exception:
                logger.exception("stalled-job reaper failed")
    threading.Thread(target=loop, name="ingest-reaper", daemon=True).start()
    return stop


def stalled_count() -> int:
    s = get_settings()
    with get_db().system() as conn:
        return conn.execute("SELECT count(*) AS n FROM ingest_jobs WHERE coalesce(started_at, enqueued_at) < "
                            "now() - make_interval(secs => %s)", (s.task_time_limit_sec + 60,)).fetchone()["n"]


def queue_depth() -> dict:
    with get_db().system() as conn:
        rows = conn.execute("SELECT state, count(*) AS n FROM ingest_jobs GROUP BY state").fetchall()
    return {r["state"]: r["n"] for r in rows}


def wait_idle(timeout: float = 600.0) -> bool:
    """Block until no job is pending (used by tests and the bulk-ingest CLI)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not queue_depth():
            return True
        time.sleep(0.2)
    return False
