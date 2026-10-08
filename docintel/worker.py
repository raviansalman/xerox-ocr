"""Celery worker for multi-node ingestion: ``celery -A docintel.worker worker -Q docintel.ingest``."""
from __future__ import annotations

import psycopg
from celery import Celery
from celery.signals import worker_init, worker_process_init

from docintel.config import get_settings
from docintel.logging_setup import configure_logging

_s = get_settings()
_s.validate_for_startup("worker")
app = Celery("docintel", broker=_s.require("redis_url"), backend=None)
app.conf.update(
    task_acks_late=True,                 # redeliver if a worker dies mid-task
    task_reject_on_worker_lost=True,
    task_time_limit=_s.task_time_limit_sec,          # a hung parser is killed and the job retried or failed
    task_soft_time_limit=max(60, _s.task_time_limit_sec - 60),
    worker_prefetch_multiplier=1,        # long tasks: do not hoard jobs
    task_default_queue=_s.ingest_queue,
    broker_connection_retry_on_startup=True,
    # an unacknowledged job (its worker died) is redelivered after this; it must outlast the longest job. The
    # heartbeat reaper usually requeues such a job within minutes, long before this.
    broker_transport_options={"visibility_timeout": _s.task_time_limit_sec + 300},
    task_serializer="json", accept_content=["json"],
)


@worker_process_init.connect
def _init(**_):
    configure_logging()


@worker_init.connect
def _serve_metrics(**_):
    """Processing, OCR and embedding metrics are recorded in the worker's child processes; the parent serves them,
    aggregated, on DOCINTEL_WORKER_METRICS_PORT (needs PROMETHEUS_MULTIPROC_DIR, emptied here at start)."""
    import os
    import shutil

    port = get_settings().worker_metrics_port
    folder = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not port:
        return
    if not folder:
        raise RuntimeError("DOCINTEL_WORKER_METRICS_PORT needs PROMETHEUS_MULTIPROC_DIR")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    from prometheus_client import CollectorRegistry, multiprocess, start_http_server
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(port, registry=registry)


@app.task(name="docintel.process_document", bind=True, max_retries=5)
def process_document(self, tenant_id: str, document_id: str) -> dict:
    from docintel.ingest import pipeline

    try:
        return pipeline.process(tenant_id, document_id)
    except (pipeline.TransientError, psycopg.OperationalError) as e:   # a dependency is down: try again later
        raise self.retry(exc=e, countdown=min(300, 10 * 2 ** self.request.retries)) from e
