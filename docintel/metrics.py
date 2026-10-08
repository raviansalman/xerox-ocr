"""Prometheus metrics for ingestion, processing stages, retrieval and queries.

Exposed at ``/metrics`` (optionally behind a bearer token). With several API worker processes, set
``PROMETHEUS_MULTIPROC_DIR`` so the client aggregates them. Labels carry no tenant, user or document identifiers.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess

_STAGE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600)
_QUERY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)

DOCUMENTS = Counter("docintel_documents_processed_total", "Documents processed", ["kind", "outcome"])
PAGES = Counter("docintel_pages_processed_total", "Pages processed", ["kind"])
STAGE = Histogram("docintel_processing_stage_seconds", "Time per processing stage", ["stage"], buckets=_STAGE_BUCKETS)
QUERIES = Counter("docintel_queries_total", "Queries answered", ["intent", "outcome"])
QUERY_SECONDS = Histogram("docintel_query_seconds", "Query latency", ["intent"], buckets=_QUERY_BUCKETS)
RETRIEVER_SECONDS = Histogram("docintel_retriever_seconds", "Retriever latency", ["retriever"], buckets=_QUERY_BUCKETS)
RETRIEVER_ERRORS = Counter("docintel_retriever_errors_total", "Retriever failures (query degraded)", ["retriever"])
UPLOADS = Counter("docintel_uploads_total", "Uploads", ["outcome"])
QUEUE_DEPTH = Gauge("docintel_queue_depth", "Ingest jobs waiting or running", ["state"], multiprocess_mode="max")
ANSWERS = Counter("docintel_answers_total", "Generated answers", ["provider", "outcome"])


def observe_document(kind: str, pages: int, timings: dict[str, float], outcome: str) -> None:
    DOCUMENTS.labels(kind, outcome).inc()
    if pages:
        PAGES.labels(kind).inc(pages)
    for stage, seconds in timings.items():
        STAGE.labels(stage).observe(seconds)


def render() -> tuple[bytes, str]:
    import os
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
