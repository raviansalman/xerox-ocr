#!/usr/bin/env python3
"""
Ultimate Search Processor Celery Configuration

This module configures Celery for background task processing in the Ultimate Search Processor.
Uses Redis as the message broker and result backend.
"""

import os
from celery import Celery

# Get Redis URL from environment or use default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery instance for Ultimate Search Processor
celery_app = Celery(
    "ultimate_search_processor",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.ultimate_tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes
    task_soft_time_limit=1500,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    task_reject_on_worker_lost=True,
    worker_disable_rate_limits=True,
    result_expires=3600,  # 1 hour
    broker_transport_options={
        # Ensure unacked tasks are re-queued if a worker is lost/stuck.
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "7200")),
    },
    # Celery 6 compatibility: keep startup retry behavior explicit.
    broker_connection_retry_on_startup=True,
    task_default_queue=os.getenv("CELERY_DEFAULT_QUEUE", "ultimate_processing"),
    task_default_exchange=os.getenv("CELERY_DEFAULT_QUEUE", "ultimate_processing"),
    task_default_exchange_type="direct",
    task_default_routing_key=os.getenv("CELERY_DEFAULT_QUEUE", "ultimate_processing"),
)

# Optional configuration for better monitoring
celery_app.conf.update(
    worker_send_task_events=True,
    task_send_sent_event=True,
)

if __name__ == "__main__":
    celery_app.start()
