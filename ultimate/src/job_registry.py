#!/usr/bin/env python3
"""
Redis Job Registry – unified lifecycle tracking for document processing jobs.

Key schema:  job:{user_id}:{file_id}  →  JSON blob
Index key:   job_index:{user_id}  →  Redis Set of file_ids (for list_jobs)

This complements the existing process_lock / heartbeat mechanism:
  - process_lock  : prevents duplicate workers
  - heartbeat     : lets monitors detect stuck tasks
  - job_registry  : single source of truth for job state history + admin UI
"""

import json
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FAILED_STALE = "FAILED_STALE"
    CANCELLED = "CANCELLED"


# Default TTL for job entries (24 hours keeps the registry lean).
_JOB_TTL_SECONDS = int(os.getenv("JOB_REGISTRY_TTL_SECONDS", str(24 * 3600)))


class JobRegistry:
    """Redis-backed store tracking full lifecycle of document processing jobs."""

    KEY_PREFIX = "job"
    INDEX_PREFIX = "job_index"

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, user_id: str, file_id: str) -> str:
        return f"{self.KEY_PREFIX}:{user_id}:{file_id}"

    def _index_key(self, user_id: str) -> str:
        return f"{self.INDEX_PREFIX}:{user_id}"

    TASK_LOOKUP_PREFIX = "task_lookup"

    def _load(self, user_id: str, file_id: str) -> Optional[Dict]:
        """Load job record from Redis. Returns None if not found."""
        raw = self._redis.get(self._key(user_id, file_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] Failed to parse job record: {exc}")
            return None

    def _save(self, user_id: str, file_id: str, record: Dict) -> None:
        """Persist job record to Redis with TTL."""
        key = self._key(user_id, file_id)
        self._redis.setex(key, _JOB_TTL_SECONDS, json.dumps(record))
        # Maintain a per-user index so list_jobs is O(members) not O(SCAN).
        idx_key = self._index_key(user_id)
        self._redis.sadd(idx_key, file_id)
        self._redis.expire(idx_key, _JOB_TTL_SECONDS)
        # Reverse index: task_id -> (user_id, file_id) for status lookup when Celery result expires
        task_id_val = record.get("task_id")
        if task_id_val:
            lookup_key = f"{self.TASK_LOOKUP_PREFIX}:{task_id_val}"
            self._redis.setex(
                lookup_key,
                _JOB_TTL_SECONDS,
                json.dumps({"user_id": user_id, "file_id": file_id}),
            )

    def _now(self) -> float:
        return time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, user_id: str, file_id: str, task_id: str) -> bool:
        """Create or reset a job entry in PENDING state.

        Returns:
            True  – entry created / reset (caller should proceed with enqueue).
            False – existing PROCESSING entry with a fresh heartbeat; caller
                    should NOT enqueue a new task and should return the
                    existing task_id to the client.
        """
        try:
            existing = self._load(user_id, file_id)
            if existing and existing.get("status") == JobStatus.PROCESSING:
                # Check if the task still has a live heartbeat.
                existing_task_id = existing.get("task_id", "")
                if existing_task_id:
                    hb_key = f"heartbeat:{existing_task_id}"
                    hb_raw = self._redis.get(hb_key)
                    if hb_raw:
                        try:
                            hb = json.loads(hb_raw)
                            age = self._now() - float(hb.get("timestamp", 0))
                            stuck_threshold = int(os.getenv("STUCK_JOB_THRESHOLD", "300"))
                            if age < stuck_threshold:
                                logger.info(
                                    f"[JOB REGISTRY] Job {file_id} already PROCESSING by {existing_task_id} "
                                    f"(heartbeat age={age:.0f}s). Skipping duplicate enqueue."
                                )
                                return False
                        except Exception:
                            pass

            now = self._now()
            record = {
                "status": JobStatus.PENDING,
                "task_id": task_id,
                "attempts": (existing.get("attempts", 0) + 1) if existing else 1,
                "last_heartbeat_ts": now,
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
                "error": "",
                "chunks": 0,
            }
            self._save(user_id, file_id, record)
            logger.info(f"[JOB REGISTRY] Registered job file_id={file_id} task_id={task_id}")
            return True
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] register() failed (non-fatal): {exc}")
            return True  # Fail-open: don't block processing on registry errors.

    def mark_processing(self, user_id: str, file_id: str, task_id: str) -> None:
        """Transition job to PROCESSING state."""
        try:
            existing = self._load(user_id, file_id) or {}
            now = self._now()
            record = {
                **existing,
                "status": JobStatus.PROCESSING,
                "task_id": task_id,
                "last_heartbeat_ts": now,
                "updated_at": now,
                "error": "",
            }
            self._save(user_id, file_id, record)
            logger.debug(f"[JOB REGISTRY] {file_id} → PROCESSING")
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] mark_processing() failed (non-fatal): {exc}")

    def mark_success(self, user_id: str, file_id: str, task_id: str, chunks: int = 0) -> None:
        """Transition job to SUCCESS state."""
        try:
            existing = self._load(user_id, file_id) or {}
            now = self._now()
            record = {
                **existing,
                "status": JobStatus.SUCCESS,
                "task_id": task_id,
                "last_heartbeat_ts": now,
                "updated_at": now,
                "error": "",
                "chunks": chunks,
            }
            self._save(user_id, file_id, record)
            logger.info(f"[JOB REGISTRY] {file_id} → SUCCESS (chunks={chunks})")
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] mark_success() failed (non-fatal): {exc}")

    def mark_failed(
        self, user_id: str, file_id: str, task_id: str, error: str = ""
    ) -> None:
        """Transition job to FAILED state."""
        try:
            existing = self._load(user_id, file_id) or {}
            now = self._now()
            record = {
                **existing,
                "status": JobStatus.FAILED,
                "task_id": task_id,
                "last_heartbeat_ts": now,
                "updated_at": now,
                "error": error[:2000],  # Truncate to keep Redis entries small.
            }
            self._save(user_id, file_id, record)
            logger.info(f"[JOB REGISTRY] {file_id} → FAILED: {error[:100]}")
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] mark_failed() failed (non-fatal): {exc}")

    def get_status(self, user_id: str, file_id: str) -> Optional[Dict]:
        """Return the current job record, or None if not found."""
        try:
            return self._load(user_id, file_id)
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] get_status() failed: {exc}")
            return None

    def get_job(self, user_id: str, file_id: str) -> Optional[Dict]:
        """Load job record by user_id and file_id. Returns None if not found."""
        rec = self._load(user_id, file_id)
        if rec:
            rec["file_id"] = file_id
        return rec

    def get_job_by_task_id(self, task_id: str) -> Optional[Dict]:
        """Load job record by Celery task_id. Used when Celery result backend has expired."""
        try:
            lookup_key = f"{self.TASK_LOOKUP_PREFIX}:{task_id}"
            raw = self._redis.get(lookup_key)
            if not raw:
                return None
            info = json.loads(raw)
            uid = info.get("user_id")
            fid = info.get("file_id")
            if uid and fid:
                rec = self._load(uid, fid)
                if rec:
                    rec["file_id"] = fid
                    return rec
        except Exception as exc:
            logger.debug(f"[JOB REGISTRY] get_job_by_task_id failed: {exc}")
        return None

    def list_jobs(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Return recent job records for a user (most-recently-updated first)."""
        try:
            idx_key = self._index_key(user_id)
            file_ids = self._redis.smembers(idx_key)
            records = []
            for fid in file_ids:
                if isinstance(fid, bytes):
                    fid = fid.decode("utf-8")
                rec = self._load(user_id, fid)
                if rec:
                    rec["file_id"] = fid
                    records.append(rec)
            # Sort by updated_at desc, newest first.
            records.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
            return records[:limit]
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] list_jobs() failed: {exc}")
            return []

    def mark_stale_processing_jobs_failed(
        self, stuck_threshold_seconds: int = 300
    ) -> int:
        """
        Scan all job registry entries across ALL users and mark any job that is
        stuck in PROCESSING state with a heartbeat older than
        ``stuck_threshold_seconds`` as FAILED_STALE.

        Called on worker startup so that files leftover from a crashed container
        can be safely re-queued rather than forever blocked by a dead heartbeat.

        Returns the number of records updated.
        """
        updated = 0
        try:
            # Scan all job keys matching the prefix pattern
            pattern = f"{self.KEY_PREFIX}:*"
            cursor = 0
            now = self._now()
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=200)
                for raw_key in keys:
                    if isinstance(raw_key, bytes):
                        raw_key = raw_key.decode("utf-8")
                    # key format: job:{user_id}:{file_id}
                    parts = raw_key.split(":", 2)
                    if len(parts) != 3:
                        continue
                    _, user_id, file_id = parts
                    try:
                        rec = self._load(user_id, file_id)
                        if not rec:
                            continue
                        if rec.get("status") != JobStatus.PROCESSING:
                            continue
                        age = now - float(rec.get("last_heartbeat_ts", 0))
                        if age < stuck_threshold_seconds:
                            continue
                        # Mark stale
                        rec["status"] = JobStatus.FAILED_STALE
                        rec["updated_at"] = now
                        rec["error"] = (
                            f"Marked FAILED_STALE on worker restart after {age:.0f}s "
                            f"with no heartbeat (threshold={stuck_threshold_seconds}s)"
                        )
                        self._save(user_id, file_id, rec)
                        updated += 1
                        logger.info(
                            f"[JOB REGISTRY] Marked stale job {file_id} (user={user_id}, "
                            f"age={age:.0f}s) as FAILED_STALE"
                        )
                        
                        # Sync failure state to external workflow tracking API
                        if os.getenv("WORKFLOW_ENABLED", "false").lower() == "true":
                            try:
                                from src.workflow_manager import get_workflow_manager
                                workflow_manager = get_workflow_manager()
                                workflow_manager.update_failed(file_id, error_message=rec["error"])
                                logger.info(f"[JOB REGISTRY] Synced FAILED_STALE to Workflow API for {file_id}")
                            except Exception as wf_exc:
                                logger.warning(f"[JOB REGISTRY] Failed to sync Workflow API for {file_id}: {wf_exc}")
                                
                    except Exception as inner_exc:
                        logger.debug(f"[JOB REGISTRY] Skipping key {raw_key}: {inner_exc}")
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] mark_stale_processing_jobs_failed() error: {exc}")
        return updated


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_REGISTRY: Optional[JobRegistry] = None


def get_job_registry() -> Optional[JobRegistry]:
    """Return a process-wide JobRegistry, or None if Redis is unavailable."""
    global _REGISTRY
    if _REGISTRY is None:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
            client.ping()  # Fail fast if Redis is down at boot.
            _REGISTRY = JobRegistry(client)
            logger.info("[JOB REGISTRY] Initialized job registry")
        except Exception as exc:
            logger.warning(f"[JOB REGISTRY] Redis unavailable, registry disabled: {exc}")
            _REGISTRY = None
    return _REGISTRY
