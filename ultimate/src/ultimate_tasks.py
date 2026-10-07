#!/usr/bin/env python3
"""
Ultimate Search Processor - Celery Background Tasks
Handles:
  • File download (if URL provided)
  • OCR / text extraction
  • Vector embedding & storage
  • Optional CLIP image embedding storage
"""

import os
import time
import logging
import tempfile
import shutil
import re
import threading
from pathlib import Path
from urllib.parse import urlparse, unquote, quote
from typing import Dict, Any, Optional

import requests
import redis
from celery.exceptions import SoftTimeLimitExceeded
from dotenv import load_dotenv
from celery import current_task
from src.ultimate_celery_app import celery_app
from celery.signals import worker_ready, worker_shutting_down
from src.scalability_utils import (
    HeartbeatMonitor,
    should_retry_task,
    calculate_retry_delay,
    MAX_RETRIES,
    detect_stuck_jobs,
)
from src.job_registry import get_job_registry
from src.net_safety import UnsafeURLError, max_download_bytes, redact_url, safe_get

# Load env
load_dotenv()

# Globally disable BLIP image captioning in Celery workers to
# prevent heavy model initialization and OOMs during document processing.
# All captioning-related code paths should see this flag and NO-OP.
os.environ["SKIP_IMAGE_CAPTIONING_IN_PROCESSOR"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional workflow integration
WORKFLOW_API_URL = os.getenv("WORKFLOW_API_URL", "http://localhost:3333")
WORKFLOW_ACCESS_KEY = os.getenv("WORKFLOW_ACCESS_KEY", "")
WORKFLOW_STOR_API_KEY = os.getenv("WORKFLOW_STOR_API_KEY", "")
WORKFLOW_ENABLED = os.getenv("WORKFLOW_ENABLED", "false").lower() == "true"
WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "10"))

# Threshold for considering a task "stuck" based on heartbeat (seconds)
LOCK_STUCK_THRESHOLD = int(os.getenv("STUCK_JOB_THRESHOLD", "300"))

# Processing modules
# Heavy imports moved inside tasks to save baseline RAM
# from src.ultimate_search_processor import DocumentProcessor
# from src.ultimate_vector_integration import UltimateVectorIntegration

# Semantic metadata extractors (temporal + entities + governing law)
from src.semantic.temporal_engine import TemporalReasoningEngine
from src.semantic.semantic_components import EntityExtractor
from src.semantic.query_enhancement import extract_governing_law
from src.semantic.temporal_engine import IngestionTemporalNormalizer

# Initialize shared engines once per worker process
_TEMPORAL_ENGINE = TemporalReasoningEngine()
_ENTITY_EXTRACTOR = EntityExtractor()
_INGESTION_TEMPORAL_NORMALIZER = IngestionTemporalNormalizer(allow_filename_dates=True)

# Reuse vector integration per worker process to reduce Milvus reconnect churn.
_VECTOR_INTEGRATION: Optional["UltimateVectorIntegration"] = None # Type hint needs to be string for lazy import
_VECTOR_INTEGRATION_LOCK = threading.Lock()
_MILVUS_CIRCUIT_OPEN_UNTIL: float = 0.0
_MILVUS_CIRCUIT_LOCK = threading.Lock()


def get_worker_vector_integration(force_recreate: bool = False) -> "UltimateVectorIntegration":
    global _VECTOR_INTEGRATION
    from src.ultimate_vector_integration import UltimateVectorIntegration # Lazy import
    with _VECTOR_INTEGRATION_LOCK:
        if force_recreate or _VECTOR_INTEGRATION is None:
            _VECTOR_INTEGRATION = UltimateVectorIntegration()
        return _VECTOR_INTEGRATION


# ---------------------------------------------------------------------------
# PRE-WARM: Load model + Milvus connection once at worker startup.
# This guarantees the SentenceTransformer is in memory before the first task
# arrives, eliminating cold-start latency and preventing concurrent reloads.
# ---------------------------------------------------------------------------
@worker_ready.connect
def prewarm_worker(sender, **kwargs):
    """Pre-load models on startup to avoid delay on first task."""
    logger.info("[PREWARM] Worker starting — pre-loading embedding model and Milvus connection...")
    try:
        from src.ultimate_vector_integration import UltimateVectorIntegration
        _ = UltimateVectorIntegration()
        logger.info("[PREWARM] Embedding model and Milvus connection ready.")
    except Exception as e:
        logger.error(f"[PREWARM] Failed: {e}")

    # Clear any stale PROCESSING locks from a previous crashed worker instance.
    # Without this, duplicate-detection blocks re-processing of files that
    # were in-flight when the old container was killed.
    try:
        reg = get_job_registry()
        if reg:
            cleared = reg.mark_stale_processing_jobs_failed(
                stuck_threshold_seconds=int(os.getenv("STUCK_JOB_THRESHOLD", "300"))
            )
            if cleared:
                logger.info(f"[PREWARM] Cleared {cleared} stale PROCESSING job(s) from previous run.")
    except Exception as exc:
        logger.warning(f"[PREWARM] Stale-job cleanup failed (non-fatal): {exc}")



@worker_shutting_down.connect
def on_worker_shutdown(sender, **kwargs):
    """Cleanly close the Milvus connection when the worker shuts down."""
    global _VECTOR_INTEGRATION
    with _VECTOR_INTEGRATION_LOCK:
        if _VECTOR_INTEGRATION is not None:
            try:
                from pymilvus import connections
                connections.disconnect("default")
                logger.info("[SHUTDOWN] Milvus connection closed cleanly.")
            except Exception:
                pass


def _is_milvus_connectivity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "milvus",
        "connection refused",
        "failed to connect",
        "statuscode.unavailable",
        "unavailable",
        "rpc error",
    )
    return any(marker in text for marker in markers)


def _open_milvus_circuit(seconds: int | None = None) -> None:
    """Open circuit breaker. Set MILVUS_CIRCUIT_COOLDOWN_SEC=0 to disable."""
    cooldown = int(os.getenv("MILVUS_CIRCUIT_COOLDOWN_SEC", "20"))
    if cooldown <= 0:
        return
    sec = seconds if seconds is not None else cooldown
    global _MILVUS_CIRCUIT_OPEN_UNTIL
    with _MILVUS_CIRCUIT_LOCK:
        _MILVUS_CIRCUIT_OPEN_UNTIL = max(_MILVUS_CIRCUIT_OPEN_UNTIL, time.time() + sec)


def _assert_milvus_circuit_closed() -> None:
    """Skip if circuit disabled (MILVUS_CIRCUIT_COOLDOWN_SEC=0)."""
    if int(os.getenv("MILVUS_CIRCUIT_COOLDOWN_SEC", "20")) <= 0:
        return
    with _MILVUS_CIRCUIT_LOCK:
        if time.time() < _MILVUS_CIRCUIT_OPEN_UNTIL:
            remaining = int(_MILVUS_CIRCUIT_OPEN_UNTIL - time.time())
            raise RuntimeError(f"Milvus circuit open; retry after ~{remaining}s")


def _is_connectivity_retry(exc: Exception) -> bool:
    """Fast-path retries for transient infrastructure outages."""
    return _is_milvus_connectivity_error(exc) or isinstance(
        exc, (ConnectionError, TimeoutError, OSError)
    )

# Optional image processing (lazy import to avoid errors if not available)
def _get_image_utils():
    try:
        from src.semantic.semantic_components import get_image_embedding, analyze_image
        return get_image_embedding, analyze_image
    except ImportError:
        logger.warning("Image processing utilities not available")
        return None, None


# ---------------------------------------------------------------------------
# FILE DOWNLOAD UTILITY
# ---------------------------------------------------------------------------
def _safe_filename(name: str, fallback: str) -> str:
    """
    Sanitize filename by removing query parameters, special characters, and path components.
    
    Args:
        name: Original filename (may contain query parameters or path components)
        fallback: Fallback name if sanitization results in empty string
        
    Returns:
        Safe filename (max 120 chars)
    """
    if not name:
        return fallback[:120]
    
    # Remove query parameters (e.g., "file.svg?X-Amz-Algorithm=...")
    if "?" in name:
        name = name.split("?")[0]
    
    # Remove URL fragments (e.g., "file.svg#section")
    if "#" in name:
        name = name.split("#")[0]
    
    # Extract just the basename (remove any path components)
    safe = os.path.basename(name)
    
    # Replace spaces and other problematic characters
    safe = safe.replace(" ", "_")
    # Remove any remaining special characters that could cause issues
    import re
    safe = re.sub(r'[<>:"|?*\\]', '_', safe)
    
    # Remove leading/trailing dots and spaces
    safe = safe.strip('. ')
    
    if not safe:
        safe = fallback
    
    # Cap to avoid OS filename limits (leave room for prefix/suffix)
    return safe[:120]


def download_file(url: str, file_id: str, filename: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Download file from URL and save with sanitized filename.
    
    Args:
        url: URL to download from
        file_id: Unique file identifier
        filename: Optional original filename (will be sanitized)
        
    Returns:
        Path to downloaded file
    """
    temp_dir = tempfile.mkdtemp(prefix="ultimate_")
    
    # Handle double-encoded S3 URLs (e.g. %2520 instead of %20)
    is_s3_url = "s3.amazonaws.com" in url or "s3." in url or ".s3." in url
    if is_s3_url:
        parsed_original = urlparse(url)
        if "%2520" in parsed_original.path or "%2525" in parsed_original.path:
            logger.warning("[DOWNLOAD] Detected double-encoded S3 URL path, attempting to fix")
            try:
                fixed_path = unquote(parsed_original.path)
                fixed_path_encoded = quote(fixed_path, safe="/")
                if parsed_original.query:
                    url = f"{parsed_original.scheme}://{parsed_original.netloc}{fixed_path_encoded}?{parsed_original.query}"
                else:
                    url = f"{parsed_original.scheme}://{parsed_original.netloc}{fixed_path_encoded}"
                logger.info("[DOWNLOAD] Fixed double-encoded path in S3 URL")
            except Exception as decode_error:
                logger.warning(f"[DOWNLOAD] Could not fix double-encoding, using original URL: {decode_error}")
    
    parsed = urlparse(url)
    
    # Extract filename from URL path, removing query parameters
    url_path = parsed.path
    if "?" in url_path:
        url_path = url_path.split("?")[0]
    if "#" in url_path:
        url_path = url_path.split("#")[0]
    url_name = os.path.basename(url_path) if url_path else None
    
    # Prefer provided filename/URL-derived name initially; may be overridden by Content-Disposition
    local_name = _safe_filename(filename or url_name or "", f"{file_id}")
    path = os.path.join(temp_dir, local_name)

    logger.info(f"[DOWNLOAD] {redact_url(url)}")

    # S3 pre-signed URLs are sensitive to request shape; keep headers simple and deterministic
    headers = {}
    if is_s3_url:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DocumentProcessor/1.0)",
            "Accept": "*/*",
        }
    
    try:
        r = safe_get(url, timeout=60, headers=headers)  # SSRF guard, redirects re-checked
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            error_msg = (
                "403 Forbidden while downloading file. Possible causes: expired pre-signed URL, "
                "invalid signature, or double-encoded path."
            )
            logger.error(f"[DOWNLOAD ERROR] {error_msg} url={redact_url(url)}")
            raise ValueError(f"{error_msg} Original error: {e}")
        raise  # any other HTTP error: never index an error page as the document
    
    # Extract filename from Content-Disposition when present
    disposition_filename: Optional[str] = None
    content_disposition = r.headers.get("Content-Disposition", "")
    if content_disposition:
        # Supports filename="x.ext" and filename*=UTF-8''x.ext
        filename_match = re.search(r'filename[*]?=["\']?([^"\';]+)["\']?', content_disposition, re.IGNORECASE)
        if filename_match:
            disposition_filename = filename_match.group(1).strip()
            try:
                disposition_filename = unquote(disposition_filename)
            except Exception:
                pass
            logger.info(f"[DOWNLOAD] Extracted filename from Content-Disposition: {disposition_filename}")

    limit = max_download_bytes()
    declared = r.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > limit:
        r.close()
        raise ValueError(f"Download exceeds MAX_DOWNLOAD_BYTES ({limit} bytes)")
    written = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(65536):
            if chunk:
                written += len(chunk)
                if written > limit:
                    r.close()
                    raise ValueError(f"Download exceeds MAX_DOWNLOAD_BYTES ({limit} bytes)")
                f.write(chunk)

    if os.path.getsize(path) == 0:
        raise ValueError("Downloaded file is empty")

    # If Content-Disposition has a known extension, enforce it on the local file.
    if disposition_filename:
        disposition_ext = os.path.splitext(disposition_filename)[1].lower()
        known_exts = {
            ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".pptm", ".ppsx", ".potx", ".potm", ".ppsm",
            ".xls", ".xlsx", ".xlsm", ".xltx", ".xltm", ".xlsb", ".ods",
            ".txt", ".md", ".html", ".rtf", ".csv",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"
        }
        if disposition_ext and disposition_ext in known_exts:
            current_ext = os.path.splitext(path)[1].lower()
            if disposition_ext != current_ext:
                new_path = os.path.splitext(path)[0] + disposition_ext
                os.rename(path, new_path)
                path = new_path
                logger.info(f"[DOWNLOAD] Set extension '{disposition_ext}' from Content-Disposition filename")

    if disposition_filename:
        logger.info(f"[DOWNLOAD] Returning extracted filename for indexing: {disposition_filename}")
    return path, disposition_filename


def _trim_text_for_embedding(
    text: str,
    *,
    char_limit: int,
    line_limit: int,
) -> str:
    if not text or char_limit <= 0:
        return text
    if len(text) <= char_limit and line_limit <= 0:
        return text

    lines = text.splitlines()
    if line_limit > 0 and len(lines) > line_limit:
        keep_head = max(1, int(line_limit * 0.6))
        keep_tail = max(1, line_limit - keep_head)
        lines = lines[:keep_head] + lines[-keep_tail:]

    trimmed = "\n".join(lines)
    if len(trimmed) > char_limit:
        head = max(1, int(char_limit * 0.7))
        tail = max(1, char_limit - head)
        trimmed = f"{trimmed[:head]}\n...\n{trimmed[-tail:]}"

    return trimmed


# ---------------------------------------------------------------------------
# PROCESS DOCUMENT TASK
# ---------------------------------------------------------------------------
@celery_app.task(
    bind=True,
    name="src.ultimate_tasks.process_ultimate_document_task",
    # Crash-safe: message is only acked after the task body returns.
    # If the worker is OOM-killed mid-task the broker will redeliver.
    acks_late=True,
    reject_on_worker_lost=True,
    # Hard cap on automatic retries (matches MAX_RETRIES constant used in
    # should_retry_task so the two guards are always consistent).
    max_retries=15,
)
def process_ultimate_document_task(
    self,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    file_id: Optional[str] = None,
    user_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    file_type: Optional[str] = None,
    bucket_id: Optional[str] = None,
    path: Optional[str] = None,
    connection_id: Optional[str] = None,
) -> Dict[str, Any]:

    task_id = self.request.id
    delivery = self.request.delivery_info or {}
    queue_name = delivery.get("routing_key") or delivery.get("exchange") or "unknown"
    logger.info(
        f"[TASK START] {task_id} file_id={file_id} queue={queue_name} "
        f"bucket_id={bucket_id} path={path} connection_id={connection_id}"
    )

    # Heartbeat monitor for stuck-task detection & status tracking
    heartbeat_monitor = HeartbeatMonitor(task_id=task_id, file_id=file_id or "")

    # Set state to PROCESSING immediately so status API shows "processing" status
    # This ensures the UI sees "uploaded" -> "processing" -> "completed" progression
    self.update_state(state="PROCESSING", meta={"file_id": file_id, "status": "Starting processing...", "progress": 0})

    # Update job registry to PROCESSING so /admin/jobs reflects real state.
    try:
        _registry = get_job_registry()
        if _registry and user_id and file_id:
            _registry.mark_processing(user_id=user_id, file_id=file_id, task_id=task_id)
    except Exception as _reg_exc:
        logger.warning(f"[JOB REGISTRY] mark_processing failed (non-fatal): {_reg_exc}")
    
    # Note: Workflow API status is now updated immediately when task is submitted (in ultimate_ui.py)
    # This ensures backend sees "processing" status right away, even if task is queued
    # We still update here as a safety net in case the immediate update failed
    if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
        try:
            from src.workflow_manager import get_workflow_manager
            workflow_manager = get_workflow_manager()
            # Only update if not already PROCESSING (avoid redundant API calls)
            workflow_manager.update_processing(file_id)
            logger.debug(f"[WORKFLOW] Confirmed {file_id} is in PROCESSING status (safety check)")
        except Exception as e:
            logger.warning(f"[WORKFLOW] Failed to confirm PROCESSING status: {e}")
    
    # Lazy imports to save baseline RAM
    from src.ultimate_search_processor import DocumentProcessor
    from src.ultimate_vector_integration import UltimateVectorIntegration

    temp_file = None
    terminal_cleanup_ok = False
    lock_key = f"process_lock:{user_id}:{file_id}" if file_id else None  # tenant-scoped
    lock_acquired = False
    lock_ttl = int(os.getenv("PROCESS_LOCK_TTL", "10800"))  # 3 hours

    # Best-effort Redis lock to prevent duplicate processing of same file_id.
    # If an existing lock appears stale based on heartbeat, we recover it
    # instead of skipping the task, so that failed/stuck jobs can be retried.
    redis_client = None
    try:
        redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        if lock_key:
            # Try to acquire a fresh lock
            lock_acquired = bool(redis_client.set(lock_key, task_id, nx=True, ex=lock_ttl))
            if not lock_acquired:
                # Someone already holds the lock; check if that task is still healthy.
                existing_task_id = None
                try:
                    existing_task_id = redis_client.get(lock_key)
                    if isinstance(existing_task_id, bytes):
                        existing_task_id = existing_task_id.decode("utf-8")
                except Exception:
                    existing_task_id = None

                is_stale = False
                if existing_task_id:
                    hb_key = f"heartbeat:{existing_task_id}"
                    try:
                        hb_raw = redis_client.get(hb_key)
                        if hb_raw:
                            import json as _json
                            try:
                                hb = _json.loads(hb_raw)
                                last_ts = float(hb.get("timestamp", 0))
                            except Exception:
                                last_ts = 0
                            if last_ts <= 0 or (time.time() - last_ts) > LOCK_STUCK_THRESHOLD:
                                is_stale = True
                        else:
                            # No heartbeat found for existing task -> treat as stale
                            is_stale = True
                    except Exception as hb_err:
                        logger.warning(f"[LOCK] Failed to inspect heartbeat for {existing_task_id}: {hb_err}")
                else:
                    # No known owner -> treat as stale and recover
                    is_stale = True

                if is_stale:
                    # Recover stale lock for this new task
                    logger.warning(
                        f"[LOCK RECOVER] Overriding stale lock for file_id={file_id} "
                        f"(previous_task={existing_task_id})"
                    )
                    redis_client.set(lock_key, task_id, ex=lock_ttl)
                    lock_acquired = True
                else:
                    # Another healthy worker is already processing this file; mark this
                    # duplicate task as skipped to avoid double work.
                    logger.warning(
                        f"[TASK DUPLICATE] {task_id} file_id={file_id} already processing "
                        f"by task={existing_task_id}; skipping duplicate task instance"
                    )
                    self.update_state(
                        state="SKIPPED",
                        meta={"file_id": file_id, "status": "Already processing", "progress": 0},
                    )
                    return {"success": True, "file_id": file_id, "task_id": task_id, "skipped": True}
    except Exception as e:
        logger.warning(f"[LOCK] Redis lock unavailable, proceeding without lock: {e}")

    def update(msg, progress):
        self.update_state(state="PROCESSING", meta={"file_id": file_id, "status": msg, "progress": progress})
        # Also emit heartbeat so external monitors can see progress
        try:
            heartbeat_monitor.update(progress=progress, status="processing", details=msg)
        except Exception:
            # Heartbeat failures must never break core processing
            pass
        logger.info(f"[STATE] {msg} ({progress}%)")

    try:
        update("Preparing file...", 5)

        # Download if needed
        if url:
            update("Downloading file...", 10)
            temp_file, extracted_filename = download_file(url, file_id, original_filename)
            file_path = temp_file
            if extracted_filename:
                original_filename = extracted_filename
                logger.info(f"[TASK] Using extracted filename from Content-Disposition: {original_filename}")
        elif not file_path:
            raise ValueError("No file source (url or file_path) provided.")

        # Check if this is an image file
        is_image = file_type and file_type.startswith("image/")
        
        # For images, extract text using OCR for vector search.
        # BLIP captioning is globally disabled via SKIP_IMAGE_CAPTIONING_IN_PROCESSOR=true
        # to keep workers lightweight and avoid OOMs.
        # For documents, use text extraction.
        if is_image:
            # Image processing: Extract text using OCR first (for vector search)
            update("Extracting text from image (OCR)...", 30)
            processor = DocumentProcessor()
            result = processor.process_document(file_path=file_path, target_words=[], file_type=file_type)
            
            if not result or not result.text_content:
                logger.warning(f"[IMAGE OCR] No text extracted, trying captioning only...")
                # Fallback to captioning if OCR fails
                get_img_emb, analyze_img = _get_image_utils()
                if get_img_emb and analyze_img:
                    img_analysis = analyze_img(file_path)
                    caption = img_analysis.get("description", "image content")
                    result.text_content = caption
                    result.extraction_method = "image_captioning"
                else:
                    raise ValueError("No text extracted from image and captioning not available")
            
            logger.info(f"[IMAGE TEXT OK] chars={len(result.text_content)}")
            
            # Store extracted text as document for vector search (this is the key for vector search on images)
            update("Storing image text in Milvus for vector search...", 65)
            uvi = get_worker_vector_integration()
            
            store_meta = {
                # Preserve original filename for filename-based search/display
                "filename": original_filename or file_id,
                "original_filename": original_filename or file_id,
                # Canonical identifier for this file across all systems
                "file_id": file_id,
                "source_file": file_id,
                "file_type": file_type,
                "extraction_method": result.extraction_method,
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                "is_image": True,
                # Tenant / bucket / connection scoping
                "bucket_id": bucket_id,
                "path": path,
                "connection_id": connection_id,
            }
            
            _assert_milvus_circuit_closed()
            try:
                store_res = uvi.upsert_document(
                    file_id=file_id,
                    text_content=result.text_content,
                    metadata=store_meta,
                    user_id=user_id,
                )
            except Exception as first_store_error:
                if _is_milvus_connectivity_error(first_store_error):
                    _open_milvus_circuit()
                logger.warning(
                    f"[MILVUS RETRY] image upsert failed once, recreating integration: {first_store_error}"
                )
                uvi = get_worker_vector_integration(force_recreate=True)
                store_res = uvi.upsert_document(
                    file_id=file_id,
                    text_content=result.text_content,
                    metadata=store_meta,
                    user_id=user_id,
                )
            
            if not store_res.get("success"):
                logger.error(f"[IMAGE TEXT STORE ERROR] {store_res}")
                raise RuntimeError(f"Milvus image text upsert failed: {store_res}")
            else:
                logger.info(f"[IMAGE TEXT STORE OK] chunks={store_res.get('inserted_chunks')}")
            
            # Optional: Also store CLIP embedding for semantic image search.
            # This step is heavy and can cause OOM on constrained machines.
            # It is now guarded by an environment flag so OCR + text-indexing
            # always work, and image-CLIP embeddings are only enabled
            # explicitly when enough resources are available.
            if os.getenv("ENABLE_IMAGE_CLIP", "false").lower() == "true":
                update("Storing image CLIP embedding (optional)...", 80)
                get_img_emb, analyze_img = _get_image_utils()
                if get_img_emb and analyze_img:
                    try:
                        # Process in smaller steps to reduce memory pressure
                        import gc
                        emb = get_img_emb(file_path)
                        if emb is not None and emb.size > 0:
                            img_analysis = analyze_img(file_path)
                            uvi.upsert_image_vector(
                                file_id=file_id,
                                caption=img_analysis.get("description", ""),
                                analysis=img_analysis,
                                user_id=user_id,
                                text_to_image_embedder=lambda _: emb,
                                bucket_id=bucket_id,
                                path=path,
                                connection_id=connection_id,
                            )
                            logger.info(f"[IMAGE CLIP EMBEDDING STORED] file_id={file_id}")
                            # Clean up memory
                            del emb, img_analysis
                            gc.collect()
                    except MemoryError as mem_err:
                        logger.warning(f"[IMAGE CLIP EMBEDDING SKIP - MEMORY] {mem_err}")
                        import gc
                        gc.collect()
                    except Exception as img_err:
                        logger.warning(f"[IMAGE CLIP EMBEDDING SKIP] {img_err}")
                        import gc
                        gc.collect()
            else:
                logger.info("[IMAGE CLIP EMBEDDING SKIP] ENABLE_IMAGE_CLIP is false.")
        
        else:
            # Document processing: Use text extraction
            update("Extracting text / OCR...", 30)
            import time as time_module
            extraction_start = time_module.time()
            processor = DocumentProcessor()
            logger.info(f"[EXTRACTION] Starting extraction for file: {file_path}")
            result = processor.process_document(file_path=file_path, target_words=[], file_type=file_type)
            extraction_time = time_module.time() - extraction_start
            logger.info(f"[TIMING] Text extraction: {extraction_time:.2f}s")

            if not result:
                raise ValueError("process_document returned None")
            
            if not result.text_content:
                logger.warning(f"[EXTRACTION] No text content extracted from {file_path}. Proceeding with metadata indexing only.")
            else:
                logger.info(f"[TEXT OK] chars={len(result.text_content):,}")

            # Store embeddings + chunks in Milvus
            update("Storing document embeddings & metadata in Milvus...", 65)
            import time as time_module
            embedding_start = time_module.time()
            uvi = get_worker_vector_integration()

            # ------------------------------------------------------
            # FULL METADATA EXTRACTION (persons, orgs, dates, etc.)
            # ------------------------------------------------------
            text_content = result.text_content or ""
            file_ext = Path(original_filename or file_path or "").suffix.lower()
            is_spreadsheet = (
                file_type in {
                    "text/csv",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.oasis.opendocument.spreadsheet",
                }
                or file_ext in {".csv", ".xls", ".xlsx", ".ods", ".xlsb"}
            )

            # Hard caps for embedding input to prevent huge CSV/XLSX from blocking.
            csv_char_limit = int(os.getenv("CSV_EMBED_CHAR_LIMIT", "2000000"))
            csv_line_limit = int(os.getenv("CSV_EMBED_ROW_LIMIT", "200000"))
            sheet_char_limit = int(os.getenv("SPREADSHEET_EMBED_CHAR_LIMIT", "2000000"))
            sheet_line_limit = int(os.getenv("SPREADSHEET_EMBED_LINE_LIMIT", "200000"))
            # Hard caps for stability under concurrent load (independent of env overrides).
            csv_char_limit = min(csv_char_limit, 250000)
            csv_line_limit = min(csv_line_limit, 25000)
            sheet_char_limit = min(sheet_char_limit, 250000)
            sheet_line_limit = min(sheet_line_limit, 25000)
            global_char_limit = 300000
            global_line_limit = 30000

            if is_spreadsheet:
                if file_ext == ".csv" or file_type == "text/csv":
                    # Keep CSV embedding payload bounded so workers stay healthy under parallel loads.
                    original_chars = len(text_content)
                    original_rows = text_content.count('\n')
                    text_content = _trim_text_for_embedding(
                        text_content,
                        char_limit=csv_char_limit,
                        line_limit=csv_line_limit,
                    )
                    if len(text_content) < original_chars:
                        logger.warning(
                            f"[CSV] Trimmed embedding input for stability "
                            f"(chars {original_chars}->{len(text_content)}, rows {original_rows}->{text_content.count(chr(10))})"
                        )
                else:
                    # For Excel files, still apply limits (they're less row-heavy)
                    text_content = _trim_text_for_embedding(
                        text_content,
                        char_limit=sheet_char_limit,
                        line_limit=sheet_line_limit,
                    )
            else:
                # Hard global bound for non-spreadsheet types to avoid OOM under parallel loads.
                text_content = _trim_text_for_embedding(
                    text_content,
                    char_limit=global_char_limit,
                    line_limit=global_line_limit,
                )
            # Skip metadata for spreadsheets AND presentations (fast path)
            is_presentation = (
                file_type in {
                    "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
                }
                or file_ext in {".ppt", ".pptx", ".ppsx", ".pptm"}
            )
            is_word = (
                "word" in (file_type or "").lower() or "msword" in (file_type or "").lower()
                or file_ext in {".doc", ".docx"}
            )
            metadata_max_chars = int(os.getenv("METADATA_MAX_CHARS", "200000"))
            skip_metadata = (
                (os.getenv("SPREADSHEET_SKIP_METADATA", "true").lower() == "true" and is_spreadsheet)
                or is_presentation
                or (is_word and len(text_content) < 35000)
                or (len(text_content) > metadata_max_chars)
            )

            if skip_metadata:
                logger.info(
                    "[META SKIP] Skipping temporal/entity extraction "
                    f"(spreadsheet={is_spreadsheet}, presentation={is_presentation}, chars={len(text_content)})"
                )
                normalized_temporal = {}
                years = []
                primary_dates = []
                full_dates = []
                month_years = []
                month_only = None
                organizations = []
                persons = []
                locations = []
                governing_law = None
                signers = []
            else:
                # 1) Temporal profile (years, full dates, month-year, month-only)
                #    Use ingestion-time normalizer as canonical source, with temporal engine as helper.
                normalized_temporal = _INGESTION_TEMPORAL_NORMALIZER.normalize(
                    metadata=result.metadata or {},
                    full_text=text_content,
                    file_id=file_id or "",
                )

                years = normalized_temporal.get("years", []) or []
                primary_dates = normalized_temporal.get("primary_dates", []) or []
                month_years = normalized_temporal.get("month_years", []) or []
                month_only = None
                if normalized_temporal.get("month_year"):
                    # month_year is (month_name, year)
                    month_only = normalized_temporal["month_year"][0]

                # Backward-compat: also consult temporal engine for month_only if not set
                if not month_only:
                    month_only = _TEMPORAL_ENGINE.extract_month_only(text_content)

                full_dates = primary_dates or []

                # 2) Entity profile (organizations, persons, locations)
                entity_profile = _ENTITY_EXTRACTOR.extract_all_entities(
                    text_content,
                    metadata=result.metadata or {},
                )
                organizations = entity_profile.get("organizations", []) or []
                persons = entity_profile.get("persons", []) or []
                locations = entity_profile.get("locations", []) or []

                # 3) Governing law (from body text)
                governing_law = extract_governing_law(text_content) or result.metadata.get("governing_law")

                # 4) Heuristic signers (for now, align with persons/signature patterns)
                signers = list(persons)
            
            store_meta = {
                # Preserve original filename for filename-based search/display
                "filename": original_filename or file_id,
                "original_filename": original_filename or file_id,
                # Canonical identifier for this file across all systems
                "file_id": file_id,
                "source_file": file_id,
                "file_type": file_type,
                "extraction_method": result.extraction_method,
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                # Temporal metadata
                "years": years,
                "full_dates": full_dates,
                "month_years": month_years,
                "month_only": month_only,
                "primary_dates": primary_dates,
                "earliest_date": normalized_temporal.get("earliest_date"),
                "latest_date": normalized_temporal.get("latest_date"),
                "is_expired": normalized_temporal.get("is_expired"),
                # Entity metadata
                "organizations": organizations,
                "persons": persons,
                "signers": signers,
                "locations": locations,
                "governing_law": governing_law,
                # Tenant / bucket / connection scoping
                "bucket_id": bucket_id,
                "path": path,
                "connection_id": connection_id,
            }

            _assert_milvus_circuit_closed()
            try:
                store_res = uvi.upsert_document(
                    file_id=file_id,
                    text_content=result.text_content,
                    metadata=store_meta,
                    user_id=user_id
                )
            except Exception as first_store_error:
                if _is_milvus_connectivity_error(first_store_error):
                    _open_milvus_circuit()
                logger.warning(
                    f"[MILVUS RETRY] document upsert failed once, recreating integration: {first_store_error}"
                )
                uvi = get_worker_vector_integration(force_recreate=True)
                store_res = uvi.upsert_document(
                    file_id=file_id,
                    text_content=result.text_content,
                    metadata=store_meta,
                    user_id=user_id
                )
            embedding_time = time_module.time() - embedding_start
            logger.info(f"[TIMING] Embedding + indexing: {embedding_time:.2f}s")

            if not store_res.get("success"):
                logger.error(f"[VECTOR STORE ERROR] {store_res}")
                raise RuntimeError(f"Milvus document upsert failed: {store_res}")
            else:
                logger.info(f"[VECTOR STORE OK] chunks={store_res.get('inserted_chunks')}")

        update("Completed", 100)
        terminal_cleanup_ok = True

        # Workflow completion callback
        if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
            try:
                from src.workflow_manager import get_workflow_manager
                workflow_manager = get_workflow_manager()
                workflow_manager.update_completed(file_id)
                logger.info(f"[WORKFLOW] Updated {file_id} to COMPLETED status")
            except Exception as e:
                logger.warning(f"[WORKFLOW] Failed to update COMPLETED status: {e}")

        # Compute inserted chunks for registry
        _inserted_chunks = 0
        if 'store_res' in locals() and isinstance(store_res, dict):
            _inserted_chunks = store_res.get('inserted_chunks', 0) or 0

        # Update job registry to SUCCESS
        try:
            _registry = get_job_registry()
            if _registry and user_id and file_id:
                _registry.mark_success(user_id=user_id, file_id=file_id, task_id=task_id, chunks=_inserted_chunks)
        except Exception as _reg_exc:
            logger.warning(f"[JOB REGISTRY] mark_success failed (non-fatal): {_reg_exc}")

        # Rebuild metadata inverted index + push Redis blob so the search server
        # never serves a stale temporal/location/person index after re-ingestion.
        if user_id and not is_image:
            try:
                schedule_user_metadata_index_rebuild(user_id)
            except Exception as _meta_sched_exc:
                logger.debug(
                    f"[METADATA SYNC] schedule after ingest skipped: {_meta_sched_exc}"
                )

        # Return success response
        if is_image:
            return {
                "success": True,
                "file_id": file_id,
                "text_length": len(result.text_content) if 'result' in locals() and result else 0,
                "task_id": task_id,
                "is_image": True
            }
        else:
            return {
                "success": True,
                "file_id": file_id,
                "text_length": len(result.text_content),
                "task_id": task_id
            }

    except SoftTimeLimitExceeded as e:
        logger.error(f"[TASK TIMEOUT] {e}")
        # Decide whether to retry or mark as permanently failed
        attempt = getattr(self.request, "retries", 0)
        if should_retry_task(e, attempt):
            delay = calculate_retry_delay(attempt)
            if _is_connectivity_retry(e):
                # Keep retry cadence tight for infra hiccups so queues do not stall.
                delay = min(10 * (attempt + 1), 30)
            msg = f"Retrying after timeout (attempt {attempt + 1}/{MAX_RETRIES}) in {delay}s"
            logger.warning(f"[TASK RETRY] {msg}")
            # Mark intermediate state as RETRYING for visibility
            self.update_state(
                state="RETRYING",
                meta={"file_id": file_id, "status": msg, "progress": 0},
            )
            if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
                try:
                    from src.workflow_manager import get_workflow_manager
                    workflow_manager = get_workflow_manager()
                    workflow_manager.update_failed(file_id, error_message=msg)
                except Exception as workflow_err:
                    logger.warning(f"[WORKFLOW] Failed to record retry status: {workflow_err}")
            raise self.retry(exc=e, countdown=delay)
        else:
            update("FAILED: Processing timed out", 0)
            terminal_cleanup_ok = True
            if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
                try:
                    from src.workflow_manager import get_workflow_manager
                    workflow_manager = get_workflow_manager()
                    workflow_manager.update_failed(file_id, error_message="Processing timed out")
                    logger.info(f"[WORKFLOW] Updated {file_id} to FAILED status (timeout)")
                except Exception as workflow_err:
                    logger.warning(f"[WORKFLOW] Failed to update FAILED status: {workflow_err}")
            raise e
    except Exception as e:
        logger.error(f"[TASK ERROR] {e}")
        attempt = getattr(self.request, "retries", 0)
        if should_retry_task(e, attempt):
            delay = calculate_retry_delay(attempt)
            if _is_connectivity_retry(e):
                # Fail fast and retry fast for connectivity issues (Milvus/Redis/network).
                delay = min(10 * (attempt + 1), 30)
            msg = f"Retrying after error (attempt {attempt + 1}/{MAX_RETRIES}) in {delay}s: {str(e)}"
            logger.warning(f"[TASK RETRY] {msg}")
            self.update_state(
                state="RETRYING",
                meta={"file_id": file_id, "status": msg, "progress": 0},
            )
            if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
                try:
                    from src.workflow_manager import get_workflow_manager
                    workflow_manager = get_workflow_manager()
                    workflow_manager.update_failed(file_id, error_message=msg)
                except Exception as workflow_err:
                    logger.warning(f"[WORKFLOW] Failed to record retry status: {workflow_err}")
            raise self.retry(exc=e, countdown=delay)
        else:
            update(f"FAILED: {str(e)}", 0)
            terminal_cleanup_ok = True
            # Update external workflow API to FAILED status (if enabled)
            if WORKFLOW_ENABLED and WORKFLOW_ACCESS_KEY and WORKFLOW_STOR_API_KEY and file_id:
                try:
                    from src.workflow_manager import get_workflow_manager
                    workflow_manager = get_workflow_manager()
                    workflow_manager.update_failed(file_id, error_message=str(e))
                    logger.info(f"[WORKFLOW] Updated {file_id} to FAILED status")
                except Exception as workflow_err:
                    logger.warning(f"[WORKFLOW] Failed to update FAILED status: {workflow_err}")
            # Update job registry to FAILED
            try:
                _registry = get_job_registry()
                if _registry and user_id and file_id:
                    _registry.mark_failed(user_id=user_id, file_id=file_id, task_id=task_id, error=str(e))
            except Exception as _reg_exc:
                logger.warning(f"[JOB REGISTRY] mark_failed failed (non-fatal): {_reg_exc}")
            raise e

    finally:
        if temp_file:
            try:
                shutil.rmtree(os.path.dirname(temp_file), ignore_errors=True)
            except:
                pass
        # For uploaded files, remove /app/temp_uploads artifact only when task reached
        # a terminal state (success or permanent failure). Keep file for retries.
        if terminal_cleanup_ok and file_path:
            try:
                upload_root = os.path.abspath(os.getenv("UPLOAD_DIR", "/app/temp_uploads"))
                abs_file_path = os.path.abspath(str(file_path))
                if abs_file_path.startswith(upload_root + os.sep) and os.path.isfile(abs_file_path):
                    os.remove(abs_file_path)
                    logger.info(f"[UPLOAD CLEANUP] Removed temp upload file: {abs_file_path}")
            except Exception as cleanup_err:
                logger.warning(f"[UPLOAD CLEANUP] Failed for file_path={file_path}: {cleanup_err}")
        if redis_client and lock_key and lock_acquired:
            try:
                redis_client.delete(lock_key)
            except Exception:
                pass
        # Always clean up heartbeat data at the very end
        try:
            heartbeat_monitor.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# METADATA INDEX — publish to Redis after ingestion (processing worker only)
# ---------------------------------------------------------------------------
def schedule_user_metadata_index_rebuild(user_id: Optional[str]) -> None:
    """Queue a Milvus→MetadataIndex→Redis publish. No-op if not publisher."""
    if not user_id:
        return
    if os.getenv("METADATA_INDEX_PUBLISHER", "false").lower() not in ("true", "1", "yes"):
        return
    try:
        # Shorter default so search nodes pick up re-ingested docs faster after bulk uploads.
        countdown = int(os.getenv("METADATA_REBUILD_COUNTDOWN_SEC", "12"))
    except ValueError:
        countdown = 25
    try:
        rebuild_user_metadata_index_task.apply_async(args=[user_id], countdown=countdown)
        logger.info(
            f"[METADATA SYNC] Queued metadata rebuild for user={user_id[:12]}… "
            f"in {countdown}s"
        )
    except Exception as e:
        logger.warning(f"[METADATA SYNC] Failed to queue metadata rebuild: {e}")


@celery_app.task(
    bind=True,
    name="src.ultimate_tasks.rebuild_user_metadata_index_task",
    ignore_result=True,
    acks_late=True,
)
def rebuild_user_metadata_index_task(self, user_id: str) -> Dict[str, Any]:
    """
    Full metadata index rebuild from Milvus + save_to_disk (pushes Redis blob).
    Serialized per user via Redis lock so burst uploads coalesce into sequential runs.
    """
    if not user_id:
        return {"ok": False, "reason": "no user_id"}
    if os.getenv("METADATA_INDEX_PUBLISHER", "false").lower() not in ("true", "1", "yes"):
        return {"ok": False, "reason": "not_publisher"}

    url = os.getenv("REDIS_URL", "").strip()
    r = None
    if url:
        try:
            r = redis.from_url(url, socket_connect_timeout=5, socket_timeout=120)
        except Exception:
            r = None

    def _run() -> Dict[str, Any]:
        import os as _os

        from src.semantic.semantic_components import metadata_cache_path

        cache_file = metadata_cache_path(user_id)
        try:
            if cache_file and _os.path.isfile(cache_file):
                _os.remove(cache_file)
        except OSError:
            pass
        from src.semantic.semantic_pipeline import SemanticPipeline

        sp = SemanticPipeline()
        sp._init_milvus_connections()
        if not sp.doc_db:
            logger.warning("[METADATA SYNC] doc_db unavailable; skip rebuild")
            return {"ok": False, "reason": "no_doc_db"}
        sp._ensure_metadata_index_for_user(user_id, force_refresh=True)
        nd = len(sp.metadata_index.docs) if sp.metadata_index else 0
        logger.info(f"[METADATA SYNC] Rebuilt metadata index user={user_id} ({nd} docs)")
        return {"ok": True, "docs": nd}

    try:
        if r is not None:
            lock = r.lock(
                f"ultimate:meta_rebuild_lock:{user_id}",
                timeout=900,
                blocking_timeout=240,
            )
            with lock:
                return _run()
        return _run()
    except Exception as e:
        logger.warning(f"[METADATA SYNC] rebuild_user_metadata_index_task failed: {e}")
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# SEARCH TASK (OPTIONAL BACKGROUND SEARCH)
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="src.ultimate_tasks.search_ultimate_documents_task")
def search_ultimate_documents_task(self, query: str, user_id: Optional[str] = None, limit: int = 10):
    try:
        from src.semantic.semantic_pipeline import SemanticPipeline
        pipeline = SemanticPipeline()
        results = pipeline.search_documents(query, top_k=limit, user_id=user_id)
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
    except ImportError as e:
        logger.error(f"Semantic pipeline not available: {e}")
        return {
            "success": False,
            "query": query,
            "results": [],
            "count": 0,
            "error": "Semantic search not available",
        }


@celery_app.task(bind=True, name="src.ultimate_tasks.monitor_stuck_jobs")
def monitor_stuck_jobs(self) -> Dict[str, Any]:
    """
    Periodic monitoring task for stuck jobs.

    This task is intended to be scheduled via Celery beat or triggered manually.
    It does NOT modify task state; it only logs and returns the current view of
    stuck jobs based on heartbeat data.
    """
    try:
        jobs = detect_stuck_jobs()
        if jobs:
            logger.warning(f"[MONITOR] Detected {len(jobs)} stuck jobs")
        else:
            logger.info("[MONITOR] No stuck jobs detected")
        return {
            "success": True,
            "count": len(jobs),
            "jobs": jobs,
        }
    except Exception as e:
        logger.error(f"[MONITOR] Failed to monitor stuck jobs: {e}")
        return {
            "success": False,
            "error": str(e),
        }
