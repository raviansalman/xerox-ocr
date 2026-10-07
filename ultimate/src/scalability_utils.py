#!/usr/bin/env python3
"""
Scalability Utilities
Provides heartbeat monitoring, stuck job detection, and retry mechanisms
"""

import os
import time
import json
import logging
import redis
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from celery import current_task

logger = logging.getLogger(__name__)

# Redis client for heartbeat tracking
_redis_client = None

def get_redis_client():
    """Get or create Redis client"""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    return _redis_client

# Configuration
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # seconds
STUCK_JOB_THRESHOLD = int(os.getenv("STUCK_JOB_THRESHOLD", "300"))  # 5 minutes
MAX_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "15"))
RETRY_DELAY_BASE = int(os.getenv("RETRY_DELAY_BASE", "10"))  # seconds
RETRY_DELAY_MAX = 45  # hard cap to keep queues moving under transient outages


class HeartbeatMonitor:
    """Manages heartbeat updates during task processing"""
    
    def __init__(self, task_id: str, file_id: Optional[str] = None):
        self.task_id = task_id
        self.file_id = file_id
        self.redis_client = get_redis_client()
        self.last_heartbeat = time.time()
        self.heartbeat_key = f"heartbeat:{task_id}"
        self.start_time = time.time()
    
    def update(self, progress: int = 0, status: str = "processing", details: str = ""):
        """Update heartbeat with current progress"""
        try:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            heartbeat_data = {
                "task_id": self.task_id,
                "file_id": self.file_id or "",
                "timestamp": current_time,
                "progress": progress,
                "status": status,
                "details": details,
                "elapsed_seconds": elapsed
            }
            
            # Store heartbeat in Redis with TTL (JSON-encoded for safe parsing)
            self.redis_client.setex(
                self.heartbeat_key,
                STUCK_JOB_THRESHOLD + 60,  # TTL slightly longer than threshold
                json.dumps(heartbeat_data)
            )
            
            # Also update Celery task state
            if current_task:
                current_task.update_state(
                    state="PROCESSING",
                    meta={
                        "file_id": self.file_id,
                        "status": status,
                        "progress": progress,
                        "details": details,
                        "elapsed_seconds": elapsed
                    }
                )
            
            self.last_heartbeat = current_time
            logger.debug(f"[HEARTBEAT] {self.task_id} progress={progress}% status={status}")
            
        except Exception as e:
            logger.warning(f"[HEARTBEAT] Failed to update heartbeat: {e}")
    
    def check_if_stuck(self) -> bool:
        """Check if task appears to be stuck (no heartbeat within threshold)"""
        try:
            heartbeat_data = self.redis_client.get(self.heartbeat_key)
            if not heartbeat_data:
                # No heartbeat found - might be stuck
                return True
            
            # Parse heartbeat data
            import json
            try:
                data = json.loads(heartbeat_data)
                last_timestamp = data.get("timestamp", 0)
                time_since_heartbeat = time.time() - last_timestamp
                
                if time_since_heartbeat > STUCK_JOB_THRESHOLD:
                    logger.warning(f"[STUCK JOB] Task {self.task_id} appears stuck (no heartbeat for {time_since_heartbeat:.0f}s)")
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.warning(f"[STUCK JOB] Failed to check stuck status: {e}")
            return False
    
    def cleanup(self):
        """Clean up heartbeat data"""
        try:
            self.redis_client.delete(self.heartbeat_key)
        except Exception as e:
            logger.warning(f"[HEARTBEAT] Failed to cleanup: {e}")


def detect_stuck_jobs() -> list:
    """Detect all stuck jobs across the system"""
    try:
        redis_client = get_redis_client()
        stuck_jobs = []
        
        # Find all heartbeat keys
        heartbeat_keys = redis_client.keys("heartbeat:*")
        
        current_time = time.time()
        for key in heartbeat_keys:
            try:
                heartbeat_data = redis_client.get(key)
                if heartbeat_data:
                    import json
                    data = json.loads(heartbeat_data)
                    last_timestamp = data.get("timestamp", 0)
                    time_since_heartbeat = current_time - last_timestamp
                    
                    if time_since_heartbeat > STUCK_JOB_THRESHOLD:
                        task_id = data.get("task_id", key.split(":")[-1])
                        stuck_jobs.append({
                            "task_id": task_id,
                            "file_id": data.get("file_id", ""),
                            "time_since_heartbeat": time_since_heartbeat,
                            "last_status": data.get("status", "unknown")
                        })
            except Exception as e:
                logger.warning(f"[STUCK JOB] Error checking {key}: {e}")
        
        return stuck_jobs
        
    except Exception as e:
        logger.error(f"[STUCK JOB] Failed to detect stuck jobs: {e}")
        return []


def calculate_retry_delay(attempt: int) -> int:
    """Calculate exponential backoff delay for retries"""
    return min(RETRY_DELAY_BASE * (2 ** attempt), RETRY_DELAY_MAX)


def should_retry_task(exception: Exception, attempt: int) -> bool:
    """Determine if task should be retried based on exception type and attempt count"""
    if attempt >= MAX_RETRIES:
        return False
    
    # Don't retry on certain exceptions
    non_retryable_exceptions = (
        ValueError,  # Invalid input
        TypeError,   # Programming errors
        AttributeError,  # Programming errors
    )
    
    if isinstance(exception, non_retryable_exceptions):
        return False
    
    # Retry on transient errors
    retryable_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,
        MemoryError,
    )
    
    if isinstance(exception, retryable_exceptions):
        return True
    
    # Retry on SoftTimeLimitExceeded (task timeout)
    from celery.exceptions import SoftTimeLimitExceeded
    if isinstance(exception, SoftTimeLimitExceeded):
        return True
    
    # Default: retry on unknown errors
    return True


