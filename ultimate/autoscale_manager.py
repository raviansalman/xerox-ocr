#!/usr/bin/env python3
"""
Ultimate Auto-Scaler & Performance Monitor
------------------------------------------
1. Monitors Redis queue lengths for each specialized file type.
2. Dynamically scales Celery workers using `docker compose up --scale`.
3. Limits scaling to prevent exceeding 80% CPU capacity.
4. Provides a real-time dashboard of system capability.
"""

import time
import subprocess
import os
import json
import redis
from datetime import datetime

# --- CONFIGURATION ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CHECK_INTERVAL_SEC = 20
MAX_CPU_PERCENT = 80  # Target limit
HOST_CORES = 16       # Total CPU cores available on this machine

# Scaling thresholds (Queen length -> Number of workers)
SCALE_MAP = {
    "ultimate_pdf":         [{"queue": 5, "workers": 2}, {"queue": 15, "workers": 3}],
    "ultimate_spreadsheet": [{"queue": 3, "workers": 2}, {"queue": 10, "workers": 3}],
    "ultimate_image":       [{"queue": 10, "workers": 2}],
    "ultimate_word":        [{"queue": 10, "workers": 2}],
    "ultimate_ocr":         [{"queue": 5, "workers": 2}],
    "total_pressure":       [{"queue": 10, "workers": 2}, {"queue": 40, "workers": 3}]
}

# Worker resource estimation (CPU cores per worker instance)
CPU_WEIGHTS = {
    "ultimate-celery-pdf": 0.5,
    "ultimate-celery-spreadsheet": 0.8,
    "ultimate-celery-word": 0.5,
    "ultimate-celery-ppt": 0.5,
    "ultimate-celery-image": 0.8,
    "ultimate-celery-ocr": 0.5,
    "ultimate-search": 0.5,
    "ultimate-embedder": 2.0,
    "milvus": 1.0,
    "redis": 0.2
}

def get_queue_lengths(r):
    queues = ["ultimate_pdf", "ultimate_spreadsheet", "ultimate_image", "ultimate_word", "ultimate_ocr"]
    lengths = {}
    for q in queues:
        lengths[q] = r.llen(q)
    return lengths

def calculate_current_cpu_allocation(current_scales):
    total = 0
    for service, scale in current_scales.items():
        weight = CPU_WEIGHTS.get(service, 0.5)
        total += (weight * scale)
    return (total / HOST_CORES) * 100

def scale_system(targets):
    """Executes docker compose up --scale command."""
    cmd = ["docker", "compose", "-f", "docker-compose.ultimate.yml", "up", "-d"]
    for service, scale in targets.items():
        cmd.extend(["--scale", f"{service}={scale}"])
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Applying new scales: {targets}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        print(f"✅ Scaling command output: {res.stdout.decode().strip()}")
    except subprocess.TimeoutExpired:
        print("⚠️ Scaling command timed out!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Scaling failed: {e.stderr.decode()}")

def main():
    print("=" * 60)
    print("      Ultimate System Auto-Scaler & CPU Monitor")
    print(f"      Target CPU: {MAX_CPU_PERCENT}% | Cores: {HOST_CORES}")
    print("=" * 60)

    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        return

    # Tracking current state to avoid redundant scaling commands
    current_scales = {
        "ultimate-celery-pdf": 1,
        "ultimate-celery-spreadsheet": 1,
        "ultimate-celery-word": 1,
        "ultimate-celery-ppt": 1,
        "ultimate-celery-image": 1,
        "ultimate-celery-ocr": 1,
        "ultimate-embedder": 1
    }

    while True:
        lengths = get_queue_lengths(r)
        new_scales = current_scales.copy()
        
        # 1. Determine desired scales for each worker
        for q_name, limits in SCALE_MAP.items():
            if q_name == "total_pressure":
                total_len = sum(lengths.values())
                target_e = 1
                for limit in sorted(limits, key=lambda x: x["queue"], reverse=True):
                    if total_len >= limit["queue"]:
                        target_e = limit["workers"]
                        break
                new_scales["ultimate-embedder"] = target_e
                continue

            service_name = q_name.replace("_", "-").replace("ultimate", "ultimate-celery")
            q_len = lengths.get(q_name, 0)
            
            target_w = 1
            for limit in sorted(limits, key=lambda x: x["queue"], reverse=True):
                if q_len >= limit["queue"]:
                    target_w = limit["workers"]
                    break
            new_scales[service_name] = target_w
        
        print(f"Target scales: {new_scales}")

        # 2. Check CPU constraint
        estimated_cpu = calculate_current_cpu_allocation(new_scales)
        
        if estimated_cpu > MAX_CPU_PERCENT:
            print(f"⚠️ Scale-up throttled! Estimated CPU ({estimated_cpu:.1f}%) exceeds limit ({MAX_CPU_PERCENT}%).")
            # Revert to current scales if we would blow the budget
            new_scales = current_scales 
        
        # 3. Apply changes if different from current
        if new_scales != current_scales:
            scale_system(new_scales)
            current_scales = new_scales
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitor: Load Stable. Queues: {lengths} | CPU Allocation: {estimated_cpu:.1f}%")

        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
