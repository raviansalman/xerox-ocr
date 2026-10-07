#!/usr/bin/env python3
"""
Resource-aware scaling for Ultimate stack.

Detects host CPU and memory, reserves fixed amounts for API/Milvus/Redis,
then computes per-worker memory and CPU limits so the full stack fits
without OOM. Writes docker-compose.scale.override.yml and optionally
prints which workers to run when memory is tight.

Usage:
  python scripts/scale_by_resources.py [--dry-run] [--output FILE]
  cd ultimate && python scripts/scale_by_resources.py
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


# Fixed memory (GB) reserved for non-worker services
RESERVED_API_GB = 8
RESERVED_MILVUS_GB = 10
RESERVED_REDIS_GB = 4
RESERVED_OS_GB = 2
RESERVED_TOTAL_GB = RESERVED_API_GB + RESERVED_MILVUS_GB + RESERVED_REDIS_GB + RESERVED_OS_GB

# CPU reserved for API + Milvus (not workers)
RESERVED_CPUS = 2.0

# Worker pool: all Celery worker service names (must match docker-compose.ultimate.yml)
WORKER_SERVICES = [
    "ultimate-celery-worker",
    "ultimate-celery-worker-large",
    "ultimate-celery-worker-spreadsheet",
    "ultimate-celery-worker-spreadsheet-large",
    "ultimate-celery-worker-image",
    "ultimate-celery-worker-image-large",
    "ultimate-celery-worker-pdf",
    "ultimate-celery-worker-pdf-2",
    "ultimate-celery-worker-pdf-large",
    "ultimate-celery-worker-word",
    "ultimate-celery-worker-word-2",
    "ultimate-celery-worker-word-large",
    "ultimate-celery-worker-powerpoint",
    "ultimate-celery-worker-powerpoint-2",
    "ultimate-celery-worker-powerpoint-large",
    "ultimate-celery-worker-ocr",
    "ultimate-celery-worker-ocr-large",
]

# Bounds for per-worker memory (GB)
PER_WORKER_MEM_MIN_GB = 1.0
PER_WORKER_MEM_MAX_GB = 8.0
# Per-worker CPU (fraction)
PER_WORKER_CPU_DEFAULT = 0.8
PER_WORKER_CPU_MIN = 0.25


def get_host_memory_gb() -> float:
    """Return total host physical memory in GB."""
    if platform.system() == "Darwin":
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
        )
        return int(out.strip()) / (1024**3)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except (FileNotFoundError, OSError):
        pass
    return 0.0


def get_host_cpu_count() -> int:
    """Return number of logical CPUs."""
    if platform.system() == "Darwin":
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.ncpu"], text=True, timeout=5
        )
        return int(out.strip())
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return f.read().count("processor\t")
    except (FileNotFoundError, OSError):
        pass
    return 0


def scale_resources(
    total_ram_gb: float,
    ncpu: int,
    num_workers: int | None = None,
) -> tuple[float, float, int, float]:
    """
    Compute per-worker memory (GB), per-worker CPU, recommended worker count,
    and worker memory budget (GB). Shares worker budget equally across all
    worker services so total stack fits in host RAM.

    Returns:
        (per_worker_mem_gb, per_worker_cpu, recommended_num_workers, worker_budget_gb)
    """
    worker_budget_gb = max(0.0, total_ram_gb - RESERVED_TOTAL_GB)
    cpu_for_workers = max(0.0, ncpu - RESERVED_CPUS)
    n = num_workers or len(WORKER_SERVICES)

    # Per-worker memory: equal share of budget, clamped so each worker has enough headroom
    per_worker_mem_gb = worker_budget_gb / n if n else 0.0
    per_worker_mem_gb = max(
        PER_WORKER_MEM_MIN_GB,
        min(PER_WORKER_MEM_MAX_GB, per_worker_mem_gb),
    )
    # How many workers can we run at per_worker_mem_min without exceeding budget?
    max_by_mem = int(worker_budget_gb / PER_WORKER_MEM_MIN_GB) if worker_budget_    gb >= PER_WORKER_MEM_MIN_GB else 0
    max_by_cpu = int(cpu_for_workers / PER_WORKER_CPU_MIN) if cpu_for_workers >= PER_WORKER_CPU_MIN else 0
    recommended_n = min(n, max(1, max_by_mem), max(1, max_by_cpu))

    per_worker_cpu = min(
        PER_WORKER_CPU_DEFAULT,
        max(PER_WORKER_CPU_MIN, cpu_for_workers / n),
    )
    per_worker_cpu = round(per_worker_cpu, 2)

    return per_worker_mem_gb, per_worker_cpu, recommended_n, worker_budget_gb


def format_mem_gb(gb: float) -> str:
    """Format memory for compose (e.g. 1.5 -> '1536m', 2 -> '2g')."""
    if gb >= 1.0 and gb == int(gb):
        return f"{int(gb)}g"
    mb = int(gb * 1024)
    return f"{mb}m"


def generate_override(
    per_worker_mem_gb: float,
    per_worker_cpu: float,
) -> str:
    """Generate docker-compose override YAML for worker resources. OCR workers get extra memory (Tesseract) to avoid OOM (exit 137)."""
    res_mem = int(per_worker_mem_gb * 0.25 * 1024)
    res_mem_str = f"{max(256, res_mem)}m"
    # OCR workers need more memory for Tesseract + models to avoid OOM
    OCR_MEM_GB = min(2.0, max(per_worker_mem_gb * 1.5, 1.5))

    lines = [
        "# Generated by scripts/scale_by_resources.py — do not edit by hand",
        "# Adjusts worker memory and CPU to fit host resources.",
        "",
        "services:",
    ]
    for svc in WORKER_SERVICES:
        mem_gb = OCR_MEM_GB if "ocr" in svc else per_worker_mem_gb
        mem_str = format_mem_gb(mem_gb)
        res_for_svc = int(mem_gb * 0.25 * 1024)
        res_for_svc_str = f"{max(256, res_for_svc)}m"
        lines.append(f"  {svc}:")
        lines.append(f"    mem_limit: {mem_str}")
        lines.append(f"    mem_reservation: {res_for_svc_str}")
        lines.append("    deploy:")
        lines.append("      resources:")
        lines.append("        limits:")
        lines.append(f"          memory: {mem_str.upper()}")
        lines.append(f"          cpus: \"{per_worker_cpu}\"")
        lines.append("        reservations:")
        lines.append(f"          memory: {res_for_svc_str.upper()}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate resource-aware docker-compose override from host CPU/RAM."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed values and override content only.",
    )
    parser.add_argument(
        "--output",
        default="docker-compose.scale.override.yml",
        help="Output override file path (default: docker-compose.scale.override.yml).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of worker slots (default: use all worker services).",
    )
    args = parser.parse_args()

    total_ram_gb = get_host_memory_gb()
    ncpu = get_host_cpu_count()

    if total_ram_gb <= 0 or ncpu <= 0:
        print("Could not detect host memory or CPU.", file=sys.stderr)
        return 1

    per_worker_mem_gb, per_worker_cpu, recommended_n, worker_budget_gb = (
        scale_resources(total_ram_gb, ncpu, num_workers=args.workers)
    )

    total_worker_mem = per_worker_mem_gb * len(WORKER_SERVICES)
    total_stack_mem = RESERVED_TOTAL_GB + total_worker_mem

    print("Host resources:")
    print(f"  Total RAM: {total_ram_gb:.2f} GB")
    print(f"  CPU cores: {ncpu}")
    print("Reserved (API + Milvus + Redis + OS):")
    print(f"  {RESERVED_TOTAL_GB} GB, {RESERVED_CPUS} CPUs")
    print("Worker budget and scaling:")
    print(f"  Worker budget: {worker_budget_gb:.2f} GB")
    print(f"  Recommended worker slots (by memory/CPU): {recommended_n}")
    print(f"  Per-worker memory: {per_worker_mem_gb:.2f} GB ({format_mem_gb(per_worker_mem_gb)})")
    print(f"  Per-worker CPU: {per_worker_cpu}")
    print(f"  Total worker memory if all {len(WORKER_SERVICES)} run: {total_worker_mem:.2f} GB")
    print(f"  Total stack (reserved + workers): {total_stack_mem:.2f} GB")

    if total_stack_mem > total_ram_gb:
        print(
            "\nNote: Total stack memory exceeds host RAM. Consider running fewer workers "
            "(e.g. disable optional replicas or large-only workers) or add more RAM.",
            file=sys.stderr,
        )

    override_content = generate_override(per_worker_mem_gb, per_worker_cpu)

    if args.dry_run:
        print("\n--- Generated override (dry-run) ---\n")
        print(override_content)
        return 0

    out_path = Path(args.output)
    if not out_path.is_absolute():
        ultimate_dir = Path(__file__).resolve().parent.parent
        if (ultimate_dir / "docker-compose.ultimate.yml").exists():
            out_path = ultimate_dir / Path(args.output).name
        else:
            out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(override_content, encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print("Use: docker compose -f docker-compose.ultimate.yml -f docker-compose.scale.override.yml up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
