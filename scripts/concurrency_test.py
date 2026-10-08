"""Query latency and throughput under concurrency, with the resource use of every component sampled alongside.

Each level runs N closed-loop users (each sends its next question as soon as the previous answer arrives) for a
fixed time over a mix of question kinds, after a short warm-up. Per level it reports p50, p95 and p99 latency,
throughput, errors, and the CPU and memory of the processes and containers named on the command line plus the
database connections in use, so the component that saturates first can be identified.

    python scripts/concurrency_test.py --url http://localhost:8000 --key $KEY --levels 1,8,16,32,64,100 \\
        --duration 60 --proc api="docintel.cli serve" --proc embedder="docintel.embedder" \\
        --proc postgres=postgres --container milvus=di-milvus --pg-dsn postgresql://postgres@localhost/postgres

Questions come from the load test corpus (scripts/load_test.py); run that first to load a tenant.
Linux only (reads /proc). Nothing is uploaded or changed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import QUERIES, pct

TICKS = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def _pids(pattern: str) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmd:
            out.append(int(d))
    return out


def _proc_sample(pids: list[int]) -> tuple[float, int]:
    """(cpu seconds, resident bytes) summed over the processes that still exist."""
    cpu, rss = 0.0, 0
    for p in pids:
        try:
            f = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += (int(f[11]) + int(f[12])) / TICKS
            rss += int(Path(f"/proc/{p}/statm").read_text().split()[1]) * PAGE
        except (OSError, IndexError, ValueError):
            pass
    return cpu, rss


def _host_cpu() -> tuple[float, float]:
    f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = f[3] + f[4]
    return float(sum(f)), float(idle)


class Sampler(threading.Thread):
    def __init__(self, procs: dict[str, str], containers: dict[str, str], pg_dsn: str | None, pg_db: str | None):
        super().__init__(daemon=True)
        self.procs, self.containers, self.pg_dsn, self.pg_db = procs, containers, pg_dsn, pg_db
        self.stop = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:
        pids = {k: _pids(v) for k, v in self.procs.items()}
        prev = {k: _proc_sample(p) for k, p in pids.items()}
        prev_host, prev_t = _host_cpu(), time.monotonic()
        conn = None
        if self.pg_dsn:
            import psycopg
            conn = psycopg.connect(self.pg_dsn, autocommit=True)
        while not self.stop.wait(2.0):
            now = time.monotonic()
            dt = now - prev_t
            s: dict = {"procs": {}, "containers": {}}
            for k, p in pids.items():
                cpu, rss = _proc_sample(p)
                s["procs"][k] = {"cpu_pct": round((cpu - prev[k][0]) / dt * 100, 1), "rss_mb": round(rss / 2**20)}
                prev[k] = (cpu, rss)
            host = _host_cpu()
            total, idle = host[0] - prev_host[0], host[1] - prev_host[1]
            s["host_cpu_pct"] = round((1 - idle / total) * 100, 1) if total else None
            prev_host, prev_t = host, now
            if self.containers:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}",
                                      *self.containers.values()], capture_output=True, text=True, check=False).stdout
                for line in out.splitlines():
                    name, cpu, mem = line.split(" ", 2)
                    label = next((k for k, v in self.containers.items() if v == name), name)
                    s["containers"][label] = {"cpu_pct": float(cpu.rstrip("%")), "mem": mem.split(" / ")[0]}
            if conn is not None:
                rows = conn.execute("SELECT state, count(*) FROM pg_stat_activity WHERE datname = %s GROUP BY state",
                                    (self.pg_db,)).fetchall()
                s["db_connections"] = {str(st): n for st, n in rows}
            self.samples.append(s)
        if conn is not None:
            conn.close()

    def summary(self) -> dict:
        def avg_max(values):
            values = [v for v in values if v is not None]
            return {"avg": round(statistics.mean(values), 1), "max": round(max(values), 1)} if values else None
        out: dict = {"host_cpu_pct": avg_max([s["host_cpu_pct"] for s in self.samples])}
        for k in self.procs:
            out[k] = {"cpu_pct": avg_max([s["procs"][k]["cpu_pct"] for s in self.samples]),
                      "rss_mb_max": max((s["procs"][k]["rss_mb"] for s in self.samples), default=None)}
        for k in self.containers:
            vals = [s["containers"].get(k) for s in self.samples if s["containers"].get(k)]
            out[k] = {"cpu_pct": avg_max([v["cpu_pct"] for v in vals]), "mem_last": vals[-1]["mem"] if vals else None}
        if self.pg_dsn:
            out["db_connections"] = {
                "active": avg_max([s["db_connections"].get("active", 0) for s in self.samples]),
                "total_max": max((sum(s["db_connections"].values()) for s in self.samples), default=None)}
        return out


def run_level(a, users: int, questions: list[tuple[str, str]]) -> dict:
    deadline_warm = time.monotonic() + a.warmup
    stop_at = deadline_warm + a.duration
    lock = threading.Lock()
    lat: dict[str, list[float]] = {}
    errors: dict[str, int] = {}

    def user(seed: int) -> None:
        r = random.Random(seed)
        with httpx.Client(base_url=a.url, headers={"X-API-Key": a.key}, timeout=a.timeout) as c:
            while (now := time.monotonic()) < stop_at:
                kind, q = r.choice(questions)
                t = time.perf_counter()
                try:
                    resp = c.post("/api/v1/query", json={"q": q, "limit": 10})
                    status = str(resp.status_code) if resp.status_code != 200 else None
                except httpx.TimeoutException:
                    status = "timeout"
                except httpx.HTTPError as e:
                    status = type(e).__name__
                ms = (time.perf_counter() - t) * 1000
                if now < deadline_warm:
                    continue
                with lock:
                    if status:
                        errors[status] = errors.get(status, 0) + 1
                    else:
                        lat.setdefault(kind, []).append(ms)

    sampler = Sampler(a.proc, a.container, a.pg_dsn, a.pg_db)
    threads = [threading.Thread(target=user, args=(i,), daemon=True) for i in range(users)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    sampler.start()
    for t in threads:
        t.join()
    sampler.stop.set()
    sampler.join()
    every = [x for v in lat.values() for x in v]
    done = len(every)
    return {
        "users": users,
        "requests": done,
        "errors": errors,
        "error_rate": round(sum(errors.values()) / max(1, done + sum(errors.values())), 4),
        "throughput_rps": round(done / a.duration, 1),
        "latency_ms": {"p50": pct(every, 50), "p95": pct(every, 95), "p99": pct(every, 99)} if every else None,
        "by_kind_p95_ms": {k: pct(v, 95) for k, v in sorted(lat.items())},
        "resources": sampler.summary(),
    }


def _pairs(values: list[str]) -> dict[str, str]:
    out = {}
    for v in values:
        k, _, pattern = v.partition("=")
        out[k] = pattern
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", required=True, help="API key with the reader role for the loaded tenant")
    ap.add_argument("--levels", default="1,8,16,32,64,100")
    ap.add_argument("--duration", type=float, default=60, help="measured seconds per level")
    ap.add_argument("--warmup", type=float, default=10, help="unmeasured seconds at the start of each level")
    ap.add_argument("--timeout", type=float, default=60, help="client timeout per request (counted as an error)")
    ap.add_argument("--kinds", default="", help="comma list of question kinds (default: all)")
    ap.add_argument("--proc", action="append", default=[], help="label=command-line substring of processes to sample")
    ap.add_argument("--container", action="append", default=[], help="label=docker container name to sample")
    ap.add_argument("--pg-dsn", help="connection with access to pg_stat_activity, to count database connections")
    ap.add_argument("--pg-db", help="database whose connections are counted")
    a = ap.parse_args()
    a.proc, a.container = _pairs(a.proc), _pairs(a.container)
    kinds = [k for k in a.kinds.split(",") if k] or list(QUERIES)
    questions = [(k, q) for k in kinds for q in QUERIES[k]]
    report = {"levels": [], "duration_s": a.duration, "kinds": kinds, "cpus": os.cpu_count()}
    for users in [int(x) for x in a.levels.split(",")]:
        result = run_level(a, users, questions)
        report["levels"].append(result)
        print(json.dumps({k: result[k] for k in ("users", "throughput_rps", "latency_ms", "error_rate")}), file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
