#!/usr/bin/env python3
"""Large-tenant probe for the CURRENT search code (measurement only, nothing is changed).

For each tenant size it ingests synthetic business text plus planted "needle" chunks that each contain
one unique exact phrase, then measures through POST /search:
  * exact-phrase recall@10 for the needles (both / vector / semantic modes)
  * search latency p50 / p95 per mode
  * the full-scan primitive the legacy exact supplements depend on (query_all_chunks, cap 16384):
    rows returned and time

Runs against Milvus + Redis on localhost and an embedder at EMBEDDER_URL. Without EMBEDDER_URL it starts the
deterministic stand-in embedder (tests/support/fake_embedder.py). The stand-in is lexical by construction, so
its exact-phrase recall is an optimistic upper bound for MPNet; latency and cap behaviour do not depend on it.

    cd ultimate && python scripts/forensics/scale_probe.py --sizes 1000 5000 20000 --out probe.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ULTIMATE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ULTIMATE))

CHUNKS_PER_FILE = 10
NEEDLES = 20
WORDS = ("printer toner fleet invoice contract agreement uptime duplex network firmware maintenance service "
         "renewal payment vendor department warehouse shipment approval budget quarter region policy device "
         "colour scanner fuser drum tray paper report audit schedule customer support ticket escalation lease "
         "procurement delivery warranty compliance security access badge office floor building").split()


def _env():
    os.environ.setdefault("QUEUE_SHARD_COUNT", "0")
    os.environ.setdefault("WORKFLOW_ENABLED", "false")
    os.environ.setdefault("ENABLE_QUERY_NER", "0")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
    for a, b in (("DOC_COLLECTION", "MILVUS_DOC_COLLECTION"), ("IMG_COLLECTION", "MILVUS_IMG_COLLECTION")):
        v = os.environ.setdefault(a, f"xocr_scale_probe_{a[:3].lower()}")
        os.environ.setdefault(b, v)
    if not os.getenv("EMBEDDER_URL"):
        import uvicorn

        from tests.support.fake_embedder import app

        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=18081, log_level="warning"))
        threading.Thread(target=srv.run, daemon=True).start()
        time.sleep(1.5)
        os.environ["EMBEDDER_URL"] = os.environ["EMBEDDER_SEARCH_URL"] = "http://127.0.0.1:18081/embed"
        return "standin"
    os.environ.setdefault("EMBEDDER_SEARCH_URL", os.environ["EMBEDDER_URL"])
    return "external"


def _paragraph(rng, n_words=170):
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def _ingest(vi, tenant, n_chunks, rng):
    files = max(1, n_chunks // CHUNKS_PER_FILE)
    needle_files = set(rng.sample(range(files), min(NEEDLES, files)))
    needles = {}
    for f in range(files):
        paras = [_paragraph(rng) for _ in range(CHUNKS_PER_FILE)]
        if f in needle_files:
            k = len(needles)
            phrase = f"reference code {k:02d} {rng.choice(['amber', 'cobalt', 'violet', 'saffron'])} kestrel ledger"
            i = rng.randrange(CHUNKS_PER_FILE)
            paras[i] = paras[i][:500] + f" {phrase}. " + paras[i][500:]
            needles[phrase] = f"{tenant}_f{f:05d}"
        vi.upsert_document(file_id=f"{tenant}_f{f:05d}", text_content="\n\n".join(paras),
                           metadata={"filename": f"{tenant}_f{f:05d}.txt"}, user_id=tenant)
    vi.vector_db.collection.flush()
    return needles


def _pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    ap.add_argument("--out", default="scale_probe.json")
    args = ap.parse_args()
    embedder = _env()

    from fastapi.testclient import TestClient

    from tests.support.auth import api_keys_json, headers
    os.environ["API_KEYS"] = api_keys_json()
    import ultimate_ui
    from src.ultimate_tasks import get_worker_vector_integration

    vi = get_worker_vector_integration()
    client = TestClient(ultimate_ui.app)
    rng = random.Random(1234)
    report = {"embedder": embedder, "chunks_per_file": CHUNKS_PER_FILE, "sizes": {}}
    for n in args.sizes:
        tenant = f"scale_{n}"
        t0 = time.time()
        needles = _ingest(vi, tenant, n, rng)
        vi.vector_db._ensure_collection_loaded()
        ingest_s = round(time.time() - t0, 1)
        t0 = time.time()
        rows = vi.vector_db.query_all_chunks(user_id=tenant, limit=16384, include_text=True)
        scan = {"rows": len(rows), "seconds": round(time.time() - t0, 2)}
        res = {}
        for mode in ("both", "vector", "semantic"):
            hits, lat = 0, []
            for phrase, fid in needles.items():
                t0 = time.time()
                r = client.post("/search", headers=headers("service"),
                                json={"userId": tenant, "query": phrase, "limit": 10, "searchMethod": mode})
                lat.append(time.time() - t0)
                ids = [x["file_id"] for x in r.json().get("results", [])] if r.status_code == 200 else []
                hits += fid in ids
            res[mode] = {"needle_recall_at_10": f"{hits}/{len(needles)}", "p50_s": _pct(lat, 50), "p95_s": _pct(lat, 95)}
        report["sizes"][n] = {"chunks_ingested": n, "ingest_seconds": ingest_s, "full_scan": scan, "search": res}
        print(json.dumps({n: report["sizes"][n]}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
