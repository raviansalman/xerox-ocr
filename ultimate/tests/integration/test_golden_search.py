"""Golden search baseline (search forensics milestone).

Ingests tests/golden/cases.py's corpus through the real Celery task, runs every case in every search
mode through POST /search, grades it, and writes a report.

Two kinds of assertions:
  * Hard invariant: no case, in any mode, may return another tenant's document.
  * Baseline: the pass/fail map must equal the recorded baseline for the active embedder
    (tests/golden/baseline_<GOLDEN_EMBEDDER>.json). A case that starts passing fails the test too,
    exactly like a strict xfail, so every search behaviour change is deliberate and re-recorded.

Embedder: the stand-in (default, GOLDEN_EMBEDDER=standin) or the real all-mpnet-base-v2 service
(export EMBEDDER_URL / EMBEDDER_SEARCH_URL and GOLDEN_EMBEDDER=mpnet). Re-record a baseline with
GOLDEN_RECORD=1. The full report goes to GOLDEN_REPORT (default .pytest_cache/golden_<embedder>.json).
"""
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.golden import cases as G
from tests.support.auth import headers

pytestmark = pytest.mark.integration

EMBEDDER = os.getenv("GOLDEN_EMBEDDER", "standin")
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
BASELINE = GOLDEN_DIR / f"baseline_{EMBEDDER}.json"
REPORT = Path(os.getenv("GOLDEN_REPORT", Path(__file__).resolve().parents[2] / ".pytest_cache" / f"golden_{EMBEDDER}.json"))


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    import ultimate_ui
    from src.ultimate_tasks import get_worker_vector_integration, process_ultimate_document_task as task

    d = tmp_path_factory.mktemp("golden")
    for doc in G.CORPUS:
        p = doc.build(d / doc.filename)
        r = task.apply(kwargs=dict(file_path=str(p), file_id=doc.file_id, user_id=doc.tenant,
                                   original_filename=doc.filename, file_type=doc.file_type,
                                   bucket_id=doc.bucket_id, path=doc.path))
        assert r.state == "SUCCESS" and not (r.result or {}).get("skipped"), (doc.file_id, r.result)
    vi = get_worker_vector_integration()
    vi.vector_db.collection.flush()
    vi.vector_db._ensure_collection_loaded()
    time.sleep(1)

    client = TestClient(ultimate_ui.app)
    results = {}
    for case in G.CASES:
        for mode in G.MODES:
            body = {"userId": case.tenant, "query": case.query, "limit": 10, "searchMethod": mode, **case.extra}
            t0 = time.time()
            r = client.post("/search", json=body, headers=headers("service"))
            took = round(time.time() - t0, 3)
            if r.status_code != 200:
                results[f"{case.id}|{mode}"] = {"status": r.status_code, "ids": [], "fails": [f"HTTP {r.status_code}"],
                                                "seconds": took, "top": []}
                continue
            res = r.json()["results"]
            ids = [x["file_id"] for x in res]
            results[f"{case.id}|{mode}"] = {
                "status": 200, "ids": ids, "fails": G.grade(case, ids), "seconds": took,
                "top": [{"file_id": x["file_id"], "score": round(float(x.get("similarity_score") or 0), 4),
                         "search_method": x.get("search_method")} for x in res[:5]],
            }
    yield results
    # Delete only this module's tenants: other modules in the session reuse the cached collection handle.
    tenants = sorted({doc.tenant for doc in G.CORPUS})
    coll = vi.vector_db.collection
    pks = [r["id"] for r in coll.query(expr=f"user_id in {json.dumps(tenants)}", output_fields=["id"], limit=16384)]
    for i in range(0, len(pks), 500):  # Milvus 2.3 deletes by primary key only
        coll.delete(expr=f"id in {json.dumps(pks[i:i + 500])}")
    coll.flush()


def _write_report(results):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    by_case = {}
    for key, v in results.items():
        cid, mode = key.split("|")
        by_case.setdefault(cid, {})[mode] = v
    meta = {c.id: {"category": c.category, "query": c.query, "tenant": c.tenant, "note": c.note} for c in G.CASES}
    REPORT.write_text(json.dumps({"embedder": EMBEDDER, "cases": meta, "results": by_case},
                                 ensure_ascii=False, indent=1), encoding="utf-8")


def test_no_case_returns_another_tenants_documents(golden):
    _write_report(golden)
    leaks = {k: v["fails"] for k, v in golden.items() if any("FOREIGN-TENANT" in f for f in v["fails"])}
    assert leaks == {}
    errors = {k: v["status"] for k, v in golden.items() if v["status"] not in (200, 400)}
    assert errors == {}


def test_golden_pass_fail_map_matches_baseline(golden):
    current = {k: not v["fails"] for k, v in sorted(golden.items())}
    if os.getenv("GOLDEN_RECORD") == "1" or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"baseline recorded to {BASELINE.name}")
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))
    changed = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v}
    assert changed == {}, "golden pass/fail changed (recorded, now); re-record deliberately with GOLDEN_RECORD=1"
