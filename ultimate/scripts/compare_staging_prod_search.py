#!/usr/bin/env python3
"""
POST /search on staging vs production (main-byoc-v1 style: searchMethod=both).

Compare latency, total_results, query_meta (persons, locations, dates), top file_ids.

Parity checklist (search-separation + disk cache):
  • Same code: ultimate_ui.py, query_enhancement.py, semantic_pipeline.py on both hosts
  • Same tuning: ultimate/docker_byoc.env vs docker_byoc_staging.env (search block matches)
  • Search container: ./data volume → data/metadata_cache/ for MetadataIndex disk load
  • Cold first request may build cache; use --repeat 2 on staging to see warm latency

Usage:
  SSL_INSECURE=1 python ultimate/scripts/compare_staging_prod_search.py
  SSL_INSECURE=1 python ultimate/scripts/compare_staging_prod_search.py --staging-only --repeat 2
  python ultimate/scripts/compare_staging_prod_search.py --category nda_temporal

Env overrides:
  STAGING_URL  PROD_URL  STAGING_USER_ID  PROD_USER_ID  LIMIT  TIMEOUT_SEC  SSL_INSECURE=1
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

STAGING_URL = os.environ.get(
    "STAGING_URL", "https://api-vector-byoc-stage.storagechain.io/search"
)
PROD_URL = os.environ.get(
    "PROD_URL", "https://api-vector-byoc.storagechain.io/search"
)
STAGING_USER = os.environ.get("STAGING_USER_ID", "")
PROD_USER = os.environ.get("PROD_USER_ID", "")

# (category, query) — main-byoc style: NDA, agreements, entities, locations, dates
QUERY_SUITE: List[Tuple[str, str]] = [
    ("people_agreements", "David subar agreements"),
    ("people_agreements", "agreements from david subar"),
    ("people_agreements", "agreements with David Subar"),
    ("people_agreements", "documents signed by Jane Doe"),
    ("location", "agreements signed in Austin TX"),
    ("location", "agreements signed in Austin"),
    ("location", "agreements austin texas"),
    ("location", "contracts signed in Texas"),
    ("location", "California employment agreement"),
    ("nda_location", "NDA austin Texas"),
    ("nda_location", "show me NDA from Austin Texas"),
    ("nda", "NDA"),
    ("nda", "mutual non-disclosure agreement"),
    ("nda", "confidentiality agreement 2023"),
    ("agreements", "master services agreement"),
    ("agreements", "subscription agreement"),
    ("agreements", "lease agreement"),
    ("temporal_date", "January 30, 2019"),
    ("temporal_date", "March 5, 2025"),
    ("temporal_year", "documents from 2019"),
    ("temporal_year", "contracts from 2020"),
    ("temporal_range", "documents between 2018 and 2020"),
    ("temporal_range", "agreements signed in Q1 2024"),
    ("entity_filetype", "PDF contracts"),
    ("entity_filetype", "PowerPoint presentations about revenue"),
    ("governing_law", "governing law New York"),
    ("governing_law", "Delaware law"),
]


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def _post_json(
    url: str, body: Dict[str, Any], timeout: float, *, ssl_insecure: bool = False
) -> Tuple[int, float, Dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    ctx = _ssl_context(ssl_insecure)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - t0
            return resp.status, elapsed, json.loads(raw)
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        try:
            raw = e.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw.strip().startswith("{") else {"detail": raw[:500]}
        except Exception:
            payload = {"error": str(e)}
        return e.code, elapsed, payload
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e) and not ssl_insecure:
            return _post_json(url, body, timeout, ssl_insecure=True)
        raise


def _summarize_meta(qm: Any) -> Dict[str, Any]:
    if not isinstance(qm, dict):
        return {}
    keys = (
        "persons",
        "location",
        "locations",
        "organizations",
        "date",
        "date_range",
        "month_year",
        "years",
        "normalized_query",
        "vector_query",
    )
    return {k: qm.get(k) for k in keys if k in qm and qm.get(k) not in (None, [], {})}


def _top_ids(results: Any, n: int) -> List[str]:
    out: List[str] = []
    if not isinstance(results, list):
        return out
    for r in results[:n]:
        if not isinstance(r, dict):
            continue
        fid = r.get("file_id") or r.get("source_file") or ""
        sm = r.get("search_method") or ""
        sc = r.get("similarity_score", r.get("score"))
        out.append(f"{fid[:72]!s} | {sm!s} | {sc}")
    return out


def _fid_only(lines: List[str]) -> List[str]:
    return [x.split(" | ")[0].strip() for x in lines if x]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare staging vs prod /search (BYOC parity).")
    ap.add_argument(
        "--staging-only",
        action="store_true",
        help="Only hit staging (faster; use with --repeat for warm-cache timing).",
    )
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each query N times (staging URL only).")
    ap.add_argument("--category", type=str, default="", help="Filter by category prefix, e.g. nda, location.")
    args = ap.parse_args()

    limit = int(os.environ.get("LIMIT", "15"))
    timeout_sec = float(os.environ.get("TIMEOUT_SEC", "120"))
    ssl_flag = os.environ.get("SSL_INSECURE", "").strip() in ("1", "true", "yes")

    if not STAGING_USER or (not args.staging_only and not PROD_USER):
        print("Set STAGING_USER_ID (and PROD_USER_ID unless --staging-only) in the environment.", file=sys.stderr)
        return 2

    suite = [(c, q) for c, q in QUERY_SUITE if not args.category or c.startswith(args.category)]

    print("Staging URL:", STAGING_URL)
    if not args.staging_only:
        print("Production URL:", PROD_URL)
    print("SSL_INSECURE (env):", ssl_flag)
    print("Staging userId:", STAGING_USER)
    if not args.staging_only:
        print("Production userId:", PROD_USER)
    print("limit=", limit, "timeout_sec=", timeout_sec, "repeat=", args.repeat)
    print("queries:", len(suite))
    print("-" * 88)

    mismatches = 0
    staging_times: List[float] = []

    for cat, q in suite:
        body_s = {
            "userId": STAGING_USER,
            "query": q,
            "searchMethod": "both",
            "limit": limit,
        }
        times_run: List[float] = []
        js: Dict[str, Any] = {}
        sc = 0
        for _ in range(max(1, args.repeat)):
            sc, ts, js = _post_json(STAGING_URL, body_s, timeout_sec, ssl_insecure=ssl_flag)
            times_run.append(ts)
            staging_times.append(ts)
        ts = times_run[-1]
        if args.repeat > 1:
            ts_stats = f"runs={times_run} mean={statistics.mean(times_run):.2f}s min={min(times_run):.2f}s max={max(times_run):.2f}s"
        else:
            ts_stats = f"{ts:.2f}s"

        meta_s = _summarize_meta(js.get("query_meta") if isinstance(js, dict) else None)
        ids_s = _top_ids(js.get("results") if isinstance(js, dict) else [], 5)
        ns = js.get("total_results") if isinstance(js, dict) else None
        sto = js.get("semantic_timed_out") if isinstance(js, dict) else None

        if args.staging_only:
            print(f"\n[{cat}] {q!r}")
            print(f"  staging: HTTP {sc}  {ts_stats}  total={ns}  semantic_to={sto}")
            if meta_s:
                print(f"  query_meta: {json.dumps(meta_s, default=str)}")
            print("  top5:", ids_s or "(none)")
            continue

        body_p = {
            "userId": PROD_USER,
            "query": q,
            "searchMethod": "both",
            "limit": limit,
        }
        pc, tp, jp = _post_json(PROD_URL, body_p, timeout_sec, ssl_insecure=ssl_flag)

        np_ = jp.get("total_results") if isinstance(jp, dict) else None
        meta_p = _summarize_meta(jp.get("query_meta") if isinstance(jp, dict) else None)

        same_meta = meta_s == meta_p
        ids_p = _top_ids(jp.get("results") if isinstance(jp, dict) else [], 5)
        same_top = _fid_only(ids_s) == _fid_only(ids_p)

        if not same_top or not same_meta:
            mismatches += 1

        flag = "OK" if (same_top and same_meta and sc == 200 and pc == 200) else "DIFF"
        print(f"\n[{flag}] [{cat}] {q!r}")
        print(f"  staging: HTTP {sc}  {ts:.2f}s  total={ns}  semantic_to={sto}")
        print(f"  prod:    HTTP {pc}  {tp:.2f}s  total={np_}  semantic_to={jp.get('semantic_timed_out') if isinstance(jp, dict) else 'n/a'}")
        if meta_s or meta_p:
            print(f"  query_meta staging: {json.dumps(meta_s, default=str)}")
            print(f"  query_meta prod:    {json.dumps(meta_p, default=str)}")
        print("  top5 staging:", ids_s or "(none)")
        print("  top5 prod:   ", ids_p or "(none)")

    print("\n" + "=" * 88)
    if staging_times:
        print(
            f"Staging latency: mean={statistics.mean(staging_times):.2f}s "
            f"min={min(staging_times):.2f}s max={max(staging_times):.2f}s (n={len(staging_times)})"
        )
    if not args.staging_only:
        print(
            f"DIFF rows ≈ meta or top-5 mismatch: {mismatches}/{len(suite)} "
            "(expected when prod lacks latest query_enhancement or corpora differ)."
        )
    print(
        "Disk cache: second request often faster once data/metadata_cache exists for user_id "
        "(METADATA_CACHE_DIR); use --staging-only --repeat 2."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
