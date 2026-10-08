"""Command line: ``docintel migrate | rollback | serve | ingest | reprocess | query | eval | keygen``.

Workers run with Celery: ``celery -A docintel.worker worker -Q docintel.ingest``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from docintel.config import get_settings


def _ctx(tenant: str):
    from docintel.security import ROLE_IMPLIES, AuthContext, is_valid_tenant_id
    if not is_valid_tenant_id(tenant):
        sys.exit("invalid tenant id")
    return AuthContext(principal="cli", tenant_id=tenant, roles=ROLE_IMPLIES["admin"])


def cmd_migrate(a) -> int:
    from docintel.storage.db import get_db
    db = get_db()
    applied = db.migrate(a.to)
    out: dict = {"applied": applied}
    if warning := db.collation_check():
        out["warning"] = warning
    print(json.dumps(out))
    return 0


def cmd_rollback(a) -> int:
    from docintel.storage.db import get_db
    print(json.dumps({"reverted": get_db().rollback(a.to)}))
    return 0


def _thread_mode_claim():
    """In thread mode this process will run jobs itself: make sure no other process does (see claim_thread_mode)."""
    from docintel.ingest import dispatch
    return dispatch.claim_thread_mode() if get_settings().task_mode == "thread" else None


def cmd_reprocess(a) -> int:
    """Queue documents for processing again: all of a tenant's, or only those whose stored version is stale."""
    try:
        claim = _thread_mode_claim()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        return _cmd_reprocess(a)
    finally:
        if claim is not None:
            claim.close()


def _cmd_reprocess(a) -> int:
    from docintel import settings_store
    from docintel.ingest import dispatch, pipeline
    from docintel.packs import get_domain
    from docintel.storage.db import get_db

    ctx = _ctx(a.tenant)
    with get_db().tenant(ctx.tenant_id) as conn:
        current = pipeline.components(get_domain(tuple(settings_store.enabled_packs(conn))))
        rows = conn.execute("SELECT id, components FROM documents WHERE status IN ('indexed', 'failed') ORDER BY created_at").fetchall()
    chosen = [str(r["id"]) for r in rows if not a.stale or pipeline.stale_reasons(r["components"], current)]
    for doc_id in chosen:
        dispatch.submit(ctx.tenant_id, doc_id)
    print(json.dumps({"documents": len(rows), "queued": len(chosen)}))
    if a.wait and get_settings().task_mode != "celery":
        dispatch.wait_idle(timeout=a.timeout)
    return 0


def cmd_eval_run(a) -> int:
    from docintel.evaluation import HttpTarget, evaluate, load_dataset
    from docintel.evaluation.runner import render_summary
    keys = json.loads(Path(a.keys).read_text(encoding="utf-8"))
    report = evaluate(load_dataset(a.dataset), HttpTarget(a.url, keys), isolation=not a.no_isolation)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(render_summary(report))
    return 0 if report["passed"] else 1


def cmd_eval_compare(a) -> int:
    from docintel.evaluation import compare
    result = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.baseline, a.candidate)))
    print(json.dumps(result, indent=2))
    return 1 if any(m["regression"] for m in result["metrics"].values()) else 0


def cmd_ingest(a) -> int:
    """Register every supported file under a folder and process them (thread or Celery workers)."""
    try:
        claim = _thread_mode_claim()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        return _cmd_ingest(a)
    finally:
        if claim is not None:
            claim.close()


def _cmd_ingest(a) -> int:
    from docintel.ingest import dispatch, pipeline
    from docintel.processing.parsers import ParseError
    from docintel.storage.db import get_db

    get_db().migrate()
    ctx = _ctx(a.tenant)
    files = [p for p in Path(a.folder).rglob("*") if p.is_file() and not p.name.startswith(".")]
    t0, new, dup, rejected = time.time(), 0, 0, 0
    for p in files:
        try:
            with open(p, "rb") as f:
                reg = pipeline.register(ctx, p.name, f, collection=a.collection)
        except (ParseError, ValueError) as e:
            rejected += 1
            print(f"rejected {p}: {e}", file=sys.stderr)
            continue
        if reg.duplicate:
            dup += 1
        else:
            new += 1
            dispatch.submit(ctx.tenant_id, str(reg.document["id"]))
    print(json.dumps({"registered": new, "duplicates": dup, "rejected": rejected, "seconds": round(time.time() - t0, 1)}))
    if a.wait and get_settings().task_mode != "celery":
        dispatch.wait_idle(timeout=a.timeout)
        print(json.dumps({"processed_in_seconds": round(time.time() - t0, 1)}))
    return 0


def cmd_query(a) -> int:
    from docintel.query import get_engine
    out = get_engine().run(_ctx(a.tenant), a.question, a.limit, explain=a.explain)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    from docintel.config import get_settings
    if a.workers > 1 and get_settings().task_mode == "thread":
        # thread mode keeps its job queue in process memory: several processes cannot tell another process's
        # waiting jobs from stalled ones, so each would requeue and process the others' backlog again
        print("DOCINTEL_TASK_MODE=thread runs in one process; use --workers 1, or Celery mode for more API processes",
              file=sys.stderr)
        return 2
    bind = {"host": a.host} if a.host else {}
    uvicorn.run("docintel.api.app:app", port=a.port, workers=a.workers, proxy_headers=True, **bind)
    return 0


def cmd_keygen(_a) -> int:
    from docintel.security import _main
    return _main(["", "generate"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="docintel")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="apply pending database migrations")
    m.add_argument("--to", help="stop after the migration whose name starts with this prefix")
    m.set_defaults(fn=cmd_migrate)
    rb = sub.add_parser("rollback", help="revert migrations newer than --to")
    rb.add_argument("--to", required=True, help="the last migration to keep (a name prefix such as 003, or none)")
    rb.set_defaults(fn=cmd_rollback)
    rp = sub.add_parser("reprocess", help="process a tenant's documents again")
    rp.add_argument("--tenant", required=True)
    rp.add_argument("--stale", action="store_true", help="only documents produced by another pipeline, packs or model")
    rp.add_argument("--wait", action="store_true")
    rp.add_argument("--timeout", type=float, default=86400)
    rp.set_defaults(fn=cmd_reprocess)
    ev = sub.add_parser("eval", help="evaluate retrieval quality against a labelled dataset")
    evs = ev.add_subparsers(dest="eval_cmd", required=True)
    er = evs.add_parser("run")
    er.add_argument("--dataset", required=True, help="YAML dataset (see docintel/evaluation/dataset.py)")
    er.add_argument("--url", required=True, help="base URL of the API")
    er.add_argument("--keys", required=True, help="JSON file mapping each tenant of the dataset to a reader API key")
    er.add_argument("--out", help="write the JSON report here")
    er.add_argument("--no-isolation", action="store_true", help="skip asking every question as every other tenant")
    er.set_defaults(fn=cmd_eval_run)
    ec = evs.add_parser("compare")
    ec.add_argument("baseline")
    ec.add_argument("candidate")
    ec.set_defaults(fn=cmd_eval_compare)
    s = sub.add_parser("serve")
    s.add_argument("--host", help="interface to bind (default: the loopback interface; containers pass their own)")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve)
    i = sub.add_parser("ingest")
    i.add_argument("folder")
    i.add_argument("--tenant", required=True)
    i.add_argument("--collection")
    i.add_argument("--wait", action="store_true")
    i.add_argument("--timeout", type=float, default=86400)
    i.set_defaults(fn=cmd_ingest)
    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("--tenant", required=True)
    q.add_argument("--limit", type=int, default=10)
    q.add_argument("--explain", action="store_true")
    q.set_defaults(fn=cmd_query)
    sub.add_parser("keygen").set_defaults(fn=cmd_keygen)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
