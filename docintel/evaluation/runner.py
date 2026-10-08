"""Run a dataset against a target and compute the report."""
from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean
from typing import Any, Protocol

from docintel.evaluation import metrics as M
from docintel.evaluation.dataset import LOWER_IS_BETTER, Dataset, Question


class Target(Protocol):
    def query(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    def documents(self, tenant: str) -> dict[str, str]: ...          # document id -> file name
    def semantic_enabled(self) -> bool: ...


class HttpTarget:
    """A running API. ``keys`` maps each tenant of the dataset to an API key with the reader role."""

    def __init__(self, base_url: str, keys: dict[str, str], timeout: float = 30.0):
        import httpx
        self.keys = keys
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _h(self, tenant: str) -> dict[str, str]:
        if tenant not in self.keys:
            raise KeyError(f"no API key for tenant {tenant!r}")
        return {"X-API-Key": self.keys[tenant]}

    def query(self, tenant: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post("/api/v1/query", headers=self._h(tenant), json=payload)
        r.raise_for_status()
        return r.json()

    def documents(self, tenant: str) -> dict[str, str]:
        out, offset = {}, 0
        while True:
            r = self.http.get("/api/v1/documents", headers=self._h(tenant), params={"limit": 500, "offset": offset})
            r.raise_for_status()
            page = r.json()
            out.update({d["id"]: d["filename"] for d in page["documents"]})
            offset += len(page["documents"])
            if not page["documents"] or offset >= page["total"]:
                return out

    def semantic_enabled(self) -> bool:
        checks = self.http.get("/health/ready").json().get("checks", {})
        return (checks.get("vector_index") or {}).get("status") != "disabled"


def _score(q: Question, out: dict[str, Any], names: dict[str, str], own: set[str], k: int) -> dict[str, Any]:
    results = out.get("results") or []
    ranked = [names.get(r["document_id"], r["document_id"]) for r in results]
    leaked = [r["document_id"] for r in results if r["document_id"] not in own]
    row: dict[str, Any] = {"id": q.id, "category": q.category, "question": q.question, "tenant": q.tenant,
                           "results": ranked[:k], "leaked": len(leaked), "engine": out.get("engine"),
                           "ms": (out.get("timings_ms") or {}).get("total")}
    if q.unanswerable:
        row["false_positive"] = bool(results) and (out.get("answer") or {}).get("kind") != "none"
        row["passed"] = not row["false_positive"] and not leaked
        return row
    if q.answer is not None:
        row["answer"] = {k: (out.get("answer") or {}).get(k) for k in q.answer}
        row["answer_correct"] = M.answer_matches(q.answer, out.get("answer") or {})
    if q.relevant:
        row.update(precision=M.precision_at(ranked, q.relevant, 5), recall=M.recall_at(ranked, q.relevant, k),
                   mrr=M.reciprocal_rank(ranked, q.relevant), ndcg=M.ndcg_at(ranked, q.relevant, k))
        row["missing"] = sorted(set(q.relevant) - set(ranked[:k]))
    row["passed"] = (not leaked and row.get("answer_correct", True) and
                     (not q.relevant or (not row["missing"] if q.exact else row["mrr"] > 0)))
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [r for r in rows if "mrr" in r]
    precision = [r["precision"] for r in ranked if r["precision"] is not None]
    exact = [r for r in rows if r.get("exact")]
    computed = [r for r in rows if "answer_correct" in r]
    negative = [r for r in rows if "false_positive" in r]
    ms = [r["ms"] for r in rows if r.get("ms") is not None]
    avg = lambda xs: round(mean(xs), 4) if xs else None  # noqa: E731
    return {
        "questions": len(rows), "passed": sum(r["passed"] for r in rows),
        "precision_at_5": avg(precision), "recall_at_k": avg([r["recall"] for r in ranked]),
        "mrr": avg([r["mrr"] for r in ranked]), "ndcg_at_k": avg([r["ndcg"] for r in ranked]),
        "exact_recall": avg([r["recall"] for r in exact]),
        "computed_accuracy": avg([1.0 if r["answer_correct"] else 0.0 for r in computed]),
        "abstention_fp_rate": avg([1.0 if r["false_positive"] else 0.0 for r in negative]),
        "leakage": sum(r["leaked"] for r in rows), "p50_ms": M.percentile(ms, 50), "p95_ms": M.percentile(ms, 95),
    }


def evaluate(dataset: Dataset, target: Target, isolation: bool = True) -> dict[str, Any]:
    """Ask every question (and, with ``isolation``, every question as every tenant) and build the report."""
    started = time.time()
    names: dict[str, str] = {}
    own: dict[str, set[str]] = {}
    for tenant in dataset.tenants:
        docs = target.documents(tenant)
        names.update(docs)
        own[tenant] = set(docs)
    semantic = target.semantic_enabled()
    rows, skipped = [], []
    for q in dataset.questions:
        if "semantic" in q.requires and not semantic:
            skipped.append(q.id)
            continue
        out = target.query(q.tenant, {"q": q.question, "limit": q.limit})
        rows.append({**_score(q, out, names, own[q.tenant], dataset.k), "exact": q.exact})
    sweep = 0
    if isolation:                      # every question asked by every other tenant: nothing foreign may come back
        for q in dataset.questions:
            for tenant in dataset.tenants:
                if tenant == q.tenant:
                    continue
                out = target.query(tenant, {"q": q.question, "limit": q.limit})
                foreign = [r["document_id"] for r in out.get("results") or [] if r["document_id"] not in own[tenant]]
                sweep += len(foreign)
    by_category: dict[str, list] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    summary = _summary(rows)
    summary["leakage"] += sweep
    report = {"dataset": dataset.name, "source": dataset.source, "k": dataset.k, "semantic_enabled": semantic,
              "engine": next((r["engine"] for r in rows if r.get("engine")), None), "started": started,
              "seconds": round(time.time() - started, 1), "summary": summary,
              "by_category": {c: _summary(rs) for c, rs in sorted(by_category.items())},
              "isolation_queries": sum(len(dataset.tenants) - 1 for _ in dataset.questions) if isolation else 0,
              "skipped": skipped, "questions": rows}
    report["gates"] = check_gates(summary, dataset.gates)
    report["passed"] = all(g["passed"] for g in report["gates"].values())
    if dataset.source == "synthetic":
        report["note"] = "Synthetic dataset: these numbers measure regressions, not quality on real documents."
    return report


def check_gates(summary: dict[str, Any], gates: dict[str, float]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, threshold in gates.items():
        value = summary.get(name)
        lower = name in LOWER_IS_BETTER
        passed = value is not None and (value <= threshold if lower else value >= threshold)
        if value is None and name in ("abstention_fp_rate", "computed_accuracy", "exact_recall"):
            passed = True              # nothing of that kind in the dataset
        out[name] = {"value": value, "threshold": threshold, "rule": "max" if lower else "min", "passed": passed}
    return out


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Metric deltas and per-question changes between two reports of the same dataset."""
    deltas = {}
    for name, base in baseline["summary"].items():
        cand = candidate["summary"].get(name)
        if isinstance(base, int | float) and isinstance(cand, int | float):
            delta = round(cand - base, 4)
            worse = delta > 0 if name in LOWER_IS_BETTER else delta < 0
            deltas[name] = {"baseline": base, "candidate": cand, "delta": delta,
                            "regression": worse and name not in ("p50_ms", "p95_ms", "questions", "passed")}
    base_q = {q["id"]: q for q in baseline["questions"]}
    regressions, improvements = [], []
    for q in candidate["questions"]:
        b = base_q.get(q["id"])
        if not b:
            continue
        if (b["passed"] and not q["passed"]) or q.get("mrr", 1) < b.get("mrr", 1):
            regressions.append({"id": q["id"], "question": q["question"], "baseline": b["results"][:3],
                                "candidate": q["results"][:3]})
        elif (q["passed"] and not b["passed"]) or q.get("mrr", 0) > b.get("mrr", 0):
            improvements.append({"id": q["id"], "question": q["question"], "baseline": b["results"][:3],
                                 "candidate": q["results"][:3]})
    return {"baseline": {"engine": baseline.get("engine"), "dataset": baseline["dataset"]},
            "candidate": {"engine": candidate.get("engine"), "dataset": candidate["dataset"]},
            "metrics": deltas, "regressions": regressions, "improvements": improvements}


def render_summary(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"dataset {report['dataset']} ({report['source']}), engine {report['engine']}, "
             f"semantic {'on' if report['semantic_enabled'] else 'off'}: {s['passed']}/{s['questions']} passed"]
    for name in ("precision_at_5", "recall_at_k", "mrr", "ndcg_at_k", "exact_recall", "computed_accuracy",
                 "abstention_fp_rate", "leakage", "p50_ms", "p95_ms"):
        lines.append(f"  {name:20s} {s.get(name)}")
    lines.append("  per category: " + ", ".join(f"{c} {v['passed']}/{v['questions']}" for c, v in report["by_category"].items()))
    for name, g in report["gates"].items():
        lines.append(f"  gate {name}: {g['value']} ({g['rule']} {g['threshold']}) {'ok' if g['passed'] else 'FAILED'}")
    failed = [q for q in report["questions"] if not q["passed"]]
    for q in failed[:20]:
        lines.append(f"  failed {q['id']} [{q['category']}] {q['question']!r}: got {q['results'][:3]}")
    if report.get("note"):
        lines.append(report["note"])
    return "\n".join(lines)
