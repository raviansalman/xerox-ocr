"""Deterministic computations over extracted data: counts, sums, averages, extremes, groupings and comparisons.

Every computed answer carries its calculation (operation, field, filters, grouping, periods, documents considered
and documents that had a value) and the supporting records (document, value, page and character span), so the
number can be checked against the source. Values are never estimated from text or produced by a language model.
Currencies are never added together.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import psycopg

from docintel.search import retrieval as R
from docintel.search.plan import DateFilter, QueryPlan

MAX_RECORDS = 200


def _period_label(d: DateFilter) -> str:
    if d.start and d.end and d.start.month == 1 and d.start.day == 1 and d.end.month == 12 and d.end.day == 31:
        return str(d.start.year) if d.start.year == d.end.year else f"{d.start.year}-{d.end.year}"
    return f"{d.start or '…'} to {d.end or '…'}"


def document_dates(conn: psycopg.Connection, ids: list[str], field: str) -> dict[str, date]:
    """The date that places each document in time for comparisons (the requested date field, else any date)."""
    names = list(R.DATE_FIELDS) if field == "any" else [field]
    rows = conn.execute("SELECT document_id, min(value_date) AS d FROM fields WHERE document_id = ANY(%s::uuid[]) "
                        "AND name = ANY(%s) AND value_date IS NOT NULL GROUP BY document_id", (ids, names)).fetchall()
    return {str(r["document_id"]): r["d"] for r in rows}


def document_parties(conn: psycopg.Connection, ids: list[str]) -> dict[str, str]:
    rows = conn.execute("SELECT DISTINCT ON (document_id) document_id, value_text FROM fields WHERE document_id = ANY(%s::uuid[]) "
                        "AND name = 'party' ORDER BY document_id, confidence DESC, id", (ids,)).fetchall()
    return {str(r["document_id"]): r["value_text"] for r in rows}


def document_attribute(conn: psycopg.Connection, ids: list[str], key: str) -> dict[str, str]:
    """Group key per document for type, language, status (document columns) or jurisdiction (extracted field)."""
    if key == "jurisdiction":
        rows = conn.execute("SELECT DISTINCT ON (document_id) document_id, value_text AS v FROM fields WHERE "
                            "document_id = ANY(%s::uuid[]) AND name = 'jurisdiction' ORDER BY document_id, confidence DESC, id",
                            (ids,)).fetchall()
    else:
        column = {"type": "doc_type", "language": "language", "status": "status"}[key]
        rows = conn.execute(f"SELECT id AS document_id, {column} AS v FROM documents WHERE id = ANY(%s::uuid[])",
                            (ids,)).fetchall()
    return {str(r["document_id"]): r["v"] or "unknown" for r in rows}


_ATTRIBUTE_GROUPS = ("type", "language", "status", "jurisdiction")


def _group_key(plan: QueryPlan, doc: str, dates: dict[str, date], parties: dict[str, str]) -> str | None:
    if plan.group_by == "party":
        return parties.get(doc, "unknown")
    if plan.group_by in _ATTRIBUTE_GROUPS:
        return parties.get(doc, "unknown")              # filled with document_attribute() by amounts()
    if plan.group_by in ("year", "month"):
        d = dates.get(doc)
        return None if d is None else (str(d.year) if plan.group_by == "year" else f"{d.year}-{d.month:02d}")
    return None


def _periods(plan: QueryPlan, ids: list[str], dates: dict[str, date]) -> dict[str, list[str]] | None:
    if not plan.compare:
        return None
    out: dict[str, list[str]] = {}
    for p in plan.compare:
        out[_period_label(p)] = [d for d in ids if d in dates and (p.start is None or dates[d] >= p.start)
                                 and (p.end is None or dates[d] <= p.end)]
    return out


def calculation(plan: QueryPlan, operation: str, considered: int, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"operation": operation, "filters": _filters_dict(plan), "documents_considered": considered}
    if plan.text.strip() or plan.phrases or plan.identifiers:
        out["text_condition"] = " ".join([*plan.phrases, plan.text]).strip()
    if plan.group_by:
        out["group_by"] = plan.group_by
    if plan.compare:
        out["periods"] = [_period_label(p) for p in plan.compare]
    out.update(extra)
    return out


def _filters_dict(plan: QueryPlan) -> dict[str, Any]:
    d = plan.to_dict().get("filters") or {}
    return {k: v for k, v in d.items() if v not in (None, [], False) and k != "doc_types_hard"}


def count(conn: psycopg.Connection, plan: QueryPlan, ids: list[str]) -> dict[str, Any]:
    """Count answer for a document set, per period when the question compares periods."""
    if plan.compare:
        field = plan.compare[0].field
        dates = document_dates(conn, ids, field)
        periods = _periods(plan, ids, dates) or {}
        rows = [{"period": k, "count": len(v)} for k, v in periods.items()]
        text = "; ".join(f"{r['period']}: {r['count']}" for r in rows)
        return {"kind": "table", "rows": rows, "text": f"Documents per period: {text}.",
                "calculation": calculation(plan, "count", len(ids), date_field=field)}
    return {"kind": "number", "value": len(ids), "unit": "documents",
            "calculation": calculation(plan, "count", len(ids))}


def amounts(conn: psycopg.Connection, plan: QueryPlan, ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Sum, average, minimum or maximum of the documents' amounts, per currency (and group or period)."""
    op = plan.intent
    names = [plan.aggregate_field or "total_amount", "amount"]
    rows = conn.execute(
        "SELECT DISTINCT ON (document_id, unit) document_id, unit, value_num, value_text, name, page, char_start, char_end, "
        "snippet FROM fields WHERE document_id = ANY(%s::uuid[]) AND name = ANY(%s) AND value_num IS NOT NULL "
        "ORDER BY document_id, unit, (name = %s) DESC, value_num DESC", (ids, names, names[0])).fetchall()
    dates = document_dates(conn, ids, plan.compare[0].field if plan.compare else "any") \
        if plan.compare or plan.group_by in ("year", "month") else {}
    parties = (document_parties(conn, ids) if plan.group_by == "party" else
               document_attribute(conn, ids, plan.group_by) if plan.group_by in _ATTRIBUTE_GROUPS else {})
    periods = _periods(plan, ids, dates)
    period_of = {d: label for label, docs in (periods or {}).items() for d in docs}
    buckets: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    no_currency: set[str] = set()
    for r in rows:
        doc = str(r["document_id"])
        if not r["unit"]:                       # an amount of unknown currency cannot be added to anything
            no_currency.add(doc)
            continue
        if periods is not None and doc not in period_of:
            continue
        group = period_of.get(doc) if periods is not None else _group_key(plan, doc, dates, parties)
        currency = r["unit"]
        buckets[(group, currency)].append((doc, float(r["value_num"])))
    table = []
    for (group, currency), vals in sorted(buckets.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        nums = [v for _, v in vals]
        value = sum(nums) if op == "sum" else sum(nums) / len(nums) if op == "average" else max(nums) if op == "max" else min(nums)
        row: dict[str, Any] = {"currency": currency, "value": round(value, 2), "documents": len(vals)}
        if group is not None and plan.group_by != "currency":
            row = {("period" if periods is not None else plan.group_by or "group"): group, **row}
        table.append(row)
    valued = sorted({doc for vals in buckets.values() for doc, _ in vals})
    valued_set = set(valued)
    records = [{"document_id": str(r["document_id"]), "field": r["name"], "value": float(r["value_num"]),
                "currency": r["unit"], "text": r["value_text"], "page": r["page"], "char_start": r["char_start"],
                "char_end": r["char_end"], "snippet": r["snippet"]}
               for r in rows if str(r["document_id"]) in valued_set][:MAX_RECORDS]
    calc = calculation(plan, op, len(ids), field=names[0], fallback_field=names[1],
                       documents_with_values=len(valued), documents_without_values=len(set(ids) - valued_set - no_currency),
                       documents_with_amounts_of_unknown_currency=len(no_currency - valued_set),
                       rule="per document and currency, the value of the requested field (else a generic amount); "
                            "currencies are never added together")
    if not table:
        return {"kind": "none", "text": "No amounts were found in the matching documents.", "calculation": calc,
                "records": []}, valued
    label = {"sum": "Total", "average": "Average", "max": "Largest", "min": "Smallest"}[op]
    parts = []
    for r in table:
        key = r.get("period") if periods is not None else None if plan.group_by == "currency" else r.get(plan.group_by or "")
        prefix = f"{key}: " if key else ""
        parts.append(f"{prefix}{r['currency']} {r['value']:,.2f} ({r['documents']} document{'s' if r['documents'] != 1 else ''})")
    answer = {"kind": "table", "rows": table, "text": f"{label}: " + "; ".join(parts) + ".", "calculation": calc,
              "records": records, "note": "Amounts are grouped by currency; currencies are never added together."}
    if calc["documents_with_amounts_of_unknown_currency"]:
        answer["note"] += (f" {calc['documents_with_amounts_of_unknown_currency']} document(s) state an amount without a "
                           "currency and are not included.")
    if calc["documents_without_values"]:
        answer["note"] += f" {calc['documents_without_values']} matching document(s) had no amount and are not included."
    return answer, valued


def group(conn: psycopg.Connection, plan: QueryPlan, where: R.Where, type_label) -> list[dict]:
    if plan.group_by == "type":
        rows = conn.execute(f"SELECT coalesce(d.doc_type,'other') AS key, count(*) AS n FROM documents d WHERE {where.sql} "
                            "GROUP BY 1 ORDER BY 2 DESC, 1", where.args).fetchall()
        return [{"key": r["key"], "label": type_label(r["key"]), "count": r["n"]} for r in rows]
    if plan.group_by in ("month", "year"):
        fmt = "YYYY-MM" if plan.group_by == "month" else "YYYY"
        rows = conn.execute(
            f"SELECT to_char(x.dt, '{fmt}') AS key, count(*) AS n FROM (SELECT d.id, (SELECT min(f.value_date) FROM fields f "
            f"WHERE f.document_id = d.id AND f.name = ANY(%s)) AS dt FROM documents d WHERE {where.sql}) x "
            "WHERE x.dt IS NOT NULL GROUP BY 1 ORDER BY 1", [list(R.DATE_FIELDS), *where.args]).fetchall()
        return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
    if plan.group_by in ("jurisdiction", "party"):
        rows = conn.execute(
            f"SELECT x.value_text AS key, count(DISTINCT d.id) AS n FROM documents d JOIN fields x ON x.document_id = d.id "
            f"AND x.name = %s WHERE {where.sql} GROUP BY 1 ORDER BY 2 DESC, 1", [plan.group_by, *where.args]).fetchall()
        return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
    if plan.group_by == "currency":
        rows = conn.execute(
            f"SELECT x.unit AS key, count(DISTINCT d.id) AS n FROM documents d JOIN fields x ON x.document_id = d.id "
            f"AND x.value_num IS NOT NULL AND x.unit IS NOT NULL WHERE {where.sql} GROUP BY 1 ORDER BY 2 DESC, 1",
            where.args).fetchall()
        return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
    col = "d.language" if plan.group_by == "language" else "d.status"
    base = where.sql if plan.group_by != "status" else "true"
    rows = conn.execute(f"SELECT coalesce({col}, 'unknown') AS key, count(*) AS n FROM documents d WHERE {base} GROUP BY 1 "
                        "ORDER BY 2 DESC, 1", where.args if plan.group_by != "status" else []).fetchall()
    return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
