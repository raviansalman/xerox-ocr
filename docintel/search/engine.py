"""Query execution: plan → retrieval/aggregation → answer with evidence.

Ranking: documents with an exact phrase, identifier or file-name match form the first tier (they cannot be lost to
vector similarity); the rest are fused with reciprocal rank fusion of lexical and semantic ranks. Semantic-only
documents must clear the embedding model's similarity floor, so unknown queries return no results instead of the
"closest" ones. Counts and sums are computed in SQL over extracted fields, never estimated from text.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import psycopg

from docintel import settings_store
from docintel import text as T
from docintel.config import get_settings
from docintel.indexing.embeddings import get_embedder
from docintel.indexing.vectors import get_vector_store
from docintel.model_registry import model_spec
from docintel.packs import get_domain
from docintel.search import retrieval as R
from docintel.search.plan import Filters, QueryPlan
from docintel.search.planner import plan_query
from docintel.security import AuthContext
from docintel.storage import repo
from docintel.storage.db import get_db

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="query")
RRF_K = 60
SEMANTIC_MARGIN = 0.22     # semantic-only documents must score within this of the best semantic match


@dataclass
class DocScore:
    document_id: str
    tier: int = 9
    match_types: set[str] = field(default_factory=set)
    lex_rank: int | None = None
    vec_rank: int | None = None
    vec_score: float = 0.0
    chunks: list[tuple[int, str, str, int, str]] = field(default_factory=list)   # (priority, chunk_id, match, page, text)

    def fused(self, w_lex: float, w_vec: float) -> float:
        s = 0.0
        if self.lex_rank is not None:
            s += w_lex / (RRF_K + self.lex_rank)
        if self.vec_rank is not None:
            s += w_vec / (RRF_K + self.vec_rank)
        return s


def _type_label(label: str | None, plan: QueryPlan) -> str:
    return get_domain(plan.packs).type_label(label)


def _snippet(text: str, terms: list[str], size: int = 360) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    pos = min((low.find(t.lower()) for t in terms if t and low.find(t.lower()) >= 0), default=0)
    start = max(0, pos - size // 3)
    end = min(len(flat), start + size)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def _terms(plan: QueryPlan) -> list[str]:
    out = [p for p in plan.phrases]
    words = T.content_words(plan.text)
    if len(words) > 1:
        out.append(" ".join(T.tokens(plan.text)))
    out += [w for w in words if len(w) > 1]
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/.#]*[A-Za-z0-9]", plan.text)
    out += [r for r in raw if any(c.isdigit() for c in r)]
    return list(dict.fromkeys(out))


_PHRASE_EVIDENCE = {"exact_phrase", "exact_identifier", "ocr_tolerant", "fuzzy", "filename"}


def _identifier_only(plan: QueryPlan) -> bool:
    """True when every word of the query is one of its identifiers."""
    words = plan.text.split()
    return bool(plan.identifiers) and bool(words) and all(
        re.sub(r"[^0-9a-z]", "", w.casefold()) in plan.identifiers for w in words)


class QueryEngine:
    def __init__(self):
        self.settings = get_settings()
        self.spec = model_spec(self.settings.embedding_model)

    # ------------------------------------------------------------------------------------------------ public

    def run(self, ctx: AuthContext, question: str, limit: int | None = None, explain: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        limit = max(1, min(limit or self.settings.result_limit, 100))
        with get_db().tenant(ctx.tenant_id) as conn:
            domain = get_domain(tuple(settings_store.enabled_packs(conn)))
            plan = self._prepare(conn, ctx, plan_query(question, domain=domain), domain)
            timings: dict[str, float] = {"plan": round((time.perf_counter() - t0) * 1000, 1)}
            if plan.intent in ("count", "percentage", "group"):
                out = self._count(conn, ctx, plan, limit, timings)
            elif plan.intent in ("sum", "average", "min", "max"):
                out = self._amounts(conn, ctx, plan, limit, timings)
            elif plan.intent == "lookup":
                out = self._lookup(conn, ctx, plan, limit, timings)
            else:
                out = self._search(conn, ctx, plan, limit, timings)
                if plan.intent == "fact":
                    out["answer"] = self._fact(plan, out) or out["answer"]
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        out.update(query=question, intent=plan.intent, timings_ms=timings, terms=_terms(plan), engine="v1")
        if explain:
            out["plan"] = plan.to_dict()
        logger.info("query", extra={"tenant_id": ctx.tenant_id, "intent": plan.intent, "results": len(out["results"]),
                                    "ms": timings["total"]})
        return out

    def _prepare(self, conn, ctx, plan: QueryPlan, domain) -> QueryPlan:
        """Hook to refine the plan with tenant data before execution (the v2 engine resolves entities)."""
        return plan

    # ------------------------------------------------------------------------------------------------ search

    def _vector(self, ctx: AuthContext, text: str, allowed: list[str] | None, k: int) -> tuple[list, float]:
        t = time.perf_counter()
        if not self.settings.semantic_enabled or not text.strip() or (allowed is not None and not allowed):
            return [], 0.0
        vec = get_embedder().embed_query(text)
        ids = allowed if allowed is not None and len(allowed) <= self.settings.structured_id_limit else None
        hits = get_vector_store().search(ctx.tenant_id, vec, k, document_ids=ids)
        if allowed is not None and ids is None:
            keep = set(allowed)
            hits = [h for h in hits if h.document_id in keep]
        return hits, (time.perf_counter() - t) * 1000

    def _rank(self, conn: psycopg.Connection, ctx: AuthContext, plan: QueryPlan, allowed: list[str] | None,
              limit: int, timings: dict) -> tuple[list[DocScore], bool]:
        """Fused document ranking. Returns (scores, semantic_available)."""
        sem_text = plan.semantic_text if plan.text or plan.phrases else ""
        fut = _pool.submit(self._vector, ctx, sem_text, allowed, self.settings.vector_top_k)
        t = time.perf_counter()
        lex = R.lexical(conn, plan, allowed, self.settings.lexical_top_k)
        fname = R.filename_hits(conn, plan, allowed, 50)
        timings["lexical"] = round((time.perf_counter() - t) * 1000, 1)
        semantic_ok = True
        try:
            vhits, vms = fut.result(timeout=self.settings.query_timeout_ms / 1000)
            timings["semantic"] = round(vms, 1)
        except Exception as e:                            # degrade to lexical-only, report it
            logger.warning("semantic retrieval unavailable", extra={"error": str(e)})
            vhits, semantic_ok = [], False

        docs: dict[str, DocScore] = {}
        lex_order: list[str] = []
        prio = {"exact_phrase": 0, "exact_identifier": 0, "exact_term": 1, "joined_form": 1, "partial_identifier": 2,
                "all_terms": 2, "ocr_tolerant": 3, "fuzzy": 4}
        for h in sorted(lex, key=lambda h: (R.TIER[h.match_type], prio.get(h.match_type, 5), -h.rank)):
            d = docs.setdefault(h.document_id, DocScore(h.document_id))
            d.tier = min(d.tier, R.TIER[h.match_type])
            d.match_types.add(h.match_type)
            d.chunks.append((prio.get(h.match_type, 5), h.chunk_id, h.match_type, h.page, h.text))
            if h.document_id not in lex_order:
                lex_order.append(h.document_id)
        for doc_id, mt in fname:
            d = docs.setdefault(doc_id, DocScore(doc_id))
            d.tier = min(d.tier, R.TIER[mt])
            d.match_types.add(mt)
            if doc_id not in lex_order:
                lex_order.append(doc_id)
        for i, doc_id in enumerate(lex_order):
            docs[doc_id].lex_rank = i
        floor = self.spec.semantic_floor
        if vhits:                                    # semantic-only hits must also be close to the best semantic hit
            floor = max(floor, max(h.score for h in vhits) - SEMANTIC_MARGIN)
        seen_vec: list[str] = []
        for h in vhits:
            if h.document_id in seen_vec:
                d = docs[h.document_id]
                if len(d.chunks) < 4:
                    d.chunks.append((6, h.chunk_id, "semantic", 0, ""))
                continue
            seen_vec.append(h.document_id)
            d = docs.setdefault(h.document_id, DocScore(h.document_id))
            d.vec_rank, d.vec_score = len(seen_vec) - 1, h.score
            if h.score >= floor:
                d.match_types.add("semantic")
                d.tier = min(d.tier, 1)
            d.chunks.append((5 if h.score >= floor else 7, h.chunk_id, "semantic", 0, ""))

        exact_query = plan.exact_intent
        if _identifier_only(plan):
            # "AB-1234": embeddings of a bare code carry no meaning, so only documents with lexical evidence count
            docs = {k: d for k, d in docs.items() if d.match_types - {"semantic"}}
        elif plan.phrases:
            # a quoted phrase must appear (exactly, OCR-tolerant or near-exact); scattered words are not the phrase
            docs = {k: d for k, d in docs.items() if d.match_types & _PHRASE_EVIDENCE}
        w_lex, w_vec = (1.0, 0.6) if exact_query else (0.7, 1.0)
        soft_types = set(plan.filters.doc_types) if plan.filters.doc_types and not plan.filters.doc_types_hard else set()
        doc_meta = repo.get_documents(conn, docs.keys()) if soft_types else {}
        ranked = []
        for d in docs.values():
            if d.tier == 9:                          # semantic below floor and no lexical evidence
                continue
            score = d.fused(w_lex, w_vec)
            if soft_types and (doc_meta.get(d.document_id) or {}).get("doc_type") in soft_types:
                score += 0.006
            ranked.append((d.tier if exact_query or d.tier == 0 else min(d.tier, 1), -score, d))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return [r[2] for r in ranked], semantic_ok

    def _listing(self, conn: psycopg.Connection, where: R.Where, limit: int) -> list[DocScore]:
        rows = conn.execute(f"SELECT d.id FROM documents d WHERE {where.sql} ORDER BY d.indexed_at DESC NULLS LAST, d.id LIMIT %s",
                            where.args + [limit]).fetchall()
        return [DocScore(str(r["id"]), tier=0, match_types={"structured"}) for r in rows]

    def _search(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        f = plan.filters
        if plan.intent == "fact":                    # facts: filters steer ranking, they do not exclude documents
            f = Filters(doc_types=f.doc_types)
        where = R.filter_where(f)
        allowed = R.allowed_documents(conn, where, self.settings.structured_id_limit)
        has_text = bool(T.tokens(plan.text) or plan.phrases or plan.identifiers)
        semantic_ok = True
        if has_text:
            scored, semantic_ok = self._rank(conn, ctx, plan, allowed, limit, timings)
        elif where.sql != R.Where().sql:
            scored = self._listing(conn, where, limit)
        else:
            scored = []
        scored = scored[:limit]
        results = self._hydrate(conn, scored, plan)
        total = len(results)
        if not results:
            answer = {"kind": "none", "text": "No sufficiently reliable evidence was found for this question."}
        elif not has_text:
            answer = {"kind": "documents", "text": f"{total} document{'s match' if total != 1 else ' matches'} the filters."}
        else:
            top = results[0]
            answer = {"kind": "documents", "text": f"{total} matching document{'s' if total != 1 else ''}; best match: "
                      f"{top['title'] or top['filename']} ({', '.join(top['match_types'])})."}
        if not semantic_ok:
            answer["degraded"] = "semantic search unavailable; lexical results only"
        return {"answer": answer, "results": results, "total": total}

    def _hydrate(self, conn, scored: list[DocScore], plan: QueryPlan) -> list[dict]:
        if not scored:
            return []
        ids = [d.document_id for d in scored]
        meta = repo.get_documents(conn, ids)
        need = [c[1] for d in scored for c in d.chunks if not c[4]]
        texts = R.chunk_texts(conn, need)
        firsts = R.first_chunks(conn, [d.document_id for d in scored if not d.chunks])
        fields = R.key_fields(conn, ids)
        clauses = R.clause_snippets(conn, ids, plan.filters.clause_types)
        terms = _terms(plan)
        out = []
        for rank, d in enumerate(scored):
            m = meta.get(d.document_id)
            if not m:
                continue
            snippets, used = [], set()
            for c in clauses.get(d.document_id, [])[:2]:
                snippets.append({"page": c["page"], "text": _snippet(c["text"], terms, 500), "match_type": f"clause:{c['clause_type']}"})
            for _prio, cid, mt, page, text in sorted(d.chunks, key=lambda c: c[0]):
                if cid in used or len(snippets) >= 3:
                    continue
                if not text:
                    row = texts.get(cid)
                    if not row:
                        continue
                    text, page = row["text"], row["page_start"]
                used.add(cid)
                snippets.append({"chunk_id": cid, "page": page, "text": _snippet(text, terms), "match_type": mt})
            if not snippets and d.document_id in firsts:
                fc = firsts[d.document_id]
                snippets.append({"chunk_id": fc["id"], "page": fc["page_start"], "text": _snippet(fc["text"], terms), "match_type": "structured"})
            mts = sorted(d.match_types, key=lambda x: (R.TIER.get(x, 1), x))
            conf = 0.97 if d.tier == 0 else 0.85 if mts and mts[0] in R.TIER and R.TIER[mts[0]] == 1 else \
                round(min(0.9, 0.4 + d.vec_score), 2) if "semantic" in mts else 0.6
            out.append({
                "rank": rank + 1, "document_id": d.document_id, "filename": m["filename"], "title": m["title"],
                "doc_type": m["doc_type"], "doc_type_label": _type_label(m["doc_type"], plan),
                "doc_type_confidence": m["doc_type_confidence"], "page_count": m["page_count"], "kind": m["kind"],
                "has_signature": m["has_signature"], "match_types": mts, "tier": d.tier, "confidence": conf,
                "semantic_score": round(d.vec_score, 4) if d.vec_rank is not None else None,
                "snippets": snippets, "fields": fields.get(d.document_id, {}),
            })
        return out

    # ------------------------------------------------------------------------------------------------ aggregation

    def _doc_set(self, conn, ctx, plan: QueryPlan, filters: Filters) -> list[str]:
        """All documents satisfying the filters and, if the question names terms, containing them."""
        where = R.filter_where(filters)
        words = T.tokens(plan.text)
        if words or plan.phrases or plan.identifiers:
            sub = plan.__class__(**{**plan.__dict__, "filters": filters})
            allowed = R.allowed_documents(conn, where, 10_000_000)
            hits = R.lexical(conn, sub, allowed, 100_000)
            strong = {h.document_id for h in hits if h.match_type != "fuzzy"}
            return sorted(strong)
        rows = conn.execute(f"SELECT d.id FROM documents d WHERE {where.sql}", where.args).fetchall()
        return [str(r["id"]) for r in rows]

    def _evidence(self, conn, plan: QueryPlan, ids: list[str], limit: int) -> list[dict]:
        return self._hydrate(conn, [DocScore(i, tier=0, match_types={"structured"}) for i in ids[:limit]], plan)

    def _uncertain(self, conn, ids: list[str], plan: QueryPlan) -> int:
        if not ids or not plan.filters.doc_types:
            return 0
        r = conn.execute("SELECT count(*) AS n FROM documents WHERE id = ANY(%s::uuid[]) AND doc_type_confidence < 0.6",
                         (ids,)).fetchone()
        return int(r["n"])

    def _pending(self, conn) -> int:
        return int(conn.execute("SELECT count(*) AS n FROM documents WHERE status IN ('queued','processing')").fetchone()["n"])

    def _count(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        t = time.perf_counter()
        pending = self._pending(conn)
        if plan.intent == "group":
            rows = self._group(conn, plan)
            total = sum(r["count"] for r in rows)
            answer = {"kind": "table", "group_by": plan.group_by, "rows": rows,
                      "text": f"{total} documents in {len(rows)} groups by {plan.group_by}."}
            timings["aggregate"] = round((time.perf_counter() - t) * 1000, 1)
            return {"answer": answer, "results": [], "total": total}
        ids = self._doc_set(conn, ctx, plan, plan.filters)
        if plan.intent == "percentage":
            base_ids = set(self._doc_set(conn, ctx, plan.__class__(**{**plan.__dict__, "text": "", "phrases": [], "identifiers": []}),
                                         plan.base or Filters()))
            num = [i for i in ids if i in base_ids]
            pct = round(100.0 * len(num) / len(base_ids), 1) if base_ids else 0.0
            answer = {"kind": "number", "value": pct, "unit": "%", "numerator": len(num), "denominator": len(base_ids),
                      "text": f"{pct}% ({len(num)} of {len(base_ids)} documents)."}
            ids = num
        else:
            unc = self._uncertain(conn, ids, plan)
            text = f"{len(ids)} document{'s' if len(ids) != 1 else ''}"
            if unc:
                text += f" ({unc} with uncertain classification)"
            answer = {"kind": "number", "value": len(ids), "unit": "documents", "uncertain": unc, "text": text + "."}
        if pending:
            answer["pending"] = pending
            answer["text"] += f" {pending} document{'s are' if pending != 1 else ' is'} still being processed."
        timings["aggregate"] = round((time.perf_counter() - t) * 1000, 1)
        return {"answer": answer, "results": self._evidence(conn, plan, ids, limit), "total": len(ids)}

    def _group(self, conn, plan: QueryPlan) -> list[dict]:
        where = R.filter_where(plan.filters)
        if plan.group_by == "type":
            sql = f"SELECT coalesce(d.doc_type,'other') AS key, count(*) AS n FROM documents d WHERE {where.sql} GROUP BY 1 ORDER BY 2 DESC"
            rows = conn.execute(sql, where.args).fetchall()
            return [{"key": r["key"], "label": _type_label(r["key"], plan), "count": r["n"]} for r in rows]
        if plan.group_by in ("month", "year"):
            fmt = "YYYY-MM" if plan.group_by == "month" else "YYYY"
            sql = (f"SELECT to_char(x.dt, '{fmt}') AS key, count(*) AS n FROM (SELECT d.id, (SELECT min(f.value_date) FROM fields f "
                   f"WHERE f.document_id = d.id AND f.name = ANY(%s)) AS dt FROM documents d WHERE {where.sql}) x "
                   f"WHERE x.dt IS NOT NULL GROUP BY 1 ORDER BY 1")
            rows = conn.execute(sql, [list(R.DATE_FIELDS)] + where.args).fetchall()
            return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
        if plan.group_by == "jurisdiction":
            sql = (f"SELECT x.value_text AS key, count(DISTINCT d.id) AS n FROM documents d JOIN fields x ON x.document_id = d.id "
                   f"AND x.name = 'jurisdiction' WHERE {where.sql} GROUP BY 1 ORDER BY 2 DESC")
            rows = conn.execute(sql, where.args).fetchall()
            return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]
        col = "d.language" if plan.group_by == "language" else "d.status"
        base = where.sql if plan.group_by != "status" else "true"
        rows = conn.execute(f"SELECT coalesce({col}, 'unknown') AS key, count(*) AS n FROM documents d WHERE {base} GROUP BY 1 ORDER BY 2 DESC",
                            where.args if plan.group_by != "status" else []).fetchall()
        return [{"key": r["key"], "label": r["key"], "count": r["n"]} for r in rows]

    def _amounts(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        t = time.perf_counter()
        ids = self._doc_set(conn, ctx, plan, plan.filters)
        names = [plan.aggregate_field or "total_amount", "amount"]
        rows = conn.execute(
            "SELECT DISTINCT ON (document_id, unit) document_id, unit, value_num, name FROM fields "
            "WHERE document_id = ANY(%s::uuid[]) AND name = ANY(%s) AND value_num IS NOT NULL "
            "ORDER BY document_id, unit, (name = %s) DESC, value_num DESC", (ids, names, names[0])).fetchall()
        per_cur: dict[str, list[tuple[str, float]]] = {}
        for r in rows:
            per_cur.setdefault(r["unit"] or "?", []).append((str(r["document_id"]), float(r["value_num"])))
        op = plan.intent
        table = []
        for cur, vals in sorted(per_cur.items()):
            nums = [v for _, v in vals]
            value = sum(nums) if op == "sum" else sum(nums) / len(nums) if op == "average" else max(nums) if op == "max" else min(nums)
            table.append({"currency": cur, "value": round(value, 2), "documents": len(vals)})
        valued = {doc for vals in per_cur.values() for doc, _ in vals}
        with_values = [d for d in ids if d in valued]
        if not table:
            answer = {"kind": "none", "text": "No amounts were found in the matching documents."}
        else:
            label = {"sum": "Total", "average": "Average", "max": "Largest", "min": "Smallest"}[op]
            parts = [f"{r['currency']} {r['value']:,.2f} ({r['documents']} document{'s' if r['documents'] != 1 else ''})" for r in table]
            answer = {"kind": "table", "rows": table, "text": f"{label}: " + "; ".join(parts) + ".",
                      "note": "Amounts are grouped by currency; currencies are never added together."}
        timings["aggregate"] = round((time.perf_counter() - t) * 1000, 1)
        return {"answer": answer, "results": self._evidence(conn, plan, with_values, limit), "total": len(with_values)}

    # ------------------------------------------------------------------------------------------------ lookup / facts

    def _lookup(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        found = self._search(conn, ctx, plan, min(limit, 5), timings)
        ids = [r["document_id"] for r in found["results"][:3]]
        rows = conn.execute(
            "SELECT document_id, name, value_text, page, snippet, confidence FROM fields WHERE document_id = ANY(%s::uuid[]) "
            "AND name = %s ORDER BY confidence DESC, id", (ids, plan.lookup_field)).fetchall() if ids else []
        by_doc: dict[str, list[dict]] = {}
        for r in rows:
            by_doc.setdefault(str(r["document_id"]), []).append(r)
        values = []
        for res in found["results"][:3]:
            for r in by_doc.get(res["document_id"], [])[:3]:
                values.append({"document_id": res["document_id"], "filename": res["filename"], "value": r["value_text"],
                               "page": r["page"], "snippet": r["snippet"], "confidence": r["confidence"]})
        if values:
            v = values[0]
            answer = {"kind": "fields", "field": plan.lookup_field, "rows": values,
                      "text": f"{plan.lookup_field.replace('_', ' ').capitalize()}: {v['value']} "
                              f"({v['filename']}, page {v['page']})."}
        else:
            answer = {"kind": "none", "field": plan.lookup_field,
                      "text": f"No {plan.lookup_field.replace('_', ' ')} was found in the matching documents."}
        found["answer"] = answer
        return found

    def _fact(self, plan: QueryPlan, out: dict) -> dict | None:
        """Quote a figure stated in a top passage for "how many X" questions (never computed)."""
        m = re.search(r"\bhow\s+many\s+(?P<noun>[a-z][a-z\-]*)", plan.question, re.I)
        if not m:
            return None
        noun = m.group("noun").lower()
        stem = noun[:-1] if noun.endswith("s") and len(noun) > 3 else noun
        pats = [re.compile(rf"\b{re.escape(stem)}\w*\b[^.\n\d]{{0,40}}?(\d[\d,]*(?:\.\d+)?)", re.I),
                re.compile(rf"\(?(\d[\d,]*(?:\.\d+)?)\)?\s+(?:[a-z]+\s+){{0,2}}{re.escape(stem)}\w*\b", re.I)]
        for res in out["results"][:5]:
            for snip in res["snippets"]:
                for p in pats:
                    mm = p.search(snip["text"])
                    if mm:
                        return {"kind": "figure", "value": mm.group(1), "unit": noun, "document_id": res["document_id"],
                                "filename": res["filename"], "page": snip["page"], "snippet": snip["text"],
                                "text": f"{mm.group(1)} {noun}, as stated in {res['filename']} (page {snip['page']}). "
                                        "This figure is quoted from the document, not computed."}
        return None


_engine: QueryEngine | None = None


def get_engine() -> QueryEngine:
    global _engine
    if _engine is None:
        _engine = QueryEngine()
    return _engine
