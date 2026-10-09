"""Query engine v2: independent retrievers, candidate fusion and verified evidence.

The plan decides which retrievers run (docs/design/QUERY_PLANNER.md); each implements the retrieval contract
(docintel/retrieval/contracts.py). Exact, lexical, fuzzy, entity, contextual and metadata retrieval read PostgreSQL
through the tenant-scoped connection; semantic retrieval runs concurrently against the vector index. A retriever that
fails or is disabled is reported in the response and the others still answer. Counts, sums and lookups reuse the
deterministic computations of the base engine, with document sets from the v2 exact and lexical retrievers.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg

from docintel import metrics
from docintel import text as T
from docintel.config import get_settings
from docintel.model_registry import model_spec
from docintel.packs import get_domain
from docintel.query import compute
from docintel.query.validation import PlanError, validate
from docintel.retrieval import evidence as EV
from docintel.retrieval.contracts import Budget, Candidate, RetrievalContext, Retriever, Scope
from docintel.retrieval.fusion import DocumentResult, fuse
from docintel.retrieval.lexical import ExactRetriever, FuzzyRetriever, LexicalRetriever, analyze
from docintel.retrieval.postings import Postings
from docintel.retrieval.rerank import get_reranker, rerank
from docintel.retrieval.semantic import SemanticRetriever, SemanticUnavailable
from docintel.retrieval.structured import (
    ContextualRetriever,
    EntityRetriever,
    MetadataRetriever,
    StructuredRetriever,
    question_ngrams,
)
from docintel.search import retrieval as R
from docintel.search.engine import QueryEngine, _terms, _type_label
from docintel.search.plan import Filters, QueryPlan
from docintel.search.planner import strategies_for
from docintel.security import AuthContext
from docintel.storage import repo
from docintel.understanding.entities import norm_org

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="query-v2")
STRONG_SEMANTIC_DELTA = 0.15     # similarity above the model's floor at which a semantic match is strong evidence
_MATCH_LABEL = {"exact_phrase": "exact phrase", "exact_identifier": "identifier", "filename": "file name",
                "structured": "filters", "context": "document opening", "exact_term": "exact word", "joined_form": "joined words", "all_terms": "all words",
                "concept": "related wording", "concept_clause": "clause", "relation": "stated relation", "entity": "name",
                "metadata": "title or key fields", "filename_terms": "file name words", "partial_identifier": "part of an identifier",
                "ocr_tolerant": "OCR-tolerant", "fuzzy": "spelling variant", "most_terms": "most words", "semantic": "meaning"}
_DOC_SET_TYPES = {"exact_phrase", "exact_identifier", "exact_term", "joined_form", "partial_identifier", "all_terms",
                  "ocr_tolerant", "filename", "filename_terms"}


class QueryEngineV2(QueryEngine):
    version = "v2"

    def __init__(self):
        super().__init__()
        self.structured = StructuredRetriever()
        self.lexical_retrievers: list[Retriever] = [ExactRetriever(), LexicalRetriever(), FuzzyRetriever(),
                                                    EntityRetriever(), ContextualRetriever(), MetadataRetriever()]
        self.semantic = SemanticRetriever()

    def run(self, ctx: AuthContext, question: str, limit: int | None = None, explain: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = super().run(ctx, question, limit, explain)
        out["engine"] = self.version
        metrics.QUERIES.labels(out["intent"], "answered" if out["results"] or out["answer"].get("kind") != "none" else "none").inc()
        metrics.QUERY_SECONDS.labels(out["intent"]).observe(time.perf_counter() - t0)
        return out

    # ------------------------------------------------------------------------------------------------ retrieval

    def _context(self, conn: psycopg.Connection, ctx: AuthContext, plan: QueryPlan) -> RetrievalContext:
        return RetrievalContext(conn, ctx, get_domain(plan.packs))

    def _run_retriever(self, r: Retriever, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget,
                       timings: dict, failed: list[str]) -> list[Candidate]:
        t = time.perf_counter()
        try:
            found = r.retrieve(rc, plan, scope, budget)
        except Exception:
            logger.exception("retriever failed", extra={"retriever": r.name})
            metrics.RETRIEVER_ERRORS.labels(r.name).inc()
            failed.append(r.name)
            return []
        finally:
            elapsed = time.perf_counter() - t
            timings[f"retriever.{r.name}"] = round(elapsed * 1000, 1)
            metrics.RETRIEVER_SECONDS.labels(r.name).observe(elapsed)
        return found

    def _semantic(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> tuple[list[Candidate], float]:
        t = time.perf_counter()
        found = self.semantic.retrieve(rc, plan, scope, budget)
        return found, (time.perf_counter() - t) * 1000

    def _rank(self, conn, ctx, plan: QueryPlan, allowed: list[str] | None, limit: int, timings: dict):
        s = get_settings()
        rc = self._context(conn, ctx, plan)
        scope = Scope(None if allowed is None else tuple(allowed))
        budget = Budget(max(limit, s.lexical_top_k // 25 or 8), time.monotonic() + s.query_timeout_ms / 1000)
        use_semantic = s.semantic_enabled and "semantic" in plan.strategies
        sem_future = _pool.submit(self._semantic, rc, plan, scope, budget) if use_semantic else None
        failed: list[str] = []
        candidates: list[Candidate] = []
        t = time.perf_counter()
        for r in self.lexical_retrievers:
            if r.name in plan.strategies:
                candidates += self._run_retriever(r, rc, plan, scope, budget, timings, failed)
        timings["lexical"] = round((time.perf_counter() - t) * 1000, 1)
        semantic_ok = True
        if sem_future is None:                         # not needed, or switched off by configuration: not a failure
            timings["semantic"] = 0.0
        else:
            try:
                found, ms = sem_future.result(timeout=s.query_timeout_ms / 1000)
                candidates += found
                timings["semantic"] = round(ms, 1)
                metrics.RETRIEVER_SECONDS.labels("semantic").observe(ms / 1000)
            except SemanticUnavailable:
                semantic_ok = False
            except Exception as e:
                logger.warning("semantic retrieval unavailable", extra={"error": str(e)})
                metrics.RETRIEVER_ERRORS.labels("semantic").inc()
                semantic_ok, timings["semantic"] = False, 0.0
                failed.append("semantic")
        soft = plan.filters.doc_types and not plan.filters.doc_types_hard
        meta = repo.get_documents(conn, {c.document_id for c in candidates}) if soft else {}
        strong = model_spec(s.embedding_model).semantic_floor + STRONG_SEMANTIC_DELTA
        unmatched = self._unmatched_terms(rc, plan) if plan.concepts else []
        if unmatched:
            plan.notes.append(f"words found in no document: {', '.join(unmatched)}")
        ranked = fuse(plan, candidates, {k: v["doc_type"] for k, v in meta.items()}, strong, bool(unmatched))
        reranker = get_reranker()
        if reranker is not None and len(ranked) > 1:
            t = time.perf_counter()
            try:
                ranked = rerank(conn, plan.question, ranked, reranker, s.rerank_top_n)
            except Exception as e:                      # the fused order stands
                logger.warning("reranker unavailable", extra={"error": str(e)[:200]})
                metrics.RETRIEVER_ERRORS.labels("reranker").inc()
                failed.append("reranker")
            timings["rerank"] = round((time.perf_counter() - t) * 1000, 1)
        timings["_failed"] = failed                     # per-request; removed by _search before the response
        return ranked, semantic_ok

    def _unmatched_terms(self, rc: RetrievalContext, plan: QueryPlan) -> list[str]:
        """Content words of the question that occur in none of the tenant's documents (in any indexed form), apart
        from the words that named a concept or a known entity."""
        consumed = {w for c in plan.concepts if c in rc.domain.concepts for v in rc.domain.concepts[c].variants
                    for w in T.search_text(v).split()}
        consumed |= {w for e in plan.entities or [] for w in e.split()}
        words = [w for w in dict.fromkeys(analyze(rc, plan.text).words)
                 if w not in T.STOPWORDS and w not in consumed and len(w) > 2 and not w.isdigit()]
        if not words:
            return []
        rows = rc.conn.execute("SELECT w, ts_lexize('english_stem', w) AS s FROM unnest(%s::text[]) AS w", (words,)).fetchall()
        stems = {r["w"]: T.arabic_stem(r["w"]) if T.arabic_stem(r["w"]) != r["w"] else (r["s"] or [r["w"]])[0]
                 for r in rows}                       # Arabic words are indexed under their light stems
        df = Postings(rc.conn, rc.auth.tenant_id).df([*(f"w:{w}" for w in words), *(f"f:{w}" for w in words),
                                                      *(f"s:{stems[w]}" for w in words)])
        return [w for w in words if not (df[f"w:{w}"] or df[f"f:{w}"] or df[f"s:{stems[w]}"])]

    def _listing(self, conn, where: R.Where, limit: int) -> list[DocumentResult]:
        rows = conn.execute(f"SELECT d.id FROM documents d WHERE {where.sql} ORDER BY d.indexed_at DESC NULLS LAST, d.id LIMIT %s",
                            [*where.args, limit]).fetchall()
        return [DocumentResult(str(r["id"]), [Candidate(str(r["id"]), None, "structured", "structured", 1.0)]) for r in rows]

    def _search(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        out = super()._search(conn, ctx, plan, limit, timings)
        if plan.intent == "fact":
            out["answer"] = self._relation_fact(conn, plan, out) or out["answer"]
        if out["results"] and all(r["tier"] == 4 for r in out["results"]):
            out["answer"]["note"] = "These documents are related by meaning; none of them contains the question's wording."
        failed = [f for f in timings.pop("_failed", []) if f != "semantic"]
        if failed:
            out["answer"]["degraded"] = f"some retrievers failed ({', '.join(failed)}); results may be incomplete"
        return out

    # ------------------------------------------------------------------------------------------------ results

    def _hydrate(self, conn, scored, plan: QueryPlan) -> list[dict]:
        if not scored:
            return []
        if scored and not isinstance(scored[0], DocumentResult):        # base-engine callers pass ids only
            scored = [DocumentResult(d.document_id, [Candidate(d.document_id, None, "structured", "structured", 1.0)])
                      for d in scored]
        ids = [d.document_id for d in scored]
        meta = repo.get_documents(conn, ids)
        if plan.filters.any() and any(c.match_type == "structured" and not c.spans for d in scored for c in d.candidates):
            rc = self._context(conn, None, plan)
            listed = {c.document_id: c for c in self.structured.retrieve(rc, plan, Scope(tuple(ids)), Budget())}
            for d in scored:
                for c in d.candidates:
                    if c.match_type == "structured" and not c.spans and d.document_id in listed:
                        c.spans = listed[d.document_id].spans
        terms = _terms(plan)
        evidence = EV.build(conn, scored, terms)
        fields = R.key_fields(conn, ids)
        firsts = R.first_chunks(conn, [d for d in ids if not evidence.get(d)])
        out = []
        for d in scored:
            m = meta.get(d.document_id)
            if not m or m["status"] != "indexed":
                continue
            ev = evidence.get(d.document_id, [])
            if not ev:
                if not any(c.match_type == "structured" for c in d.candidates):
                    # matched through its text, but no evidence survived verification: it cannot prove itself
                    logger.warning("result dropped: evidence failed verification", extra={"document_id": d.document_id})
                    continue
                if d.document_id in firsts:            # matched by filters: show the opening as context, not proof
                    fc = firsts[d.document_id]
                    ev = [{"page": fc["page_start"], "text": fc["text"][:EV.WINDOW], "match_type": "context",
                           "retriever": "structured", "chunk_id": fc["id"]}]
            mts = d.match_types
            tier = d.tier
            conf = 0.97 if tier == 1 else 0.85 if tier == 2 else round(min(0.9, 0.4 + d.vec_score), 2) if tier == 4 else 0.6
            seen_labels = []
            for e in ev:
                label = f"{_MATCH_LABEL.get(e['match_type'], e['match_type'])} on page {e['page']}"
                if label not in seen_labels:
                    seen_labels.append(label)
            out.append({
                "rank": len(out) + 1, "document_id": d.document_id, "filename": m["filename"], "title": m["title"],
                "doc_type": m["doc_type"], "doc_type_label": _type_label(m["doc_type"], plan),
                "doc_type_confidence": m["doc_type_confidence"], "page_count": m["page_count"], "kind": m["kind"],
                "has_signature": m["has_signature"], "match_types": mts, "retrievers": d.retrievers, "tier": tier,
                "confidence": conf, "semantic_score": round(d.vec_score, 4) if d.vec_rank is not None else None,
                "snippets": [{"page": e["page"], "text": e["text"], "match_type": e["match_type"],
                              **({"chunk_id": e["chunk_id"]} if "chunk_id" in e else {})} for e in ev],
                "evidence": ev, "explanation": "; ".join(seen_labels).capitalize() or None,
                "fields": fields.get(d.document_id, {}), "version": m.get("current_version"),
            })
        return out

    def _evidence(self, conn, plan: QueryPlan, ids: list[str], limit: int) -> list[dict]:
        return self._hydrate(conn, [DocumentResult(i, [Candidate(i, None, "structured", "structured", 1.0)]) for i in ids[:limit]], plan)

    # ------------------------------------------------------------------------------------------------ planning

    def _prepare(self, conn, ctx, plan: QueryPlan, domain) -> QueryPlan:
        """Resolve names in the question against the tenant's extracted entities, then validate the plan."""
        grams = question_ngrams(plan.question)
        if grams:
            rows = resolve_entities(conn, grams)
            names = sorted({r["value_norm"] for r in rows}, key=len, reverse=True)
            names = [n for i, n in enumerate(names) if not any(f" {n} " in f" {m} " for m in names[:i])]
            plan.entities = names
            if plan.intent in ("count", "percentage", "sum", "average", "min", "max") and names:
                # "How much did <organization> spend": a known organization becomes a party condition, not text
                residue = T.search_text(plan.text)
                orgs = [n for n in names if any(r["value_norm"] == n and r["type"] == "organization" for r in rows)]
                for n in orgs:
                    said = next((g for g in sorted(grams, key=len, reverse=True)
                                 if n == g or n == norm_org(g) or n.startswith(g + " ")), None)
                    if said and f" {said} " in f" {residue} ":
                        plan.filters.parties.append(n)
                        residue = " ".join(f" {residue} ".replace(f" {said} ", " ").split())
                if residue != T.search_text(plan.text):
                    plan.text = residue
                    plan.identifiers = sorted(T.identifiers(residue))
                    plan.notes.append(f"party condition: {', '.join(plan.filters.parties)}")
                    plan.strategies = strategies_for(plan)
        try:
            return validate(plan, domain)
        except PlanError:
            logger.exception("planner produced an invalid plan", extra={"intent": plan.intent})
            raise

    # ------------------------------------------------------------------------------------------------ aggregation

    def _count(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        out = super()._count(conn, ctx, plan, limit, timings)
        answer = out["answer"]
        if plan.intent == "group":
            if plan.text.strip() or plan.phrases or plan.identifiers:     # "invoices mentioning toner per year"
                ids = self._doc_set(conn, ctx, plan, plan.filters)
                where = R.filter_where(plan.filters)
                where.add("d.id = ANY(%s::uuid[])", ids)
                rows = compute.group(conn, plan, where, lambda k: _type_label(k, plan))
                total = sum(r["count"] for r in rows)
                answer.update(rows=rows, text=f"{total} documents in {len(rows)} groups by {plan.group_by}.")
                out["total"] = total
            answer["calculation"] = compute.calculation(plan, "count", out["total"])
            return out
        ids = [r["document_id"] for r in out["results"]] if plan.intent == "percentage" else None
        if plan.compare and plan.intent == "count":
            all_ids = self._doc_set(conn, ctx, plan, plan.filters)
            per = compute.count(conn, plan, all_ids)
            answer.update(kind="table", rows=per["rows"], text=per["text"], calculation=per["calculation"])
        else:
            answer["calculation"] = compute.calculation(
                plan, plan.intent, answer.get("denominator", answer.get("value")),
                **({"numerator": answer["numerator"], "denominator": answer["denominator"]} if plan.intent == "percentage" else {}))
        if ids is not None:
            answer["calculation"]["documents_counted"] = len(ids)
        return out

    def _group(self, conn, plan: QueryPlan) -> list[dict]:
        return compute.group(conn, plan, R.filter_where(plan.filters), lambda k: _type_label(k, plan))

    def _amounts(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        t = time.perf_counter()
        ids = self._doc_set(conn, ctx, plan, plan.filters)
        answer, valued = compute.amounts(conn, plan, ids)
        timings["aggregate"] = round((time.perf_counter() - t) * 1000, 1)
        return {"answer": answer, "results": self._evidence(conn, plan, valued, limit), "total": len(valued)}

    def _lookup(self, conn, ctx, plan: QueryPlan, limit: int, timings: dict) -> dict:
        out = super()._lookup(conn, ctx, plan, limit, timings)
        rows = out["answer"].get("rows") or []
        tiers = {r["document_id"]: r["tier"] for r in out["results"]}
        if rows and any(t < 4 for t in tiers.values()):
            # documents that contain the question's wording answer it; documents only related by meaning do not
            rows = [r for r in rows if tiers.get(r["document_id"], 4) < 4]
            out["answer"]["rows"] = rows
        if rows:
            spans = conn.execute(
                "SELECT document_id, value_text, page, char_start, char_end FROM fields WHERE document_id = ANY(%s::uuid[]) "
                "AND name = %s", (sorted({r["document_id"] for r in rows}), plan.lookup_field)).fetchall()
            where = {(str(x["document_id"]), x["value_text"], x["page"]): x for x in spans}
            for r in rows:
                x = where.get((r["document_id"], r["value"], r["page"]))
                if x:
                    r.update(char_start=x["char_start"], char_end=x["char_end"])
        return out

    def _relation_fact(self, conn, plan: QueryPlan, out: dict) -> dict | None:
        """A figure stated as a relation qualifier in a top document ("notice: sixty (60) days"). Never computed."""
        noun = re.search(r"\bhow\s+many\s+(?P<noun>[a-z][a-z\-]*)", plan.question, re.I)
        ids = [r["document_id"] for r in out["results"][:3]]
        if not (noun and ids and plan.concepts):
            return None
        unit = noun.group("noun").lower()
        stem = unit[:-1] if unit.endswith("s") else unit
        domain = get_domain(plan.packs)
        predicates = [domain.concepts[c].predicate for c in plan.concepts if domain.concepts.get(c) and domain.concepts[c].predicate]
        if not predicates:
            return None
        rows = conn.execute("SELECT document_id, predicate, qualifiers, page, char_start, char_end, snippet FROM relations "
                            "WHERE document_id = ANY(%s::uuid[]) AND predicate = ANY(%s)", (ids, predicates)).fetchall()
        rows.sort(key=lambda r: ids.index(str(r["document_id"])))
        for r in rows:
            for q in (r["qualifiers"] or {}).values():
                m = re.search(rf"(\d[\d,]*(?:\.\d+)?)\)?\s*{re.escape(stem)}", str(q), re.I)
                if not m:
                    continue
                res = next(x for x in out["results"] if x["document_id"] == str(r["document_id"]))
                return {"kind": "figure", "value": m.group(1), "unit": unit, "document_id": str(r["document_id"]),
                        "filename": res["filename"], "page": r["page"], "char_start": r["char_start"],
                        "char_end": r["char_end"], "snippet": r["snippet"], "source": f"relation:{r['predicate']}",
                        "text": f"{m.group(1)} {unit}, as stated in {res['filename']} (page {r['page']}). "
                                "This figure is quoted from the document, not computed."}
        return None

    def _fact(self, plan: QueryPlan, out: dict) -> dict | None:
        if out["answer"].get("kind") == "figure":           # already answered from a relation
            return out["answer"]
        return super()._fact(plan, out)

    def _doc_set(self, conn, ctx, plan: QueryPlan, filters: Filters) -> list[str]:
        """All documents satisfying the filters and, if the question names terms, containing them (complete: no
        ranking cut-off, no semantic or typo-corrected matches)."""
        where = R.filter_where(filters)
        if plan.text.strip() or plan.phrases or plan.identifiers:
            allowed = R.allowed_documents(conn, where, 10_000_000)
            sub = plan.__class__(**{**plan.__dict__, "filters": filters})
            rc = self._context(conn, ctx, sub)
            scope = Scope(None if allowed is None else tuple(allowed))
            budget = Budget(max_candidates=400_000)
            found: set[str] = set()
            for r in self.lexical_retrievers[:3]:            # exact, lexical, fuzzy
                found |= {c.document_id for c in r.retrieve(rc, sub, scope, budget) if c.match_type in _DOC_SET_TYPES}
            if found:
                rows = conn.execute("SELECT id FROM documents WHERE id = ANY(%s::uuid[]) AND status = 'indexed'",
                                    (sorted(found),)).fetchall()
                return sorted(str(r["id"]) for r in rows)
            return []
        rows = conn.execute(f"SELECT d.id FROM documents d WHERE {where.sql}", where.args).fetchall()
        return [str(r["id"]) for r in rows]


def resolve_entities(conn, grams: list[str]) -> list[dict]:
    """Organizations and people whose normalized name equals a question n-gram, or starts with it at a word
    boundary ("riverside" → "riverside freight"). Equality and range comparisons use the (tenant_id, value_norm)
    index under row-level security."""
    values = sorted({*grams, *(norm_org(g) for g in grams)} - {""})
    prefixes = [g for g in grams if len(g) >= 4 and not g.isdigit()][:12]
    conds = ["value_norm = ANY(%s)"] + ["(value_norm >= %s AND value_norm < %s)"] * len(prefixes)
    args: list = [values]
    for g in prefixes:
        args += [g + " ", g + "!"]                    # names that continue with another word after the prefix
    rows = conn.execute(f"SELECT DISTINCT value, value_norm, type FROM entities WHERE type IN ('organization', 'person') "
                        f"AND ({' OR '.join(conds)}) LIMIT 200", args).fetchall()
    return rows


class ShadowEngine:
    """Serves the v1 engine's answer and runs v2 on the same question in the background, logging how the two
    result lists compare (overlap of the top 10, first result, latency). For safe cut-over evaluation."""
    version = "shadow"

    def __init__(self, primary: QueryEngine, candidate: QueryEngineV2):
        self.primary, self.candidate = primary, candidate

    def run(self, ctx: AuthContext, question: str, limit: int | None = None, explain: bool = False) -> dict[str, Any]:
        out = self.primary.run(ctx, question, limit, explain)
        out["engine"] = "v1"
        first = [r["document_id"] for r in out["results"][:10]]

        def compare():
            try:
                t = time.perf_counter()
                other = self.candidate.run(ctx, question, limit, False)
                second = [r["document_id"] for r in other["results"][:10]]
                overlap = len(set(first) & set(second)) / max(1, len(set(first) | set(second)))
                logger.info("shadow comparison", extra={"intent": out["intent"], "overlap_at_10": round(overlap, 3),
                                                        "same_first": bool(first and second and first[0] == second[0]),
                                                        "v1_ms": out["timings_ms"]["total"],
                                                        "v2_ms": round((time.perf_counter() - t) * 1000, 1)})
            except Exception:
                logger.exception("shadow comparison failed")
        _pool.submit(compare)
        return out
