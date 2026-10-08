"""Entity, contextual, metadata and structured retrieval.

* EntityRetriever: people, organizations and places named in the question, matched against extracted entities by
  normalized value (any n-gram of the question), with the mention's span as evidence.
* ContextualRetriever: concepts from the enabled packs ("cancel" → termination) reach clauses of that type, relations
  with that predicate, and every lexical variant of the concept, so a question finds a clause worded differently.
* MetadataRetriever: the document unit (title, document type, file name, key fields) contains every content word.
* StructuredRetriever: documents satisfying the plan's filters, with the matching fields as evidence; also the
  scope for the other retrievers.
"""
from __future__ import annotations

from docintel import text as T
from docintel.retrieval.contracts import Budget, Candidate, EvidenceSpan, RetrievalContext, Scope
from docintel.retrieval.lexical import _PHRASE, analyze
from docintel.retrieval.postings import Postings
from docintel.search import retrieval as R
from docintel.search.plan import Filters, QueryPlan
from docintel.understanding.entities import norm_org

_ENTITY_TYPES = ("person", "organization", "jurisdiction")
_SPAN_COLS = "document_id, page, char_start, char_end"


def _doc_filter(scope: Scope, args: list) -> str:
    if scope.document_ids is None:
        return ""
    args.append(list(scope.document_ids))
    return " AND document_id = ANY(%s::uuid[])"


def _span(r: dict) -> list[EvidenceSpan]:
    return [EvidenceSpan(r["page"], r["char_start"], r["char_end"])] if r.get("page") else []


def question_ngrams(question: str, max_n: int = 4) -> list[str]:
    words = T.search_text(question).split()
    out = []
    for n in range(max_n, 0, -1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if all(w in T.STOPWORDS for w in gram) or (n == 1 and (len(gram[0]) < 3 or gram[0].isdigit())):
                continue
            out.append(" ".join(gram))
    return list(dict.fromkeys(out))


class EntityRetriever:
    name = "entity"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        grams = question_ngrams(plan.question)
        if not grams:
            return []
        values = sorted({*grams, *(norm_org(g) for g in grams)} - {""})
        args: list = [values, list(_ENTITY_TYPES)]
        sql = (f"SELECT {_SPAN_COLS}, type, value, value_norm FROM entities WHERE value_norm = ANY(%s) AND type = ANY(%s)"
               + _doc_filter(scope, args) + " LIMIT %s")
        rows = rc.conn.execute(sql, [*args, budget.max_candidates * 5]).fetchall()
        # a longer matching name wins over a shorter one contained in it ("jane roe" over "roe")
        names = sorted({r["value_norm"] for r in rows}, key=len, reverse=True)
        keep = [n for i, n in enumerate(names) if not any(n in m.split() or f" {n} " in f" {m} " for m in names[:i])]
        return [Candidate(str(r["document_id"]), None, self.name, "entity", float(len(r["value_norm"])), _span(r),
                          (r["value"],)) for r in rows if r["value_norm"] in keep]


class ContextualRetriever:
    name = "contextual"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        out: list[Candidate] = []
        p = Postings(rc.conn, rc.auth.tenant_id)
        for concept in (rc.domain.concepts[c] for c in plan.concepts if c in rc.domain.concepts):
            if concept.clause_type:
                args: list = [concept.clause_type]
                rows = rc.conn.execute(f"SELECT {_SPAN_COLS}, clause_type FROM clauses WHERE clause_type = %s"
                                       + _doc_filter(scope, args) + " LIMIT %s", [*args, budget.max_candidates * 5]).fetchall()
                out += [Candidate(str(r["document_id"]), None, self.name, "concept_clause", 1.0, _span(r), (concept.name,))
                        for r in rows]
            if concept.predicate:
                args = [concept.predicate]
                rows = rc.conn.execute(f"SELECT {_SPAN_COLS}, predicate FROM relations WHERE predicate = %s"
                                       + _doc_filter(scope, args) + " LIMIT %s", [*args, budget.max_candidates * 5]).fetchall()
                out += [Candidate(str(r["document_id"]), None, self.name, "relation", 1.0, _span(r), (concept.name,))
                        for r in rows]
            for variant in concept.variants:                         # every way the concept may be written
                q = analyze(rc, variant)
                if not q.words:
                    continue
                hits = p.all_of([f"w:{w}" for w in q.words], scope.document_ids, budget.max_candidates * 5)
                keep = p.verify(list(hits), _PHRASE, T.search_text(variant)) if len(q.words) > 1 else set(hits)
                out += [Candidate(h.document_id, uid, self.name, "concept", 0.5, [], (concept.name, variant))
                        for uid, h in hits.items() if uid in keep]
        return out


class MetadataRetriever:
    name = "metadata"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        q = analyze(rc, plan.text)
        words = [w for w in q.words if w not in T.STOPWORDS]
        if not words:
            return []
        p = Postings(rc.conn, rc.auth.tenant_id)
        hits = p.all_of([f"w:{w}" for w in words], scope.document_ids, budget.max_candidates * 10)
        if not hits:
            return []
        rows = rc.conn.execute("SELECT id FROM chunks WHERE id = ANY(%s) AND unit_type = 'document'", (list(hits),)).fetchall()
        return [Candidate(hits[r["id"]].document_id, r["id"], self.name, "metadata", 1.0, [], tuple(words)) for r in rows]


class StructuredRetriever:
    name = "structured"

    def scope(self, rc: RetrievalContext, filters: Filters | None, limit: int) -> Scope:
        where = R.filter_where(filters)
        allowed = R.allowed_documents(rc.conn, where, limit)
        return Scope(None if allowed is None else tuple(allowed))

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        """Documents matching the filters (``scope``), each with the field values that satisfied them."""
        if scope.document_ids is None or not scope.document_ids:
            return []
        f = plan.filters
        names: list[str] = []
        names += [x for df in f.dates for x in (list(R.DATE_FIELDS) if df.field == "any" else [df.field])]
        if f.amounts:
            names += list(R.AMOUNT_FIELDS)
        if f.jurisdictions:
            names.append("jurisdiction")
        if f.signer or f.has_signature:
            names += ["signer", "signature_mark"]
        evidence: dict[str, list[EvidenceSpan]] = {}
        if names:
            rows = rc.conn.execute(f"SELECT {_SPAN_COLS} FROM fields WHERE document_id = ANY(%s::uuid[]) AND name = ANY(%s) "
                                   "ORDER BY document_id, confidence DESC", (list(scope.document_ids), sorted(set(names)))).fetchall()
            for r in rows:
                evidence.setdefault(str(r["document_id"]), []).extend(_span(r)[:1])
        if f.clause_types:
            rows = rc.conn.execute(f"SELECT {_SPAN_COLS} FROM clauses WHERE document_id = ANY(%s::uuid[]) AND clause_type = ANY(%s)",
                                   (list(scope.document_ids), f.clause_types)).fetchall()
            for r in rows:
                evidence.setdefault(str(r["document_id"]), []).extend(_span(r)[:1])
        return [Candidate(d, None, self.name, "structured", 1.0, evidence.get(d, [])[:3], ()) for d in scope.document_ids]
