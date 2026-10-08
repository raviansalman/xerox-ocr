"""Candidate generators. Every function receives a tenant-scoped connection (row-level security) and, for the
vector index, the tenant id, so no candidate from another tenant can enter a result."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import psycopg

from docintel import text as T
from docintel.search.plan import Filters, QueryPlan
from docintel.understanding.entities import norm_name, norm_org

# Match types and their tier (0 = exact, 1 = strong lexical, 2 = tolerant). Semantic-only hits are tier 1 when above
# the model's similarity floor.
TIER = {"exact_phrase": 0, "exact_identifier": 0, "filename": 0, "exact_term": 1, "joined_form": 1,
        "partial_identifier": 1, "all_terms": 1, "filename_terms": 1, "ocr_tolerant": 2, "fuzzy": 2}
DATE_FIELDS = ("issue_date", "effective_date", "expiry_date", "due_date", "signature_date", "date")
AMOUNT_FIELDS = ("total_amount", "amount")


@dataclass
class LexHit:
    chunk_id: str
    document_id: str
    page: int
    text: str
    match_type: str
    rank: float


@dataclass
class Where:
    sql: str = "d.status = 'indexed'"
    args: list = field(default_factory=list)

    def add(self, clause: str, *args) -> None:
        self.sql += f" AND {clause}"
        self.args.extend(args)


_AMOUNT_OPS = frozenset({">", ">=", "<", "<="})


def filter_where(f: Filters | None, include_soft_types: bool = False) -> Where:
    """SQL condition on ``documents d`` for structured filters (parameterized; no string interpolation of values)."""
    w = Where()
    if not f:
        return w
    if f.doc_types and (f.doc_types_hard or include_soft_types):
        w.add("d.doc_type = ANY(%s)", list(f.doc_types))
    if f.kinds:
        w.add("d.kind = ANY(%s)", list(f.kinds))
    if f.extensions:
        w.add("d.file_ext = ANY(%s)", list(f.extensions))
    if f.has_signature is True:
        w.add("d.has_signature IS TRUE")
    elif f.has_signature is False:
        w.add("d.has_signature IS NOT TRUE")
    if f.jurisdictions:
        w.add("EXISTS (SELECT 1 FROM fields x WHERE x.document_id = d.id AND x.name = 'jurisdiction' AND x.value_text = ANY(%s))",
              list(f.jurisdictions))
    if f.signer:
        n = norm_name(f.signer)
        w.add("EXISTS (SELECT 1 FROM entities e WHERE e.document_id = d.id AND e.type = 'person' AND e.role = 'signer' "
              "AND (e.value_norm = %s OR e.value_norm LIKE %s OR %s LIKE e.value_norm || ' %%'))", n, f"%{n}%", n)
    for party in f.parties:
        n, ph = norm_org(party), T.search_text(party)
        w.add("(EXISTS (SELECT 1 FROM entities e WHERE e.document_id = d.id AND e.type IN ('organization', 'person') "
              "AND e.value_norm LIKE %s) OR EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id "
              "AND c.tsv @@ phraseto_tsquery('simple', %s)))", f"%{n}%", ph)
    for df in f.dates:
        names = list(DATE_FIELDS) if df.field == "any" else [df.field]
        cond = "EXISTS (SELECT 1 FROM fields x WHERE x.document_id = d.id AND x.name = ANY(%s)"
        args: list = [names]
        if df.start:
            cond += " AND x.value_date >= %s"
            args.append(df.start)
        if df.end:
            cond += " AND x.value_date <= %s"
            args.append(df.end)
        w.add(cond + ")", *args)
    for af in f.amounts:
        cond = "EXISTS (SELECT 1 FROM fields x WHERE x.document_id = d.id AND x.name = ANY(%s)"
        args = [list(AMOUNT_FIELDS)]
        if af.op == "between":
            cond += " AND x.value_num BETWEEN %s AND %s"
            args += [min(af.value, af.value2 or af.value), max(af.value, af.value2 or af.value)]
        else:
            if af.op not in _AMOUNT_OPS:                     # never interpolate anything outside the closed set
                raise ValueError(f"unsupported amount operator {af.op!r}")
            cond += f" AND x.value_num {af.op} %s"
            args.append(af.value)
        if af.currency:
            cond += " AND x.unit = %s"
            args.append(af.currency)
        w.add(cond + ")", *args)
    if f.clause_types:
        w.add("EXISTS (SELECT 1 FROM clauses k WHERE k.document_id = d.id AND k.clause_type = ANY(%s))", list(f.clause_types))
    return w


def allowed_documents(conn: psycopg.Connection, where: Where, limit: int | None = None) -> list[str] | None:
    """Every document allowed by the structured filters (complete: a truncated list would silently narrow the
    search), or None when there are no filters beyond status. ``limit`` is accepted for compatibility and ignored."""
    if where.sql == "d.status = 'indexed'":
        return None
    rows = conn.execute(f"SELECT d.id FROM documents d WHERE {where.sql}", where.args).fetchall()
    return [str(r["id"]) for r in rows]


def _doc_scope(allowed: list[str] | None) -> tuple[str, list]:
    if allowed is None:
        return "c.document_id IN (SELECT id FROM documents WHERE status = 'indexed')", []
    return "c.document_id = ANY(%s::uuid[])", [allowed]


def lexical(conn: psycopg.Connection, plan: QueryPlan, allowed: list[str] | None, limit: int) -> list[LexHit]:
    """Exact and lexical candidates, independent of vector similarity.

    All tiers are evaluated in a single pass over the tenant's chunks: under row-level security PostgreSQL scans
    the tenant's rows for these operators anyway, so one pass that tests every tier costs far less than one
    statement per tier. Each chunk is reported once, with its best (lowest) tier."""
    words = T.tokens(plan.text)
    if not words and not plan.identifiers:
        return []
    qs = " ".join(words)
    tiers: list[tuple[str, str, list]] = []         # (match type, SQL condition, args)
    rank_sql, rank_args = "0", []
    # 0. quoted phrases are mandatory exact phrases
    for ph in plan.phrases:
        tiers.append(("exact_phrase", "c.tsv @@ phraseto_tsquery('simple', %s)", [T.search_text(ph)]))
    # 1. identifiers in canonical form (INV-1024-77 == inv102477)
    if plan.identifiers:
        tiers.append(("exact_identifier", "c.idents && %s::text[]", [list(plan.identifiers)]))
    # 2. the whole residual text as an exact phrase (single word: exact token)
    if words:
        tiers.append(("exact_phrase" if len(words) > 1 else "exact_term", "c.tsv @@ phraseto_tsquery('simple', %s)", [qs]))
        rank_sql, rank_args = "ts_rank_cd(c.tsv_en, plainto_tsquery('english', %s))", [qs]
    # 3. joined/split variants ("Data Vault" ~ "DataVault")
    for v in T.join_variants(words):
        tiers.append(("joined_form", "c.tsv @@ phraseto_tsquery('simple', %s)", [v]))
    # 4. partial identifiers ("00481")
    for w in words:
        if len(w) >= 4 and any(ch.isdigit() for ch in w) and not re.fullmatch(r"(?:19|20)\d{2}", w):
            tiers.append(("partial_identifier", "c.search_text ILIKE %s", [f"%{w}%"]))
    # 5. all terms (English stemming)
    if len(T.content_words(qs)) >= 1 and len(words) > 1:
        tiers.append(("all_terms", "c.tsv_en @@ plainto_tsquery('english', %s)", [qs]))
    # 6. OCR-tolerant phrase (both sides folded)
    tiers.append(("ocr_tolerant", "c.tsv_fold @@ phraseto_tsquery('simple', %s)", [T.fold(qs)]))

    scope, sargs = _doc_scope(allowed)
    hits = _tiered(conn, tiers, scope, sargs, rank_sql, rank_args, limit)
    # 7. typo tolerance for names and rare words, only when nothing matched lexically
    if not hits:
        long_words = [w for w in words if len(w) >= 5 and not w.isdigit()]
        if long_words and len(words) <= 6:
            conds, args = [], []
            for w in words:
                if w in long_words:
                    conds.append("(c.tsv @@ plainto_tsquery('simple', %s) OR %s <%% c.search_text)")
                    args += [w, w]
                elif w not in T.STOPWORDS:
                    conds.append("c.tsv @@ plainto_tsquery('simple', %s)")
                    args.append(w)
            conn.execute("SET LOCAL pg_trgm.word_similarity_threshold = 0.6")
            hits = _tiered(conn, [("fuzzy", " AND ".join(conds), args)], scope, sargs, "0", [], 50)
    return hits


def _tiered(conn: psycopg.Connection, tiers: list[tuple[str, str, list]], scope: str, sargs: list,
            rank_sql: str, rank_args: list, limit: int) -> list[LexHit]:
    """One statement: every chunk matching any tier, labelled with the first tier it matches, best tiers first."""
    if not tiers:
        return []
    case = " ".join(f"WHEN {cond} THEN {i}" for i, (_, cond, _) in enumerate(tiers))
    case_args = [a for _, _, args in tiers for a in args]
    # the rank is computed in the outer query, so only for matching chunks
    sql = (f"SELECT id, document_id, page_start, text, tier, {rank_sql.replace('c.', 'm.')} AS rank FROM ("
           f"  SELECT c.id, c.document_id, c.page_start, c.text, c.tsv_en, CASE {case} END AS tier"
           f"  FROM chunks c WHERE {scope}) m "
           f"WHERE tier IS NOT NULL ORDER BY tier, rank DESC LIMIT %s")
    rows = conn.execute(sql, rank_args + case_args + sargs + [limit * 4]).fetchall()
    return [LexHit(r["id"], str(r["document_id"]), r["page_start"], r["text"], tiers[r["tier"]][0], float(r["rank"] or 0))
            for r in rows]


def filename_hits(conn: psycopg.Connection, plan: QueryPlan, allowed: list[str] | None, limit: int) -> list[tuple[str, str]]:
    """(document_id, match_type) for documents whose file name contains the query."""
    qs = T.search_text(" ".join([*plan.phrases, plan.text])).strip()
    if len(qs) < 3:
        return []
    cond, args = "d.status = 'indexed'", []
    if allowed is not None:
        cond += " AND d.id = ANY(%s::uuid[])"
        args.append(allowed)
    out = [(str(r["id"]), "filename") for r in conn.execute(
        f"SELECT d.id FROM documents d WHERE {cond} AND d.filename_search LIKE %s LIMIT %s", args + [f"%{qs}%", limit]).fetchall()]
    words = [w for w in qs.split() if w not in T.STOPWORDS]
    if len(words) > 1:
        like = " AND ".join(["d.filename_search LIKE %s"] * len(words))
        seen = {d for d, _ in out}
        for r in conn.execute(f"SELECT d.id FROM documents d WHERE {cond} AND {like} LIMIT %s",
                              args + [f"%{w}%" for w in words] + [limit]).fetchall():
            if str(r["id"]) not in seen:
                out.append((str(r["id"]), "filename_terms"))
    return out


def chunk_texts(conn: psycopg.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    rows = conn.execute("SELECT id, document_id, page_start, text FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()
    return {r["id"]: r for r in rows}


def first_chunks(conn: psycopg.Connection, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    rows = conn.execute("SELECT DISTINCT ON (document_id) id, document_id, page_start, text FROM chunks "
                        "WHERE document_id = ANY(%s::uuid[]) ORDER BY document_id, ordinal", (doc_ids,)).fetchall()
    return {str(r["document_id"]): r for r in rows}


def key_fields(conn: psycopg.Connection, doc_ids: list[str]) -> dict[str, dict]:
    """A compact set of extracted fields per document for result cards."""
    if not doc_ids:
        return {}
    rows = conn.execute(
        "SELECT document_id, name, value_text, page, confidence FROM fields WHERE document_id = ANY(%s::uuid[]) "
        "AND name IN ('invoice_number','contract_number','po_number','issue_date','effective_date','expiry_date',"
        "'due_date','total_amount','jurisdiction','signer','party') ORDER BY document_id, confidence DESC, id",
        (doc_ids,)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        d = out.setdefault(str(r["document_id"]), {})
        d.setdefault(r["name"], r["value_text"])
    return out


def clause_snippets(conn: psycopg.Connection, doc_ids: list[str], clause_types: list[str]) -> dict[str, list[dict]]:
    if not doc_ids or not clause_types:
        return {}
    rows = conn.execute("SELECT document_id, clause_type, ref, heading, page, text, confidence FROM clauses "
                        "WHERE document_id = ANY(%s::uuid[]) AND clause_type = ANY(%s) ORDER BY document_id, page, id",
                        (doc_ids, clause_types)).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r["document_id"]), []).append(r)
    return out
