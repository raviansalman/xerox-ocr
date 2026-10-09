"""Data access for documents and their derived content. Every function takes a tenant-scoped connection from
``Database.tenant``; row-level security guarantees it only sees that tenant's rows."""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import psycopg

from docintel import text as T
from docintel.models import Chunk, Clause, Entity, FieldValue, Page, Relation

DOC_COLUMNS = ("id, tenant_id, collection, filename, title, mime_type, file_ext, size_bytes, sha256, storage_key, "
               "source, source_url, status, error, kind, page_count, word_count, language, doc_type, "
               "doc_type_confidence, doc_type_method, has_signature, pipeline_version, processing_ms, created_at, "
               "updated_at, indexed_at, current_version, components")


def new_id() -> uuid.UUID:
    return uuid.uuid4()


def insert_document(conn: psycopg.Connection, *, tenant_id: str, filename: str, size_bytes: int, sha256: str,
                    storage_key: str, mime_type: str | None, file_ext: str | None, source: str = "upload",
                    source_url: str | None = None, collection: str | None = None) -> dict:
    doc_id = new_id()
    row = conn.execute(
        f"""INSERT INTO documents (id, tenant_id, collection, filename, filename_search, mime_type, file_ext,
                size_bytes, sha256, storage_key, source, source_url, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued')
            ON CONFLICT (tenant_id, sha256) DO NOTHING
            RETURNING {DOC_COLUMNS}""",
        (doc_id, tenant_id, collection, filename, T.search_text(filename), mime_type, file_ext, size_bytes, sha256,
         storage_key, source, source_url)).fetchone()
    return row


def find_by_sha(conn: psycopg.Connection, sha256: str) -> dict | None:
    return conn.execute(f"SELECT {DOC_COLUMNS} FROM documents WHERE sha256 = %s", (sha256,)).fetchone()


def get_document(conn: psycopg.Connection, doc_id: str | uuid.UUID) -> dict | None:
    return conn.execute(f"SELECT {DOC_COLUMNS} FROM documents WHERE id = %s", (str(doc_id),)).fetchone()


def get_documents(conn: psycopg.Connection, ids: Iterable[str]) -> dict[str, dict]:
    ids = [str(i) for i in ids]
    if not ids:
        return {}
    rows = conn.execute(f"SELECT {DOC_COLUMNS} FROM documents WHERE id = ANY(%s::uuid[])", (ids,)).fetchall()
    return {str(r["id"]): r for r in rows}


def list_documents(conn: psycopg.Connection, *, status: str | None = None, doc_type: str | None = None,
                   q: str | None = None, collection: str | None = None, limit: int = 50, offset: int = 0
                   ) -> tuple[list[dict], int]:
    where, args = ["true"], []
    if status:
        where.append("status = %s")
        args.append(status)
    if doc_type:
        where.append("doc_type = %s")
        args.append(doc_type)
    if collection:
        where.append("collection = %s")
        args.append(collection)
    if q:
        where.append("filename_search ILIKE %s")
        args.append(f"%{T.search_text(q)}%")
    w = " AND ".join(where)
    total = conn.execute(f"SELECT count(*) AS n FROM documents WHERE {w}", args).fetchone()["n"]
    rows = conn.execute(f"SELECT {DOC_COLUMNS} FROM documents WHERE {w} ORDER BY created_at DESC, id LIMIT %s OFFSET %s",
                        args + [limit, offset]).fetchall()
    return rows, total


_STATUS_EXTRA = frozenset({"kind", "page_count", "word_count", "language", "title", "doc_type", "doc_type_confidence",
                           "doc_type_method", "has_signature", "processing_ms", "pipeline_version", "components",
                           "current_version"})


def set_status(conn: psycopg.Connection, doc_id: str, status: str, error: str | None = None, **extra: Any) -> None:
    cols = ["status = %s", "error = %s", "updated_at = now()"]
    args: list[Any] = [status, error]
    for k, v in extra.items():
        if k not in _STATUS_EXTRA:
            raise ValueError(f"unknown document column {k!r}")
        cols.append(f"{k} = %s")
        args.append(v)
    if status == "indexed":
        cols.append("indexed_at = now()")
    conn.execute(f"UPDATE documents SET {', '.join(cols)} WHERE id = %s", args + [str(doc_id)])


@dataclass
class Content:
    """Everything processing derives from one document version."""
    pages: list[Page]
    chunks: list[Chunk]
    chunk_ids: list[str]
    fields: list[FieldValue]
    entities: list[Entity]
    clauses: list[Clause]
    relations: list[Relation] = dc_field(default_factory=list)


_NUM = re.compile(r"^[^\d\-]{0,4}(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*%?[^\d]{0,4}$")


def _cell_number(text: str) -> float | None:
    m = _NUM.match(text.strip())
    if not m:
        return None
    try:
        return float(re.sub(r"[,\s]", "", m.group(1)))
    except ValueError:
        return None


def _span_cols(span) -> tuple:
    return (span.char_start, span.char_end, span.block) if span else (None, None, None)


DERIVED_TABLES = ("pages", "chunks", "fields", "entities", "clauses", "blocks", "doc_tables", "table_cells", "relations",
                  "unit_terms")


def replace_content(conn: psycopg.Connection, *, tenant_id: str, doc_id: str, content: Content) -> None:
    """Replace every derived row of a document in one transaction (idempotent for retries and reprocessing):
    pages, blocks, tables and cells, units, annotations with their spans, relations, and the lexical postings.
    Writers of the same document are serialized (a redelivered job, or a reprocess while processing), so the last
    one replaces the content instead of colliding with it."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('docintel.content:' || %s, 0))", (str(doc_id),))
    old_terms = document_words(conn, doc_id)
    for table in DERIVED_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE document_id = %s", (doc_id,))
    pages = content.pages
    _bulk(conn, "pages", ("document_id", "tenant_id", "page_number", "kind", "width", "height", "ocr_confidence", "text", "words"),
          [(doc_id, tenant_id, p.number, p.kind, p.width, p.height, p.ocr_confidence, p.text,
            json.dumps([[w.text, round(w.x0, 1), round(w.y0, 1), round(w.x1, 1), round(w.y1, 1), round(w.conf, 1)]
                        for w in p.words]) if p.words else None) for p in pages])
    block_rows, table_rows, cell_rows = [], [], []
    for p in pages:
        for i, (b, (start, _end)) in enumerate(zip(p.blocks, p.block_offsets(), strict=True)):
            block_rows.append((doc_id, tenant_id, p.number, i, b.type, b.text, start, list(b.bbox) if b.bbox else None,
                               b.confidence, b.table))
        table_block = {b.table: i for i, b in enumerate(p.blocks) if b.type == "table" and b.table is not None}
        for ti, t in enumerate(p.tables):
            table_rows.append((doc_id, tenant_id, p.number, ti, table_block.get(ti), len(t.rows), t.n_cols, t.header_rows,
                               list(t.bbox) if t.bbox else None, t.caption))
            header = t.rows[t.header_rows - 1] if 0 < t.header_rows <= len(t.rows) else []
            for r, row in enumerate(t.rows):
                for c, val in enumerate(row):
                    if not val:
                        continue
                    h = header[c] if r >= t.header_rows and c < len(header) and header[c] else None
                    cell_rows.append((doc_id, tenant_id, p.number, ti, r, c, val, h, _cell_number(val) if h else None))
    _bulk(conn, "blocks", ("document_id", "tenant_id", "page", "ordinal", "block_type", "text", "char_start", "bbox",
                           "confidence", "table_ordinal"), block_rows)
    _bulk(conn, "doc_tables", ("document_id", "tenant_id", "page", "ordinal", "block_ordinal", "n_rows", "n_cols",
                               "header_rows", "bbox", "caption"), table_rows)
    _bulk(conn, "table_cells", ("document_id", "tenant_id", "page", "table_ordinal", "row_index", "col_index", "text",
                                "header", "value_num"), cell_rows)
    rows = []
    for cid, c in zip(content.chunk_ids, content.chunks, strict=True):
        searchable = f"{c.context} {c.text}" if c.context else c.text
        rows.append((cid, doc_id, tenant_id, c.ordinal, c.page_start, c.page_end, c.heading, c.text,
                     T.search_text(searchable), T.stem_text(searchable), T.fold(searchable),
                     sorted(T.identifiers(searchable)), c.unit_type, c.char_start, c.char_end, c.block_from, c.block_to,
                     c.context))
    _bulk(conn, "chunks", ("id", "document_id", "tenant_id", "ordinal", "page_start", "page_end", "heading", "text",
                           "search_text", "stem_text", "fold_text", "idents", "unit_type", "char_start", "char_end",
                           "block_from", "block_to", "context"), rows)
    _bulk(conn, "fields", ("document_id", "tenant_id", "name", "value_text", "value_num", "value_date", "unit", "page",
                           "snippet", "confidence", "method", "char_start", "char_end", "block_ordinal"),
          [(doc_id, tenant_id, f.name, f.value_text, f.value_num, f.value_date, f.unit, f.page, f.snippet, f.confidence,
            f.method, *_span_cols(f.span)) for f in content.fields])
    _bulk(conn, "entities", ("document_id", "tenant_id", "type", "value", "value_norm", "role", "page", "snippet",
                             "confidence", "method", "char_start", "char_end", "block_ordinal"),
          [(doc_id, tenant_id, e.type, e.value, e.value_norm, e.role, e.page, e.snippet, e.confidence, e.method,
            *_span_cols(e.span)) for e in content.entities])
    _bulk(conn, "clauses", ("document_id", "tenant_id", "clause_type", "ref", "heading", "page", "text", "confidence",
                            "char_start", "char_end", "block_ordinal"),
          [(doc_id, tenant_id, c.clause_type, c.ref, c.heading, c.page, c.text, c.confidence, *_span_cols(c.span))
           for c in content.clauses])
    _bulk(conn, "relations", ("document_id", "tenant_id", "predicate", "subject_type", "subject", "subject_norm",
                              "object_type", "object", "object_norm", "qualifiers", "page", "char_start", "char_end",
                              "snippet", "confidence", "extractor"),
          [(doc_id, tenant_id, r.predicate, r.subject_type, r.subject[:500], T.search_text(r.subject)[:500], r.object_type,
            r.object[:500], T.search_text(r.object)[:500], json.dumps(r.qualifiers), r.span.page if r.span else None,
            r.span.char_start if r.span else None, r.span.char_end if r.span else None, r.snippet, r.confidence,
            r.extractor) for r in content.relations])
    build_postings(conn, doc_id)
    prune_vocabulary(conn, old_terms - document_words(conn, doc_id))


# Postings are generated from the units' stored full-text vectors, inside the database, so the lexical index always
# matches exactly what PostgreSQL's own text matching would find in the unit.
_POSTINGS_SQL = """
INSERT INTO unit_terms (tenant_id, term, unit_id, document_id, tf)
SELECT c.tenant_id, 'w:' || u.lexeme, c.id, c.document_id, least(32767, coalesce(array_length(u.positions, 1), 1))
  FROM chunks c, unnest(c.tsv) u WHERE c.document_id = %(doc)s
UNION ALL
SELECT c.tenant_id, 's:' || u.lexeme, c.id, c.document_id, least(32767, coalesce(array_length(u.positions, 1), 1))
  FROM chunks c, unnest(c.tsv_en) u WHERE c.document_id = %(doc)s
UNION ALL
SELECT c.tenant_id, 'f:' || u.lexeme, c.id, c.document_id, least(32767, coalesce(array_length(u.positions, 1), 1))
  FROM chunks c, unnest(c.tsv_fold) u WHERE c.document_id = %(doc)s AND NOT (u.lexeme = ANY (tsvector_to_array(c.tsv)))
UNION ALL
SELECT c.tenant_id, 'i:' || i, c.id, c.document_id, 1
  FROM chunks c, unnest(c.idents) i WHERE c.document_id = %(doc)s
UNION ALL
SELECT c.tenant_id, 'r:' || reverse(u.lexeme), c.id, c.document_id, 1
  FROM chunks c, unnest(c.tsv) u WHERE c.document_id = %(doc)s AND u.lexeme ~ '[0-9]' AND length(u.lexeme) >= 3
UNION ALL
SELECT c.tenant_id, 'n:' || u.lexeme, c.id, c.document_id, 1
  FROM chunks c JOIN documents d ON d.id = c.document_id, unnest(to_tsvector('simple', d.filename_search)) u
 WHERE c.document_id = %(doc)s AND c.unit_type = 'document'
"""


def build_postings(conn: psycopg.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM unit_terms WHERE document_id = %s", (doc_id,))
    conn.execute(_POSTINGS_SQL, {"doc": doc_id})
    conn.execute("UPDATE chunks c SET n_terms = coalesce((SELECT sum(coalesce(array_length(u.positions, 1), 1)) "
                 "FROM unnest(c.tsv) u), 0) WHERE c.document_id = %s", (doc_id,))
    conn.execute("INSERT INTO vocabulary (tenant_id, term, length) "
                 "SELECT DISTINCT tenant_id, substr(term, 3), length(term) - 2 FROM unit_terms "
                 "WHERE document_id = %s AND term >= 'w:' AND term < 'w;' AND length(term) BETWEEN 6 AND 34 "
                 "AND substr(term, 3) ~ '^[a-z]+$' ON CONFLICT DO NOTHING", (doc_id,))


def document_words(conn: psycopg.Connection, doc_id: str) -> set[str]:
    rows = conn.execute("SELECT DISTINCT substr(term, 3) AS w FROM unit_terms WHERE document_id = %s "
                        "AND term >= 'w:' AND term < 'w;'", (doc_id,)).fetchall()
    return {r["w"] for r in rows}


def prune_vocabulary(conn: psycopg.Connection, words: set[str]) -> None:
    """Remove words that no unit of the tenant contains any more (after a delete or reprocess)."""
    if not words:
        return
    # the posting keys are passed in rather than built in SQL: under row-level security a computed join key
    # ('w:' || v.term) cannot use the postings index, and each word no unit contains any more cost a scan of every
    # posting (4.7 s per delete at 50,000 documents; 2 ms this way)
    words = sorted(words)
    conn.execute("DELETE FROM vocabulary v USING unnest(%s::text[], %s::text[]) AS t(w, key) "
                 "WHERE v.term = t.w AND NOT EXISTS (SELECT 1 FROM unit_terms u WHERE u.term = t.key)",
                 (words, [f"w:{w}" for w in words]))


def _bulk(conn: psycopg.Connection, table: str, cols: tuple[str, ...], rows: list[tuple]) -> None:
    """Fast bulk insert that keeps row-level security: COPY into a temporary table (COPY is not allowed on RLS
    tables), then INSERT ... SELECT, which applies the tenant policy's WITH CHECK to every row."""
    if not rows:
        return
    collist = ", ".join(cols)
    tmp = f"_bulk_{table}"
    conn.execute(f"CREATE TEMP TABLE {tmp} ON COMMIT DROP AS SELECT {collist} FROM {table} WITH NO DATA")
    with conn.cursor() as cur:
        with cur.copy(f"COPY {tmp} ({collist}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
    conn.execute(f"INSERT INTO {table} ({collist}) SELECT {collist} FROM {tmp}")
    conn.execute(f"DROP TABLE {tmp}")


def delete_document(conn: psycopg.Connection, doc_id: str) -> dict | None:
    """Delete a document and, through foreign keys, every row derived from it; then prune vocabulary words that
    only it contained."""
    words = document_words(conn, doc_id)
    row = conn.execute(f"DELETE FROM documents WHERE id = %s RETURNING {DOC_COLUMNS}", (doc_id,)).fetchone()
    prune_vocabulary(conn, words)
    return row


def document_detail(conn: psycopg.Connection, doc_id: str) -> dict | None:
    doc = get_document(conn, doc_id)
    if not doc:
        return None
    doc = dict(doc)
    doc["fields"] = conn.execute(
        "SELECT name, value_text, value_num, value_date, unit, page, snippet, confidence, method, char_start, char_end, "
        "block_ordinal AS block FROM fields WHERE document_id = %s ORDER BY name, page NULLS LAST, id", (doc_id,)).fetchall()
    doc["entities"] = conn.execute(
        "SELECT type, value, role, page, snippet, confidence, method, char_start, char_end, block_ordinal AS block "
        "FROM entities WHERE document_id = %s ORDER BY type, value, id", (doc_id,)).fetchall()
    doc["clauses"] = conn.execute(
        "SELECT clause_type, ref, heading, page, text, confidence, char_start, char_end, block_ordinal AS block "
        "FROM clauses WHERE document_id = %s ORDER BY page, id", (doc_id,)).fetchall()
    doc["relations"] = conn.execute(
        "SELECT predicate, subject_type, subject, object_type, object, qualifiers, page, char_start, char_end, snippet, "
        "confidence, extractor FROM relations WHERE document_id = %s ORDER BY page NULLS LAST, id LIMIT 500",
        (doc_id,)).fetchall()
    doc["tables"] = conn.execute(
        "SELECT page, ordinal, n_rows, n_cols, header_rows, caption FROM doc_tables WHERE document_id = %s "
        "ORDER BY page, ordinal", (doc_id,)).fetchall()
    doc["versions"] = versions(conn, doc_id)
    doc["pages"] = conn.execute(
        "SELECT page_number, kind, ocr_confidence, length(text) AS chars FROM pages WHERE document_id = %s ORDER BY page_number",
        (doc_id,)).fetchall()
    return doc


def get_page(conn: psycopg.Connection, doc_id: str, page_number: int) -> dict | None:
    """A page with its canonical text, OCR words, typed blocks (offsets into the text, boxes) and tables with cells."""
    page = conn.execute("SELECT page_number, kind, width, height, ocr_confidence, text, words FROM pages "
                        "WHERE document_id = %s AND page_number = %s", (doc_id, page_number)).fetchone()
    if not page:
        return None
    page = dict(page)
    page["blocks"] = conn.execute(
        "SELECT ordinal, block_type AS type, char_start, char_start + length(text) AS char_end, bbox, confidence, "
        "table_ordinal AS table FROM blocks WHERE document_id = %s AND page = %s ORDER BY ordinal",
        (doc_id, page_number)).fetchall()
    tables = conn.execute("SELECT ordinal, n_rows, n_cols, header_rows, bbox, caption FROM doc_tables "
                          "WHERE document_id = %s AND page = %s ORDER BY ordinal", (doc_id, page_number)).fetchall()
    cells = conn.execute("SELECT table_ordinal, row_index, col_index, text FROM table_cells WHERE document_id = %s "
                         "AND page = %s ORDER BY table_ordinal, row_index, col_index", (doc_id, page_number)).fetchall()
    out = []
    for t in tables:
        grid = [[""] * t["n_cols"] for _ in range(t["n_rows"])]
        for c in cells:
            if c["table_ordinal"] == t["ordinal"] and c["row_index"] < t["n_rows"] and c["col_index"] < t["n_cols"]:
                grid[c["row_index"]][c["col_index"]] = c["text"]
        out.append({**t, "rows": grid})
    page["tables"] = out
    return page


def stats(conn: psycopg.Connection) -> dict:
    by_status = {r["status"]: r["n"] for r in conn.execute(
        "SELECT status, count(*) AS n FROM documents GROUP BY status").fetchall()}
    by_type = {r["doc_type"] or "unclassified": r["n"] for r in conn.execute(
        "SELECT doc_type, count(*) AS n FROM documents WHERE status = 'indexed' GROUP BY doc_type ORDER BY n DESC").fetchall()}
    totals = conn.execute("SELECT count(*) AS documents, coalesce(sum(page_count),0) AS pages FROM documents "
                          "WHERE status = 'indexed'").fetchone()
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return {"by_status": by_status, "by_type": by_type, "indexed_documents": totals["documents"],
            "indexed_pages": int(totals["pages"]), "chunks": chunks}


def audit(conn: psycopg.Connection, tenant_id: str, actor: str, action: str, target: str | None = None,
          detail: dict | None = None) -> None:
    conn.execute("INSERT INTO audit_events (tenant_id, actor, action, target, detail) VALUES (%s,%s,%s,%s,%s)",
                 (tenant_id, actor, action, target, json.dumps(detail) if detail else None))


def now() -> datetime:
    return datetime.now(UTC)


def start_version(conn: psycopg.Connection, tenant_id: str, doc_id: str, components: dict) -> int:
    """Record a new processing run of a document; returns its version number."""
    row = conn.execute(
        "INSERT INTO document_versions (document_id, tenant_id, number, status, components) "
        "SELECT %s, %s, coalesce(max(number), 0) + 1, 'building', %s FROM document_versions WHERE document_id = %s "
        "RETURNING number", (doc_id, tenant_id, json.dumps(components), doc_id)).fetchone()
    return int(row["number"])


def finish_version(conn: psycopg.Connection, doc_id: str, number: int, status: str, components: dict | None,
                   error: str | None = None) -> None:
    """Mark a run current (superseding the previous current one) or failed."""
    if status == "current":
        conn.execute("UPDATE document_versions SET status = 'superseded' WHERE document_id = %s AND status = 'current' "
                     "AND number <> %s", (doc_id, number))
    conn.execute("UPDATE document_versions SET status = %s, finished_at = now(), error = %s, "
                 "components = coalesce(%s::jsonb, components) WHERE document_id = %s AND number = %s",
                 (status, error, json.dumps(components) if components else None, doc_id, number))


def versions(conn: psycopg.Connection, doc_id: str) -> list[dict]:
    return conn.execute("SELECT number, status, components, started_at, finished_at, error FROM document_versions "
                        "WHERE document_id = %s ORDER BY number DESC LIMIT 20", (doc_id,)).fetchall()
