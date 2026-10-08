"""Ingestion: register an upload, then process it into the canonical model and every index.

``register`` stores the original (content-addressed, never deleted by processing) and creates the document row.
``process`` parses, OCRs, chunks, understands, embeds and indexes it. Processing is idempotent: retries and
reprocessing replace all derived rows and vectors of the document.
"""
from __future__ import annotations

import contextlib
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import psycopg

from docintel import metrics, settings_store
from docintel.config import get_settings
from docintel.indexing.embeddings import EmbeddingError, get_embedder
from docintel.indexing.vectors import get_vector_store
from docintel.model_registry import model_spec
from docintel.models import Chunk, ParsedDocument, Understanding
from docintel.packs import Domain, get_domain
from docintel.processing.chunking import chunk_pages, document_unit
from docintel.processing.detect import MIME, detect
from docintel.processing.parsers import ParseError, parse_file
from docintel.security import AuthContext
from docintel.storage import repo
from docintel.storage.db import get_db
from docintel.storage.objects import get_object_store
from docintel.understanding import analyze

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "2.0.0"
DOCUMENT_VECTOR_MIN_PASSAGES = 3


class TransientError(RuntimeError):
    """A dependency is temporarily unavailable; the job should be retried."""


class VectorIndexError(RuntimeError):
    """Writing vectors failed (the vector index is down or refused the write)."""


@dataclass
class Registered:
    document: dict
    duplicate: bool


def safe_filename(name: str | None) -> str:
    """The last path component, without control characters or quotes (it ends up in headers and logs)."""
    cleaned = "".join(ch for ch in (name or "") if ch.isprintable() and ch not in '"<>')
    base = Path(cleaned.replace("\\", "/")).name.strip()
    return base[:255] if base not in ("", ".", "..") else "document"


def register(ctx: AuthContext, filename: str, stream: BinaryIO, *, collection: str | None = None,
             source: str = "upload", source_url: str | None = None) -> Registered:
    s = get_settings()
    filename = safe_filename(filename)
    store = get_object_store()
    key, sha, size = store.put_stream(ctx.tenant_id, stream, s.max_upload_bytes)
    try:
        if size == 0:
            raise ValueError("empty file")
        fmt = detect(store.path(key), filename)
    except ValueError as e:
        with get_db().tenant(ctx.tenant_id) as conn:
            referenced = repo.find_by_sha(conn, sha)
        if not referenced:
            store.delete(key)
        raise ParseError(str(e)) from e
    db = get_db()
    with db.tenant(ctx.tenant_id) as conn:
        existing = repo.find_by_sha(conn, sha)
        if existing:
            return Registered(existing, True)
        doc = repo.insert_document(conn, tenant_id=ctx.tenant_id, filename=filename, size_bytes=size, sha256=sha,
                                   storage_key=key, mime_type=MIME.get(fmt), file_ext=Path(filename).suffix.lower() or None,
                                   source=source, source_url=source_url, collection=collection)
        if doc is None:                                   # concurrent upload of the same content
            return Registered(repo.find_by_sha(conn, sha), True)
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                     (doc["id"], ctx.tenant_id))
        repo.audit(conn, ctx.tenant_id, ctx.principal, "document.upload", str(doc["id"]),
                   {"filename": filename, "size": size, "source": source})
    return Registered(doc, False)


def _embedding_input(title: str, chunk: Chunk) -> str:
    """Unit text with its context (document title, heading or table headers), as embedded."""
    if chunk.unit_type == "document":
        return chunk.text
    parts = [title] if title else []
    if chunk.unit_type == "table_row" and chunk.context:
        parts.append(chunk.context)
    else:
        if chunk.heading and chunk.heading not in chunk.text[:200]:
            parts.append(chunk.heading)
        parts.append(chunk.text)
    return "\n".join(parts)


def components(domain: Domain, parsed: ParsedDocument | None = None) -> dict:
    """What produced a document version: compared with the current configuration to find stale documents."""
    s = get_settings()
    out = {"pipeline": PIPELINE_VERSION, "packs": list(domain.packs),
           "embedding_model": s.embedding_model if s.semantic_enabled else None}
    if parsed is not None:
        out["parser"] = parsed.parser
        if any(p.kind == "ocr" for p in parsed.pages):
            from docintel.processing.ocr import get_ocr
            out["ocr"] = get_ocr().name
    else:                                   # the configuration documents are compared against
        import fitz
        out["pdf_parser"] = f"pymupdf-{fitz.VersionBind}"
        try:
            from docintel.processing.ocr import get_ocr
            out["ocr"] = get_ocr().name
        except Exception:                   # no OCR engine on this host: OCR staleness is not judged
            logger.debug("OCR engine unavailable; OCR staleness not checked", exc_info=True)
    return out


def stale_reasons(stored: dict | None, current: dict) -> list[str]:
    """Why a document's stored version differs from what processing would produce now."""
    if not stored:
        return ["processed before versioning"]
    out = [f"{k} changed" for k in ("pipeline", "packs", "embedding_model") if stored.get(k) != current.get(k)]
    if stored.get("ocr") and current.get("ocr") and stored["ocr"] != current["ocr"]:   # engine version or languages
        out.append("ocr changed")
    parser = str(stored.get("parser") or "")
    if parser.startswith("pymupdf-") and current.get("pdf_parser") and parser != current["pdf_parser"]:
        out.append("parser changed")
    return out


def _key_fields(understanding: Understanding) -> list[str]:
    keep = ("invoice_number", "contract_number", "po_number", "reference_number", "total_amount", "issue_date",
            "expiry_date", "jurisdiction", "party", "signer")
    out = []
    for f in understanding.fields:
        if f.name in keep and f.value_text and len(out) < 12:
            out.append(f"{f.name.replace('_', ' ')}: {f.value_text}")
    return out


def process(tenant_id: str, document_id: str) -> dict:
    """Process one document into its canonical model and every index. Raises TransientError for retryable
    dependency failures; any other failure marks the document failed with the reason. The document is marked
    indexed only after the database rows and the vectors are both written."""
    s = get_settings()
    db, store = get_db(), get_object_store()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    with db.system() as conn:
        conn.execute("UPDATE ingest_jobs SET state = 'running', attempts = attempts + 1, started_at = now() "
                     "WHERE document_id = %s", (document_id,))
    with db.tenant(tenant_id) as conn:
        doc = repo.get_document(conn, document_id)
        if doc is None:
            _finish_job(document_id)
            return {"document_id": document_id, "status": "missing"}
        repo.set_status(conn, document_id, "processing")
        domain = get_domain(tuple(settings_store.enabled_packs(conn)))
        version = repo.start_version(conn, tenant_id, document_id, components(domain))
    work = Path(tempfile.mkdtemp(prefix="docintel-"))
    try:
        local = work / Path(doc["filename"]).name
        store.copy_to(doc["storage_key"], local)
        t = time.perf_counter()
        parsed = parse_file(local, doc["filename"])
        timings["parse"] = time.perf_counter() - t
        spec = model_spec(s.embedding_model)
        embedder = get_embedder() if s.semantic_enabled else None
        t = time.perf_counter()
        understanding = analyze(parsed, doc["filename"], embed=embedder.embed if embedder else None,
                                embed_key=spec.key, domain=domain)
        timings["understand"] = time.perf_counter() - t
        title = understanding.title or doc["filename"]
        chunks = chunk_pages(parsed.pages, spec.chunk_chars)
        chunks.append(document_unit(len(chunks), title, doc["filename"],
                                    domain.type_label(understanding.classification.label), _key_fields(understanding)))
        chunk_ids = [f"{document_id}:{c.ordinal}" for c in chunks]
        # a short document's passages already carry its title, so a summary vector would only repeat them (and double
        # the embedding cost); longer documents also get one for the document as a whole
        passages = sum(1 for c in chunks if c.unit_type == "passage")
        kinds = ("passage", "document") if passages >= DOCUMENT_VECTOR_MIN_PASSAGES else ("passage",)
        embedded = [(cid, c) for cid, c in zip(chunk_ids, chunks, strict=True) if c.unit_type in kinds]
        vectors = None
        if embedder is not None:
            t = time.perf_counter()
            vectors = embedder.embed([_embedding_input(title, c) for _, c in embedded])
            timings["embed"] = time.perf_counter() - t
        words = sum(len(p.text.split()) for p in parsed.pages)
        comp = components(domain, parsed)
        t = time.perf_counter()
        with db.tenant(tenant_id) as conn:
            repo.replace_content(conn, tenant_id=tenant_id, doc_id=document_id, content=repo.Content(
                parsed.pages, chunks, chunk_ids, understanding.fields, understanding.entities, understanding.clauses,
                understanding.relations))
            repo.set_status(conn, document_id, "processing", kind=parsed.kind, page_count=len(parsed.pages),
                            word_count=words, language=understanding.language, title=title[:300],
                            doc_type=understanding.classification.label,
                            doc_type_confidence=understanding.classification.confidence,
                            doc_type_method=understanding.classification.method,
                            has_signature=understanding.has_signature)
        timings["store"] = time.perf_counter() - t
        if vectors is not None:
            t = time.perf_counter()
            try:
                get_vector_store().upsert(tenant_id, document_id, understanding.classification.label,
                                          [cid for cid, _ in embedded], vectors)
            except ValueError:                            # invalid input to the store: retrying cannot help
                raise
            except Exception as e:                        # an index outage is retried, never a permanent failure
                raise VectorIndexError(f"{type(e).__name__}: {e}") from e
            timings["vectors"] = time.perf_counter() - t
        ms = int((time.perf_counter() - started) * 1000)
        with db.tenant(tenant_id) as conn:
            repo.set_status(conn, document_id, "indexed", error=("; ".join(parsed.warnings)[:500] or None),
                            processing_ms=ms, pipeline_version=PIPELINE_VERSION, components=json.dumps(comp),
                            current_version=version)
            repo.finish_version(conn, document_id, version, "current", comp)
        _finish_job(document_id)
        metrics.observe_document(parsed.kind, len(parsed.pages), timings, "indexed")
        logger.info("document indexed", extra={"tenant_id": tenant_id, "document_id": document_id, "pages": len(parsed.pages),
                                               "units": len(chunks), "ms": ms, "doc_type": understanding.classification.label})
        return {"document_id": document_id, "status": "indexed", "pages": len(parsed.pages), "units": len(chunks), "ms": ms}
    except ParseError as e:
        _fail(tenant_id, document_id, str(e), version=version)
        metrics.observe_document("unknown", 0, timings, "failed")
        return {"document_id": document_id, "status": "failed", "error": str(e)}
    except psycopg.OperationalError as e:               # the database went away mid-run (restart, failover)
        logger.warning("database unavailable during processing", extra={"document_id": document_id})
        with contextlib.suppress(psycopg.OperationalError):   # still down: the job row is untouched and is retried
            _fail(tenant_id, document_id, f"database unavailable: {type(e).__name__}", keep_job=True, version=version)
        metrics.observe_document("unknown", 0, timings, "retry")
        raise TransientError(str(e)) from e
    except VectorIndexError as e:
        _fail(tenant_id, document_id, f"vector index unavailable: {e}", keep_job=True, version=version)
        metrics.observe_document("unknown", 0, timings, "retry")
        raise TransientError(str(e)) from e
    except EmbeddingError as e:
        _fail(tenant_id, document_id, f"embedding service unavailable: {e}", keep_job=True, version=version)
        metrics.observe_document("unknown", 0, timings, "retry")
        raise TransientError(str(e)) from e
    except Exception as e:  # unexpected: record and surface; the document is never left looking indexed
        logger.exception("document processing failed", extra={"tenant_id": tenant_id, "document_id": document_id})
        _fail(tenant_id, document_id, f"processing error: {type(e).__name__}: {e}", version=version)
        metrics.observe_document("unknown", 0, timings, "failed")
        return {"document_id": document_id, "status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fail(tenant_id: str, document_id: str, error: str, keep_job: bool = False, version: int | None = None) -> None:
    with get_db().tenant(tenant_id) as conn:
        repo.set_status(conn, document_id, "failed", error=error[:1000])
        if version is not None:
            repo.finish_version(conn, document_id, version, "failed", None, error[:1000])
    if not keep_job:
        _finish_job(document_id)


def _finish_job(document_id: str) -> None:
    with get_db().system() as conn:
        # a reprocess requested while this run was going re-enqueued the job: leave that row for the next run
        conn.execute("DELETE FROM ingest_jobs WHERE document_id = %s AND (started_at IS NULL OR enqueued_at <= started_at)",
                     (document_id,))


def requeue(ctx: AuthContext, document_id: str) -> dict | None:
    with get_db().tenant(ctx.tenant_id) as conn:
        doc = repo.get_document(conn, document_id)
        if not doc:
            return None
        repo.set_status(conn, document_id, "queued")
        conn.execute("INSERT INTO ingest_jobs (document_id, tenant_id) VALUES (%s, %s) ON CONFLICT (document_id) "
                     "DO UPDATE SET state = 'queued', enqueued_at = now()", (document_id, ctx.tenant_id))
        repo.audit(conn, ctx.tenant_id, ctx.principal, "document.reprocess", document_id)
        return repo.get_document(conn, document_id)


def delete(ctx: AuthContext, document_id: str) -> bool:
    db = get_db()
    with db.tenant(ctx.tenant_id) as conn:
        doc = repo.get_document(conn, document_id)
        if not doc:
            return False
    get_vector_store().delete_document(ctx.tenant_id, document_id)
    with db.tenant(ctx.tenant_id) as conn:
        removed = repo.delete_document(conn, document_id)
        repo.audit(conn, ctx.tenant_id, ctx.principal, "document.delete", document_id, {"filename": doc["filename"]})
    with db.system() as conn:
        conn.execute("DELETE FROM ingest_jobs WHERE document_id = %s", (document_id,))
    if removed:
        get_object_store().delete(removed["storage_key"])
    return bool(removed)
