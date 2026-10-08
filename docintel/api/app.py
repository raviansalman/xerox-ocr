"""HTTP API and web UI.

All data routes require an API key; the tenant always comes from the key (``docintel.security``). Handlers are
synchronous functions, so FastAPI runs them in its thread pool and slow work never blocks the event loop.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from docintel import __version__, metrics, settings_store
from docintel.api.limits import BodyLimitMiddleware
from docintel.config import get_settings
from docintel.ingest import dispatch, pipeline
from docintel.logging_setup import configure_logging, request_id
from docintel.packs import Domain, available_packs, get_domain
from docintel.processing.parsers import ParseError
from docintel.query import get_engine
from docintel.security import AuthContext, AuthError, authenticate
from docintel.storage import repo
from docintel.storage.db import get_db
from docintel.storage.objects import get_object_store

logger = logging.getLogger("docintel.api")
STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    s = get_settings()
    s.validate_for_startup("api")
    db = get_db()
    if s.auto_migrate:
        db.migrate()
    claim = None
    if s.task_mode == "thread":
        claim = dispatch.claim_thread_mode()
        dispatch.recover()
    reaper = dispatch.start_reaper() if s.task_mode != "inline" else None
    logger.info("docintel started", extra={"version": __version__, "environment": s.environment, "task_mode": s.task_mode})
    yield
    if reaper is not None:
        reaper.set()
    if claim is not None:
        claim.close()


app = FastAPI(title="Document Intelligence Engine", version=__version__, lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
app.add_middleware(BodyLimitMiddleware)
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"],
                       allow_headers=["Authorization", "X-API-Key", "X-Tenant-Id", "Content-Type"])


_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


@app.middleware("http")
async def context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or ""
    if not _REQUEST_ID.fullmatch(rid):              # never echo or log arbitrary client text
        rid = uuid.uuid4().hex[:16]
    token = request_id.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
                                                       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
    if not request.url.path.startswith(("/static", "/health")):
        logger.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                                      "ms": round((time.perf_counter() - t0) * 1000, 1), "request_id": rid})
    return response


@app.exception_handler(AuthError)
async def _auth_error(_request: Request, exc: AuthError):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def auth(request: Request) -> AuthContext:
    return authenticate(request.headers)


Ctx = Annotated[AuthContext, Depends(auth)]


def _domain(conn) -> Domain:
    return get_domain(tuple(settings_store.enabled_packs(conn)))


def _doc_out(d: dict, domain: Domain | None = None) -> dict:
    return {
        "id": str(d["id"]), "filename": d["filename"], "title": d.get("title"), "status": d["status"],
        "error": d.get("error"), "doc_type": d.get("doc_type"), "doc_type_label": domain.type_label(d.get("doc_type")) if domain and d.get("doc_type") else None,
        "doc_type_confidence": d.get("doc_type_confidence"), "doc_type_method": d.get("doc_type_method"),
        "kind": d.get("kind"), "page_count": d.get("page_count"), "word_count": d.get("word_count"),
        "language": d.get("language"), "has_signature": d.get("has_signature"), "size_bytes": d.get("size_bytes"),
        "mime_type": d.get("mime_type"), "collection": d.get("collection"), "source": d.get("source"),
        "sha256": d.get("sha256"), "processing_ms": d.get("processing_ms"),
        "created_at": d["created_at"].isoformat() if d.get("created_at") else None,
        "indexed_at": d["indexed_at"].isoformat() if d.get("indexed_at") else None,
    }


# ---------------------------------------------------------------------------------------------------- health

@app.get("/health/live", include_in_schema=False)
def live():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request):
    """Prometheus metrics (no tenant, user or document labels). Bearer token required when DOCINTEL_METRICS_TOKEN is set."""
    s = get_settings()
    if not s.metrics_enabled:
        raise HTTPException(404, "not found")
    if s.metrics_token:
        import hmac
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {s.metrics_token}".encode()):
            raise HTTPException(401, "metrics token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        depth = dispatch.queue_depth()
        for state in ("queued", "running"):
            metrics.QUEUE_DEPTH.labels(state).set(depth.get(state, 0))
    except Exception:                       # metrics stay available when the database is not
        logger.warning("queue depth unavailable for metrics")
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)


@app.get("/health/ready", include_in_schema=False)
def ready():
    checks, ok = {}, True
    try:
        get_db().ping()
        checks["database"] = {"ok": True}
    except Exception as e:
        ok, checks["database"] = False, {"ok": False, "error": type(e).__name__}
    if not get_settings().semantic_enabled:
        checks["vector_index"] = checks["embedder"] = {"ok": True, "status": "disabled"}
    else:
        try:
            from docintel.indexing.vectors import get_vector_store
            get_vector_store().ping()
            checks["vector_index"] = {"ok": True}
        except Exception as e:
            ok, checks["vector_index"] = False, {"ok": False, "error": type(e).__name__}
        try:
            from docintel.indexing.embeddings import get_embedder
            get_embedder().verify()
            checks["embedder"] = {"ok": True, "model": get_settings().embedding_model}
        except Exception as e:
            ok, checks["embedder"] = False, {"ok": False, "error": str(e)[:200]}
    try:
        get_object_store().check_writable()
        checks["object_store"] = {"ok": True}
    except Exception as e:
        ok, checks["object_store"] = False, {"ok": False, "error": type(e).__name__}
    try:
        checks["queue"] = {"ok": True, "depth": dispatch.queue_depth(), "stalled": dispatch.stalled_count()}
    except Exception as e:
        checks["queue"] = {"ok": False, "error": type(e).__name__}
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


# ---------------------------------------------------------------------------------------------------- documents

@app.get("/api/v1/me")
def me(ctx: Ctx):
    return {"principal": ctx.principal, "tenant_id": ctx.tenant_id, "roles": sorted(ctx.roles)}


@app.post("/api/v1/documents", status_code=201)
def upload(ctx: Ctx, files: list[UploadFile] = File(...), collection: str | None = Form(default=None)):
    ctx.require("uploader")
    out = []
    for f in files:
        try:
            reg = pipeline.register(ctx, f.filename or "document", f.file, collection=collection)
        except (ParseError, ValueError) as e:
            metrics.UPLOADS.labels("rejected").inc()
            out.append({"filename": f.filename, "status": "rejected", "error": str(e)})
            continue
        finally:
            f.file.close()
        metrics.UPLOADS.labels("duplicate" if reg.duplicate else "accepted").inc()
        if not reg.duplicate:
            dispatch.submit(ctx.tenant_id, str(reg.document["id"]))
        doc = _doc_out(reg.document)
        doc["duplicate"] = reg.duplicate
        out.append(doc)
    return {"documents": out}


class UrlIngest(BaseModel):
    url: str = Field(max_length=4096)
    filename: str | None = None
    collection: str | None = None


@app.post("/api/v1/documents/url", status_code=201)
def upload_url(ctx: Ctx, body: UrlIngest):
    ctx.require("uploader")
    s = get_settings()
    if not s.url_fetch_enabled:
        raise HTTPException(403, "URL ingestion is disabled on this server")
    from docintel.net_safety import IterStream, UnsafeURLError, fetch, redact_url
    try:
        chunks, name = fetch(body.url, s.max_upload_bytes)
        reg = pipeline.register(ctx, body.filename or name, IterStream(chunks), collection=body.collection,
                                source="url", source_url=redact_url(body.url))
    except (UnsafeURLError, ParseError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    if not reg.duplicate:
        dispatch.submit(ctx.tenant_id, str(reg.document["id"]))
    doc = _doc_out(reg.document)
    doc["duplicate"] = reg.duplicate
    return doc


@app.get("/api/v1/documents")
def list_docs(ctx: Ctx, status: str | None = None, doc_type: str | None = None, q: str | None = None,
              collection: str | None = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500))
    with get_db().tenant(ctx.tenant_id) as conn:
        rows, total = repo.list_documents(conn, status=status, doc_type=doc_type, q=q, collection=collection,
                                          limit=limit, offset=max(0, offset))
        domain = _domain(conn)
    return {"documents": [_doc_out(r, domain) for r in rows], "total": total, "limit": limit, "offset": offset}


def _uuid(doc_id: str) -> str:
    try:
        return str(uuid.UUID(doc_id))
    except ValueError:
        raise HTTPException(404, "document not found") from None


@app.get("/api/v1/documents/{doc_id}")
def get_doc(ctx: Ctx, doc_id: str):
    with get_db().tenant(ctx.tenant_id) as conn:
        d = repo.document_detail(conn, _uuid(doc_id))
        domain = _domain(conn)
    if not d:
        raise HTTPException(404, "document not found")
    out = _doc_out(d, domain)
    out["fields"] = [{**f, "value_date": f["value_date"].isoformat() if f["value_date"] else None,
                      "value_num": float(f["value_num"]) if f["value_num"] is not None else None} for f in d["fields"]]
    out["entities"], out["clauses"], out["pages"] = d["entities"], d["clauses"], d["pages"]
    out["relations"], out["tables"], out["versions"] = d["relations"], d["tables"], d["versions"]
    out["stale"] = pipeline.stale_reasons(d.get("components"), pipeline.components(domain)) if d["status"] == "indexed" else []
    return out


@app.get("/api/v1/documents/{doc_id}/pages/{page}")
def get_page(ctx: Ctx, doc_id: str, page: int):
    with get_db().tenant(ctx.tenant_id) as conn:
        p = repo.get_page(conn, _uuid(doc_id), page)
    if not p:
        raise HTTPException(404, "page not found")
    return p


@app.get("/api/v1/documents/{doc_id}/original")
def original(ctx: Ctx, doc_id: str):
    with get_db().tenant(ctx.tenant_id) as conn:
        d = repo.get_document(conn, _uuid(doc_id))
    if not d:
        raise HTTPException(404, "document not found")
    # always a download, and sandboxed should a browser render it anyway (uploaded HTML must never run on this origin)
    return FileResponse(get_object_store().path(d["storage_key"]), filename=d["filename"],
                        media_type=d.get("mime_type") if d.get("mime_type") and "*" not in d["mime_type"] else "application/octet-stream",
                        content_disposition_type="attachment",
                        headers={"Content-Security-Policy": "default-src 'none'; sandbox", "Cache-Control": "private, no-store"})


@app.get("/api/v1/documents/{doc_id}/pages/{page}/image")
def page_image(ctx: Ctx, doc_id: str, page: int):
    with get_db().tenant(ctx.tenant_id) as conn:
        d = repo.get_document(conn, _uuid(doc_id))
    if not d:
        raise HTTPException(404, "document not found")
    from docintel.processing.render import render_page_png
    png = render_page_png(get_object_store().path(d["storage_key"]), d["filename"], page)
    if png is None:
        raise HTTPException(404, "no page image for this document type")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


@app.delete("/api/v1/documents/{doc_id}")
def delete_doc(ctx: Ctx, doc_id: str):
    ctx.require("uploader")
    if not pipeline.delete(ctx, _uuid(doc_id)):
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


@app.post("/api/v1/documents/{doc_id}/reprocess")
def reprocess(ctx: Ctx, doc_id: str):
    ctx.require("uploader")
    d = pipeline.requeue(ctx, _uuid(doc_id))
    if not d:
        raise HTTPException(404, "document not found")
    dispatch.submit(ctx.tenant_id, str(d["id"]))
    return _doc_out(d)


# ---------------------------------------------------------------------------------------------------- query

class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a "tenant" or any other unknown field is an error, not ignored

    q: str = Field(min_length=1, max_length=1000)
    limit: int | None = Field(default=None, ge=1, le=100)
    explain: bool = False
    answer: bool = False          # also write a grounded answer from the evidence (docintel.answering)


@app.post("/api/v1/query")
def query(ctx: Ctx, body: QueryIn):
    from docintel.query.validation import PlanError
    try:
        out = get_engine().run(ctx, body.q, body.limit, body.explain)
    except PlanError as e:                    # a question the engine cannot express as a valid plan
        raise HTTPException(400, f"the question could not be planned: {e}") from e
    if body.answer:
        from docintel import answering
        out["generated_answer"] = answering.generate(body.q, out)
    return out


@app.get("/api/v1/stats")
def stats(ctx: Ctx):
    with get_db().tenant(ctx.tenant_id) as conn:
        out = repo.stats(conn)
        domain = _domain(conn)
    out["by_type"] = [{"type": k, "label": domain.type_label(k), "count": v} for k, v in out["by_type"].items()]
    return out


@app.get("/api/v1/taxonomy")
def get_taxonomy(ctx: Ctx):
    with get_db().tenant(ctx.tenant_id) as conn:
        domain = _domain(conn)
    return {"packs": list(domain.packs),
            "types": [{"type": k, "label": t.display, "family": t.family, "pack": t.pack} for k, t in domain.types.items()]}


@app.get("/api/v1/settings")
def get_tenant_settings(ctx: Ctx):
    """The caller's tenant settings: enabled packs (effective and configured) and the UI's example questions."""
    with get_db().tenant(ctx.tenant_id) as conn:
        cfg = settings_store.load(conn)
        effective = list(_domain(conn).packs)
    return {"packs": cfg.packs, "effective_packs": effective, "available_packs": list(available_packs()),
            "examples": [e.model_dump() for e in cfg.examples]}


@app.put("/api/v1/settings")
def put_tenant_settings(ctx: Ctx, body: settings_store.TenantSettings):
    """Replace the tenant's settings (admin role). Changing packs affects documents processed from now on;
    reprocess existing documents to apply new packs to them."""
    ctx.require("admin")
    with get_db().tenant(ctx.tenant_id) as conn:
        settings_store.save(conn, ctx.tenant_id, body, ctx.principal)
        repo.audit(conn, ctx.tenant_id, ctx.principal, "settings.update", None,
                   {"packs": body.packs, "examples": len(body.examples)})
    return get_tenant_settings(ctx)


# ---------------------------------------------------------------------------------------------------- UI

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
