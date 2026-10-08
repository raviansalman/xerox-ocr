"""Embedding service: serves the configured model from the registry over HTTP (offline, CPU or GPU).

    DOCINTEL_EMBEDDING_MODEL=all-mpnet-base-v2 DOCINTEL_EMBEDDER_CACHE=/models uvicorn docintel.embedder.app:app --port 8080

The model is loaded from the local cache only (``HF_HUB_OFFLINE``); place the files with ``scripts/fetch_model.py``
or bake them into the image. Concurrent small requests are merged into micro-batches.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from docintel.model_registry import model_spec

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_KEY = os.environ.get("DOCINTEL_EMBEDDING_MODEL", "all-mpnet-base-v2")
RERANKER_KEY = os.environ.get("DOCINTEL_RERANKER_MODEL") or None
RERANKER_PATH = os.environ.get("DOCINTEL_RERANKER_PATH") or None   # default: <cache>/<reranker key>
MAX_RERANK_TEXTS = 100
CACHE = os.environ.get("DOCINTEL_EMBEDDER_CACHE", "")
DEVICE = os.environ.get("DOCINTEL_EMBEDDER_DEVICE", "cpu")
MAX_TEXTS = 256
BATCH_TEXTS = 64            # concurrent requests are merged into one model call of up to this many texts
BATCH_WAIT_SEC = 0.004      # how long the first request waits for others to join its batch

_state: dict = {}
_lock = threading.Lock()


def _check_weights(folder: Path, sha256: str) -> None:
    """Refuse model files that do not match the checksum pinned in the registry."""
    weights = folder / "model.safetensors"
    if not weights.exists():
        raise RuntimeError(f"{weights} is missing")
    if sha256:
        h = hashlib.sha256()
        with open(weights, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != sha256:
            raise RuntimeError(f"{weights} does not match the registry checksum")


class RerankRequest(BaseModel):
    query: str = Field(max_length=2000)
    texts: list[str] = Field(max_length=MAX_RERANK_TEXTS)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(max_length=MAX_TEXTS)
    kind: str = "passage"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import torch
    from sentence_transformers import SentenceTransformer

    if not CACHE:
        raise RuntimeError("DOCINTEL_EMBEDDER_CACHE is required: the directory that holds the model files")
    spec = model_spec(MODEL_KEY)
    torch.set_num_threads(int(os.environ.get("DOCINTEL_EMBEDDER_THREADS", str(os.cpu_count() or 1))))
    folder = Path(CACHE) / MODEL_KEY                 # placed by scripts/fetch_model.py
    if folder.is_dir():
        _check_weights(folder, spec.sha256)
        model = SentenceTransformer(str(folder), device=DEVICE)
    else:                                            # a Hugging Face cache folder (development)
        model = SentenceTransformer(spec.name, device=DEVICE, cache_folder=CACHE)
    model.max_seq_length = min(model.max_seq_length or spec.max_tokens, spec.max_tokens)
    dim = model.get_sentence_embedding_dimension()
    if dim != spec.dimension:
        raise RuntimeError(f"model {spec.name} has dimension {dim}, registry says {spec.dimension}")
    _state.update(model=model, spec=spec, queue=asyncio.Queue())
    if RERANKER_KEY:
        from sentence_transformers import CrossEncoder

        from docintel.model_registry import reranker_spec
        rspec = reranker_spec(RERANKER_KEY)
        rfolder = Path(RERANKER_PATH) if RERANKER_PATH else Path(CACHE) / RERANKER_KEY
        _check_weights(rfolder, rspec.sha256)
        _state["reranker"] = CrossEncoder(str(rfolder), max_length=rspec.max_tokens, device=DEVICE)
        _state["reranker_key"] = RERANKER_KEY
    batcher = asyncio.create_task(_batcher(_state["queue"]))
    yield
    batcher.cancel()
    _state.clear()


app = FastAPI(title="docintel-embedder", lifespan=lifespan)


def _encode(texts: list[str]) -> np.ndarray:
    with _lock:
        return _state["model"].encode(texts, batch_size=32, convert_to_numpy=True,
                                      normalize_embeddings=_state["spec"].normalize, show_progress_bar=False)


async def _batcher(queue: asyncio.Queue) -> None:
    """Merge small concurrent requests (search queries) into one encode call; large requests run on their own."""
    loop = asyncio.get_running_loop()
    while True:
        texts, fut = await queue.get()
        batch = [(texts, fut)]
        size = len(texts)
        deadline = loop.time() + BATCH_WAIT_SEC
        while size < BATCH_TEXTS:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                texts, fut = await asyncio.wait_for(queue.get(), timeout)
            except TimeoutError:
                break
            batch.append((texts, fut))
            size += len(texts)
        try:
            vectors = await run_in_threadpool(_encode, [t for texts, _ in batch for t in texts])
        except Exception as e:                                  # fail every request of the batch, keep serving
            for _, f in batch:
                if not f.done():
                    f.set_exception(e)
            continue
        i = 0
        for texts, f in batch:
            if not f.done():
                f.set_result(vectors[i:i + len(texts)])
            i += len(texts)


@app.post("/embed")
async def embed(req: EmbedRequest):
    if "model" not in _state:
        raise HTTPException(503, "model not loaded")
    texts = [t or "" for t in req.texts]
    if len(texts) >= BATCH_TEXTS:
        vectors = await run_in_threadpool(_encode, texts)
    else:
        fut = asyncio.get_running_loop().create_future()
        await _state["queue"].put((texts, fut))
        vectors = await fut
    return {"embeddings": vectors.tolist(), "model": _state["spec"].key}


def _rerank(query: str, texts: list[str]) -> list[float]:
    import torch
    with _lock:
        scores = _state["reranker"].predict([(query, t) for t in texts], activation_fct=torch.nn.Sigmoid(),
                                            batch_size=32, show_progress_bar=False)
    return [float(x) for x in scores]


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if "reranker" not in _state:
        raise HTTPException(404, "no reranker is configured on this service")
    if not req.texts:
        return {"scores": [], "model": _state["reranker_key"]}
    return {"scores": await run_in_threadpool(_rerank, req.query, [t or "" for t in req.texts]),
            "model": _state["reranker_key"]}


@app.get("/info")
async def info():
    spec = _state.get("spec") or model_spec(MODEL_KEY)
    return {"model": spec.key, "name": spec.name, "revision": spec.revision, "dimension": spec.dimension,
            "normalize": spec.normalize, "device": DEVICE, "reranker": _state.get("reranker_key")}


@app.get("/health")
async def health():
    return {"status": "ok" if "model" in _state else "loading"}
