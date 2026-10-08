"""Deterministic stand-in for the embedding service (same HTTP contract as docintel.embedder.app).

Hashes word unigrams and character trigrams into 768 dimensions and L2-normalizes, so texts sharing vocabulary
are similar. Function words are skipped (as a real model gives them little weight), so "recipe for cake" is not
similar to "for immediate release". It validates plumbing and lexical behaviour, not semantic quality (see tests/quality for the real model).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

DIM = 768
KEY = "test-hash-768"
app = FastAPI()


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


_FUNCTION_WORDS = frozenset("a an and are as at be by for from has have i in is it of on or our the this to was we with "
                            "what which who how do does".split())


def embed_one(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"\w+", (text or "").lower()):
        if w in _FUNCTION_WORDS:
            continue
        v[_h("w:" + w) % DIM] += 2.0
        for i in range(max(0, len(w) - 2)):
            v[_h("c:" + w[i:i + 3]) % DIM] += 0.5
    n = np.linalg.norm(v)
    return v / n if n else v


class Req(BaseModel):
    texts: list[str]
    kind: str = "passage"


@app.post("/embed")
def embed(req: Req):
    return {"embeddings": [embed_one(t).tolist() for t in req.texts], "model": KEY}


class RerankReq(BaseModel):
    query: str
    texts: list[str]


@app.post("/rerank")
def rerank(req: RerankReq):
    """Stand-in cross-encoder: the share of the question's content words found in the text."""
    words = {w for w in re.findall(r"\w+", req.query.lower()) if w not in _FUNCTION_WORDS}
    out = []
    for t in req.texts:
        found = set(re.findall(r"\w+", t.lower()))
        out.append(len(words & found) / len(words) if words else 0.0)
    return {"scores": out, "model": "test-overlap"}


@app.get("/info")
def info():
    return {"model": KEY, "dimension": DIM, "name": "fake", "revision": "1", "normalize": True, "reranker": "test-overlap"}


@app.get("/health")
def health():
    return {"status": "ok"}
