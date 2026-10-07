"""Deterministic stand-in for src/embedder_service.py (same /embed contract).

HuggingFace is blocked in this sandbox, so MPNet cannot load. This hashes word
unigrams and character trigrams into 768 dims and L2-normalizes, so texts that
share vocabulary have higher cosine similarity. Good for plumbing tests; it says
nothing about MPNet's semantic quality.
"""
import hashlib
import re
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

DIM = 768
app = FastAPI()


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def embed_one(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    t = (text or "").lower()
    for w in re.findall(r"[\w]+", t):
        v[_h("w:" + w) % DIM] += 2.0
        for i in range(max(0, len(w) - 2)):
            v[_h("c:" + w[i:i + 3]) % DIM] += 0.5
    n = np.linalg.norm(v)
    return v / n if n else v


class EmbedRequest(BaseModel):
    texts: List[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    return {"embeddings": [embed_one(t).tolist() for t in req.texts]}


@app.get("/health")
def health():
    return {"status": "ok"}
