"""Client for the embedding service (``docintel.embedder``). Batches requests and verifies the model contract."""
from __future__ import annotations

import logging
import threading
import time

import httpx
import numpy as np

from docintel.config import get_settings
from docintel.model_registry import ModelSpec, model_spec

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, base_url: str, spec: ModelSpec, batch_size: int = 32, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._checked = False
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        return self.spec.key

    def info(self) -> dict:
        r = self._client.get(f"{self.base_url}/info")
        r.raise_for_status()
        return r.json()

    def verify(self) -> None:
        """Refuse to work with an embedder that serves a different model or dimension."""
        with self._lock:
            if self._checked:
                return
            info = self.info()
            if int(info.get("dimension", -1)) != self.spec.dimension or info.get("model") != self.spec.key:
                raise EmbeddingError(f"embedder serves {info.get('model')} ({info.get('dimension')} dims); "
                                     f"engine is configured for {self.spec.key} ({self.spec.dimension} dims)")
            self._checked = True

    def _post(self, texts: list[str], kind: str) -> np.ndarray:
        last: Exception | None = None
        for attempt in range(4):
            try:
                r = self._client.post(f"{self.base_url}/embed", json={"texts": texts, "kind": kind})
                r.raise_for_status()
                arr = np.asarray(r.json()["embeddings"], dtype=np.float32)
                if arr.shape != (len(texts), self.spec.dimension):
                    raise EmbeddingError(f"unexpected embedding shape {arr.shape}")
                return arr
            except (httpx.HTTPError, KeyError) as e:
                last = e
                time.sleep(0.5 * 2 ** attempt)
        raise EmbeddingError(f"embedding service unavailable: {last}")

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)
        self.verify()
        out = [self._post(texts[i:i + self.batch_size], kind) for i in range(0, len(texts), self.batch_size)]
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], kind="query")[0]


_client: EmbeddingClient | None = None


def get_embedder() -> EmbeddingClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = EmbeddingClient(s.require("embedder_url"), model_spec(s.embedding_model), s.embed_batch_size, s.embed_timeout_sec)
    return _client


def set_embedder(client: EmbeddingClient | None) -> None:
    global _client
    _client = client
