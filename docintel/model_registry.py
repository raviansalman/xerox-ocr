"""Model registry: the embedding (and reranking) models the engine may use, each pinned by revision and checksum,
with its dimension, limits and calibrated similarity floor. The data is ``model_registry.yaml``; a deployment can
point ``DOCINTEL_MODEL_REGISTRY`` at its own file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT = Path(__file__).resolve().parent / "model_registry.yaml"


@cache
def _registry() -> dict[str, Any]:
    path = Path(os.environ.get("DOCINTEL_MODEL_REGISTRY") or _DEFAULT)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    dimension: int
    normalize: bool
    max_tokens: int
    chunk_chars: int
    semantic_floor: float
    revision: str = ""
    sha256: str = ""
    languages: tuple[str, ...] = ("en",)


def model_spec(key: str) -> ModelSpec:
    models = _registry().get("models") or {}
    if key not in models:
        raise KeyError(f"unknown embedding model '{key}'; known: {sorted(models)}")
    m = models[key]
    return ModelSpec(key=key, name=m["name"], dimension=int(m["dimension"]), normalize=bool(m.get("normalize", True)),
                     max_tokens=int(m["max_tokens"]), chunk_chars=int(m["chunk_chars"]),
                     semantic_floor=float(m["semantic_floor"]), revision=str(m.get("revision", "")),
                     sha256=str(m.get("sha256", "")), languages=tuple(m.get("languages", ["en"])))



@dataclass(frozen=True)
class RerankerSpec:
    key: str
    name: str
    revision: str
    sha256: str
    max_tokens: int


def reranker_spec(key: str) -> RerankerSpec:
    models = _registry().get("rerankers") or {}
    if key not in models:
        raise KeyError(f"unknown reranker '{key}'; known: {sorted(models)}")
    m = models[key]
    return RerankerSpec(key=key, name=m["name"], revision=str(m.get("revision", "")), sha256=str(m.get("sha256", "")),
                        max_tokens=int(m.get("max_tokens", 512)))
