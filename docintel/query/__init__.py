"""Query execution. ``get_engine`` returns the engine selected by DOCINTEL_RETRIEVAL_ENGINE:
``v2`` (default; docintel.query.engine), ``v1`` (the previous engine, kept for one release) or ``shadow`` (serves
v1, runs v2 in the background and logs how they compare)."""
from __future__ import annotations

import threading

from docintel.config import get_settings

_lock = threading.Lock()
_engines: dict[str, object] = {}


def get_engine():
    mode = get_settings().retrieval_engine
    with _lock:
        if mode not in _engines:
            from docintel.query.engine import QueryEngineV2, ShadowEngine
            from docintel.search.engine import QueryEngine
            if mode == "v1":
                _engines[mode] = QueryEngine()
            elif mode == "v2":
                _engines[mode] = QueryEngineV2()
            else:
                _engines[mode] = ShadowEngine(QueryEngine(), QueryEngineV2())
        return _engines[mode]


def reset_engines() -> None:
    with _lock:
        _engines.clear()
