"""Shared pytest setup for the xerox-ocr regression suite.

Two tiers:
  * unit (default): no Milvus, Redis or model downloads. Run with `pytest` from ultimate/.
  * integration: real Milvus + Redis and a deterministic stand-in embedder.
    Run with `RUN_INTEGRATION=1 pytest -m integration` (see tests/README.md).

Most modules read configuration at import time, so environment defaults are set in
pytest_configure, before any test module imports application code.
"""
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ULTIMATE_DIR = Path(__file__).resolve().parents[1]
if str(ULTIMATE_DIR) not in sys.path:
    sys.path.insert(0, str(ULTIMATE_DIR))

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"
HAS_TESSERACT = shutil.which("tesseract") is not None


def _tesseract_langs():
    if not HAS_TESSERACT:
        return set()
    try:
        import pytesseract

        return set(pytesseract.get_languages(config=""))
    except Exception:
        return set()


TESSERACT_LANGS = _tesseract_langs()

requires_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")
requires_tesseract_ara = pytest.mark.skipif(
    "ara" not in TESSERACT_LANGS, reason="tesseract Arabic language pack (ara) not installed"
)


def _port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _start_fake_embedder(port):
    import uvicorn

    from tests.support.fake_embedder import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if not _port_free(port):
            return
        time.sleep(0.05)
    raise RuntimeError(f"fake embedder did not start on port {port}")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs Milvus + Redis (RUN_INTEGRATION=1)")
    config.addinivalue_line("markers", "known_defect: pins a verified bug; strict xfail until fixed")

    # Behaviour as deployed by docker-compose.*.yml (QUEUE_SHARD_COUNT=0 there; code default is 8).
    os.environ.setdefault("QUEUE_SHARD_COUNT", "0")
    os.environ.setdefault("WORKFLOW_ENABLED", "false")
    os.environ.setdefault("ENABLE_QUERY_NER", "0")
    os.environ.setdefault("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "true")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ULTIMATE_DIR / ".pytest_cache" / "st"))
    from tests.support.auth import api_keys_json

    os.environ["API_KEYS"] = api_keys_json()
    os.environ.pop("API_KEYS_FILE", None)
    os.environ.pop("AUTH_DISABLED", None)

    if RUN_INTEGRATION:
        port = int(os.getenv("XOCR_TEST_EMBEDDER_PORT", "18080"))
        if _port_free(port):
            _start_fake_embedder(port)
        url = f"http://127.0.0.1:{port}/embed"
        os.environ.setdefault("EMBEDDER_URL", url)
        os.environ.setdefault("EMBEDDER_SEARCH_URL", url)
        os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
        os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
        suffix = os.getenv("XOCR_TEST_COLLECTION_SUFFIX", f"t{os.getpid()}")
        for a, b, v in (("DOC_COLLECTION", "MILVUS_DOC_COLLECTION", f"xocr_test_docs_{suffix}"),
                        ("IMG_COLLECTION", "MILVUS_IMG_COLLECTION", f"xocr_test_img_{suffix}")):
            os.environ.setdefault(a, v)
            os.environ.setdefault(b, v)
        os.environ.setdefault("METADATA_CACHE_DIR", str(ULTIMATE_DIR / ".pytest_cache" / f"meta_{suffix}"))


def pytest_collection_modifyitems(config, items):
    if RUN_INTEGRATION:
        return
    skip = pytest.mark.skip(reason="integration test: set RUN_INTEGRATION=1 with Milvus + Redis running")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
