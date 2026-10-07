"""A slow search must not block unrelated requests; searches stay serialized as before."""
import socket
import threading
import time

import pytest
import requests
import uvicorn

from tests.support.auth import headers

SLOW = 3.0


class _SlowEngine:
    def search_documents(self, *a, **k):
        time.sleep(SLOW)
        return []


@pytest.fixture(scope="module")
def server():
    import ultimate_ui

    mp = pytest.MonkeyPatch()
    mp.setattr(ultimate_ui, "get_semantic_engine", lambda: _SlowEngine())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(ultimate_ui.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            requests.get(base + "/health/live", timeout=0.5)
            break
        except requests.ConnectionError:
            time.sleep(0.05)
    yield base
    srv.should_exit = True
    t.join(timeout=5)
    mp.undo()


def _slow_search(base, out, key):
    t = time.time()
    r = requests.post(base + "/search", headers=headers("service"), timeout=60,
                      json={"userId": "alice", "query": "press release", "searchMethod": "semantic"})
    out[key] = (r.status_code, time.time() - t, time.time())


def test_unrelated_requests_are_served_during_a_slow_search(server):
    out = {}
    th = threading.Thread(target=_slow_search, args=(server, out, "s"))
    th.start()
    time.sleep(0.5)  # search is now in progress
    for path in ("/health/live", "/"):
        t = time.time()
        assert requests.get(server + path, timeout=5).status_code == 200
        assert time.time() - t < 0.5, f"{path} waited for the search"
    th.join()
    status, took, _ = out["s"]
    assert status == 200 and took >= SLOW


def test_searches_remain_serialized(server):
    out = {}
    ths = [threading.Thread(target=_slow_search, args=(server, out, k)) for k in ("a", "b")]
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert all(v[0] == 200 for v in out.values())
    assert time.time() - t0 >= 2 * SLOW  # one search at a time per process, exactly as before
