"""The embedding service merges concurrent small requests into one model call and returns each caller its rows."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from docintel.embedder import app as E


class CountingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **_):
        self.calls.append(len(texts))
        return np.array([[float(len(t)), float(i)] for i, t in enumerate(texts)])


def test_concurrent_requests_share_one_encode_call(monkeypatch):
    model = CountingModel()

    async def scenario():
        queue = asyncio.Queue()
        monkeypatch.setitem(E._state, "model", model)
        monkeypatch.setitem(E._state, "spec", type("S", (), {"normalize": True})())
        task = asyncio.create_task(E._batcher(queue))
        loop = asyncio.get_running_loop()
        futures = []
        for i in range(10):
            f = loop.create_future()
            await queue.put(([f"q{i}", "x" * (i + 1)], f))
            futures.append(f)
        results = await asyncio.gather(*futures)
        task.cancel()
        return results

    with ThreadPoolExecutor(1) as pool:     # own thread: another test may leave an event loop running here
        results = pool.submit(asyncio.run, scenario()).result()
    assert model.calls == [20]
    for i, rows in enumerate(results):
        assert rows.shape == (2, 2) and rows[1][0] == i + 1        # each caller gets its own rows, in order
