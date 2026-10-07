"""KD-SEC-05: nothing read from Redis or shared disk is unpickled; data stores are not exposed."""
import datetime
import pickle
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import src.semantic.semantic_components as sc

ULTIMATE = Path(__file__).resolve().parents[1]

STATE = {
    "version": sc.METADATA_STATE_VERSION,
    "timestamp": 1700000000.5,
    "docs": {"f1": {"full_text": "Invoice 2024", "years": [2024], "persons": ["Lisa Riordan"]}},
    "by_year": {2024: ["f1"]},
    "by_year_range": {(2020, 2024): ["f1"]},
    "by_month_year": {("june", 2024): ["f1"]},
    "by_full_date": {datetime.date(2024, 6, 3): ["f1"]},
    "by_month_only": {}, "by_location": {"austin": ["f1"]}, "by_person": {"lisa riordan": ["f1"]},
    "by_org": {}, "by_clause": {}, "by_skill": {}, "by_amount": {12500: ["f1"]}, "by_category": {},
    "by_signer": {}, "by_document_type": {}, "by_expiry": {},
    "by_expired": {"yes": [], "no": ["f1"]},
    "by_temporal_hash": {"f1": {"years": {2024}}},
}

EXECUTED = []


def _payload():
    EXECUTED.append(True)
    return {}


class _Evil:
    def __reduce__(self):
        return (_payload, ())


class FakeRedis:
    def __init__(self):
        self.kv, self.z = {}, {}

    def ping(self):
        return True

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v, ex=None):
        self.kv[k] = v

    def zadd(self, name, mapping):
        self.z.setdefault(name, {}).update(mapping)

    def zscore(self, name, member):
        return self.z.get(name, {}).get(member)

    def pipeline(self):
        outer = self

        class P:
            def set(self, *a, **k):
                outer.set(*a, **k)
                return self

            def zadd(self, *a, **k):
                outer.zadd(*a, **k)
                return self

            def execute(self):
                return []
        return P()


def test_state_round_trips_through_json():
    raw = sc.dump_metadata_state(STATE)
    assert raw.lstrip().startswith(b"{")
    assert sc.load_metadata_state(raw) == STATE


def test_unsupported_types_are_refused():
    with pytest.raises(TypeError):
        sc.dump_metadata_state({**STATE, "docs": {"f1": object()}})


@pytest.mark.parametrize("raw", [
    pickle.dumps(STATE),
    pickle.dumps(_Evil()),
    b"not json",
    sc.dump_metadata_state({**STATE, "version": "5.0.8"}),
])
def test_foreign_or_legacy_blobs_are_rejected(raw):
    assert sc.load_metadata_state(raw) is None
    assert not EXECUTED


def test_pickle_blob_in_redis_is_never_unpickled(monkeypatch):
    fake = FakeRedis()
    fake.set(sc.MetadataIndex._REDIS_BLOB_KEY.format(user_id="alice"), zlib.compress(pickle.dumps(_Evil())))
    fake.set("metadata_index_blob:alice", zlib.compress(pickle.dumps(_Evil())))  # legacy key
    monkeypatch.setattr(sc.MetadataIndex, "_get_redis_client", staticmethod(lambda: fake))
    assert sc.MetadataIndex._load_state_from_redis("alice") == (None, -1)
    assert not EXECUTED


def _index_from(state):
    idx = object.__new__(sc.MetadataIndex)  # constructor is broken (KD-SRCH-04); build the fields directly
    for k, v in state.items():
        if k.startswith("by_") and k not in ("by_expired", "by_temporal_hash"):
            v = defaultdict(list, v)
        setattr(idx, k, v)
    return idx


def test_publisher_writes_json_to_redis_and_disk(monkeypatch, tmp_path):
    fake = FakeRedis()
    monkeypatch.setattr(sc.MetadataIndex, "_get_redis_client", staticmethod(lambda: fake))
    monkeypatch.setenv("METADATA_INDEX_PUBLISHER", "true")
    assert _index_from(STATE).save_to_disk("alice", cache_dir=str(tmp_path))
    disk = (tmp_path / "metadata_index_alice.json").read_bytes()
    blob = zlib.decompress(fake.get(sc.MetadataIndex._REDIS_BLOB_KEY.format(user_id="alice")))
    for raw in (disk, blob):
        state = sc.load_metadata_state(raw)
        assert state["docs"] == STATE["docs"] and state["by_month_year"] == STATE["by_month_year"]
    assert fake.zscore("metadata_index_changed_users", "alice")


@pytest.mark.parametrize("uid", ["../etc/x", "a/b", "..", "", 'a"b', "x" * 200])
def test_unsafe_tenant_ids_get_no_cache_file(uid, tmp_path):
    assert sc.metadata_cache_path(uid, str(tmp_path)) is None


def test_safe_tenant_id_cache_path(tmp_path):
    assert sc.metadata_cache_path("65f0c1a2b3c4d5e6f7a8b9c0", str(tmp_path)) == str(
        tmp_path / "metadata_index_65f0c1a2b3c4d5e6f7a8b9c0.json")


def test_embedding_cache_is_pickle_free(tmp_path):
    from src.embeddings import EmbeddingGenerator

    g = object.__new__(EmbeddingGenerator)
    path = str(tmp_path / "c.npz")
    cache = {"h1": np.ones(4, dtype=np.float32), "h2": np.zeros(4, dtype=np.float32)}
    g._save_embedding_cache(cache, path)
    loaded = g._load_embedding_cache(path)
    assert set(loaded) == {"h1", "h2"} and np.array_equal(loaded["h1"], cache["h1"])
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, {"h": _Evil()}, allow_pickle=True)
    assert g._load_embedding_cache(str(legacy)) == {}
    assert not EXECUTED


@pytest.mark.parametrize("compose", sorted(ULTIMATE.glob("docker-compose*.yml")))
def test_compose_never_publishes_data_stores_on_all_interfaces(compose):
    text = compose.read_text()
    for port in ("6379", "19530", "9091"):
        assert not re.search(rf'^\s*-\s*"?{port}:{port}"?\s*$', text, re.M), f"{compose.name} publishes {port}"
        assert not re.search(rf'0\.0\.0\.0:{port}', text)
    if re.search(r"^\s{2}redis:\s*$", text, re.M):
        assert "--requirepass" in text
    assert not re.search(r"\b52\.22\.242\.248\b|\b172\.31\.90\.23\b", text)
