"""Freeze the Milvus contract that existing deployments depend on.

Changing any value pinned here makes new code incompatible with collections that are
already populated (or, worse, triggers the auto-drop in _initialize_collection). If a
change is intended it needs a migration plan, not just an updated assertion.
"""
import numpy as np
import pytest
from pymilvus import DataType

import src.vector_db_milvus_server as vdb
from src.semantic.semantic_components import TextChunk


class _FakeCollection:
    def __init__(self, name=None, schema=None, using=None, shards_num=None):
        self.name, self.schema, self.shards_num = name, schema, shards_num
        self.indexes = []
        self.index_calls, self.search_calls, self.query_calls, self.inserted = [], [], [], []

    def create_index(self, field_name, index_params):
        self.index_calls.append((field_name, index_params))

    def search(self, **kw):
        self.search_calls.append(kw)
        return [[]]

    def query(self, **kw):
        self.query_calls.append(kw)
        return []

    def insert(self, data, **kw):
        self.inserted.extend(data)


def _db(monkeypatch, is_image=False, dim=768):
    monkeypatch.setattr(vdb, "Collection", _FakeCollection)
    db = object.__new__(vdb.MilvusServerVectorDatabase)
    db.collection_name = "contract_test"
    db.vector_size = dim
    db.distance_metric = "COSINE"
    db.is_image_collection = is_image
    db.collection = None
    db.temporal = None
    db._create_collection()
    db._ensure_collection_loaded = lambda: True
    db._reconnect_if_needed = lambda: True
    return db


def _fields(schema):
    out = {}
    for f in schema.fields:
        out[f.name] = (f.dtype, f.params.get("dim") or f.params.get("max_length"), f.is_primary)
    return out


def test_document_collection_schema_is_frozen(monkeypatch):
    db = _db(monkeypatch)
    assert _fields(db.collection.schema) == {
        "id": (DataType.VARCHAR, 500, True),
        "embedding": (DataType.FLOAT_VECTOR, 768, False),
        "chunk_id": (DataType.VARCHAR, 500, False),
        "text": (DataType.VARCHAR, 65535, False),
        "page_number": (DataType.INT64, None, False),
        "element_type": (DataType.VARCHAR, 500, False),
        "source_file": (DataType.VARCHAR, 500, False),
        "created_at": (DataType.VARCHAR, 500, False),
        "object_id": (DataType.VARCHAR, 500, False),
        "user_id": (DataType.VARCHAR, 500, False),
        "bucket_id": (DataType.VARCHAR, 500, False),
        "path": (DataType.VARCHAR, 2000, False),
        "connection_id": (DataType.VARCHAR, 500, False),
    }
    assert db.collection.schema.enable_dynamic_field is True
    assert db.collection.shards_num == 2


def test_image_collection_schema_is_frozen(monkeypatch):
    db = _db(monkeypatch, is_image=True, dim=512)
    assert _fields(db.collection.schema) == {
        "id": (DataType.VARCHAR, 500, True),
        "embedding": (DataType.FLOAT_VECTOR, 512, False),
        "source_file": (DataType.VARCHAR, 500, False),
        "objects": (DataType.VARCHAR, 10000, False),
        "scene": (DataType.VARCHAR, 500, False),
        "dominant_colors": (DataType.VARCHAR, 2000, False),
        "description": (DataType.VARCHAR, 5000, False),
        "created_at": (DataType.VARCHAR, 500, False),
        "user_id": (DataType.VARCHAR, 500, False),
    }


def test_hnsw_cosine_index_is_frozen(monkeypatch):
    db = _db(monkeypatch)
    # _create_collection calls _create_index once; it must be HNSW/COSINE with M=32, efConstruction=200
    assert db.collection.index_calls == [(
        "embedding",
        {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 200}},
    )]


def test_default_dimensions_models_and_collection_names_agree():
    import src.semantic.semantic_pipeline as sp
    import src.ultimate_vector_integration as uvi

    assert uvi.EMBED_MODEL == "sentence-transformers/all-mpnet-base-v2"
    assert uvi.MODEL_DIM == 768 and sp.DOC_DIM == 768
    assert uvi.IMAGE_VECTOR_DIM == 512 and sp.IMG_DIM == 512
    # Ingest and search read collection names from different env vars; their defaults must match.
    assert uvi.DOC_COLLECTION == sp.DOC_COLLECTION
    assert uvi.IMG_COLLECTION == sp.IMG_COLLECTION


@pytest.mark.parametrize("limit,expected_k,expected_ef", [(10, 10, 64), (40, 40, 80), (500, 50, 100)])
def test_search_params_limit_cap_and_ef(monkeypatch, limit, expected_k, expected_ef):
    monkeypatch.delenv("MILVUS_SEARCH_EF", raising=False)
    db = _db(monkeypatch)
    db.search_similar(np.zeros(768, dtype=np.float32), limit=limit, user_id="alice")
    call = db.collection.search_calls[-1]
    assert call["limit"] == expected_k  # hard cap of 50 chunks per vector search
    assert call["param"] == {"metric_type": "COSINE", "params": {"ef": expected_ef}}
    assert call["anns_field"] == "embedding"


def test_tenant_filter_expression_format(monkeypatch):
    db = _db(monkeypatch)
    db.search_similar(np.zeros(768, dtype=np.float32), limit=5, user_id="alice",
                      filter_conditions={"bucket_id": "b1", "path": "invoices/2024", "connection_id": "c9"})
    assert db.collection.search_calls[-1]["expr"] == (
        'user_id == "alice" and bucket_id == "b1" and '
        '(path == "invoices/2024" or path like "invoices/2024/%") and connection_id == "c9"'
    )


def test_inserted_row_shape_is_frozen(monkeypatch):
    db = _db(monkeypatch)
    chunk = TextChunk(chunk_id="f1::chunk::0::1", text="hello",
                      metadata={"file_id": "f1", "filename": "a.pdf", "page_number": 3, "years": [2024],
                                "bucket_id": "b", "path": "p", "connection_id": "c"})
    assert db.insert_chunks([chunk], np.zeros((1, 768), dtype=np.float32), user_id="alice")
    row = db.collection.inserted[0]
    assert sorted(row) == sorted([
        "id", "embedding", "chunk_id", "text", "page_number", "element_type", "source_file", "file_id",
        "filename", "original_filename", "created_at", "object_id", "user_id", "bucket_id", "path",
        "connection_id",
    ])
    assert row["user_id"] == "alice" and row["page_number"] == 3 and row["source_file"] == "f1"


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-SEC-01: user_id is interpolated into the Milvus expr unescaped")
def test_user_id_is_escaped_in_search_expr(monkeypatch):
    db = _db(monkeypatch)
    db.search_similar(np.zeros(768, dtype=np.float32), limit=5, user_id='bob" or user_id != "bob')
    assert db.collection.search_calls[-1]["expr"] == 'user_id == "bob\\" or user_id != \\"bob"'


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-SEC-01: query_all_chunks interpolates user_id unescaped")
def test_user_id_is_escaped_in_full_scan_expr(monkeypatch):
    db = _db(monkeypatch)
    db.query_all_chunks(user_id='bob" or user_id != "bob', limit=10)
    assert db.collection.query_calls[-1]["expr"] == 'user_id == "bob\\" or user_id != \\"bob"'


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-DATA-01: extracted years/persons/orgs are dropped before insert")
def test_extracted_metadata_reaches_milvus(monkeypatch):
    db = _db(monkeypatch)
    chunk = TextChunk(chunk_id="f1::chunk::0::1", text="x",
                      metadata={"file_id": "f1", "years": [2024], "persons": ["Lisa Riordan"]})
    db.insert_chunks([chunk], np.zeros((1, 768), dtype=np.float32), user_id="alice")
    row = db.collection.inserted[0]
    assert row.get("years") == [2024] and row.get("persons") == ["Lisa Riordan"]
