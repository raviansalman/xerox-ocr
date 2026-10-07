"""Characterize how documents become Milvus chunks. Search relies on every detail here
(the [FILE: ...] header in each chunk is what makes filename search work via vectors)."""
import re

import numpy as np
import pytest

import src.ultimate_vector_integration as uvi


def test_default_chunk_policy():
    # Code defaults; the deployed env file overrides CHUNK_SIZE=500 / CHUNK_OVERLAP=100.
    assert (uvi.CHUNK_SIZE, uvi.CHUNK_OVERLAP) == (1200, 50)
    assert (uvi.CSV_CHUNK_SIZE, uvi.CSV_TARGET_MAX_CHUNKS) == (6000, 10)


def test_chunk_text_spans_are_deterministic():
    chunks = uvi._chunk_text("a" * 2500, chunk_size=1200, overlap=50)
    assert [(m["chunk_index"], m["start"], m["end"]) for _, m in chunks] == [
        (0, 0, 1200), (1, 1150, 2350), (2, 2300, 2500)]
    assert uvi._chunk_text("", 1200, 50) == []


def test_csv_chunks_respect_milvus_varchar_limit_and_keep_all_rows():
    rows = [f"{i},customer {i},{i * 3.5}" for i in range(20000)]
    text = "\n".join(rows)
    chunks = uvi._chunk_text_csv(text, chunk_size=6000, overlap=0, target_max_chunks=10)
    assert all(len(c) <= 65535 for c, _ in chunks)
    rejoined = "\n".join(c for c, _ in chunks).splitlines()
    assert rejoined == rows


class _FakeEmbedder:
    def embed_texts(self, texts):
        return np.zeros((len(texts), 768), dtype=np.float32)


class _FakeDB:
    def __init__(self):
        self.chunks, self.user_id = [], None

    def insert_chunks(self, chunks, embeddings, user_id):
        self.chunks, self.user_id = chunks, user_id
        assert embeddings.shape == (len(chunks), 768)
        return True


def _integration():
    vi = object.__new__(uvi.UltimateVectorIntegration)
    vi.text_embedder = _FakeEmbedder()
    vi.vector_db = _FakeDB()
    return vi


def test_every_chunk_carries_filename_header_and_tenant_scope():
    vi = _integration()
    res = vi.upsert_document(
        file_id="65f0c1a2b3c4d5e6f7a8b9c0", text_content="Body " * 400,
        metadata={"original_filename": "Press%20Release.pdf?X-Amz-Signature=abc", "file_type": "application/pdf",
                  "bucket_id": "b1", "path": "pr/2024", "connection_id": "c1"},
        user_id="alice")
    assert res["success"] and res["inserted_chunks"] == len(vi.vector_db.chunks) > 1
    assert vi.vector_db.user_id == "alice"
    for c in vi.vector_db.chunks:
        assert c.text.startswith("[FILE: 65f0c1a2b3c4d5e6f7a8b9c0] [FILE: Press Release.pdf] | ")
        assert re.fullmatch(r"65f0c1a2b3c4d5e6f7a8b9c0::chunk::\d+::\d+", c.chunk_id)
        assert c.metadata["filename"] == "Press Release.pdf"
        assert (c.metadata["bucket_id"], c.metadata["path"], c.metadata["connection_id"]) == ("b1", "pr/2024", "c1")


def test_empty_text_still_indexes_a_filename_only_chunk():
    # Means a failed OCR shows up as SUCCESS with one header-only chunk (see KD-OCR-04 in the assessment).
    vi = _integration()
    res = vi.upsert_document(file_id="f-empty", text_content="", metadata={"filename": "scan.png"}, user_id="alice")
    assert res["inserted_chunks"] == 1
    assert vi.vector_db.chunks[0].text == "[FILE: f-empty] [FILE: scan.png] | "


def test_upsert_requires_user_id():
    with pytest.raises(ValueError, match="user_id is required"):
        _integration().upsert_document(file_id="f", text_content="x", metadata={}, user_id=None)


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-02: page boundaries are discarded; every chunk has page_number=0")
def test_chunks_record_source_page():
    vi = _integration()
    vi.upsert_document(file_id="f", text_content="page one text\fpage two text", metadata={}, user_id="alice")
    assert {c.metadata["page_number"] for c in vi.vector_db.chunks} != {0}
