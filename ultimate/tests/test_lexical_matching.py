"""Characterize UltimateVectorIntegration.search_lexical_on_chunks.

Despite its docstring ("text + filename"), the scan matches filename / file_id only;
content matches come from the /search content-scan supplement and from vectors.
"""
from types import SimpleNamespace

import pytest

import src.ultimate_vector_integration as uvi


@pytest.fixture(scope="module")
def vi():
    return object.__new__(uvi.UltimateVectorIntegration)


def chunk(fid, filename, text="", **meta):
    return SimpleNamespace(chunk_id=f"{fid}::chunk::0", text=text,
                           metadata={"file_id": fid, "filename": filename, **meta})


def hits(vi, chunks, query, **kw):
    return [(r.file_id, r.similarity_score) for r in vi.search_lexical_on_chunks(chunks, query, **kw)]


@pytest.mark.parametrize("text,expected", [
    ("StorageChain", "storage chain"),
    ("Press%20Release_2024.pdf", "press release 2024 pdf"),
    ("XMLParserV2", "xml parser v2"),
])
def test_keyword_normalization(vi, text, expected):
    assert vi._normalize_for_keyword_match(text) == expected


def test_exact_filename_scores_highest(vi):
    assert hits(vi, [chunk("f1", "storagechain")], "StorageChain") == [("f1", 0.98)]


def test_phrase_in_filename(vi):
    assert hits(vi, [chunk("f1", "StorageChain_Press_Release.pdf")], "storagechain") == [("f1", 0.95)]


def test_camelcase_filename_matches_spaced_query(vi):
    # Matches via the per-token fallback (0.91), not the phrase tier: CamelCase splitting runs on
    # the already-lowercased filename, so "storage chain" is never seen as a phrase (KD-SRCH-05).
    assert hits(vi, [chunk("f1", "StorageChainAnnouncement.pdf")], "storage chain") == [("f1", 0.91)]


def test_url_encoded_presigned_filename(vi):
    c = chunk("f1", "Press%20Release%202024.pdf?X-Amz-Signature=abc")
    assert hits(vi, [c], "press release") == [("f1", 0.95)]


def test_file_type_word_requires_matching_extension(vi):
    pdf, docx = chunk("pdf1", "Lisa_Riordan_Bio.pdf"), chunk("doc1", "Lisa_Riordan_Bio.docx")
    assert hits(vi, [pdf, docx], "riordan pdf") == [("pdf1", 0.91)]


def test_content_only_match_is_not_returned(vi):
    c = chunk("f1", "release.pdf", text="FOR IMMEDIATE RELEASE Lisa Riordan")
    assert hits(vi, [c], "Lisa Riordan") == []


def test_one_result_per_file_and_scope_filter(vi):
    cs = [chunk("f1", "Invoice_Q1.pdf", bucket_id="b1"), chunk("f1", "Invoice_Q1.pdf", bucket_id="b1"),
          chunk("f2", "Invoice_Q2.pdf", bucket_id="b2")]
    assert hits(vi, cs, "invoice", filter_conditions={"bucket_id": "b1"}) == [("f1", 0.95)]


def test_short_or_empty_query_returns_nothing(vi):
    assert hits(vi, [chunk("f1", "a.pdf")], "a") == []
    assert hits(vi, [chunk("f1", "a.pdf")], "") == []
