"""Retrieval v2: exact-recall invariants (generated from the corpus), verifiable evidence, contextual and entity
retrieval, degradation when a dependency fails, and the engine modes. Runs with and without semantic search."""
import re

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration


def ask(client, q, key="acme_reader", **kw):
    r = client.post("/api/v1/query", headers=headers(key), json={"q": q, "limit": kw.pop("limit", 20), **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _detail(client, doc_id, key="acme_reader"):
    return client.get(f"/api/v1/documents/{doc_id}", headers=headers(key)).json()


def _spellings(ident: str) -> list[str]:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", ident) if p]
    return list(dict.fromkeys([ident, ident.lower(), " ".join(parts), "".join(parts).lower()]))


def test_every_extracted_identifier_is_found_in_every_spelling(client, ingested, corpus_dir):
    """Exact recall invariant: 100% of identifiers present in the corpus, in four spellings, at tier 1."""
    checked = 0
    for d, _ in corpus_dir:
        if d.tenant != "acme":
            continue
        doc_id = ingested[d.key]
        for f in _detail(client, doc_id)["fields"]:
            if not f["name"].endswith("_number") or not f["value_text"]:
                continue
            for spelling in _spellings(f["value_text"]):
                results = ask(client, spelling)["results"]
                hit = next((r for r in results if r["document_id"] == doc_id), None)
                assert hit and hit["tier"] == 1, (d.key, f["value_text"], spelling, [r["filename"] for r in results])
                checked += 1
    assert checked >= 15          # every identifier field of the corpus, each in up to four spellings


def test_quoted_phrases_from_every_passage_are_found(client, ingested, corpus_dir):
    """Exact recall invariant: a quoted three-word phrase taken from each document's text finds that document."""
    checked = 0
    for d, _ in corpus_dir:
        if d.tenant != "acme":
            continue
        doc_id = ingested[d.key]
        page = client.get(f"/api/v1/documents/{doc_id}/pages/1", headers=headers("acme_reader")).json()
        from docintel import text as T
        words = T.search_text(page["text"]).split()
        if len(words) < 6:
            continue
        for start in (0, len(words) // 2):
            phrase = " ".join(words[start:start + 3])
            results = ask(client, f'"{phrase}"')["results"]
            assert doc_id in [r["document_id"] for r in results], (d.key, phrase)
            checked += 1
    assert checked >= 30


def test_evidence_is_the_stored_source_text(client, ingested):
    for q in ("INV-2026-00481", "Article 12.4", "documents signed by John Smith", "termination clauses"):
        for r in ask(client, q)["results"][:3]:
            assert r["evidence"] and r["explanation"], (q, r["filename"])
            for e in r["evidence"]:
                if e.get("char_start") is None:
                    continue
                page = client.get(f"/api/v1/documents/{r['document_id']}/pages/{e['page']}", headers=headers("acme_reader")).json()
                assert page["text"][e["char_start"]:e["char_end"]] == e["text"], (q, e)


def test_identifier_evidence_contains_the_identifier(client, ingested):
    r = ask(client, "INV 2026 00481")["results"][0]
    assert r["filename"] == "Tax_Invoice_INV-2026-00481.pdf" and "INV-2026-00481" in r["evidence"][0]["text"]
    assert {"exact", "fuzzy"} <= set(r["retrievers"]) and r["tier"] == 1


def test_contextual_retrieval_finds_differently_worded_clauses(client, ingested):
    out = ask(client, "Can either party cancel the contract early?")
    top = [r["filename"] for r in out["results"][:2]]
    assert set(top) == {"Service_Contract_17-2024.pdf", "Employment_Agreement_Field_Engineer.docx"}, top
    assert "contextual" in out["results"][0]["retrievers"]
    assert any(e["match_type"] in ("concept_clause", "relation") for e in out["results"][0]["evidence"])


def test_entity_retrieval_matches_people_and_organizations(client, ingested):
    out = ask(client, "agreements involving Bluefin Analytics")
    assert out["results"][0]["filename"] == "Mutual_NDA_Acme.pdf" and "entity" in out["results"][0]["retrievers"]


def test_bare_code_query_returns_only_documents_containing_it(client, ingested):
    out = ask(client, "00481")
    assert [r["filename"] for r in out["results"]] == ["Tax_Invoice_INV-2026-00481.pdf"]


def test_abstains_when_nothing_matches(client, ingested):
    out = ask(client, "zebra quantum lattice")
    assert out["results"] == [] and out["answer"]["kind"] == "none"
    assert "No sufficiently reliable evidence" in out["answer"]["text"]


def test_a_failing_vector_index_degrades_instead_of_failing(client, ingested, monkeypatch):
    from docintel.config import get_settings
    if not get_settings().semantic_enabled:
        pytest.skip("semantic search disabled in this run")
    from docintel.indexing import vectors

    class Broken:
        def search(self, *a, **k):
            raise ConnectionError("vector index down")
    monkeypatch.setattr(vectors, "_store", Broken())
    out = ask(client, "Northbridge Data")
    assert out["results"][0]["filename"] == "Press_Release_Northbridge.pdf"
    assert "semantic search unavailable" in out["answer"]["degraded"]
    out = ask(client, "INV-2026-00481")           # identifier lookups never use the vector index
    assert out["results"][0]["filename"] == "Tax_Invoice_INV-2026-00481.pdf" and "degraded" not in out["answer"]


def test_a_failing_retriever_is_reported_and_others_still_answer(client, ingested, monkeypatch):
    from docintel.retrieval.structured import EntityRetriever

    def boom(self, *a, **k):
        raise RuntimeError("entity index unavailable")
    monkeypatch.setattr(EntityRetriever, "retrieve", boom)
    out = ask(client, "Northbridge Data")
    assert out["results"][0]["filename"] == "Press_Release_Northbridge.pdf"
    assert "entity" in out["answer"]["degraded"]


@pytest.mark.parametrize("mode", ["v1", "shadow"])
def test_previous_engine_and_shadow_mode_still_answer(client, ingested, monkeypatch, mode):
    from docintel.config import get_settings
    from docintel.query import reset_engines
    monkeypatch.setattr(get_settings(), "retrieval_engine", mode)
    reset_engines()
    try:
        out = ask(client, "INV-2026-00481")
        assert out["engine"] == "v1" and out["results"][0]["filename"] == "Tax_Invoice_INV-2026-00481.pdf"
    finally:
        monkeypatch.setattr(get_settings(), "retrieval_engine", "v2")
        reset_engines()


def test_explain_reports_the_strategies_and_each_retriever(client, ingested):
    out = ask(client, "Can either party cancel the contract early?", explain=True)
    assert out["engine"] == "v2" and out["plan"]["concepts"] == ["termination"]
    assert {"exact", "lexical", "contextual", "semantic"} <= set(out["plan"]["strategies"])
    assert {f"retriever.{n}" for n in ("exact", "lexical", "fuzzy", "entity", "contextual", "metadata")} <= set(out["timings_ms"])
    plan = ask(client, "INV-2026-00481", explain=True)["plan"]
    assert "semantic" not in plan["strategies"] and "contextual" not in plan["strategies"]


def test_optional_reranker_keeps_exact_first_and_degrades_safely(client, ingested, monkeypatch):
    from docintel.config import get_settings
    from docintel.retrieval import rerank as RR
    s = get_settings()
    if not s.semantic_enabled or s.embedding_model != "test-hash-768":
        pytest.skip("uses the stand-in embedding service's test reranker")
    base = ask(client, "INV-2026-00481", explain=True)
    monkeypatch.setattr(s, "reranker_model", "test-overlap")
    RR.reset_reranker()
    try:
        out = ask(client, "INV-2026-00481", explain=True)
        assert [r["document_id"] for r in out["results"]][:1] == [r["document_id"] for r in base["results"]][:1]
        assert {r["document_id"] for r in out["results"]} == {r["document_id"] for r in base["results"]}
        out = ask(client, "printer maintenance quarterly visits", explain=True)
        assert len(out["results"]) > 1 and "rerank" in out["timings_ms"] and "degraded" not in out["answer"]
        monkeypatch.setattr(s, "reranker_model", "ms-marco-MiniLM-L-6-v2")        # not served: refused, not used
        RR.reset_reranker()
        out = ask(client, "printer maintenance quarterly visits")
        assert out["results"] and "reranker" in out["answer"]["degraded"]
    finally:
        RR.reset_reranker()


@pytest.mark.parametrize("mode", ["v1", "shadow", "v2"])
def test_semantic_off_never_touches_the_embedder_or_vector_index(client, ingested, monkeypatch, mode):
    """With the vector backend disabled, no engine may call the embedding service or the vector store."""
    from docintel.config import get_settings
    from docintel.query import reset_engines
    from docintel.retrieval import semantic
    from docintel.search import engine as v1
    s = get_settings()
    calls = []

    def forbidden(*a, **k):
        calls.append(1)
        raise AssertionError("embedder or vector store used with semantic search disabled")
    monkeypatch.setattr(s, "vector_backend", "disabled")
    for module in (v1, semantic):
        monkeypatch.setattr(module, "get_embedder", forbidden)
        monkeypatch.setattr(module, "get_vector_store", forbidden)
    monkeypatch.setattr(s, "retrieval_engine", mode)
    reset_engines()
    try:
        for q in ("INV-2026-00481", "rules for working from home", "Can either party cancel the contract early?"):
            out = ask(client, q)
            assert "degraded" not in out["answer"], (mode, q, out["answer"])
        assert ask(client, "INV-2026-00481")["results"][0]["filename"] == "Tax_Invoice_INV-2026-00481.pdf"
        assert calls == []
    finally:
        monkeypatch.setattr(s, "retrieval_engine", "v2")
        reset_engines()


def test_results_whose_evidence_fails_verification_are_dropped(client, ingested, monkeypatch):
    from docintel.retrieval import evidence as EV
    monkeypatch.setattr(EV, "build", lambda conn, results, terms, max_evidence=3: {})
    out = ask(client, "INV-2026-00481")
    assert out["results"] == [] and out["answer"]["kind"] == "none"
    listed = ask(client, "scanned documents")["results"]                # matched by filters: kept, shown as context
    assert listed and all(e["match_type"] == "context" for r in listed for e in r["evidence"])
