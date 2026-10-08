"""Reranking reorders only within a fusion group, never moves exact matches, and never adds or drops documents."""
from docintel.retrieval.contracts import Candidate
from docintel.retrieval.fusion import DocumentResult
from docintel.retrieval.rerank import rerank


class FakeConn:
    def __init__(self, texts):
        self.texts = texts

    def execute(self, sql, args):
        ids = args[0]
        rows = [{"id": i, "text": self.texts[i], "context": ""} for i in ids if i in self.texts]
        return type("R", (), {"fetchall": lambda _self: rows})()


class FakeClient:
    def __init__(self, scores):
        self.scores_by_text = scores
        self.calls = 0

    def scores(self, query, texts):
        self.calls += 1
        return [self.scores_by_text[t] for t in texts]


def _doc(doc_id, key, match_type="all_terms"):
    d = DocumentResult(doc_id, [Candidate(doc_id, f"u-{doc_id}", "lexical", match_type, 1.0)])
    d.rank_key = key
    return d


def test_reorders_within_groups_only():
    ranked = [_doc("a", 1, "exact_phrase"), _doc("b", 2), _doc("c", 2), _doc("d", 3), _doc("e", 3)]
    conn = FakeConn({f"u-{x}": f"text {x}" for x in "abcde"})
    client = FakeClient({"text a": 0.0, "text b": 0.1, "text c": 0.9, "text d": 0.2, "text e": 0.8})
    out = rerank(conn, "q", ranked, client, top_n=10)
    assert [d.document_id for d in out] == ["a", "c", "b", "e", "d"]
    assert out[1].rerank_score == 0.9 and out[0].rerank_score is None


def test_only_the_head_is_reranked_and_nothing_is_lost():
    ranked = [_doc(x, 2) for x in "abcd"]
    conn = FakeConn({f"u-{x}": f"text {x}" for x in "abcd"})
    client = FakeClient({"text a": 0.1, "text b": 0.9, "text c": 1.0, "text d": 1.0})
    out = rerank(conn, "q", ranked, client, top_n=2)
    assert [d.document_id for d in out] == ["b", "a", "c", "d"]


def test_documents_without_text_keep_their_place():
    ranked = [_doc("a", 2), _doc("b", 2), _doc("c", 2)]
    conn = FakeConn({"u-a": "text a", "u-c": "text c"})
    out = rerank(conn, "q", ranked, FakeClient({"text a": 0.1, "text c": 0.9}), top_n=10)
    assert [d.document_id for d in out] == ["c", "b", "a"]
