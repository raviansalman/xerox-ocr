"""Ranking metrics, answer matching, dataset validation and gates of the evaluation harness."""
import pytest

from docintel.evaluation import metrics as M
from docintel.evaluation.dataset import DatasetError, load_dataset
from docintel.evaluation.runner import check_gates, compare

REL = {"a": 3, "b": 1}


def test_ranking_metrics():
    assert M.precision_at(["a", "x", "b"], REL, 5) == pytest.approx(2 / 3)
    assert M.precision_at([], REL, 5) is None
    assert M.recall_at(["x", "a"], REL, 10) == 0.5
    assert M.reciprocal_rank(["x", "b", "a"], REL) == 0.5
    assert M.ndcg_at(["a", "b"], REL, 10) == pytest.approx(1.0)
    assert M.ndcg_at(["b", "a"], REL, 10) < 1.0 and M.ndcg_at(["x"], REL, 10) == 0.0


def test_answer_matching():
    assert M.answer_matches({"value": 2}, {"value": 2, "unit": "documents"})
    assert M.answer_matches({"value": "60"}, {"value": "60"}) and not M.answer_matches({"value": 3}, {"value": 2})
    rows = [{"currency": "SAR", "value": 418750.0, "documents": 1}, {"currency": "USD", "value": 12500.0}]
    assert M.answer_matches({"rows": [{"currency": "USD", "value": 12500}, {"currency": "SAR", "value": "418,750"}]}, {"rows": rows})
    assert not M.answer_matches({"rows": [{"currency": "USD", "value": 12500}]}, {"rows": rows})


def test_percentile():
    assert M.percentile([5, 1, 3, 2, 4], 50) == 3 and M.percentile([1.0] * 19 + [100.0], 95) == 1.0
    assert M.percentile([], 50) is None


def test_dataset_validation(tmp_path):
    bad = {
        "no source": "questions: [{id: a, tenant: t, question: q, unanswerable: true}]",
        "duplicate": "source: synthetic\nquestions: [{id: a, tenant: t, question: q, unanswerable: true}, "
                     "{id: a, tenant: t, question: r, unanswerable: true}]",
        "no label": "source: synthetic\nquestions: [{id: a, tenant: t, question: q}]",
        "grade": "source: real\nquestions: [{id: a, tenant: t, question: q, relevant: {x.pdf: 5}}]",
        "contradiction": "source: real\nquestions: [{id: a, tenant: t, question: q, unanswerable: true, relevant: {x: 1}}]",
    }
    for name, text in bad.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(text)
        with pytest.raises(DatasetError):
            load_dataset(p)


def test_fixture_dataset_loads():
    from pathlib import Path
    d = load_dataset(Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "fixtures.yaml")
    assert d.source == "synthetic" and len(d.questions) >= 70 and set(d.tenants) == {"acme", "globex", "haystack"}
    assert sum(q.unanswerable for q in d.questions) >= 5 and d.gates["leakage"] == 0


def test_gates_and_compare():
    gates = check_gates({"mrr": 0.8, "leakage": 1, "exact_recall": None}, {"mrr": 0.9, "leakage": 0, "exact_recall": 1.0})
    assert not gates["mrr"]["passed"] and not gates["leakage"]["passed"] and gates["exact_recall"]["passed"]
    base = {"dataset": "d", "summary": {"mrr": 0.9, "leakage": 0}, "questions": [
        {"id": "q1", "question": "x", "passed": True, "mrr": 1.0, "results": ["a"]}]}
    cand = {"dataset": "d", "summary": {"mrr": 0.8, "leakage": 0}, "questions": [
        {"id": "q1", "question": "x", "passed": False, "mrr": 0.5, "results": ["b", "a"]}]}
    c = compare(base, cand)
    assert c["metrics"]["mrr"]["regression"] and not c["metrics"]["leakage"]["regression"]
    assert [r["id"] for r in c["regressions"]] == ["q1"]
