"""Evaluation harness: labelled questions, retrieval and answer metrics, quality gates and report comparison.

A dataset (YAML) lists questions with graded relevant documents, expected computed answers or "unanswerable".
The runner asks every question through the HTTP API (or any object with the same two methods), computes Precision@K,
Recall@K, MRR, nDCG@K, exact recall, abstention false positives, computed-answer accuracy and tenant leakage, checks
the dataset's gates and writes a JSON report. ``compare`` lists metric deltas and per-question regressions between
two reports. A report states whether its dataset is synthetic or real; synthetic numbers say little about real
documents.
"""
from docintel.evaluation.dataset import Dataset, Question, load_dataset
from docintel.evaluation.runner import HttpTarget, compare, evaluate

__all__ = ["Dataset", "HttpTarget", "Question", "compare", "evaluate", "load_dataset"]
