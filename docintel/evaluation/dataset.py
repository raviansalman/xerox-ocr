"""Evaluation dataset format.

```yaml
dataset: fixtures
source: synthetic            # synthetic | real
k: 10
gates:                       # metric: minimum (or maximum for *_rate and leakage)
  exact_recall: 1.0
  leakage: 0
questions:
  - id: id-001
    tenant: tenant-a         # the tenant whose key asks the question
    category: identifier     # identifier, phrase, keyword, typo, semantic, contextual, entity, structured,
                             # computed, unanswerable (free-form; metrics are reported per category)
    question: INV-2024-00017
    relevant: {invoice_00017.pdf: 3}   # file name -> grade (1 to 3)
    exact: true              # the relevant documents must be found (exact-recall gate)
    requires: [semantic]     # skipped when the target has semantic search disabled
  - id: cnt-001
    question: How many invoices do we have?
    answer: {value: 2}       # computed answer: every key must match the response's answer
  - id: neg-001
    question: recipe for chocolate cake
    unanswerable: true       # any result is a false positive
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOWER_IS_BETTER = ("leakage", "abstention_fp_rate", "p95_ms", "p50_ms")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    tenant: str
    category: str
    relevant: dict[str, int] = field(default_factory=dict)
    exact: bool = False
    unanswerable: bool = False
    answer: dict[str, Any] | None = None
    requires: tuple[str, ...] = ()
    limit: int = 20


@dataclass(frozen=True)
class Dataset:
    name: str
    source: str
    k: int
    gates: dict[str, float]
    questions: list[Question]

    @property
    def tenants(self) -> list[str]:
        return sorted({q.tenant for q in self.questions})


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path) -> Dataset:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if data.get("source") not in ("synthetic", "real"):
        raise DatasetError("dataset 'source' must be 'synthetic' or 'real'")
    default_tenant = data.get("tenant")
    questions, seen = [], set()
    for raw in data.get("questions") or []:
        qid = str(raw.get("id") or "")
        if not qid or qid in seen:
            raise DatasetError(f"question ids must be unique and non-empty ({qid!r})")
        seen.add(qid)
        tenant = raw.get("tenant") or default_tenant
        if not tenant or not raw.get("question"):
            raise DatasetError(f"{qid}: 'question' and 'tenant' are required")
        relevant = {str(k): int(v) for k, v in (raw.get("relevant") or {}).items()}
        if any(not 1 <= g <= 3 for g in relevant.values()):
            raise DatasetError(f"{qid}: relevance grades are 1 to 3")
        unanswerable = bool(raw.get("unanswerable"))
        if unanswerable and (relevant or raw.get("answer")):
            raise DatasetError(f"{qid}: an unanswerable question has no relevant documents or answer")
        if not (unanswerable or relevant or raw.get("answer")):
            raise DatasetError(f"{qid}: give relevant documents, an answer, or unanswerable: true")
        questions.append(Question(qid, str(raw["question"]), str(tenant), str(raw.get("category") or "general"),
                                  relevant, bool(raw.get("exact")), unanswerable, raw.get("answer"),
                                  tuple(raw.get("requires") or ()), int(raw.get("limit") or 20)))
    if not questions:
        raise DatasetError("the dataset has no questions")
    return Dataset(str(data.get("dataset") or Path(path).stem), data["source"], int(data.get("k") or 10),
                   {str(k): float(v) for k, v in (data.get("gates") or {}).items()}, questions)
