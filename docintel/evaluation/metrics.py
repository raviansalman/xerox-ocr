"""Ranking metrics over file names (graded relevance 1 to 3)."""
from __future__ import annotations

import math
from typing import Any


def precision_at(ranked: list[str], relevant: dict[str, int], k: int) -> float | None:
    """Share of the returned top-k results that are relevant (None when nothing was returned)."""
    top = ranked[:k]
    return None if not top else sum(1 for d in top if d in relevant) / len(top)


def recall_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(1 for d in set(ranked[:k]) if d in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranked) if d in relevant), 0.0)


def ndcg_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def answer_matches(expected: dict[str, Any], answer: dict[str, Any]) -> bool:
    """Every expected key matches; numbers compare to 0.01, rows compare as sets of the expected keys."""
    for key, want in expected.items():
        got = answer.get(key)
        if key == "rows" and isinstance(want, list):
            if not isinstance(got, list):
                return False
            keys = sorted({k for row in want for k in row})
            if _rows(want, keys) != _rows(got, keys):
                return False
        elif _norm(want) != _norm(got):
            return False
    return True


def _rows(rows: list[dict[str, Any]], keys: list[str]) -> list[tuple]:
    return sorted(tuple(_norm(r.get(k)) for k in keys) for r in rows)


def _norm(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        return round(float(v), 2)
    try:
        return round(float(str(v).replace(",", "")), 2)
    except ValueError:
        return str(v).strip().lower()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))], 1)
