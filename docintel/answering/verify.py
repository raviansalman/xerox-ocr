"""Verification of a generated answer against the evidence it cites.

A sentence survives only if: it cites at least one evidence item and every cited id exists; every number in it appears
in the cited evidence; every quoted string appears in the cited evidence; and most of its content words occur in the
cited evidence or the question. Everything else is dropped. With nothing left, the answer abstains.
"""
from __future__ import annotations

import re
from typing import Any

from docintel import text as T
from docintel.answering.evidence import EvidenceItem, warnings

MAX_SENTENCES = 8
MAX_SENTENCE_CHARS = 600
MIN_SUPPORT = 0.6
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTED = re.compile(r"[\"“]([^\"”]{3,200})[\"”]")


def _norm_number(n: str) -> str:
    n = n.replace(",", "")
    return n[:-3] if n.endswith(".00") else n


def _numbers(text: str) -> set[str]:
    out = set()
    for n in map(_norm_number, _NUMBER.findall(text)):
        out.add(n)
        out |= set(n.split("."))          # parts too, so a date written 1.3.2026 supports "2026"
    return out


def _stems(text: str) -> set[str]:
    return {w[:5] for w in T.search_text(text).split() if w not in T.STOPWORDS and len(w) > 2 and not w.isdigit()}


def check_sentence(text: str, cited: list[EvidenceItem], question: str) -> str | None:
    """None if the sentence is supported by the cited evidence, else the reason it is not."""
    if not cited:
        return "cites no evidence"
    if len(text) > MAX_SENTENCE_CHARS:
        return "too long"
    source = " ".join(i.clean_text for i in cited)          # withheld sentences support nothing
    for n in map(_norm_number, _NUMBER.findall(text)):
        if n not in _numbers(source):
            return f"number {n} is not in the cited evidence"
    norm_source = f" {T.search_text(source)} "
    for q in _QUOTED.findall(text):
        if f" {T.search_text(q)} " not in norm_source:
            return "quoted text is not in the cited evidence"
    words = _stems(text)
    if words:
        supported = len(words & (_stems(source) | _stems(question))) / len(words)
        if supported < MIN_SUPPORT:
            return f"only {supported:.0%} of its words are supported by the cited evidence"
    return None


def verify(proposed: dict[str, Any], items: list[EvidenceItem], question: str) -> dict[str, Any]:
    """Keep the supported sentences of a proposed answer ({"abstain": bool, "sentences": [{text, citations}]})."""
    by_id = {i.id: i for i in items}
    kept, dropped, citations = [], [], {}
    for s in (proposed.get("sentences") or [])[:MAX_SENTENCES]:
        text = str(s.get("text") or "").strip()
        ids = [str(c) for c in (s.get("citations") or [])]
        if not text:
            continue
        unknown = [c for c in ids if c not in by_id]
        reason = f"cites unknown evidence {unknown}" if unknown else check_sentence(text, [by_id[c] for c in ids], question)
        if reason:
            dropped.append({"text": text[:200], "reason": reason})
            continue
        kept.append({"text": text, "citations": ids})
        for c in ids:
            citations[c] = by_id[c].citation()
    out: dict[str, Any] = {"sentences": kept, "citations": citations}
    if dropped:
        out["dropped"] = dropped
    if not kept:
        out.update(status="abstained", reason="the model declined to answer from this evidence" if proposed.get("abstain")
                   else "no generated sentence was supported by the evidence")
    else:
        out["status"] = "answered"
    if warnings(items):
        out["warnings"] = warnings(items)
    return out
