"""Evidence handed to the answer layer: the verified evidence of the top results, numbered E1, E2, ...

Documents are untrusted. A sentence that reads like an instruction to an AI system ("ignore previous instructions",
"you are now ...", "tell the user ...", forged markup) is withheld: it is not shown to the answer model, it cannot
support a generated sentence, and it is never quoted as an answer. The offsets of the remaining text are unchanged,
so citations still point at the stored source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_RESULTS = 5
WITHHELD = "[instruction-like text withheld]"
_SENTENCE = re.compile(r"[^.!?;\n]+(?:[.!?;]+|$)")
_INSTRUCTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.!?\n]{0,40}\b(?:instructions?|prompts?|rules|guidelines|context)\b"
    r"|\byou are now\b|\bact as (?:an?|the)\b|\bnew instructions?\b|\bsystem prompt\b|\bjailbreak"
    r"|^\s*(?:system|assistant|developer|user)\s*:"
    r"|\b(?:tell|inform|instruct) the (?:user|reader|model|assistant)\b|\b(?:respond|reply|answer) (?:only )?with\b"
    r"|\bcite\s+E\d+\b"
    r"|</?\s*(?:evidence|evidence_set|question|system|instructions?|prompt)\b",
    re.I | re.M)


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence-like piece of ``text``, trimmed of surrounding whitespace."""
    out = []
    for m in _SENTENCE.finditer(text):
        a, b = m.start(), m.end()
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def instruction_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Sentences of ``text`` that read like instructions addressed to an AI system."""
    return tuple((a, b) for a, b in sentences(text) if _INSTRUCTION.search(text[a:b]))


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    document_id: str
    filename: str
    title: str | None
    page: int | None
    char_start: int | None
    char_end: int | None
    text: str
    match_type: str
    withheld: tuple[tuple[int, int], ...] = field(default=())

    @property
    def clean_text(self) -> str:
        """The text with withheld sentences blanked (same length, so offsets are unchanged)."""
        chars = list(self.text)
        for a, b in self.withheld:
            chars[a:b] = " " * (b - a)
        return "".join(chars)

    @property
    def model_text(self) -> str:
        """The text shown to an answer model: withheld sentences replaced by a marker."""
        out, pos = [], 0
        for a, b in self.withheld:
            out += [self.text[pos:a], WITHHELD]
            pos = b
        return "".join([*out, self.text[pos:]])

    def overlaps_withheld(self, a: int, b: int) -> bool:
        return any(a < wb and wa < b for wa, wb in self.withheld)

    def citation(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in ("document_id", "filename", "title", "page", "char_start", "char_end",
                                             "text", "match_type")}
        if self.withheld:
            out["withheld"] = [list(r) for r in self.withheld]
        return out


def collect(out: dict[str, Any], limit: int) -> list[EvidenceItem]:
    """Evidence items of the top results, best result first, without duplicates."""
    items: list[EvidenceItem] = []
    seen: set[tuple] = set()
    for r in (out.get("results") or [])[:MAX_RESULTS]:
        for e in r.get("evidence") or r.get("snippets") or []:     # v1 results carry snippets only
            text = e.get("text") or ""
            key = (r["document_id"], e.get("page"), e.get("char_start"), text.strip()[:80])
            if not text.strip() or key in seen:
                continue
            seen.add(key)
            items.append(EvidenceItem(f"E{len(items) + 1}", r["document_id"], r.get("filename") or "", r.get("title"),
                                      e.get("page"), e.get("char_start"), e.get("char_end"), text,
                                      e.get("match_type") or "", instruction_ranges(text)))
            if len(items) >= limit:
                return items
    return items


def warnings(items: list[EvidenceItem]) -> list[str]:
    flagged = [i.id for i in items if i.withheld]
    return [f"Instruction-like text in {', '.join(flagged)} was withheld from the answer."] if flagged else []
