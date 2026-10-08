"""Clause detection: numbered clauses (Article 12.4, Section 5, 7.2) and unnumbered clauses recognized from their
heading or opening sentence. Types come from the enabled domain packs (clause_types)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from docintel.packs import Domain, default_domain

_NUMBERED = re.compile(
    r"(?:(?<=\n)|^|(?<=[.:;]\s))(?P<head>(?:(?i:article|section|clause)\s+(?P<ref>\d+(?:\.\d+)*[a-z]?)"
    r"|(?P<ref2>\d+\.\d+(?:\.\d+)*)\.?)(?=[\s:.\-–])\s*[:.\-–]?\s*(?P<title>[A-Z][A-Za-z ,'&/\-]{2,70}?)?)(?=[:.\n]|\s+[a-z(]|$)")
_TITLED = re.compile(r"(?:(?<=\n)|^)(?P<title>[A-Z][A-Za-z ,'&/\-]{3,60})\s*[:.]\s+(?=[A-Z])")


@dataclass
class FoundClause:
    clause_type: str
    ref: str | None
    heading: str | None
    start: int
    end: int
    text: str
    confidence: float


def classify_clause(heading: str, body: str, domain: Domain | None = None) -> tuple[str | None, float]:
    head, first = heading.lower(), body[:240].lower()
    best, score = None, 0.0
    for ctype, words in (domain or default_domain()).clause_types.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", head):
                s = 0.95
            elif re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", first):
                s = 0.75
            else:
                continue
            if s > score:
                best, score = ctype, s
    return best, score


def find_clauses(text: str, domain: Domain | None = None) -> list[FoundClause]:
    domain = domain or default_domain()
    heads = []
    for m in _NUMBERED.finditer(text):
        ref = m.group("ref") or m.group("ref2")
        title = (m.group("title") or "").strip(" -–:.")
        heads.append((m.start("head"), m.end(), ref, title))
    for m in _TITLED.finditer(text):
        if any(abs(m.start() - h[0]) < 3 for h in heads):
            continue
        title = m.group("title").strip()
        if len(title.split()) <= 6:
            heads.append((m.start("title"), m.end(), None, title))
    heads.sort()
    out = []
    for i, (start, body_start, ref, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        end = min(end, body_start + 1500)
        para_end = text.find("\n\n", body_start)
        if 0 < para_end < end:
            end = para_end
        body = text[body_start:end].strip()
        ctype, conf = classify_clause(title, body, domain)
        if not ctype or not body:
            continue
        out.append(FoundClause(ctype, ref, title or None, start, end, text[start:end].strip(), conf))
    sentence_rules = [r for r in _SENTENCE_RULES if r[0] in domain.clause_types]
    for ctype, rx in sentence_rules:
        for m in re.finditer(rx, text, re.I):
            if any(c.clause_type == ctype and c.start <= m.start() < c.end for c in out):
                continue
            out.append(FoundClause(ctype, None, None, m.start(), m.end(), m.group(0).strip(), 0.7))
    out.sort(key=lambda c: c.start)
    return out


# Sentences that state a governing law or a termination right without a clause heading (applied only when the
# enabled packs define the clause type)
_SENTENCE_RULES = (("governing_law", r"[^.\n]*\bgoverned\s+by\s+the\s+laws?\s+of\b[^.\n]*(?:[.\n]|$)"),
                   ("governing_law", r"\bgoverning\s+law\s*:[^.\n]*(?:[.\n]|$)"),
                   ("termination", r"[^.\n]*\b(?:may|shall|can)\s+(?:terminate|end\s+(?:the|this)\s+(?:employment|agreement|contract|lease|engagement))\b[^.\n]*(?:[.\n]|$)"))
