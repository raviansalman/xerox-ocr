"""People, organizations, jurisdictions, emails and phone numbers.

Extraction is cue-based and conservative (high precision): a name is recorded when a configured cue such as
"Signed by", "Bill to" or "governed by the laws of" introduces it. No names are hardcoded; vocabulary comes from
the enabled domain packs (``docintel/packs``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from docintel import text as T
from docintel.packs import Domain, default_domain

NAME = r"(?P<name>[A-Z][a-z'’\-]+(?:[ \t]+(?:[A-Z]\.|[A-Z][a-z'’\-]+|(?:van|von|de|del|da|al|bin|ibn)(?=[ \t]+[A-Z]))){1,3})"
ORG_SUFFIX = (r"(?:LLC|L\.L\.C\.|Inc\.?|Incorporated|Corp\.?|Corporation|Co\.|Company|Ltd\.?|Limited|LLP|L\.P\.|LP|PLC|"
              r"GmbH|AG|S\.A\.|SAS|B\.V\.|N\.V\.|Pty\.? Ltd\.?|Holdings|Group|Systems|Technologies|Solutions|Partners|Bank)")
ORG = re.compile(rf"\b((?:[A-Z][A-Za-z0-9&'’.\-]*[ \t]+){{0,5}}{ORG_SUFFIX})(?=[\s,.;:)]|$)")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{2,4}[\s.\-]\d{3,4}[\s.\-]\d{3,4}(?![\w])")
NOT_NAMES = frozenset("""the this that date page section article clause agreement contract invoice total amount company
party parties customer client vendor supplier chief executive officer director general counsel manager president
signature signed name title by and of for to from dear sir madam mr mrs ms dr""".split())
_ORG_STOP = re.compile(r"^(?:the|this|that|and|or|by|between|with|from|to|of)\s+", re.I)
_JUR_PATTERNS = [
    (re.compile(r"governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?(?:the\s+)?laws?\s+of\s+(?:the\s+)?"
                r"(?P<j>[A-Za-z][A-Za-z .&'’\-]{2,60}?)(?=\s*[.,;\n)]|\s+(?:and|without|excluding|applicable|as)\b|$)", re.I), 0.95),
    (re.compile(r"governing\s+law\s*[:.\-–]?\s*(?:this\s+\w+\s+shall\s+be\s+governed\s+by\s+)?(?:the\s+)?(?:laws?\s+of\s+)?(?:the\s+)?"
                r"(?P<j>[A-Za-z][A-Za-z .&'’\-]{2,60}?)(?=\s*[.,;\n)]|$)", re.I), 0.9),
    (re.compile(r"laws\s+of\s+(?:the\s+)?(?P<j>[A-Za-z][A-Za-z .&'’\-]{2,60}?)\s+shall\s+govern", re.I), 0.9),
    (re.compile(r"under\s+the\s+(?P<j>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Labor|Labour|Civil|Commercial|Business|Corporations)\s+Code", 0), 0.75),
]
_JUR_PREFIX = re.compile(r"^(?:state|commonwealth|kingdom|republic|province|emirate|federal republic)\s+of\s+(?:the\s+)?", re.I)


@dataclass
class FoundEntity:
    type: str
    value: str
    role: str | None
    start: int
    end: int
    confidence: float
    code: str | None = None


def norm_name(value: str) -> str:
    return T.search_text(value)


def norm_org(value: str) -> str:
    v = T.search_text(value)
    v = re.sub(r"\b(?:llc|l l c|inc|incorporated|corp|corporation|co|company|ltd|limited|llp|lp|plc|gmbh|ag|sa|sas|bv|nv|pty|the)\b", " ", v)
    return " ".join(v.split()) or T.search_text(value)


def match_jurisdiction(phrase: str, domain: Domain | None = None) -> tuple[str, str] | None:
    """Canonical (name, code) for a phrase such as "the State of Califomia" (OCR-tolerant), else None."""
    folded = " " + T.fold(_JUR_PREFIX.sub("", phrase.strip())) + " "
    for alias, name, code in (domain or default_domain()).gazetteer:
        if f" {alias} " in folded:
            return name, code
    return None


def _clean_name(raw: str) -> str | None:
    words = raw.split()
    while words and words[-1].lower().strip(".,") in NOT_NAMES:
        words.pop()
    while words and words[0].lower().strip(".,") in NOT_NAMES:
        words.pop(0)
    if len(words) < 2 or len(words) > 4:
        return None
    return " ".join(words)


def find_people(text: str, domain: Domain | None = None) -> list[FoundEntity]:
    out: list[FoundEntity] = []
    cues = (domain or default_domain()).person_cues
    for cue in sorted(cues, key=len, reverse=True):
        role = cues[cue]
        # cue matched case-insensitively, the name itself must be capitalized
        rx = re.compile(rf"(?<![A-Za-z])(?i:{re.escape(cue)})\s*[:\-–]?\s*(?:(?i:by)\s+)?{NAME}")
        for m in rx.finditer(text):
            name = _clean_name(m.group("name"))
            if not name or any(e.start <= m.start("name") < e.end for e in out):
                continue
            out.append(FoundEntity("person", name, role, m.start("name"), m.start("name") + len(name), 0.85))
    return out


def find_organizations(text: str, domain: Domain | None = None) -> list[FoundEntity]:
    out: list[FoundEntity] = []

    def add(value: str, role: str | None, start: int, conf: float):
        value = _ORG_STOP.sub("", value.strip(" ,.;:-–")).strip()
        if len(value) < 3 or len(value) > 90 or not value[0].isupper() or value.lower() in NOT_NAMES:
            return
        if any(norm_org(e.value) == norm_org(value) for e in out):
            return
        out.append(FoundEntity("organization", value, role, start, start + len(value), conf))

    cues = (domain or default_domain()).party_cues
    for cue in sorted(cues, key=len, reverse=True):
        rx = re.compile(rf"(?<![A-Za-z])(?i:{re.escape(cue)})\s*:\s*(?P<v>[A-Z][^\n,;]{{2,80}})")
        for m in rx.finditer(text):
            add(m.group("v"), cues[cue], m.start("v"), 0.85)
    for m in re.finditer(r"\bbetween\s+(?P<a>[A-Z][^,;\n]{2,80}?)\s+and\s+(?P<b>[A-Z][^,;.\n(]{2,80}?)(?=\s*[,.;(\n]|\s+for\b|\s+dated\b|$)", text):
        add(m.group("a"), "party", m.start("a"), 0.8)
        add(m.group("b"), "party", m.start("b"), 0.8)
    for m in ORG.finditer(text):
        add(m.group(1), None, m.start(1), 0.7)
    return out


def find_jurisdictions(text: str, domain: Domain | None = None) -> list[FoundEntity]:
    out: list[FoundEntity] = []
    for rx, conf in _JUR_PATTERNS:
        for m in rx.finditer(text):
            hit = match_jurisdiction(m.group("j"), domain)
            if not hit or any(e.code == hit[1] and abs(e.start - m.start()) < 200 for e in out):
                continue
            out.append(FoundEntity("jurisdiction", hit[0], "governing_law", m.start(), m.end(), conf, hit[1]))
    return out


def find_contacts(text: str) -> list[FoundEntity]:
    out = [FoundEntity("email", m.group(0), None, m.start(), m.end(), 0.99) for m in EMAIL.finditer(text)]
    for m in PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 8 <= len(digits) <= 15 and not re.fullmatch(r"(?:19|20)\d{2}[01]\d[0-3]\d", digits):
            out.append(FoundEntity("phone", m.group(0).strip(), None, m.start(), m.end(), 0.7))
    return out
