"""Deterministic date extraction with roles (issue, due, effective, expiry, signature) taken from nearby cue words."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from docintel.packs import Domain, default_domain

MONTHS = {m: i for i, names in enumerate([
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
    ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
    ("december", "dec")], start=1) for m in names}
_MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"

PATTERNS = [
    (re.compile(rf"\b{_DAY}\s+(?:day\s+of\s+)?{_MON},?\s+{_YEAR}\b", re.I), 0.95),            # 14 January 2026
    (re.compile(rf"\b{_MON}\s+{_DAY},?\s+{_YEAR}\b", re.I), 0.95),                             # January 14, 2026
    (re.compile(rf"\b{_YEAR}-(?P<m>[01]?\d)-(?P<d>[0-3]?\d)\b"), 0.95),                        # 2026-01-14
    (re.compile(r"\b(?P<a>[0-3]?\d)[/.\-](?P<b>[01]?\d)[/.\-](?P<year>(?:19|20)\d{2})\b"), 0.7),  # 14/01/2026
    (re.compile(rf"\b{_MON}\s+{_YEAR}\b", re.I), 0.6),                                         # March 2027
]


@dataclass
class FoundDate:
    value: date
    role: str
    start: int
    end: int
    confidence: float
    precision: str          # day | month


def _clause_window(window: str) -> str:
    """Keep only the part of the window after the last sentence or line boundary."""
    cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
    return window[cut + 1:] if cut >= 0 else window


def _role(text: str, start: int, roles: dict[str, list[str]]) -> str:
    """Role of the date from the cue closest before it; at equal distance the longer cue wins
    ("effective date" over "date")."""
    window = _clause_window(text[max(0, start - 70):start].lower())
    best, best_key = "date", (-1, 0)
    for role, cues in roles.items():
        for cue in cues:
            for m in re.finditer(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", window):
                key = (m.end(), len(cue))
                if key > best_key:
                    best, best_key = role, key
    return best


def _month(name: str) -> int:
    n = name.lower().rstrip(".")
    return MONTHS[n] if n in MONTHS else MONTHS[n[:3]]


def _make(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str, day_first: bool = True, domain: Domain | None = None) -> list[FoundDate]:
    roles = (domain or default_domain()).date_roles
    found: list[FoundDate] = []
    taken: list[tuple[int, int]] = []
    for rx, base_conf in PATTERNS:
        for m in rx.finditer(text):
            conf = base_conf
            if any(s <= m.start() < e or s < m.end() <= e for s, e in taken):
                continue
            g = m.groupdict()
            precision = "day"
            if g.get("mon") and g.get("day"):
                value = _make(int(g["year"]), _month(g["mon"]), int(g["day"]))
            elif g.get("m"):
                value = _make(int(g["year"]), int(g["m"]), int(g["d"]))
            elif g.get("a"):
                a, b = int(g["a"]), int(g["b"])
                if a > 12:
                    value = _make(int(g["year"]), b, a)
                elif b > 12:
                    value = _make(int(g["year"]), a, b)
                    conf = 0.8
                else:
                    value = _make(int(g["year"]), b, a) if day_first else _make(int(g["year"]), a, b)
                    conf = 0.6
            else:
                if re.search(r"\d\s*$", text[:m.start()]):     # "31 February 2026": an invalid full date, not a month
                    continue
                value = _make(int(g["year"]), _month(g["mon"]), 1)
                precision = "month"
            if value is None:
                continue
            taken.append((m.start(), m.end()))
            found.append(FoundDate(value, _role(text, m.start(), roles), m.start(), m.end(), conf, precision))
    found.sort(key=lambda f: f.start)
    return found
