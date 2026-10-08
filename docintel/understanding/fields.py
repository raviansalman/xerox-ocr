"""Amounts with currency and role, and labelled identifiers (invoice, contract, PO, reference numbers)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from docintel.packs import Domain, default_domain

CURRENCY_CODES = ("USD|SAR|EUR|GBP|AED|QAR|KWD|BHD|OMR|EGP|INR|PKR|JPY|CNY|CAD|AUD|CHF|SGD|HKD|ZAR|NGN|KES|TRY|SEK|NOK|DKK|"
                  "BRL|MXN|NZD|JOD")
SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "C$": "CAD", "A$": "AUD"}
_NUM = r"(?P<num>\d{1,3}(?:[,  ]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?P<mult>million|thousand|mn|bn|billion|[km](?![a-z]))?"
_AMOUNT = re.compile(
    rf"(?:(?P<code1>{CURRENCY_CODES})\s?|(?P<sym>US\$|C\$|A\$|[$€£¥₹])\s?){_NUM}"
    rf"|{_NUM.replace('num', 'num2').replace('mult', 'mult2')}\s?(?P<code2>{CURRENCY_CODES})\b",
    re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


@dataclass
class FoundAmount:
    value: float
    currency: str
    role: str
    start: int
    end: int


def _clause_window(window: str) -> str:
    """Keep only the part of the window after the last sentence or line boundary."""
    cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
    return window[cut + 1:] if cut >= 0 else window


def _role(text: str, start: int, roles: dict[str, list[str]]) -> str:
    window = _clause_window(text[max(0, start - 60):start].lower())
    best, pos = "amount", -1
    for role, cues in roles.items():
        for cue in cues:
            for m in re.finditer(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", window):
                if m.start() > pos:
                    best, pos = role, m.start()
    return best


def find_amounts(text: str, domain: Domain | None = None) -> list[FoundAmount]:
    roles = (domain or default_domain()).amount_roles
    out = []
    for m in _AMOUNT.finditer(text):
        g = m.groupdict()
        num = g.get("num") or g.get("num2")
        mult = (g.get("mult") or g.get("mult2") or "").lower()
        cur = (g.get("code1") or g.get("code2") or "").upper() or SYMBOLS.get(g.get("sym") or "", "")
        if not num or not cur:
            continue
        try:
            value = float(re.sub(r"[,  ]", "", num)) * _MULT.get(mult, 1.0)
        except ValueError:
            continue
        out.append(FoundAmount(value, cur, _role(text, m.start(), roles), m.start(), m.end()))
    return out


_ID_TOKEN = r"(?P<id>[A-Z0-9][A-Z0-9\-/._]{1,40}[A-Z0-9])"  # noqa: S105 (a pattern, not a password)


def find_identifiers(text: str, domain: Domain | None = None) -> list[tuple[str, str, int, int]]:
    """(field_name, identifier, start, end) for identifiers that follow a configured cue."""
    out, seen = [], set()
    for name, cues in (domain or default_domain()).identifier_fields.items():
        for cue in sorted(cues, key=len, reverse=True):
            rx = re.compile(rf"(?<![A-Za-z]){re.escape(cue)}\s*[:#.]?\s*(?:no\.?\s*|number\s*)?[:#]?\s*{_ID_TOKEN}", re.I)
            for m in rx.finditer(text):
                ident = m.group("id").rstrip(".")
                if not any(c.isdigit() for c in ident) or (m.start("id"), name) in seen:
                    continue
                if any(s == m.start("id") for _, _, s, _ in out):
                    continue
                seen.add((m.start("id"), name))
                out.append((name, ident, m.start("id"), m.end("id")))
    return out
