"""Document understanding: classification, fields, entities, clauses and signature cues, with provenance."""
from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from docintel.models import Classification, Clause, Entity, FieldValue, Page, ParsedDocument, Span, Understanding
from docintel.packs import Domain, default_domain
from docintel.understanding import clauses as C
from docintel.understanding import entities as E
from docintel.understanding import relations as REL
from docintel.understanding.classify import classify
from docintel.understanding.dates import find_dates
from docintel.understanding.fields import find_amounts, find_identifiers

_SIGNATURE_TEXT = re.compile(r"(?:^|\n)\s*(?:/s/|signature\s*:|signed\s*:|signed\s+by\b|in\s+witness\s+whereof)", re.I)


def _snippet(text: str, start: int, end: int, pad: int = 70) -> str:
    a, b = max(0, start - pad), min(len(text), end + pad)
    return ("…" if a else "") + " ".join(text[a:b].split()) + ("…" if b < len(text) else "")


def detect_language(text: str) -> str:
    from docintel.understanding.language import detect
    return detect(text)


def _sp(page: Page, start: int, end: int) -> Span:
    return Span(page.number, start, end, page.block_at(start))


def guess_title(parsed: ParsedDocument, filename: str) -> str:
    if parsed.title:
        return parsed.title.strip()[:200]
    for page in parsed.pages[:1]:
        for line in page.text.splitlines():
            s = line.strip()
            if 3 <= len(s) <= 120 and sum(c.isalpha() for c in s) >= 3:
                return s
    return filename


def analyze(parsed: ParsedDocument, filename: str,
            embed: Callable[[list[str]], np.ndarray] | None = None, embed_key: str = "",
            domain: Domain | None = None) -> Understanding:
    domain = domain or default_domain()
    fields: list[FieldValue] = []
    entities: list[Entity] = []
    clauses: list[Clause] = []
    relations: list = []
    has_signature = False
    for page in parsed.pages:
        t = page.text
        if not t.strip() and not page.regions:
            continue
        for d in find_dates(t, domain=domain):
            fields.append(FieldValue(d.role, value_text=d.value.isoformat(), value_date=d.value,
                                     unit=None if d.precision == "day" else d.precision, page=page.number,
                                     snippet=_snippet(t, d.start, d.end), confidence=d.confidence,
                                     span=_sp(page, d.start, d.end)))
        for a in find_amounts(t, domain):
            fields.append(FieldValue(a.role, value_text=f"{a.currency} {a.value:,.2f}", value_num=a.value, unit=a.currency,
                                     page=page.number, snippet=_snippet(t, a.start, a.end), confidence=0.9,
                                     span=_sp(page, a.start, a.end)))
        for name, ident, s, e in find_identifiers(t, domain):
            fields.append(FieldValue(name, value_text=ident, page=page.number, snippet=_snippet(t, s, e), confidence=0.9,
                                     span=_sp(page, s, e)))
            entities.append(Entity("identifier", ident, ident.casefold(), name, page.number, _snippet(t, s, e), 0.9,
                                   span=_sp(page, s, e)))
        for j in E.find_jurisdictions(t, domain):
            fields.append(FieldValue("jurisdiction", value_text=j.value, unit=j.code, page=page.number,
                                     snippet=_snippet(t, j.start, j.end), confidence=j.confidence,
                                     span=_sp(page, j.start, j.end)))
            entities.append(Entity("jurisdiction", j.value, j.value.casefold(), j.role, page.number,
                                   _snippet(t, j.start, j.end), j.confidence, span=_sp(page, j.start, j.end)))
        for p in E.find_people(t, domain):
            entities.append(Entity("person", p.value, E.norm_name(p.value), p.role, page.number,
                                   _snippet(t, p.start, p.end), p.confidence, span=_sp(page, p.start, p.end)))
            if p.role == "signer":
                has_signature = True
                fields.append(FieldValue("signer", value_text=p.value, page=page.number,
                                         snippet=_snippet(t, p.start, p.end), confidence=p.confidence,
                                         span=_sp(page, p.start, p.end)))
        for o in E.find_organizations(t, domain):
            entities.append(Entity("organization", o.value, E.norm_org(o.value), o.role, page.number,
                                   _snippet(t, o.start, o.end), o.confidence, span=_sp(page, o.start, o.end)))
            if o.role in ("customer", "supplier", "party"):
                fields.append(FieldValue("party", value_text=o.value, page=page.number,
                                         snippet=_snippet(t, o.start, o.end), confidence=o.confidence,
                                         span=_sp(page, o.start, o.end)))
        for c in E.find_contacts(t):
            entities.append(Entity(c.type, c.value, c.value.casefold(), None, page.number, _snippet(t, c.start, c.end),
                                   c.confidence, span=_sp(page, c.start, c.end)))
        for cl in C.find_clauses(t, domain):
            clauses.append(Clause(cl.clause_type, cl.text[:3000], cl.ref, cl.heading, page.number, cl.confidence,
                                  _sp(page, cl.start, cl.end)))
        if _SIGNATURE_TEXT.search(t):
            has_signature = True
        for r in page.regions:
            if r.type == "signature":
                has_signature = True
                fields.append(FieldValue("signature_mark", value_text=f"signature-like mark at ({r.x0:.0f},{r.y0:.0f})-({r.x1:.0f},{r.y1:.0f})",
                                         page=page.number, snippet=None, confidence=r.confidence, method="visual"))
        relations += REL.pattern_relations(page, domain)
        relations += REL.attribute_relations(page)

    # De-duplicate (same field/entity value repeated on several pages keeps its first occurrence)
    seen, uniq_fields = set(), []
    for f in fields:
        key = (f.name, f.value_text, f.value_num, f.unit)
        if key not in seen:
            seen.add(key)
            uniq_fields.append(f)
    seen, uniq_entities = set(), []
    for e in entities:
        key = (e.type, e.value_norm, e.role)
        if key not in seen:
            seen.add(key)
            uniq_entities.append(e)

    first_pages = "\n".join(p.text for p in parsed.pages[:3])
    cls = classify(filename, first_pages, embed, embed_key, domain)
    if cls.label == "email" or parsed.kind == "email":
        cls = Classification("email", 0.99, "format")
    relations += REL.derived_relations(uniq_fields, uniq_entities)
    return Understanding(classification=cls, fields=uniq_fields, entities=uniq_entities, clauses=clauses,
                         has_signature=has_signature, language=detect_language(parsed.text),
                         title=guess_title(parsed, filename), relations=relations)
