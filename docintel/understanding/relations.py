"""Relations: subject —predicate→ object, with qualifiers and the sentence that states them.

Two sources:
* generic relations built from what extraction already found (signer, party, governing law) and from key-value
  lines ("Technician: Jane Roe" → attribute technician = Jane Roe), which need no domain knowledge;
* pattern relations from the enabled packs (``relations.yaml``), matched sentence by sentence; named groups other
  than subject and object become qualifiers ("notice": "30 days").
"""
from __future__ import annotations

import re

from docintel import text as T
from docintel.models import Entity, FieldValue, Page, Relation, Span
from docintel.packs import Domain

_SENTENCE = re.compile(r"[^.;]+(?:[.;]|$)")
_KEY_VALUE = re.compile(r"^(?P<label>[A-Z][\w /&.'()-]{1,40}?)\s*:\s+(?P<value>\S.{0,200})$")


def _span(page: Page, start: int, end: int) -> Span:
    return Span(page.number, start, end, page.block_at(start))


def _snippet(text: str, start: int, end: int) -> str:
    return " ".join(text[start:end].split())[:400]


def pattern_relations(page: Page, domain: Domain) -> list[Relation]:
    out: list[Relation] = []
    if not domain.relation_patterns:
        return out
    text = page.text
    for (b_start, _b_end), block in zip(page.block_offsets(), page.blocks, strict=True):
        flat = block.text.replace("\n", " ")          # sentences may wrap across lines; offsets are unchanged
        for sm in _SENTENCE.finditer(flat):
            sentence = sm.group(0)
            if len(sentence.strip()) < 8:
                continue
            for rp in domain.relation_patterns:
                m = rp.pattern.search(sentence)
                if not m:
                    continue
                groups = {k: " ".join(v.split()).strip(" ,") for k, v in m.groupdict().items() if v and v.strip(" ,")}
                subject = groups.pop("subject", None)
                if not subject:
                    continue
                obj = groups.pop("object", "") or "document"
                start = b_start + sm.start() + m.start()
                end = b_start + sm.start() + m.end()
                out.append(Relation(rp.predicate, "party", subject, "text" if obj != "document" else "document", obj,
                                    groups, _snippet(text, b_start + sm.start(), b_start + sm.end()), 0.75,
                                    f"pattern:{rp.pack}", _span(page, start, end)))
    return out


def attribute_relations(page: Page) -> list[Relation]:
    """Key-value lines anywhere in the page ("Ship to: Riverside depot", "Technician: Jane Roe")."""
    out: list[Relation] = []
    offsets = page.block_offsets()
    for bi, b in enumerate(page.blocks):
        if b.type not in ("key_value", "paragraph", "list_item"):
            continue
        base = offsets[bi][0]
        pos = 0
        for line in b.text.split("\n"):
            m = _KEY_VALUE.match(line)
            if m:
                label = " ".join(m.group("label").lower().split())
                value = m.group("value").strip()
                s = base + pos + m.start("value")
                out.append(Relation("attribute", "label", label, "value", value, {}, line[:400], 0.8, "key_value",
                                    _span(page, s, s + len(value))))
            pos += len(line) + 1
    return out


def derived_relations(fields: list[FieldValue], entities: list[Entity]) -> list[Relation]:
    """Relations implied by extracted fields and entities, anchored at the same spans."""
    out: list[Relation] = []
    sig_dates = [f for f in fields if f.name == "signature_date" and f.value_date]
    for e in entities:
        if e.role == "signer":
            q = {"date": sig_dates[0].value_text} if sig_dates and sig_dates[0].value_text else {}
            out.append(Relation("signed", "person", e.value, "document", "document", q, e.snippet, e.confidence,
                                "derived", e.span))
        elif e.type == "organization" and e.role in ("party", "customer", "supplier", "employer", "employee"):
            out.append(Relation("party_to", "organization", e.value, "document", "document", {"role": e.role},
                                e.snippet, e.confidence, "derived", e.span))
    for f in fields:
        if f.name == "jurisdiction" and f.value_text:
            out.append(Relation("governed_by", "document", "document", "jurisdiction", f.value_text, {},
                                f.snippet, f.confidence, "derived", f.span))
    return out


def normalize(value: str) -> str:
    return T.search_text(value)
