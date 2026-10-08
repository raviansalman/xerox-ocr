"""Rule-based query planner: natural-language question → ``QueryPlan``.

Deterministic and explainable. Vocabulary (document types, field words, jurisdictions, clause types, concepts) comes
from the enabled domain packs (``docintel/packs``); nothing here knows any customer, person, place or domain.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from docintel import text as T
from docintel.packs import Domain, default_domain
from docintel.search.plan import AmountFilter, DateFilter, Filters, QueryPlan
from docintel.understanding.dates import MONTHS, find_dates
from docintel.understanding.fields import CURRENCY_CODES

_Q_START = re.compile(r"^\s*(?:what|how|why|who|whom|when|where|which|does|do|is|are|can|could|should|did)\b", re.I)
_COUNT = re.compile(r"\b(?:how\s+many|number\s+of|count(?:\s+of)?|total\s+number)\b", re.I)
_PCT = re.compile(r"\b(?:what\s+)?(?:percentage|percent|proportion|share|fraction|ratio)\b|%\s+of", re.I)
_SUM = re.compile(r"\b(?:total|sum|combined|overall|aggregate)\s+(?:of\s+(?:the\s+|all\s+)?)?(?:[a-z]+\s+){0,2}?(?:value|amount|worth|spend|spending|cost|price|payments?)\b"
                  r"|\bhow\s+much\b|\b(?:spend|spent|spending)\b", re.I)
_AVG = re.compile(r"\b(?:average|mean)\b", re.I)
_MAX = re.compile(r"\b(?:largest|highest|biggest|maximum|max|most\s+expensive)\b", re.I)
_MIN = re.compile(r"\b(?:smallest|lowest|minimum|min|cheapest)\b", re.I)
_GROUP = re.compile(r"\b(?:per|by|for\s+each|each|broken\s+down\s+by|breakdown\s+by|grouped\s+by)\s+(?P<key>document\s+type|type|category|month|year|"
                    r"jurisdiction|governing\s+law|language|status|vendor|supplier|customer|client|party|counterparty|currency)\b", re.I)
_GROUP_KEYS = {"document type": "type", "category": "type", "governing law": "jurisdiction", "vendor": "party", "supplier": "party",
               "customer": "party", "client": "party", "counterparty": "party"}
_COMPARE = re.compile(r"\b(?:compare|comparison|versus|vs\.?|against|change\s+from|difference\s+between)\b", re.I)
_YEAR = re.compile(r"(?<![\w/\-.#])((?:19|20)\d{2})(?![\w/\-])")
_YEAR_CUES = frozenset("in during for of from since year fy dated issued expire expires expiring expired signed between and "
                       "effective due until by to ending ended starting started".split())
_QUARTER = re.compile(r"\bq([1-4])\s*(?:of\s+)?((?:19|20)\d{2})\b", re.I)
_MONTH_YEAR = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b", re.I)
_RANGE = re.compile(r"\b(?:between|from)\s+((?:19|20)\d{2})\s+(?:and|to|until|-)\s+((?:19|20)\d{2})\b", re.I)
_BEFORE_AFTER = re.compile(r"\b(before|after|since|prior\s+to|until|by|from)\s+((?:19|20)\d{2})\b", re.I)
_RELATIVE = re.compile(r"\b(next|this|last|coming|previous)\s+(year|month|quarter)\b|\b(?:in\s+the\s+)?next\s+(\d{1,3})\s+days\b", re.I)
_CMP = re.compile(r"\b(greater\s+than|more\s+than|higher\s+than|larger\s+than|over|above|exceeding|exceeds|exceed|at\s+least|"
                  r"less\s+than|lower\s+than|smaller\s+than|under|below|at\s+most|between)\s+", re.I)
_BARE_NUM = re.compile(r"(?P<cur>" + CURRENCY_CODES + r"|[$€£])?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<mult>k|m|million|thousand|bn|billion)?\b", re.I)
_SIGNED_BY = re.compile(r"\b(?:signed|executed|approved)\s+by\s+(?P<name>[A-Za-z][\w'.\-]*(?:\s+[A-Za-z][\w'.\-]*){0,3}?)"
                        r"(?=\s+(?:and|or|governed|under|in|with|for|from|that|which|who|during|before|after|since|on|expir\w*|issued|dated)\b|[?.,;]|$)", re.I)
_UNSIGNED = re.compile(r"\b(?:unsigned|not\s+(?:been\s+)?signed|without\s+(?:a\s+|any\s+)?signatures?|missing\s+(?:a\s+)?signatures?|lack(?:s|ing)?\s+(?:a\s+)?signatures?)\b", re.I)
_SIGNED = re.compile(r"\b(?:signed|executed|contain(?:s|ing)?\s+(?:a\s+|any\s+)?signatures?|with\s+(?:a\s+)?signatures?|ha(?:ve|s)\s+(?:a\s+|been\s+)?signatures?|signatures?)\b", re.I)
_JUR_CUE = re.compile(r"\b(?:governed|governing|law|laws|jurisdiction|legal)\b", re.I)
_PARTY = re.compile(r"\b(?:with|from|for|by|to|between|issued\s+to|billed\s+to)\s+(?P<name>(?:[A-Z][\w&.'\-]*)(?:\s+(?:[A-Z][\w&.'\-]*|&|of|and))*)")
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,200})[\"“”]")
_LOOKUP = re.compile(r"^\s*(?:what(?:'s|\s+is|\s+are|\s+was|\s+were)\s+the|tell\s+me\s+the|give\s+me\s+the|find\s+the|show\s+the)\s+(?P<field>.+?)\s+(?:of|for|in|on|from)\s+(?P<rest>.+?)\??\s*$", re.I)
_WHEN_EXPIRE = re.compile(r"^\s*when\s+(?:does|do|did|will)\s+(?P<rest>.+?)\s+(?P<verb>expire|end|start|begin|commence|become\s+effective)\??\s*$", re.I)
_WHO_SIGNED = re.compile(r"^\s*who\s+signed\s+(?P<rest>.+?)\??\s*$", re.I)

SCAFFOLD = [
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:show|find|list|get|give|search(?:\s+for)?|display|retrieve|look\s+(?:for|up)|locate|fetch)(?:\s+me)?\b",
    r"\bhow\s+many\b", r"\bnumber\s+of\b", r"\bcount(?:\s+of)?\b", r"\bwhat\s+(?:is|are|was|were)\s+the\b", r"\bwhich\b", r"\bwhat\b",
    r"\bdo\s+we\s+have\b", r"\bare\s+there\b", r"\bis\s+there\b", r"\bwe\s+have\b", r"\bdo\s+i\s+have\b", r"\bin\s+the\s+system\b",
    r"\b(?:all|any|our|my|the|of|in|are|is|was|were|be|been|do|does|have|has|there|where|whose|whom)\b",
    r"\b(?:documents?|files?|docs?|records?|pdfs?|items?)\b",
    r"\b(?:that|who|whose)\b", r"\b(?:contain(?:s|ing)?|mention(?:s|ing|ed)?|include(?:s|ing)?|reference(?:s|ing)?)\b",
    r"\b(?:related\s+to|relating\s+to|about|regarding|concerning|referring\s+to|on\s+the\s+topic\s+of)\b",
    r"\bwith\s+the\s+(?:word|phrase|term|text)\b", r"\bthe\s+(?:word|phrase|term|text)\b",
]
_KIND_WORDS = [
    (re.compile(r"\bscann(?:ed|s)\b", re.I), "kinds", ["scanned", "image", "mixed"]),
    (re.compile(r"\b(?:images?|photos?|pictures?)\b", re.I), "kinds", ["image"]),
    (re.compile(r"\b(?:spreadsheets?|excel|xlsx|xls|csv)\b", re.I), "kinds", ["spreadsheet"]),
    (re.compile(r"\b(?:presentations?|slides?|slide\s+decks?|powerpoints?|pptx?)\b", re.I), "kinds", ["presentation"]),
    (re.compile(r"\b(?:word\s+documents?|docx?)\b", re.I), "extensions", [".docx", ".doc"]),
    (re.compile(r"\bpdfs?\b", re.I), "extensions", [".pdf"]),
]


def _cut(text: str, span: tuple[int, int]) -> str:
    return text[:span[0]] + " " + text[span[1]:]


def _doc_types(q: str, domain: Domain) -> tuple[list[str], list[tuple[int, int]]]:
    """Document types mentioned in the question (type query terms, then family terms)."""
    low = q.lower()
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for term, labels in domain.type_terms:
        for m in re.finditer(rf"(?<![\w-]){re.escape(term)}(?![\w-])", low):
            if any(s <= m.start() < e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            found.extend(label for label in labels if label not in found)
    return found, spans


def _date_field(q: str, doc_types: list[str], domain: Domain) -> str:
    low = q.lower()
    for fname, words in domain.field_terms.items():
        if not fname.endswith("_date"):
            continue
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                return fname
    defaults = {domain.types[t].date_field for t in doc_types if t in domain.types and domain.types[t].date_field}
    return defaults.pop() if len(defaults) == 1 else "any"


def _parse_dates(q: str, today: date, doc_types: list[str], domain: Domain) -> tuple[list[DateFilter], list[tuple[int, int]]]:
    field = _date_field(q, doc_types, domain)
    out, spans = [], []

    def add(start, end, span, source):
        out.append(DateFilter(field, start, end, source))
        spans.append(span)

    for m in find_dates(q, domain=domain):
        if m.precision == "day":
            add(m.value, m.value, (m.start, m.end), q[m.start:m.end])
    for m in _QUARTER.finditer(q):
        y, qn = int(m.group(2)), int(m.group(1))
        start = date(y, 3 * qn - 2, 1)
        end = (date(y + 1, 1, 1) if qn == 4 else date(y, 3 * qn + 1, 1)) - timedelta(days=1)
        add(start, end, m.span(), m.group(0))
    for m in _MONTH_YEAR.finditer(q):
        if any(s <= m.start() < e for s, e in spans):
            continue
        y, mo = int(m.group(2)), MONTHS[m.group(1).lower()]
        end = (date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)) - timedelta(days=1)
        add(date(y, mo, 1), end, m.span(), m.group(0))
    for m in _RANGE.finditer(q):
        add(date(int(m.group(1)), 1, 1), date(int(m.group(2)), 12, 31), m.span(), m.group(0))
    for m in _BEFORE_AFTER.finditer(q):
        if any(s <= m.start() < e for s, e in spans):
            continue
        y, w = int(m.group(2)), m.group(1).lower()
        if w in ("before", "prior to"):
            add(None, date(y - 1, 12, 31), m.span(), m.group(0))
        elif w in ("until", "by"):
            add(None, date(y, 12, 31), m.span(), m.group(0))
        elif w == "from":
            add(date(y, 1, 1), date(y, 12, 31), m.span(), m.group(0))
        else:
            add(date(y, 1, 1) if w == "since" else date(y + 1, 1, 1), None, m.span(), m.group(0))
    for m in _RELATIVE.finditer(q):
        if m.group(3):
            add(today, today + timedelta(days=int(m.group(3))), m.span(), m.group(0))
            continue
        which, unit = m.group(1).lower(), m.group(2).lower()
        delta = {"next": 1, "coming": 1, "this": 0, "last": -1, "previous": -1}[which]
        if unit == "year":
            y = today.year + delta
            add(date(y, 1, 1), date(y, 12, 31), m.span(), m.group(0))
        elif unit == "month":
            mo = today.month + delta
            y = today.year + (mo - 1) // 12
            mo = (mo - 1) % 12 + 1
            end = (date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)) - timedelta(days=1)
            add(date(y, mo, 1), end, m.span(), m.group(0))
        else:
            qn = (today.month - 1) // 3 + delta
            y = today.year + qn // 4
            qn = qn % 4
            start = date(y, 3 * qn + 1, 1)
            end = (date(y + 1, 1, 1) if qn == 3 else date(y, 3 * qn + 4, 1)) - timedelta(days=1)
            add(start, end, m.span(), m.group(0))
    for m in _YEAR.finditer(q):
        if any(s <= m.start() < e for s, e in spans):
            continue
        before = q[:m.start()].rstrip().lower().split()
        if not before or before[-1] not in _YEAR_CUES:
            continue                                   # "INV 2026 00481": part of an identifier, not a date
        y = int(m.group(1))
        add(date(y, 1, 1), date(y, 12, 31), m.span(), m.group(0))
    return out, spans


def _parse_amounts(q: str) -> tuple[list[AmountFilter], list[tuple[int, int]]]:
    out, spans = [], []
    for m in _CMP.finditer(q):
        word = " ".join(m.group(1).lower().split())
        rest = q[m.end():]
        if word == "between":
            mm = re.match(_BARE_NUM.pattern + r"\s+(?:and|to|-)\s+" + _BARE_NUM.pattern.replace("?P<cur>", "?P<cur2>")
                          .replace("?P<num>", "?P<num2>").replace("?P<mult>", "?P<mult2>"), rest, re.I)
            if not mm:
                continue
            v1 = _num(mm.group("num"), mm.group("mult"))
            v2 = _num(mm.group("num2"), mm.group("mult2"))
            cur = _cur(mm.group("cur") or mm.group("cur2"))
            out.append(AmountFilter("between", v1, v2, cur, q[m.start():m.end() + mm.end()]))
            spans.append((m.start(), m.end() + mm.end()))
            continue
        mm = _BARE_NUM.match(rest)
        if not mm or not mm.group("num"):
            continue
        tail = rest[mm.end():mm.end() + 6].strip().upper()
        cur = _cur(mm.group("cur")) or (tail[:3] if re.match(CURRENCY_CODES, tail[:3] or "-") else None)
        if not cur and not mm.group("mult") and len(re.sub(r"\D", "", mm.group("num"))) < 3:
            continue                                   # "over 5" is not an amount
        op = ">" if word in ("greater than", "more than", "higher than", "larger than", "over", "above", "exceeding",
                             "exceeds", "exceed") else ">=" if word == "at least" else "<=" if word == "at most" else "<"
        out.append(AmountFilter(op, _num(mm.group("num"), mm.group("mult")), None, cur, q[m.start():m.end() + mm.end()]))
        spans.append((m.start(), m.end() + mm.end() + (4 if cur and not mm.group("cur") else 0)))
    return out, spans


def _num(num: str, mult: str | None) -> float:
    v = float(num.replace(",", ""))
    return v * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}.get((mult or "").lower(), 1)


def _cur(sym: str | None) -> str | None:
    if not sym:
        return None
    return {"$": "USD", "€": "EUR", "£": "GBP"}.get(sym, sym.upper())


def _jurisdictions(q: str, domain: Domain) -> tuple[list[str], list[tuple[int, int]]]:
    if not _JUR_CUE.search(q):
        return [], []
    folded = T.fold(q)
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for alias, name, _code in domain.gazetteer:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", folded) and name not in found:
            found.append(name)
            # remove the alias words from the question (compared in folded form, so OCR spellings match too)
            words = alias.split()
            for mm in re.finditer(r"\b" + r"\W+".join(r"\w+" for _ in words) + r"\b", q):
                if T.fold(mm.group(0)) == alias:
                    spans.append(mm.span())
                    break
    return found, spans


def _clause_types(q: str, domain: Domain) -> tuple[list[str], list[tuple[int, int]]]:
    low = q.lower()
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for ctype, words in domain.clause_types.items():
        for w in sorted(words, key=len, reverse=True):
            m = re.search(rf"\b{re.escape(w)}\s+(?:clauses?|provisions?|sections?|terms?|articles?)\b", low)
            if m:
                found.append(ctype)
                spans.append(m.span())
                break
    return found, spans


def _strip(text: str, patterns: list[str]) -> str:
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    return " ".join(text.split())


def _residue(q: str, spans: list[tuple[int, int]]) -> str:
    for s in sorted(spans, reverse=True):
        q = _cut(q, s)
    q = re.sub(r"[?!]+", " ", q)
    q = _strip(q, SCAFFOLD)
    q = re.sub(r"\b(?:governed\s+by|governing\s+law|laws?\s+of|the\s+state\s+of|law|laws|under|signed|executed|by|with|from|for|to|and|or|issued|dated|expir\w+|due|effective|starting|valid|in|on|during|between|since|before|after|until)\b", " ", q, flags=re.I) \
        if any(spans) else q
    return " ".join(w for w in q.split() if w.strip(" ,.;:-"))


def _lookup(q: str, domain: Domain) -> tuple[str | None, str]:
    m = _LOOKUP.match(q)
    if m:
        f = " ".join(m.group("field").lower().split())
        if f in domain.lookup_words:
            return domain.lookup_words[f], m.group("rest")
    m = _WHEN_EXPIRE.match(q)
    if m:
        verb = m.group("verb").lower()
        return ("expiry_date" if verb in ("expire", "end") else "effective_date"), m.group("rest")
    m = _WHO_SIGNED.match(q)
    if m:
        return "signer", m.group("rest")
    return None, q


def _filters(q: str, today: date, intent: str, domain: Domain) -> tuple[Filters, list[tuple[int, int]], list[str]]:
    f = Filters()
    spans: list[tuple[int, int]] = []
    notes: list[str] = []
    m = _SIGNED_BY.search(q)
    if m:
        f.signer = " ".join(w.capitalize() if w.islower() else w for w in m.group("name").split())
        spans.append(m.span())
    rest = _cut(q, spans[-1]) if spans else q
    if _UNSIGNED.search(rest):
        f.has_signature = False
        spans += [mm.span() for mm in _UNSIGNED.finditer(q)]
    elif not f.signer and _SIGNED.search(rest):
        f.has_signature = True
        spans += [mm.span() for mm in _SIGNED.finditer(q)]
    f.jurisdictions, js = _jurisdictions(q, domain)
    spans += js
    f.clause_types, cs = _clause_types(q, domain)
    spans += cs
    f.doc_types, ds = _doc_types(q, domain)
    spans += ds
    f.dates, dspans = _parse_dates(q, today, f.doc_types, domain)
    spans += dspans
    f.amounts, aspans = _parse_amounts(q)
    spans += aspans
    for rx, attr, values in _KIND_WORDS:
        mm = rx.search(q)
        if mm and not any(s <= mm.start() < e for s, e in spans):
            getattr(f, attr).extend(v for v in values if v not in getattr(f, attr))
            spans.append(mm.span())
    if intent not in ("search", "lookup", "fact"):
        def taken(a: int, b: int) -> bool:
            return any(s <= a < e or s < b <= e for s, e in spans)

        for pm in _PARTY.finditer(q):
            name = pm.group("name").strip()
            low = name.lower()
            if taken(*pm.span("name")) or low in MONTHS or re.match(r"(?:19|20)\d{2}$", name):
                continue
            words = [w for w in name.split() if w.lower() not in ("and", "of", "&")]
            if words:
                f.parties.append(" ".join(words))
                spans.append(pm.span())
    return f, spans, notes


_DOC_NOUNS = frozenset("documents document files file records record docs doc pdfs pages scans".split())


def _counts_documents(q: str, domain: Domain) -> bool:
    """True when "how many X" counts documents (X is a document word or type), False for facts inside documents
    such as "how many employees were paid"."""
    m = re.search(r"\b(?:how\s+many|number\s+of|count(?:\s+of)?)\s+(?:the\s+|our\s+|of\s+)?(?P<rest>.*)$", q, re.I)
    if not m:
        return True
    rest = m.group("rest").lower()
    first = (re.findall(r"[a-z][a-z\-]*", rest) or [""])[0]
    if first in _DOC_NOUNS or first in ("signed", "unsigned", "scanned", "active", "expired", "expiring"):
        return True
    types, _ = _doc_types(" ".join(rest.split()[:3]), domain)      # a document type among the first words
    return bool(types)


def _plural_doc_ref(text: str, domain: Domain) -> bool:
    _types, spans = _doc_types(text, domain)
    return any(text.lower()[s:e].endswith("s") for s, e in spans) or bool(re.search(r"\ball\b", text, re.I))


_AGG_WORDS = [r"\b(?:average|mean|total|sum|combined|overall|amount|amounts|value|values|worth|price|payment|payments|"
              r"largest|highest|biggest|maximum|smallest|lowest|minimum|cost|costs|much|how|did|does|do|spend|spent|spending|"
              r"compare|comparison|versus|vs|per|each|by)\b"]


def _clean_residue(text: str, intent: str, filters: Filters) -> str:
    if intent in ("sum", "average", "min", "max", "count", "percentage", "group") or filters.amounts:
        text = _strip(text, _AGG_WORDS)
    return text


def _amount_question(q: str, domain: Domain) -> bool:
    """Min/max wording applies to amounts when the question names an amount word or a document type."""
    return bool(re.search(r"\b(?:value|amount|payment|price|total|cost)", q, re.I) or _doc_types(q, domain)[0])


def concepts_in(question: str, domain: Domain) -> list[str]:
    """Pack concepts the question refers to, through any of their variants ("cancel" → termination)."""
    low = " " + " ".join(T.normalize(question).split()) + " "
    found: list[str] = []
    for variant, concept in domain.concept_index:
        if concept.name not in found and re.search(rf"(?<![a-z]){re.escape(variant)}(?![a-z])", low):
            found.append(concept.name)
    return found


def strategies_for(plan: QueryPlan) -> list[str]:
    """Which retrievers the question needs. Exact retrieval always runs when there is text; semantic retrieval is
    skipped for identifier-only questions (embeddings of a code carry no meaning)."""
    from docintel.retrieval.fusion import identifier_only
    out = []
    if plan.phrases and not T.tokens(_QUOTED.sub(" ", plan.question)):
        # only quoted text: results must contain it (exactly, or OCR-tolerantly), so no other evidence can help
        return ["exact", "fuzzy"] + (["structured"] if plan.filters.any() else [])
    if T.tokens(plan.text) or plan.phrases or plan.identifiers:
        out += ["exact", "lexical", "fuzzy", "metadata", "entity"]
        if not identifier_only(plan):
            out.append("semantic")
    if plan.concepts and plan.intent in ("search", "fact", "lookup"):
        out.append("contextual")
    if plan.filters.any():
        out.append("structured")
    return out


def _ordered(d: DateFilter) -> DateFilter:
    """A range written backwards ("from 2025 to 2023") means the same years in order."""
    if not (d.start and d.end and d.start > d.end):
        return d
    start, end = d.end, d.start
    if start.month == 12 and start.day == 31 and end.month == 1 and end.day == 1:   # whole years given backwards
        start, end = date(start.year, 1, 1), date(end.year, 12, 31)
    return DateFilter(d.field, start, end, d.source)


def plan_query(question: str, today: date | None = None, domain: Domain | None = None) -> QueryPlan:
    today = today or date.today()
    domain = domain or default_domain()
    q = " ".join((question or "").split())
    phrases = [p.strip() for p in _QUOTED.findall(q) if p.strip()]
    unquoted = _QUOTED.sub(" ", q)

    intent = "search"
    group_by = None
    lookup_field, lookup_rest = _lookup(unquoted, domain)
    counted = _COUNT.search(unquoted)
    if _PCT.search(unquoted):
        intent = "percentage"
    elif _GROUP.search(unquoted) and (counted or _plural_doc_ref(unquoted, domain)) and not _SUM.search(unquoted):
        key = " ".join(_GROUP.search(unquoted).group("key").lower().split())
        intent, group_by = "group", _GROUP_KEYS.get(key, key)
    elif lookup_field and not re.search(r"\bhow\s+many\b", unquoted, re.I) and not (
            lookup_field.endswith("amount") and _SUM.search(unquoted) and _plural_doc_ref(lookup_rest, domain)):
        intent = "lookup"
    elif counted:
        intent = "count" if _counts_documents(unquoted, domain) else "fact"
    elif _SUM.search(unquoted):
        intent = "sum"
    elif _AVG.search(unquoted):
        intent = "average"
    elif _MAX.search(unquoted) and _amount_question(unquoted, domain):
        intent = "max"
    elif _MIN.search(unquoted) and _amount_question(unquoted, domain):
        intent = "min"

    if intent in ("sum", "average", "min", "max") and _GROUP.search(unquoted):
        key = " ".join(_GROUP.search(unquoted).group("key").lower().split())
        group_by = _GROUP_KEYS.get(key, key)

    base = None
    if intent == "percentage":
        m = re.search(r"\bof\s+(?:our\s+|the\s+|all\s+)?(?P<base>.+?)\s+(?:are|is|were|was|have|has|contain|contains|include|includes|that|which)\s+(?P<cond>.+)$", unquoted, re.I)
        if m:
            base, _, _ = _filters(m.group("base"), today, intent, domain)
            base.doc_types_hard = True
            unquoted = m.group("cond")
        else:
            base = Filters()

    work = lookup_rest if intent == "lookup" else unquoted
    filters, spans, notes = _filters(work, today, intent, domain)
    if group_by:
        spans += [m.span() for m in _GROUP.finditer(work)]
    text = _clean_residue(_residue(work, spans), intent, filters)
    if intent in ("search", "fact", "lookup") and T.content_words(text) and filters.doc_types:
        # the text carries meaning beyond the type ("maintenance contract"): keep the type words in the text and
        # use the type only as a ranking preference
        type_spans = set(_doc_types(work, domain)[1])
        text = _clean_residue(_residue(work, [s for s in spans if s not in type_spans]), intent, filters)
    # a quoted phrase is both a required phrase and the lexical text
    if phrases:
        text = " ".join([*phrases, text]).strip()
    semantic = _strip(question, SCAFFOLD[:1]).strip(" ?") or question
    ids = sorted(T.identifiers(" ".join([*phrases, work])))
    words = T.tokens(text)
    exact = bool(phrases or ids) or (0 < len(words) <= 6 and not _Q_START.match(question or ""))

    if intent in ("count", "percentage", "sum", "average", "min", "max", "group"):
        filters.doc_types_hard = True
    elif filters.doc_types and not T.content_words(text):
        filters.doc_types_hard = True             # "show invoices from 2026": a listing
    agg_field = None
    if intent in ("sum", "average", "min", "max"):
        agg_field = "periodic_amount" if re.search(r"\bmonthly|per\s+month\b", unquoted, re.I) else "total_amount"
    compare: list[DateFilter] = []
    if intent in ("count", "sum", "average", "min", "max") and len(filters.dates) >= 2 and \
            all(d.start and d.end for d in filters.dates) and (_COMPARE.search(unquoted) or re.search(r"\band\b", unquoted)):
        compare = sorted(filters.dates, key=lambda d: d.start)
        filters.dates = [DateFilter(compare[0].field, compare[0].start, compare[-1].end, "comparison")]
        notes.append("comparison of " + ", ".join(d.source for d in compare))
    if filters.signer and intent == "search":
        notes.append(f"signer filter: {filters.signer}")
    for f in [filters, *([base] if base else [])]:
        f.dates = [_ordered(d) for d in f.dates]
    plan = QueryPlan(question=question, intent=intent, text=text, semantic_text=semantic, phrases=phrases,
                     identifiers=ids, exact_intent=exact, filters=filters, base=base, aggregate_field=agg_field,
                     group_by=group_by, lookup_field=lookup_field if intent == "lookup" else None, notes=notes,
                     packs=domain.packs, compare=compare, concepts=concepts_in(question, domain))
    plan.strategies = strategies_for(plan)
    return plan
