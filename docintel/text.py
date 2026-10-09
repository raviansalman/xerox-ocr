"""Text normalization shared by ingestion and queries.

Ingest and query MUST use the same functions so that what is indexed is what is searched:

* ``normalize``     Unicode NFKC, unified quotes and dashes, Arabic normalization (below), case folding, collapsed
                    whitespace.
* ``search_text``   ``normalize`` with every non-alphanumeric character turned into a space, plus split variants of
                    CamelCase words ("DataVault" also indexes "data vault"). Feeds the full-text index.
* ``fold``          ``search_text`` with common OCR confusions folded (rn→m, vv→w, digits inside words, l→i), used
                    only for the OCR-tolerant tier. Both sides fold identically, so "California" and the OCR form
                    "Califomia" meet.
* ``identifiers``   canonical forms of identifier-like tokens ("INV-1024-77" → "inv102477").

Arabic normalization (part of ``normalize``, so ingest and queries agree): invisible format characters (ZWJ, ZWNJ,
bidi marks), tatweel and diacritics are removed; Arabic-Indic digits become ASCII digits and Arabic punctuation its
ASCII equivalent; Persian yeh and keheh become Arabic yeh and kaf; alef variants (آ أ إ ٱ) fold to ا, alef maqsura
(ى) to ي, ta marbuta (ة) to ه, and the hamza seats ؤ, ئ to و, ي. A query typed with or without diacritics or hamza
therefore finds the same text.

Stored source text is never rewritten; these are derived forms.
"""
from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                         "„": '"', "′": "'", "″": '"', "‐": "-", "‑": "-", "‒": "-",
                         "–": "-", "—": "-", "―": "-", "−": "-", " ": " ", "​": ""})
_ARABIC_DROP = re.compile("[\u0640\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed"
                          "\u08d3-\u08e1\u08e3-\u08ff\u061c\u200c-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_ARABIC_MAP = str.maketrans({**{chr(0x0660 + i): str(i) for i in range(10)},
                             **{chr(0x06F0 + i): str(i) for i in range(10)},
                             "٫": ".", "٬": ",", "،": ",", "؛": ";", "؟": "?",
                             "ی": "ي", "ک": "ك", "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
                             "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w]+|_+", re.UNICODE)
_CAMEL = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_CAMEL_PART = re.compile(r"[A-Z][a-z]+")
_IDENT = re.compile(r"(?<![\w])[A-Za-z0-9](?:[A-Za-z0-9]|[-/#_.](?=[A-Za-z0-9]))*[A-Za-z0-9](?![\w])")

STOPWORDS = frozenset("""a an and are as at be been by can could did do does for from had has have how i if in into is it
its me my of on or our please show tell that the their them there these this those to us was we were what when where which
who whom why will with would you your find list get give search any all""".split()
# Arabic function words, in normalized form. "علي" (on) is left out: after folding it is also the name Ali.
) | frozenset("""في من الي عن مع هذا هذه ذلك تلك التي الذي الذين و او ثم كان كانت يكون ما ماذا متي اين كيف هل لا لم لن قد
كل اي ان انه بين عند حتي اذا هو هي هم نحن انا""".split())


def arabic(text: str) -> str:
    """The Arabic part of ``normalize``: drops diacritics, tatweel and invisible marks, folds letter variants and
    maps Arabic-Indic digits and punctuation to ASCII. Other scripts pass through unchanged."""
    return _ARABIC_DROP.sub("", text).translate(_ARABIC_MAP)


def normalize(text: str | None) -> str:
    if not text:
        return ""
    t = arabic(unicodedata.normalize("NFKC", text).translate(_QUOTES))
    return _WS.sub(" ", t.casefold()).strip()


def tokens(text: str | None) -> list[str]:
    """Search tokens of a text (same tokenization as the full-text index)."""
    return [t for t in _NON_ALNUM.sub(" ", normalize(text)).split() if t]


def camel_variants(text: str | None) -> list[str]:
    """Space-split forms of CamelCase words: "DataVault" -> "data vault"."""
    if not text:
        return []
    out = []
    for m in _CAMEL.finditer(unicodedata.normalize("NFKC", text)):
        out.append(" ".join(p.lower() for p in _CAMEL_PART.findall(m.group(1))))
    return out


def search_text(text: str | None) -> str:
    base = " ".join(tokens(text))
    extra = sorted(set(camel_variants(text)))
    return base + (" " + " ".join(extra) if extra else "")


_FOLD_RULES = [
    (re.compile(r"rn"), "m"),
    (re.compile(r"vv"), "w"),
    (re.compile(r"(?<=[a-z])0(?=[a-z])|^0(?=[a-z]{2})|(?<=[a-z]{2})0$"), "o"),
    (re.compile(r"(?<=[a-z])[1l](?=[a-z])|^[1l](?=[a-z])|(?<=[a-z])[1l]$"), "i"),
    (re.compile(r"(?<=[a-z])5(?=[a-z])"), "s"),
    (re.compile(r"(?<=[a-z])8(?=[a-z])"), "b"),
]


def _fold_token(tok: str) -> str:
    if not any(c.isalpha() for c in tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters < len(tok) / 2:          # mostly digits: an identifier, keep as is
        return tok
    for rx, rep in _FOLD_RULES:
        tok = rx.sub(rep, tok)
    return tok


def fold(text: str | None) -> str:
    """OCR-tolerant form of ``search_text`` (apply to raw text or to an already normalized string)."""
    return " ".join(_fold_token(t) for t in search_text(text).split())


def identifiers(text: str | None) -> set[str]:
    """Canonical identifiers: tokens that mix letters and digits, digit groups joined by - / # _, and standalone runs
    of five or more digits (ticket and account numbers)."""
    if not text:
        return set()
    out = set()
    for m in _IDENT.finditer(arabic(unicodedata.normalize("NFKC", text).translate(_QUOTES))):
        raw = m.group(0)
        has_digit = any(c.isdigit() for c in raw)
        has_alpha = any(c.isalpha() for c in raw)
        has_sep = any(c in "-/#_." for c in raw)
        if not has_digit:
            continue
        if not has_alpha and not (has_sep and "." not in raw) and not (raw.isdigit() and len(raw) >= 5):
            continue                       # plain numbers are not identifiers, but runs of 5+ digits are codes
        canon = re.sub(r"[^0-9a-z]", "", raw.casefold())
        if len(canon) >= 3 and not canon.isalpha():
            out.add(canon)
    return out


def is_identifier_like(token: str) -> bool:
    return bool(identifiers(token)) or (token.isdigit() and len(token) >= 4)


def join_variants(words: list[str]) -> list[str]:
    """Variants that join adjacent words: ["data","vault"] -> ["datavault"]; ["multi","port","x20"] -> ["multiport x20", "multi portx20"]."""
    out = []
    if 2 <= len(words) <= 5:
        for i in range(len(words) - 1):
            out.append(" ".join(words[:i] + [words[i] + words[i + 1]] + words[i + 2:]))
        if len(words) <= 3:
            out.append("".join(words))
    return list(dict.fromkeys(v for v in out if v))


def content_words(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


_AR_WORDS = re.compile("[ء-ي]+")


def garbled_arabic(text: str | None) -> bool:
    """A PDF text layer whose Arabic was extracted wrongly, so the page should be read by OCR instead. Two failure
    modes are common in real files: glyph-positioned text that comes out as single letters ("ت قر ير" for
    "تقرير"), and lam-alef ligatures decoded in the wrong order ("األمم" for "الأمم"). In correct Arabic, under 2%
    of words are a single letter (other than و) and no word starts with an alef followed by another alef form;
    corrupted layers measured 52% and 7 to 12%."""
    words = _AR_WORDS.findall(_ARABIC_DROP.sub("", unicodedata.normalize("NFKC", text or "")))
    if len(words) < 20:
        return False
    single = sum(1 for w in words if len(w) == 1 and w != "و")
    swapped = sum(1 for w in words if len(w) > 2 and w[0] == "ا" and w[1] in "اأإآ")
    return single / len(words) > 0.2 or swapped / len(words) > 0.03


# Light Arabic stemming (Larkey et al., "light10"), applied to normalized tokens, so ة is already ه and أ/إ/آ are ا.
_AR_TOKEN = re.compile("^[ء-ي]+$")
_AR_ARTICLES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def is_arabic_word(token: str) -> bool:
    return bool(_AR_TOKEN.match(token))


def arabic_stem(token: str) -> str:
    """Light10 stem of one normalized Arabic token; anything that is not an Arabic word is returned unchanged."""
    if not _AR_TOKEN.match(token):
        return token
    w = token
    if len(w) > 3 and w.startswith("و"):
        w = w[1:]
    for art in _AR_ARTICLES:
        if w.startswith(art) and len(w) - len(art) >= 2:
            w = w[len(art):]
            break
    for suf in _AR_SUFFIXES:
        if len(w) > 2 and w.endswith(suf) and len(w) - len(suf) >= 2:
            w = w[: -len(suf)]
    return w


def stem_text(text: str | None) -> str:
    """``search_text`` with each Arabic word replaced by its light stem. Feeds the stemmed index (English words are
    stemmed by PostgreSQL's English configuration over this text), on the ingest and the query side alike."""
    return " ".join(arabic_stem(t) for t in search_text(text).split())
