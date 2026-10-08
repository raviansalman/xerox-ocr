"""Deterministic language identification for document text.

Non-Latin scripts are identified from their Unicode blocks; Latin-script text is identified among English, German,
French, Spanish, Portuguese, Italian and Dutch by stop-word frequency. Anything else is reported as ``und``
(undetermined) rather than guessed. The result is an ISO 639-1 code or a script tag ("zh-Hani" style is not used:
CJK text is reported as ``cjk``).
"""
from __future__ import annotations

import re
from collections import Counter

_SCRIPTS = (
    ("ar", re.compile(r"[؀-ۿݐ-ݿ]")),
    ("he", re.compile(r"[֐-׿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("el", re.compile(r"[Ͱ-Ͽ]")),
    ("cjk", re.compile(r"[぀-ヿ一-鿿가-힯]")),
    ("th", re.compile(r"[฀-๿]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
)
_STOPWORDS = {
    "en": "the and of to in is that for with as on are this be by or it from at an which have not will".split(),
    "de": "der die und das ist nicht ein eine zu den mit von dem sich des auf für im werden auch".split(),
    "fr": "le la les et des est une un du que en pour dans qui sur pas au par avec ce sont".split(),
    "es": "el la los las y de que en un una es por con para del se no al lo como más".split(),
    "pt": "o a os as e de que em um uma é para com não do da dos das no na por".split(),
    "it": "il la lo gli le e di che in un una è per con non del della dei sono al".split(),
    "nl": "de het een en van is dat in op te voor met zijn niet aan er die worden ook".split(),
}
_WORD = re.compile(r"[a-zà-ÿ]+")
MIN_WORDS = 8


def detect(text: str) -> str:
    sample = (text or "")[:20000]
    letters = sum(1 for c in sample if c.isalpha())
    if letters < 20:
        return "und"
    for code, rx in _SCRIPTS:
        if len(rx.findall(sample)) > letters * 0.3:
            return code
    words = _WORD.findall(sample.lower())
    if len(words) < MIN_WORDS:
        return "und"
    counts = Counter(words)
    scores = {lang: sum(counts[w] for w in sw) for lang, sw in _STOPWORDS.items()}
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < max(2, len(words) * 0.04):
        return "und"
    return best
