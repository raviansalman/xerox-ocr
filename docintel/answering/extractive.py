"""Extractive answers: the evidence sentences that best match the question, quoted verbatim with their source span."""
from __future__ import annotations

from typing import Any

from docintel import text as T
from docintel.answering.evidence import EvidenceItem, sentences, warnings

MAX_SENTENCES = 3
# match types whose relevance was established by meaning (a concept or relation), not by shared words
_CONCEPTUAL = {"concept_clause", "relation", "concept", "structured", "entity"}


def _stem(w: str) -> str:
    return w[:5]


def question_words(question: str, terms: list[str] | None = None) -> set[str]:
    words = {w for w in T.search_text(" ".join([question, *(terms or [])])).split()
             if w not in T.STOPWORDS and len(w) > 2}
    return {_stem(w) for w in words}



def extractive_answer(question: str, items: list[EvidenceItem], terms: list[str] | None = None) -> dict[str, Any]:
    wanted = question_words(question, terms)
    scored = []
    for rank, item in enumerate(items):
        for a, b in sentences(item.text):
            if b - a < 12 or item.overlaps_withheld(a, b):
                continue
            words = {_stem(w) for w in T.search_text(item.text[a:b]).split()}
            hits = len(wanted & words)
            if hits == 0 and not (item.match_type in _CONCEPTUAL and rank < 2):
                continue
            scored.append((-hits, rank, a, b, item))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen, used_docs, out_sentences, citations = [], set(), [], {}
    for _, _, a, b, item in scored:
        if len(chosen) >= MAX_SENTENCES:
            break
        if (item.document_id, item.text[a:b]) in used_docs:
            continue
        used_docs.add((item.document_id, item.text[a:b]))
        chosen.append(item)
        quote = {"text": item.text[a:b], "citations": [item.id], "quote": True}
        if item.char_start is not None:
            quote.update(char_start=item.char_start + a, char_end=item.char_start + b)
        out_sentences.append(quote)
        citations[item.id] = item.citation()
    if not out_sentences:
        out = {"provider": "extractive", "status": "abstained", "sentences": [], "citations": {},
               "reason": "none of the retrieved evidence states an answer to the question"}
    else:
        out = {"provider": "extractive", "status": "answered", "sentences": out_sentences, "citations": citations,
               "note": "Quoted from the documents; each sentence links to its page and character span."}
    if warnings(items):
        out["warnings"] = warnings(items)
    return out
