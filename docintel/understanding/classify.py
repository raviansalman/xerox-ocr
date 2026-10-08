"""Document classification against the document types of the enabled domain packs.

1. Rules: title patterns on the title zone (filename + first lines) and weighted keywords in the first pages.
2. Fallback: cosine similarity between the document and each type's description (embedding prototypes).
3. Otherwise ``other``. Confidence is reported with the method so aggregations can separate uncertain results.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np

from docintel import text as T
from docintel.models import Classification
from docintel.packs import Domain, default_domain

logger = logging.getLogger(__name__)
PROTOTYPE_FLOOR = 0.38
CONFIDENT_RULE_SCORE = 2.5


def title_zone(filename: str, first_page: str, lines: int = 6) -> str:
    stem = re.sub(r"[_\-.]+", " ", re.sub(r"\.[A-Za-z0-9]{1,5}$", "", filename or ""))
    head = [ln.strip() for ln in (first_page or "").splitlines() if ln.strip()][:lines]
    return "\n".join([stem, *head])


def rule_scores(filename: str, first_pages: str, domain: Domain | None = None) -> dict[str, float]:
    domain = domain or default_domain()
    zone = title_zone(filename, first_pages)
    body = T.normalize(first_pages[:6000])
    scores: dict[str, float] = {}
    for label, patterns in domain.title_patterns.items():
        t = domain.types[label]
        s = 0.0
        if any(p.search(zone) for p in patterns):
            s += t.title_weight
        for kw, w in t.keywords.items():
            if w and re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", body):
                s += w
        if s:
            scores[label] = s
    return scores


# Embeddings of each type description, per (embedding model, pack set). Bounded by the number of distinct pack
# sets in use; entries are immutable once computed.
_PROTOTYPES: dict[tuple[str, tuple[str, ...]], tuple[list[str], np.ndarray]] = {}


def _prototypes(embed_key: str, domain: Domain, embed: Callable[[list[str]], np.ndarray]) -> tuple[list[str], np.ndarray]:
    key = (embed_key, domain.packs)
    if key not in _PROTOTYPES:
        labels = [label for label, t in domain.types.items() if t.description]
        _PROTOTYPES[key] = (labels, embed([domain.types[label].description for label in labels]))
    return _PROTOTYPES[key]


def classify(filename: str, first_pages: str, embed: Callable[[list[str]], np.ndarray] | None = None,
             embed_key: str = "", domain: Domain | None = None) -> Classification:
    domain = domain or default_domain()
    scores = rule_scores(filename, first_pages, domain)
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top, top_s = ranked[0]
        fam = domain.types[top].family
        # the generic member of the top type's family ("contract") does not compete with the specific type
        rest = [s for label, s in ranked[1:] if not (domain.types[label].generic and domain.types[label].family == fam)]
        second = rest[0] if rest else 0.0
        if top_s >= CONFIDENT_RULE_SCORE:
            conf = min(0.98, 0.55 + 0.08 * top_s) * (1.0 - 0.45 * (second / top_s))
            return Classification(top, round(max(conf, 0.35), 3), "rules")
    if embed is not None and first_pages.strip() and domain.types:
        try:
            labels, protos = _prototypes(embed_key, domain, embed)
            vec = embed([title_zone(filename, first_pages) + "\n" + first_pages[:1500]])[0]
            sims = protos @ vec
            i = int(np.argmax(sims))
            if sims[i] >= PROTOTYPE_FLOOR:
                return Classification(labels[i], round(float(sims[i]), 3), "embedding_prototype")
        except Exception:                       # the fallback is optional; rules and the default still apply
            logger.warning("embedding-prototype classification unavailable", exc_info=True)
    if scores:
        top, _ = max(scores.items(), key=lambda kv: kv[1])
        return Classification(top, 0.4, "rules_weak")
    return Classification("other", 0.5, "default")
