"""Optional grounded answers over retrieved evidence.

The engine is complete without this layer: every query already returns ranked documents with verified evidence, and
computed answers (counts, sums, lookups) come from extracted data. When a caller asks for an answer (``answer=true``),
this layer writes a short reply from the evidence of the top results:

* ``extractive`` (default): no language model; the reply is the evidence sentences that best match the question,
  each quoted verbatim with its document, page and character span.
* ``anthropic``: a Claude model writes the reply from the same evidence. The model is untrusted: it sees only the
  question and the evidence (as delimited data it is told never to obey), it has no tools, its output is constrained
  to a JSON schema, and every sentence is verified before it is returned (citations must exist, numbers and quoted
  text must appear in the cited evidence, most content words must be supported). Unsupported sentences are dropped;
  if nothing remains the layer abstains. Any provider failure falls back to the extractive reply.

Computed answers are never rewritten by a model, and nothing is generated when retrieval found no reliable evidence.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from docintel import metrics
from docintel.answering.evidence import EvidenceItem, collect
from docintel.answering.extractive import extractive_answer
from docintel.answering.verify import verify
from docintel.config import get_settings

logger = logging.getLogger(__name__)

COMPUTED_INTENTS = ("count", "percentage", "group", "sum", "average", "min", "max", "lookup")

__all__ = ["COMPUTED_INTENTS", "EvidenceItem", "collect", "extractive_answer", "generate", "verify"]


def _abstain(provider: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"provider": provider, "status": "abstained", "sentences": [], "citations": {}, "reason": reason, **extra}


def generate(question: str, out: dict[str, Any]) -> dict[str, Any]:
    """A grounded answer for a query result (the output of ``QueryEngine.run``)."""
    s = get_settings()
    provider = s.answer_provider
    t0 = time.perf_counter()
    if provider == "disabled":
        return {"provider": provider, "status": "disabled"}
    answer = out.get("answer") or {}
    if out.get("intent") in COMPUTED_INTENTS and answer.get("kind") not in (None, "none", "documents"):
        result = {"provider": "computation", "status": "computed", "text": answer.get("text", ""), "sentences": [],
                  "citations": {}, "note": "Computed answers come from extracted data and are not rewritten by a model."}
        metrics.ANSWERS.labels("computation", "computed").inc()
        return result
    items = collect(out, s.answer_max_evidence)
    if not items:
        metrics.ANSWERS.labels(provider, "abstained").inc()
        return _abstain(provider, "no reliable evidence was retrieved for this question")
    result: dict[str, Any]
    if provider == "anthropic":
        from docintel.answering.llm import ProviderError, generate_with_claude
        try:
            proposed = generate_with_claude(question, items)
        except ProviderError as e:
            logger.warning("answer provider failed; using extractive answer", extra={"error": e.kind})
            result = extractive_answer(question, items, out.get("terms") or [])
            result["degraded"] = f"answer model unavailable ({e.kind}); showing quoted evidence instead"
        else:
            result = verify(proposed, items, question)
            result["provider"] = "anthropic"
            result["model"] = s.answer_model
    else:
        result = extractive_answer(question, items, out.get("terms") or [])
    result["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    metrics.ANSWERS.labels(result["provider"], result["status"]).inc()
    return result
