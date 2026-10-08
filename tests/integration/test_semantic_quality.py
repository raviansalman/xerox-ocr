"""Semantic retrieval quality with the production embedding model.

Skipped with the stand-in embedder. Thresholds come from the calibration run on the test corpus with
all-mpnet-base-v2: relevant paraphrases score 0.43 to 0.75, unrelated questions stay below 0.27, so the 0.30 floor
returns nothing for them.
"""
import os

import pytest

from tests.conftest import headers

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("DOCINTEL_TEST_EMBEDDING_MODEL", "test-hash-768") == "test-hash-768",
    reason="needs a real embedding model (DOCINTEL_TEST_EMBEDDER_URL, DOCINTEL_TEST_EMBEDDING_MODEL)")]

PARAPHRASES = [
    ("how do I change the toner", "toner"),
    ("rules for working from home", "remote"),
    ("bill for managed printing", "invoice"),
    ("how long does the equipment lease last", "lease"),
    ("staff salary and pay", "payroll"),
    ("confidential information protection", "nda_ca"),
    ("network outage resolved", "network"),
]
UNRELATED = ["quantum chromodynamics lecture", "recipe for banana bread", "football match results", "qwerty asdf"]


def ask(client, q):
    r = client.post("/api/v1/query", headers=headers("acme_reader"), json={"q": q, "limit": 10})
    assert r.status_code == 200
    return r.json()


@pytest.mark.parametrize("q,want", PARAPHRASES)
def test_paraphrase_finds_the_document(client, ingested, q, want):
    inv = {v: k for k, v in ingested.items()}
    got = [inv.get(r["document_id"]) for r in ask(client, q)["results"]]
    assert want in got[:3], (q, got)


@pytest.mark.parametrize("q", UNRELATED)
def test_unrelated_question_returns_nothing(client, ingested, q):
    assert ask(client, q)["results"] == []
