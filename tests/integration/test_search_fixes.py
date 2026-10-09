"""Search fixes found in the v2 test report, each with the situation that showed it. Own tenant (tenantfixes)."""
import time

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration
KEY = "fixes_uploader"
DOCS = {
    "Invoice_INV-2024-0457.txt": "Lease invoice INV-2024-0457 for the VersaLink C405. Invoice date: 14 February 2024.",
    "Q3_2024_Print_Review.txt": "Print volume review. Colour pages printed: 1.2 million. Mono pages printed: 4.8 million.",
    "Store_receipt.txt": "Store receipt dated 15 August 2024 for staples.",
    "Supplies_note.txt": "Office supplies were ordered for the print room: paper reams, staples and toner.",
}


@pytest.fixture(scope="module")
def fixes(client):
    r = client.post("/api/v1/documents", headers=headers(KEY),
                    files=[("files", (n, t.encode())) for n, t in DOCS.items()])
    assert r.status_code == 201, r.text
    ids = {d["filename"]: d["id"] for d in r.json()["documents"]}
    for doc_id in ids.values():
        for _ in range(300):
            if client.get(f"/api/v1/documents/{doc_id}", headers=headers(KEY)).json()["status"] in ("indexed", "failed"):
                break
            time.sleep(0.1)
    return ids


def ask(client, q):
    r = client.post("/api/v1/query", headers=headers(KEY), json={"q": q, "limit": 10})
    assert r.status_code == 200, r.text
    return r.json()


def test_unknown_identifier_returns_nothing_rather_than_another_invoice(client, fixes):
    out = ask(client, "INV-2024-9921")                       # shares "INV" and "2024" with INV-2024-0457
    assert out["results"] == [] and out["answer"]["kind"] == "none"
    assert [r["filename"] for r in ask(client, "INV-2024-0457")["results"]] == ["Invoice_INV-2024-0457.txt"]


def test_a_title_containing_a_period_finds_its_document(client, fixes):
    files = [r["filename"] for r in ask(client, "Q3 2024 Print Review")["results"]]
    assert files and files[0] == "Q3_2024_Print_Review.txt", files


def test_a_calculation_with_nothing_to_compute_still_shows_matching_documents(client, fixes):
    out = ask(client, "How much did we spend on office supplies?")
    assert out["intent"] == "sum" and out["answer"]["kind"] == "none" and out["answer"].get("note")
    assert "Supplies_note.txt" in [r["filename"] for r in out["results"]]


def test_readiness_reports_an_embedder_that_went_away(client, fixes, monkeypatch):
    from docintel.config import get_settings
    from docintel.indexing.embeddings import get_embedder
    if not get_settings().semantic_enabled:
        pytest.skip("no embedder in this run")
    emb = get_embedder()
    assert client.get("/health/ready").status_code == 200
    monkeypatch.setattr(emb, "base_url", "http://127.0.0.1:9")      # nothing listens there
    r = client.get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["embedder"]["ok"] is False
