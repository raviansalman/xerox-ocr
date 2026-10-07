"""API-level characterization that needs no Milvus/Redis (validation, routing, health)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import ultimate_ui

    return TestClient(ultimate_ui.app)  # no context manager: skip the Milvus warm-up startup hook


MB = 1024 * 1024


@pytest.mark.parametrize("filename,file_type,size,queue", [
    ("report.pdf", "application/pdf", 1 * MB, "ultimate_pdf"),
    ("report.pdf", "application/pdf", 6 * MB, "ultimate_pdf_large"),
    ("scan.tiff", "image/tiff", 1 * MB, "ultimate_image"),
    ("photo.jpg", "", 9 * MB, "ultimate_image_large"),
    ("contract.docx", "", 1 * MB, "ultimate_word"),
    ("legacy.doc", "application/msword", 1 * MB, "ultimate_word"),
    ("deck.pptx", "", 1 * MB, "ultimate_powerpoint"),
    ("ledger.xlsx", "", 1 * MB, "ultimate_spreadsheet"),
    ("rows.csv", "text/csv", 1 * MB, "ultimate_spreadsheet"),
    ("notes.txt", "text/plain", 1 * MB, "ultimate_ocr"),
    ("unknown.bin", "", 1 * MB, "ultimate_ocr"),
])
def test_queue_routing_table(client, filename, file_type, size, queue):
    r = client.post("/admin/route-test", json={"filename": filename, "file_type": file_type, "size_bytes": size})
    assert r.status_code == 200
    assert r.json()["queue"] == queue


def test_search_requires_user_id(client):
    r = client.post("/search", json={"query": "press release"})
    assert r.status_code == 400 and r.json()["detail"] == "userId or user_id is required"


def test_search_rejects_empty_query(client):
    r = client.post("/search", json={"userId": "alice", "query": ""})
    assert r.status_code == 400 and r.json()["detail"] == "query is required"


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-API-01: HTTPException(400) is caught by a broad except and re-raised as 500")
def test_process_without_file_url_is_a_client_error(client):
    r = client.post("/process", json={"userId": "alice", "fileId": "f1"})
    assert r.status_code == 400


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-API-01: error responses include a full server traceback")
def test_process_errors_do_not_leak_tracebacks(client):
    r = client.post("/process", json={"userId": "alice", "fileId": "f1"})
    assert "Traceback" not in r.text


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OPS-02: /health hardcodes celery/redis as healthy without checking")
def test_health_does_not_claim_unchecked_dependencies(client):
    services = client.get("/health").json()["services"]
    assert services["celery"] != "healthy"  # no worker exists in the unit-test environment


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-SEC-06: search result text is inserted into innerHTML unescaped")
def test_ui_escapes_result_text():
    import ultimate_ui

    html = ultimate_ui.create_html_ui()
    assert "${tx.substring(0, 200)}" not in html
