"""End-to-end regression: ingest -> extract -> chunk -> embed -> Milvus -> /search.

Runs against real Milvus + Redis with the deterministic stand-in embedder
(tests/support/fake_embedder.py), so it validates plumbing, tenant scoping and
exact/lexical behaviour. It does not measure MPNet's semantic quality; run the
same queries against the Docker stack for that (see tests/README.md).
"""
import os
import time

import pytest
from fastapi.testclient import TestClient

from tests.support import documents
from tests.support.auth import headers

pytestmark = pytest.mark.integration

REGRESSION_QUERIES = ["FOR IMMEDIATE RELEASE", "press release", "Lisa Riordan", "StorageChain"]


def _ingest(task, path, file_id, user, file_type):
    r = task.apply(kwargs=dict(file_path=str(path), file_id=file_id, user_id=user,
                               original_filename=path.name, file_type=file_type))
    assert r.state == "SUCCESS", r.result
    return r.result


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from pymilvus import utility

    import ultimate_ui
    from src.ultimate_tasks import get_worker_vector_integration, process_ultimate_document_task as task

    d = tmp_path_factory.mktemp("corpus")
    docs = [
        ("alice", "press_release", documents.text_pdf(d / "press_release.pdf", documents.PRESS_RELEASE_PAGES), "application/pdf"),
        ("alice", "report", documents.numbered_report_pdf(d / "report.pdf", 12), "application/pdf"),
        ("alice", "invoice", documents.invoice_docx(d / "invoice.docx"),
         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]
    memo = d / "memo.txt"
    memo.write_text("Internal memo: printer maintenance window is Saturday.\n", encoding="utf-8")
    arabic = d / "arabic.txt"
    arabic.write_text("فاتورة رقم 2024 مقدمة إلى شركة الخليج للتجارة\n", encoding="utf-8")
    bob = d / "bob_release.txt"
    bob.write_text("FOR IMMEDIATE RELEASE Bob Corp confidential merger. Contact Lisa Riordan.\n", encoding="utf-8")
    docs += [("alice", "memo", memo, "text/plain"), ("alice", "arabic", arabic, "text/plain"),
             ("bob", "bob_release", bob, "text/plain")]
    for user, fid, path, ftype in docs:
        _ingest(task, path, fid, user, ftype)

    vi = get_worker_vector_integration()
    vi.vector_db.collection.flush()
    unloaded_rows = len(vi.vector_db.query_all_chunks(user_id="alice", limit=100))
    vi.vector_db._ensure_collection_loaded()
    time.sleep(1)
    yield {"client": TestClient(ultimate_ui.app), "vi": vi, "task": task, "dir": d,
           "unloaded_rows": unloaded_rows}
    for c in (os.environ["DOC_COLLECTION"], os.environ["IMG_COLLECTION"]):
        if utility.has_collection(c):
            utility.drop_collection(c)


def search(env, query, user="alice", key="service", **extra):
    body = {"query": query, "limit": 10, **extra}
    if user is not None:
        body["userId"] = user
    r = env["client"].post("/search", json=body, headers=headers(key))
    assert r.status_code == 200, r.text
    return [x["file_id"] for x in r.json()["results"]]


@pytest.mark.parametrize("query", REGRESSION_QUERIES + ["for immediate release", "Storage Chain"])
def test_regression_queries_rank_press_release_first(env, query):
    assert search(env, query)[0] == "press_release"


@pytest.mark.parametrize("method", ["vector", "semantic", "both"])
def test_person_query_in_every_search_mode(env, method):
    assert search(env, "Lisa Riordan", searchMethod=method)[0] == "press_release"


def test_docx_table_content_is_searchable(env):
    assert search(env, "Gulf Trading invoice")[0] == "invoice"


def test_arabic_text_document_is_retrievable(env):
    assert search(env, "فاتورة")[0] == "arabic"


@pytest.mark.parametrize("query", REGRESSION_QUERIES)
def test_tenant_isolation_for_well_formed_user_ids(env, query):
    assert "bob_release" not in search(env, query, user="alice")
    assert set(search(env, query, user="bob")) <= {"bob_release"}


def test_unknown_user_sees_nothing(env):
    assert search(env, "FOR IMMEDIATE RELEASE", user="carol") == []


def test_user_id_with_quote_still_finds_own_documents(env):
    # Guards the escaping fix for KD-SEC-01: a legitimate id containing a quote must match
    # its own rows, otherwise an "escaped" expression that Milvus rejects would look like isolation.
    path = env["dir"] / "quoted.txt"
    path.write_text("Quarterly toner audit for the OBrien account.\n", encoding="utf-8")
    _ingest(env["task"], path, "quoted_doc", 'o"brien', "text/plain")
    env["vi"].vector_db.collection.flush()
    hits = env["vi"].search_documents(query="toner audit", user_id='o"brien', limit=10)
    assert [h.file_id for h in hits] == ["quoted_doc"]


def test_crafted_user_id_cannot_read_other_tenants(env):
    r = env["client"].post("/search", headers=headers("service"),
                           json={"userId": 'alice" or user_id != "alice', "query": "FOR IMMEDIATE RELEASE"})
    assert r.status_code == 400
    # Below the API the expression is escaped too (KD-SEC-01).
    hits = env["vi"].search_documents(query="FOR IMMEDIATE RELEASE", user_id='alice" or user_id != "alice', limit=10)
    assert hits == []


def test_tenant_key_searches_only_its_own_tenant(env):
    assert search(env, "FOR IMMEDIATE RELEASE", user=None, key="alice_reader")[0] == "press_release"
    assert search(env, "FOR IMMEDIATE RELEASE", user=None, key="bob_reader") == ["bob_release"]


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-02: page_number is always 0")
def test_multipage_pdf_chunks_record_page_numbers(env):
    rows = env["vi"].vector_db.query_all_chunks(user_id="alice", limit=1000)
    pages = {r.metadata["page_number"] for r in rows if r.metadata["file_id"] == "report"}
    assert pages != {0}


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-DATA-01: extracted dates/entities are not persisted")
def test_extracted_years_are_persisted(env):
    row = env["vi"].vector_db.collection.query(expr='file_id == "press_release"', output_fields=["*"], limit=1)[0]
    assert 2024 in (row.get("years") or [])


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-MLV-03: query_all_chunks does not load the collection first")
def test_full_scan_works_before_first_vector_search(env):
    assert env["unloaded_rows"] > 0


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-DATA-02: re-ingesting a file appends a second copy of its chunks")
def test_reingest_replaces_previous_chunks(env):
    vi = env["vi"]

    def count():
        return sum(1 for r in vi.vector_db.query_all_chunks(user_id="alice", limit=1000)
                   if r.metadata["file_id"] == "memo")

    before = count()
    _ingest(env["task"], env["dir"] / "memo.txt", "memo", "alice", "text/plain")
    vi.vector_db.collection.flush()
    assert count() == before


# Destructive checks last (pytest runs tests in file order).
def test_admin_purge_requires_credentials(env):
    body = {"user_id": "bob", "confirm": "purge-all-vectors-for-user"}
    assert env["client"].post("/admin/purge-user-vectors", json=body).status_code == 401
    assert env["client"].post("/admin/purge-user-vectors", json=body, headers=headers("service")).status_code == 403
    assert search(env, "FOR IMMEDIATE RELEASE", user="bob") == ["bob_release"]


def test_delete_requires_owner(env):
    assert env["client"].post("/delete-document", json={"file_id": "bob_release"}).status_code == 401
    # Alice's key can only delete inside tenant alice: Bob's file is simply not found there.
    r = env["client"].post("/delete-document", json={"file_id": "bob_release"}, headers=headers("alice_uploader"))
    assert r.status_code == 404
    assert search(env, "FOR IMMEDIATE RELEASE", user="bob") == ["bob_release"]


def test_owner_can_delete_own_document(env):
    r = env["client"].post("/delete-document", json={"file_id": "memo"}, headers=headers("alice_uploader"))
    assert r.status_code == 200 and r.json()["chunks_deleted"] >= 1
    assert "memo" not in search(env, "printer maintenance", user="alice")
