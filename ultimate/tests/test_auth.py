"""Security boundary: authentication, roles and tenant resolution (no Milvus/Redis needed)."""
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src import security
from tests.support.auth import headers

H = hashlib.sha256(b"k").hexdigest()


@pytest.fixture(scope="module")
def client():
    import ultimate_ui

    return TestClient(ultimate_ui.app)


@pytest.fixture
def fresh_keys(monkeypatch):
    """Let a test swap the key config, then restore the suite-wide keys."""
    def _set(value=None, disabled=None):
        if value is None:
            monkeypatch.delenv("API_KEYS", raising=False)
        else:
            monkeypatch.setenv("API_KEYS", value)
        if disabled is not None:
            monkeypatch.setenv("AUTH_DISABLED", disabled)
        security.reset_auth_cache()
    yield _set
    security.reset_auth_cache()


# ---- key configuration -------------------------------------------------------

@pytest.mark.parametrize("entry,error", [
    ({"key_sha256": "abc", "roles": ["service"]}, "64 hex"),
    ({"key_sha256": H, "roles": ["superuser"]}, "roles"),
    ({"key_sha256": H, "roles": []}, "roles"),
    ({"key_sha256": H, "roles": ["reader"]}, "bound to a tenant"),
    ({"key_sha256": H, "roles": ["reader"], "tenant": 'bad"tenant'}, "invalid tenant"),
])
def test_invalid_key_config_is_rejected(entry, error):
    with pytest.raises(ValueError, match=error):
        security._build_key_index([entry])


def test_duplicate_keys_are_rejected():
    e = {"key_sha256": H, "roles": ["service"]}
    with pytest.raises(ValueError, match="duplicate"):
        security._build_key_index([e, dict(e)])


def test_roles_are_hierarchical():
    idx = security._build_key_index([{"key_sha256": H, "roles": ["uploader"], "tenant": "t1"}])
    p = idx[H]
    assert p.has("reader") and p.has("uploader") and not p.has("service") and not p.has("admin")


# ---- tenant resolution -------------------------------------------------------

def _p(tenant, *roles):
    return security.Principal("t", tenant, frozenset().union(*(security.ROLE_IMPLIES[r] for r in roles)))


def test_tenant_key_uses_its_own_tenant():
    assert security.resolve_tenant(_p("alice", "reader"), None) == "alice"
    assert security.resolve_tenant(_p("alice", "reader"), "alice") == "alice"


def test_tenant_key_cannot_act_for_another_tenant():
    with pytest.raises(security.AuthError) as e:
        security.resolve_tenant(_p("alice", "reader"), "bob")
    assert e.value.status_code == 403


def test_service_key_acts_for_requested_tenant_only_if_valid():
    assert security.resolve_tenant(_p(None, "service"), "65f0c1a2b3c4d5e6f7a8b9c0") == "65f0c1a2b3c4d5e6f7a8b9c0"
    for bad, code in ((None, 400), ("", 400), ('alice" or user_id != "alice', 400), ("a b", 400)):
        with pytest.raises(security.AuthError) as e:
            security.resolve_tenant(_p(None, "service"), bad)
        assert e.value.status_code == code


# ---- HTTP boundary -----------------------------------------------------------

PROTECTED = [
    ("post", "/search", {"json": {"userId": "alice", "query": "x"}}),
    ("post", "/process", {"json": {"userId": "alice", "fileId": "f", "fileUrl": "https://example.com/a.pdf"}}),
    ("post", "/process-file", {"files": {"file": ("a.txt", b"hi", "text/plain")}, "data": {"fileId": "f"}}),
    ("post", "/delete-document", {"json": {"file_id": "f", "user_id": "alice"}}),
    ("get", "/task-status/00000000-0000-0000-0000-000000000000", {}),
    ("get", "/admin/jobs/alice", {}),
    ("get", "/admin/stuck-jobs", {}),
    ("get", "/admin/queues", {}),
    ("get", "/admin/vector-storage-by-user", {}),
    ("post", "/admin/purge-user-vectors", {"json": {"user_id": "alice", "confirm": "purge-all-vectors-for-user"}}),
    ("post", "/admin/route-test", {"json": {"filename": "a.pdf"}}),
]


@pytest.mark.parametrize("method,path,kw", PROTECTED)
def test_protected_routes_require_a_key(client, method, path, kw):
    r = getattr(client, method)(path, **kw)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("method,path,kw", PROTECTED)
def test_protected_routes_reject_unknown_keys(client, method, path, kw):
    r = getattr(client, method)(path, headers={"X-API-Key": "nope"}, **kw)
    assert r.status_code == 401


def test_bearer_token_is_accepted(client):
    r = client.post("/admin/route-test", json={"filename": "a.pdf"},
                    headers={"Authorization": "Bearer test-admin-key"})
    assert r.status_code == 200


@pytest.mark.parametrize("path", ["/admin/stuck-jobs", "/admin/queues", "/admin/vector-storage-by-user"])
@pytest.mark.parametrize("who", ["alice_reader", "alice_uploader", "service"])
def test_admin_routes_require_admin(client, path, who):
    assert client.get(path, headers=headers(who)).status_code == 403


@pytest.mark.parametrize("who", ["alice_reader", "alice_uploader", "service"])
def test_purge_requires_admin(client, who):
    r = client.post("/admin/purge-user-vectors", headers=headers(who),
                    json={"user_id": "alice", "confirm": "purge-all-vectors-for-user"})
    assert r.status_code == 403


def test_reader_cannot_ingest_or_delete(client):
    assert client.post("/process", headers=headers("alice_reader"),
                       json={"fileId": "f", "fileUrl": "https://example.com/a.pdf"}).status_code == 403
    assert client.post("/delete-document", headers=headers("alice_reader"),
                       json={"file_id": "f"}).status_code == 403


def test_tenant_key_cannot_target_another_tenant(client):
    r = client.post("/search", headers=headers("alice_reader"), json={"userId": "bob", "query": "x"})
    assert r.status_code == 403
    r = client.post("/delete-document", headers=headers("alice_uploader"), json={"file_id": "f", "user_id": "bob"})
    assert r.status_code == 403
    assert client.get("/admin/jobs/bob", headers=headers("alice_reader")).status_code == 403


def test_service_key_must_name_a_valid_tenant(client):
    r = client.post("/delete-document", headers=headers("service"), json={"file_id": "f"})
    assert r.status_code == 400
    r = client.post("/search", headers=headers("service"), json={"userId": 'a" or user_id != "a', "query": "x"})
    assert r.status_code == 400


def test_public_routes(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_fails_closed_without_key_config(client, fresh_keys):
    fresh_keys(None)
    r = client.post("/search", headers=headers("service"), json={"userId": "alice", "query": "x"})
    assert r.status_code == 503


def test_auth_disabled_is_explicit_opt_in(client, fresh_keys):
    fresh_keys(None, disabled="true")
    r = client.post("/admin/route-test", json={"filename": "a.pdf"})
    assert r.status_code == 200


def test_cli_hash_matches_config_format():
    assert security.hash_key("k") == H
    assert json.loads('{"x": "%s"}' % H)
