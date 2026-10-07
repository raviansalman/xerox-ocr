"""Test API keys. Only their SHA-256 hashes are put into the API_KEYS config, as in production."""
import hashlib
import json

KEYS = {
    "service": ("test-service-key", None, ["service"]),
    "admin": ("test-admin-key", None, ["admin"]),
    "alice_reader": ("alice-reader-key", "alice", ["reader"]),
    "alice_uploader": ("alice-uploader-key", "alice", ["uploader"]),
    "bob_reader": ("bob-reader-key", "bob", ["reader"]),
}


def api_keys_json() -> str:
    entries = []
    for name, (raw, tenant, roles) in KEYS.items():
        e = {"name": name, "key_sha256": hashlib.sha256(raw.encode()).hexdigest(), "roles": roles}
        if tenant:
            e["tenant"] = tenant
        entries.append(e)
    return json.dumps(entries)


def headers(name: str) -> dict:
    return {"X-API-Key": KEYS[name][0]}
