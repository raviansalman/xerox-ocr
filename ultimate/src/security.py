#!/usr/bin/env python3
"""
API authentication, roles and tenant resolution.

The tenant (Milvus user_id) is taken from the authenticated API key, never trusted from
the request body. Keys are configured as SHA-256 hashes:

    API_KEYS_FILE=/run/secrets/api_keys.json      (or API_KEYS='<same JSON inline>')

    [
      {"name": "acme-reader",  "key_sha256": "<hex>", "tenant": "acme", "roles": ["reader"]},
      {"name": "acme-ingest",  "key_sha256": "<hex>", "tenant": "acme", "roles": ["uploader"]},
      {"name": "backend",      "key_sha256": "<hex>", "roles": ["service"]},
      {"name": "ops",          "key_sha256": "<hex>", "roles": ["admin"]}
    ]

Roles: reader (search, task status) < uploader (+ ingest, delete own documents)
< service (acts for the userId it sends; for a trusted backend) < admin (+ /admin/*).
A key with a tenant can only ever act for that tenant.

Generate a key:  python -m src.security generate
Hash a key:      python -m src.security hash <key>

AUTH_DISABLED=true turns authentication off for local development only.
"""

import hashlib
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)

ROLE_IMPLIES: Dict[str, FrozenSet[str]] = {
    "reader": frozenset({"reader"}),
    "uploader": frozenset({"uploader", "reader"}),
    "service": frozenset({"service", "uploader", "reader"}),
    "admin": frozenset({"admin", "service", "uploader", "reader"}),
}

# Tenant ids: Mongo ObjectIds, user000-style ids, UUIDs, emails. No quotes, spaces or slashes.
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


class AuthError(Exception):
    """Raised for authentication/authorization failures; status_code is the HTTP status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    name: str
    tenant: Optional[str]
    roles: FrozenSet[str]

    def has(self, role: str) -> bool:
        return role in self.roles


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def is_valid_tenant_id(value: str) -> bool:
    return bool(value) and bool(TENANT_ID_RE.match(value))


def auth_disabled() -> bool:
    return os.getenv("AUTH_DISABLED", "false").strip().lower() in ("1", "true", "yes")


_DEV_PRINCIPAL = Principal(name="auth-disabled", tenant=None, roles=ROLE_IMPLIES["admin"])


def _load_key_config() -> list:
    path = os.getenv("API_KEYS_FILE", "").strip()
    inline = os.getenv("API_KEYS", "").strip()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if inline:
        return json.loads(inline)
    return []


def _build_key_index(entries: list) -> Dict[str, Principal]:
    index: Dict[str, Principal] = {}
    for i, e in enumerate(entries):
        name = str(e.get("name") or f"key-{i}")
        digest = (e.get("key_sha256") or "").strip().lower()
        if not digest and e.get("key"):
            logger.warning("[AUTH] key '%s' is configured in plaintext; store key_sha256 instead", name)
            digest = hash_key(str(e["key"]))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"API key '{name}': key_sha256 must be 64 hex characters")
        roles = set(e.get("roles") or [])
        unknown = roles - set(ROLE_IMPLIES)
        if not roles or unknown:
            raise ValueError(f"API key '{name}': roles must be a non-empty subset of {sorted(ROLE_IMPLIES)}")
        expanded = frozenset().union(*(ROLE_IMPLIES[r] for r in roles))
        tenant = e.get("tenant")
        tenant = str(tenant).strip() if tenant else None
        if tenant and not is_valid_tenant_id(tenant):
            raise ValueError(f"API key '{name}': invalid tenant id")
        if not tenant and "service" not in expanded:
            raise ValueError(f"API key '{name}': reader/uploader keys must be bound to a tenant")
        if digest in index:
            raise ValueError(f"API key '{name}': duplicate key")
        index[digest] = Principal(name=name, tenant=tenant, roles=expanded)
    return index


_KEY_INDEX: Optional[Dict[str, Principal]] = None


def key_index() -> Dict[str, Principal]:
    global _KEY_INDEX
    if _KEY_INDEX is None:
        _KEY_INDEX = _build_key_index(_load_key_config())
        if auth_disabled():
            logger.warning("[AUTH] AUTH_DISABLED=true: every request is treated as admin. Never use outside local development.")
        elif not _KEY_INDEX:
            logger.error("[AUTH] No API keys configured (API_KEYS_FILE / API_KEYS); protected endpoints will return 503.")
    return _KEY_INDEX


def reset_auth_cache() -> None:
    """Forget loaded keys (tests and key rotation)."""
    global _KEY_INDEX
    _KEY_INDEX = None


def _presented_key(headers) -> Optional[str]:
    key = headers.get("x-api-key")
    if key:
        return key.strip()
    authz = headers.get("authorization") or ""
    if authz.lower().startswith("bearer "):
        return authz[7:].strip()
    return None


def authenticate(headers) -> Principal:
    """Resolve request headers to a Principal or raise AuthError."""
    if auth_disabled():
        return _DEV_PRINCIPAL
    index = key_index()
    if not index:
        raise AuthError(503, "Authentication is not configured on this server")
    raw = _presented_key(headers)
    if not raw:
        raise AuthError(401, "Missing API key (X-API-Key header or Authorization: Bearer)")
    principal = index.get(hash_key(raw))
    if principal is None:
        raise AuthError(401, "Invalid API key")
    return principal


def require_role(principal: Principal, role: str) -> None:
    if not principal.has(role):
        raise AuthError(403, f"This API key lacks the '{role}' role")


def resolve_tenant(principal: Principal, requested: Optional[object],
                   missing_detail: str = "userId is required") -> str:
    """The tenant this request acts for. A tenant-bound key always acts for its own tenant; a different userId is refused."""
    req = str(requested).strip() if requested is not None else ""
    if principal.tenant:
        if req and req != principal.tenant:
            raise AuthError(403, "userId does not match the tenant of this API key")
        return principal.tenant
    if principal.has("service"):
        if not req:
            raise AuthError(400, missing_detail)
        if not is_valid_tenant_id(req):
            raise AuthError(400, "userId contains unsupported characters")
        return req
    raise AuthError(403, "This API key is not bound to a tenant")


def _main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "generate":
        key = "xocr_" + secrets.token_urlsafe(32)
        print(f"key:        {key}\nkey_sha256: {hash_key(key)}")
        return 0
    if len(argv) == 3 and argv[1] == "hash":
        print(hash_key(argv[2]))
        return 0
    print("usage: python -m src.security generate | hash <key>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
