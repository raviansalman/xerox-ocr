"""API-key authentication, roles and tenant resolution.

The tenant is always taken from the authenticated key, never from the request body. Keys are configured as
SHA-256 hashes, either inline (``DOCINTEL_API_KEYS``) or in a mounted file (``DOCINTEL_API_KEYS_FILE``)::

    [
      {"name": "<tenant>-reader", "key_sha256": "<hex>", "tenant": "<tenant>", "roles": ["reader"]},
      {"name": "<tenant>-ingest", "key_sha256": "<hex>", "tenant": "<tenant>", "roles": ["uploader"]},
      {"name": "backend",     "key_sha256": "<hex>", "roles": ["service"]},
      {"name": "<tenant>-admin", "key_sha256": "<hex>", "tenant": "<tenant>", "roles": ["admin"]}
    ]

Roles: reader (query, read documents) < uploader (+ ingest, delete, reprocess) < admin (+ tenant administration).
``service`` keys are not bound to a tenant and must name the tenant in the ``X-Tenant-Id`` header (for a trusted
backend that serves several tenants).

``DOCINTEL_AUTH_MODE=dev`` disables authentication and acts as an admin of ``DOCINTEL_DEV_TENANT``. It is refused
when ``DOCINTEL_ENVIRONMENT=production``.

    python -m docintel.security generate        # new key and its hash
    python -m docintel.security hash <key>
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from docintel.config import get_settings

logger = logging.getLogger(__name__)

ROLE_IMPLIES: dict[str, frozenset[str]] = {
    "reader": frozenset({"reader"}),
    "uploader": frozenset({"uploader", "reader"}),
    "admin": frozenset({"admin", "uploader", "reader"}),
    "service": frozenset({"service", "admin", "uploader", "reader"}),
}
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class AuthContext:
    """Immutable identity of one request. Every data access receives it."""

    principal: str
    tenant_id: str
    roles: frozenset[str]

    def has(self, role: str) -> bool:
        return role in self.roles

    def require(self, role: str) -> None:
        if role not in self.roles:
            raise AuthError(403, f"This API key lacks the '{role}' role")


@dataclass(frozen=True)
class _Key:
    name: str
    tenant: str | None
    roles: frozenset[str]


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_valid_tenant_id(value: str | None) -> bool:
    return bool(value) and bool(TENANT_ID_RE.match(value))


def _load_entries() -> list:
    s = get_settings()
    if s.api_keys_file:
        with open(s.api_keys_file, encoding="utf-8") as f:
            return json.load(f)
    if s.api_keys.strip():
        return json.loads(s.api_keys)
    return []


def build_index(entries: list) -> dict[str, _Key]:
    index: dict[str, _Key] = {}
    for i, e in enumerate(entries):
        name = str(e.get("name") or f"key-{i}")
        digest = str(e.get("key_sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"API key '{name}': key_sha256 must be 64 hex characters")
        roles = set(e.get("roles") or [])
        if not roles or roles - set(ROLE_IMPLIES):
            raise ValueError(f"API key '{name}': roles must be a non-empty subset of {sorted(ROLE_IMPLIES)}")
        expanded = frozenset().union(*(ROLE_IMPLIES[r] for r in roles))
        tenant = str(e["tenant"]).strip() if e.get("tenant") else None
        if tenant and not is_valid_tenant_id(tenant):
            raise ValueError(f"API key '{name}': invalid tenant id")
        if not tenant and "service" not in expanded:
            raise ValueError(f"API key '{name}': keys without the service role must be bound to a tenant")
        if digest in index:
            raise ValueError(f"API key '{name}': duplicate key")
        index[digest] = _Key(name=name, tenant=tenant, roles=expanded)
    return index


@lru_cache(maxsize=1)
def key_index() -> dict[str, _Key]:
    return build_index(_load_entries())


def reset_auth_cache() -> None:
    key_index.cache_clear()


def _presented_key(headers: Mapping[str, str]) -> str | None:
    key = headers.get("x-api-key")
    if key:
        return key.strip()
    authz = headers.get("authorization") or ""
    if authz.lower().startswith("bearer "):
        return authz[7:].strip()
    return None


def authenticate(headers: Mapping[str, str]) -> AuthContext:
    s = get_settings()
    if s.auth_mode == "dev":
        if s.is_production:
            raise AuthError(503, "Development authentication mode is not allowed in production")
        return AuthContext(principal="dev", tenant_id=s.dev_tenant, roles=ROLE_IMPLIES["admin"])
    try:
        index = key_index()
    except Exception:
        logger.exception("API key configuration could not be loaded")
        raise AuthError(503, "Authentication configuration is invalid; see server logs") from None
    if not index:
        raise AuthError(503, "Authentication is not configured on this server")
    raw = _presented_key(headers)
    if not raw:
        raise AuthError(401, "Missing API key (X-API-Key header or Authorization: Bearer)")
    key = index.get(hash_key(raw))
    if key is None:
        raise AuthError(401, "Invalid API key")
    requested = (headers.get("x-tenant-id") or "").strip()
    if key.tenant:
        if requested and requested != key.tenant:
            raise AuthError(403, "X-Tenant-Id does not match the tenant of this API key")
        tenant = key.tenant
    else:
        if not requested:
            raise AuthError(400, "Service keys must send the X-Tenant-Id header")
        if not is_valid_tenant_id(requested):
            raise AuthError(400, "X-Tenant-Id contains unsupported characters")
        tenant = requested
    return AuthContext(principal=key.name, tenant_id=tenant, roles=key.roles)


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "generate":
        key = "dik_" + secrets.token_urlsafe(32)
        print(f"key:        {key}\nkey_sha256: {hash_key(key)}")
        return 0
    if len(argv) == 3 and argv[1] == "hash":
        print(hash_key(argv[2]))
        return 0
    print("usage: python -m docintel.security generate | hash <key>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
