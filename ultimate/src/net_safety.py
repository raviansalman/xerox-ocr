#!/usr/bin/env python3
"""
Outbound fetch safety for caller-supplied URLs (fileUrl): SSRF guard, redirect re-checks,
download size cap and log redaction.

Environment:
  URL_FETCH_ALLOWED_HOSTS   comma list; "example.com" exact, ".amazonaws.com" suffix. Empty = any public host.
  URL_FETCH_ALLOW_PRIVATE   "true" to allow private/loopback/link-local targets (local development only).
  MAX_DOWNLOAD_BYTES        default 209715200 (200 MB).

Residual risk: DNS can change between the check and the connection (rebinding). Pair this with
network egress rules in production.
"""

import ipaddress
import os
import socket
from typing import Dict, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

MAX_REDIRECTS = 5


class UnsafeURLError(ValueError):
    """The URL is not allowed to be fetched. A ValueError, so Celery does not retry it."""


def _allowed_hosts():
    return [h.strip().lower() for h in os.getenv("URL_FETCH_ALLOWED_HOSTS", "").split(",") if h.strip()]


def _allow_private() -> bool:
    return os.getenv("URL_FETCH_ALLOW_PRIVATE", "false").strip().lower() in ("1", "true", "yes")


def max_download_bytes() -> int:
    try:
        return int(os.getenv("MAX_DOWNLOAD_BYTES", str(200 * 1024 * 1024)))
    except ValueError:
        return 200 * 1024 * 1024


def _host_allowed(host: str, allowlist) -> bool:
    for entry in allowlist:
        if entry.startswith("."):
            if host.endswith(entry) or host == entry[1:]:
                return True
        elif host == entry:
            return True
    return False


def check_url(url: str) -> None:
    """Raise UnsafeURLError unless url is an http(s) URL whose host resolves only to public addresses."""
    try:
        parts = urlsplit(url or "")
    except ValueError as e:
        raise UnsafeURLError(f"Malformed URL: {e}")
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs can be fetched")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError("URL has no host")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with credentials are not allowed")
    allowlist = _allowed_hosts()
    if allowlist and not _host_allowed(host, allowlist):
        raise UnsafeURLError(f"Host '{host}' is not in URL_FETCH_ALLOWED_HOSTS")
    if _allow_private():
        return
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError) as e:
        raise UnsafeURLError(f"Cannot resolve host '{host}': {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise UnsafeURLError(f"Host '{host}' resolves to a non-public address")


def redact_url(url: str) -> str:
    """Drop query string and fragment (presigned signatures) for logging."""
    try:
        p = urlsplit(url or "")
        return urlunsplit((p.scheme, p.hostname or "", p.path, "", "")) + ("?…" if p.query else "")
    except ValueError:
        return "<unparseable url>"


def _request(method: str, url: str, *, timeout: float, headers: Optional[Dict[str, str]] = None,
             stream: bool = False) -> requests.Response:
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        resp = requests.request(method, url, timeout=timeout, headers=headers or {},
                                stream=stream, allow_redirects=False)
        if resp.is_redirect and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            resp.close()
            continue
        return resp
    raise UnsafeURLError("Too many redirects")


def safe_get(url: str, *, timeout: float = 60, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    """Streaming GET with every redirect hop re-validated."""
    return _request("GET", url, timeout=timeout, headers=headers, stream=True)


def safe_head(url: str, *, timeout: float = 10) -> requests.Response:
    return _request("HEAD", url, timeout=timeout)
