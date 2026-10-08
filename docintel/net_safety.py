"""Outbound fetch safety for URL ingestion: SSRF guard with redirect re-checks, size cap and log redaction.

Settings: ``DOCINTEL_URL_ALLOWED_HOSTS`` (comma list; ``example.com`` exact, ``.example.com`` suffix; empty = any
public host) and ``DOCINTEL_URL_ALLOW_PRIVATE`` (development only). DNS can change between the check and the
connection (rebinding); pair this with egress rules in production.
"""
from __future__ import annotations

import ipaddress
import socket
import time
from collections.abc import Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from docintel.config import get_settings

MAX_REDIRECTS = 5
NAT64 = ipaddress.ip_network("64:ff9b::/96")
SIX_TO_FOUR = ipaddress.ip_network("2002::/16")


class UnsafeURLError(ValueError):
    pass


def _host_allowed(host: str, allowlist: list[str]) -> bool:
    for entry in allowlist:
        if entry.startswith("."):
            if host.endswith(entry) or host == entry[1:]:
                return True
        elif host == entry:
            return True
    return False


def check_url(url: str) -> None:
    s = get_settings()
    try:
        parts = urlsplit(url or "")
    except ValueError as e:
        raise UnsafeURLError(f"Malformed URL: {e}") from e
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs can be fetched")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError("URL has no host")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with credentials are not allowed")
    allowlist = [h.strip().lower() for h in s.url_allowed_hosts.split(",") if h.strip()]
    if allowlist and not _host_allowed(host, allowlist):
        raise UnsafeURLError(f"Host '{host}' is not allowed")
    if s.url_allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError) as e:
        raise UnsafeURLError(f"Cannot resolve host '{host}'") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not public_address(ip):
            raise UnsafeURLError(f"Host '{host}' resolves to a non-public address")


def public_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """A globally routable unicast address, including the IPv4 address inside IPv6 translation forms (IPv4-mapped,
    NAT64 and 6to4), which would otherwise reach private IPv4 networks through a gateway."""
    if not ip.is_global or ip.is_multicast:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        inner = None
        if ip.ipv4_mapped:
            inner = ip.ipv4_mapped
        elif ip in NAT64:
            inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip in SIX_TO_FOUR:
            inner = ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
        if inner is not None:
            return public_address(inner)
    return True


def redact_url(url: str) -> str:
    try:
        p = urlsplit(url or "")
        return urlunsplit((p.scheme, p.hostname or "", p.path, "", "")) + ("?…" if p.query else "")
    except ValueError:
        return "<unparseable url>"


def fetch(url: str, max_bytes: int, timeout: float = 60.0, deadline_sec: float = 600.0) -> tuple[Iterator[bytes], str]:
    """Stream a URL after validating every redirect hop. Returns (chunks, suggested filename). ``timeout`` bounds each
    network operation, ``deadline_sec`` the whole download (a server trickling bytes cannot hold a worker forever),
    and the body is cut off at ``max_bytes`` whatever the server declared. Proxy and credential settings from the
    environment are ignored: the address check must apply to the connection actually made."""
    client = httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False)
    try:
        return _fetch(client, url, max_bytes, time.monotonic() + deadline_sec)
    except BaseException:
        client.close()
        raise


def _fetch(client: httpx.Client, url: str, max_bytes: int, deadline: float) -> tuple[Iterator[bytes], str]:
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        req = client.build_request("GET", url)
        resp = client.send(req, stream=True)
        if resp.is_redirect and resp.headers.get("location"):
            url = urljoin(url, resp.headers["location"])
            resp.close()
            continue
        if resp.status_code >= 400:
            resp.close()
            raise UnsafeURLError(f"URL returned HTTP {resp.status_code}")
        length = int(resp.headers.get("content-length") or 0)
        if length > max_bytes:
            resp.close()
            raise UnsafeURLError(f"file exceeds the {max_bytes} byte limit")
        name = urlsplit(url).path.rsplit("/", 1)[-1] or "download"
        cd = resp.headers.get("content-disposition", "")
        if "filename=" in cd:
            name = cd.split("filename=", 1)[1].strip().strip('";\'') or name

        def chunks(resp=resp, client=client) -> Iterator[bytes]:
            received = 0
            try:
                for chunk in resp.iter_raw():           # every network read, so a trickling server meets the deadline
                    received += len(chunk)
                    if received > max_bytes:
                        raise UnsafeURLError(f"file exceeds the {max_bytes} byte limit")
                    if time.monotonic() > deadline:
                        raise UnsafeURLError("download did not finish in time")
                    yield chunk
            finally:
                resp.close()
                client.close()
        return chunks(), name
    raise UnsafeURLError("Too many redirects")


class IterStream:
    """Minimal read()-able stream over an iterator of byte chunks."""

    def __init__(self, chunks: Iterator[bytes]):
        self._it = chunks
        self._buf = b""

    def read(self, n: int = -1) -> bytes:
        while n < 0 or len(self._buf) < n:
            try:
                self._buf += next(self._it)
            except StopIteration:
                break
        if n < 0:
            out, self._buf = self._buf, b""
        else:
            out, self._buf = self._buf[:n], self._buf[n:]
        return out
