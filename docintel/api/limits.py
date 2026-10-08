"""Request body limits, enforced before a body is buffered or spooled to disk.

Uploads may be as large as ``DOCINTEL_MAX_REQUEST_BYTES`` (all files of one request together); every other request
body is limited to ``SMALL_BODY_BYTES``. A declared ``Content-Length`` over the limit is refused at once; a body sent
without one (chunked) is counted as it arrives and refused as soon as it exceeds the limit.
"""
from __future__ import annotations

import json

from docintel.config import get_settings

SMALL_BODY_BYTES = 1024 * 1024
UPLOAD_PATHS = ("/api/v1/documents",)


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS", "DELETE"):
            return await self.app(scope, receive, send)
        upload = scope["method"] == "POST" and scope["path"].rstrip("/") in UPLOAD_PATHS
        limit = get_settings().max_request_bytes if upload else SMALL_BODY_BYTES
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return await _refuse(send, limit)
        received, started, refused = 0, False, False

        async def counted_receive():
            nonlocal received, started, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:              # answer now; the application sees the client go away
                    if not started:
                        started = True
                        await _refuse(send, limit)
                    refused = True
                    return {"type": "http.disconnect"}
            return message

        async def tracked_send(message):
            nonlocal started
            if refused:
                return                            # the 413 has been sent; drop the application's late response
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except Exception:
            if not refused:
                raise


async def _refuse(send, limit: int) -> None:
    body = json.dumps({"detail": f"request body exceeds the limit of {limit} bytes"}).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
