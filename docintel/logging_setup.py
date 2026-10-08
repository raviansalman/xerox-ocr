"""Structured (JSON) logging with request correlation ids."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

from docintel.config import get_settings

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging() -> None:
    s = get_settings()
    root = logging.getLogger()
    root.handlers.clear()
    # the process's real stdout: in Celery worker children sys.stdout is Celery's redirect proxy, which drops records
    # that come back into logging through it (every child log line, errors included, was lost)
    h = logging.StreamHandler(sys.__stdout__ or sys.stdout)
    h.setFormatter(JsonFormatter() if s.log_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(h)
    root.setLevel(s.log_level.upper())
    for noisy in ("httpx", "httpcore", "pymilvus", "urllib3", "PIL", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
