from __future__ import annotations
import logging
import sys
import time
import uuid
from contextvars import ContextVar
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True
def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:  
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level)
def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id_ctx.set(rid)
    return rid
class Timer:

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        status = "error" if exc_type else "ok"
        self.logger.info("%s took %.1fms status=%s", self.label, elapsed_ms, status)
