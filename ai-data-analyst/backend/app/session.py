from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
import duckdb
from app.models import TableProfile
@dataclass
class Session:
    id: str
    conn: duckdb.DuckDBPyConnection
    tables: dict[str, TableProfile] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)  # role/content for LLM
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    def touch(self) -> None:
        self.last_active = time.time()
class SessionStore:
    def __init__(self, ttl_minutes: int = 120):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_minutes * 60

    def create(self) -> Session:
        sid = uuid.uuid4().hex
        conn = duckdb.connect(database=":memory:")
        session = Session(id=sid, conn=conn)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        self._reap_expired()
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
        return self.create()

    def _reap_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.last_active > self._ttl_seconds
            ]
            for sid in expired:
                try:
                    self._sessions[sid].conn.close()
                except Exception:
                    pass
                del self._sessions[sid]
session_store = SessionStore()
