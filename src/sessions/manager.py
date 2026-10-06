from __future__ import annotations

import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..models.state import QualificationState

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
MAX_STORED_MESSAGES = 40


@dataclass
class Session:
    id: str
    created_at: float
    last_activity: float
    history: list[dict] = field(default_factory=list)  # {"role", "content"}
    qualification: QualificationState = field(default_factory=QualificationState)
    message_count: int = 0


class SessionStore(Protocol):
    def get(self, session_id: str) -> Session | None: ...
    def put(self, session: Session) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def purge(self, older_than: float) -> int: ...
    def __len__(self) -> int: ...


class MemorySessionStore:
    """Single-process store. Swap for a Redis-backed store in Phase 8."""

    def __init__(self, max_sessions: int = 5000) -> None:
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._max = max_sessions

    def get(self, session_id: str) -> Session | None:
        return self._data.get(session_id)

    def put(self, session: Session) -> None:
        self._data[session.id] = session
        self._data.move_to_end(session.id)
        while len(self._data) > self._max:  # bound memory: drop least recent
            self._data.popitem(last=False)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def purge(self, older_than: float) -> int:
        stale = [k for k, s in self._data.items() if s.last_activity < older_than]
        for key in stale:
            del self._data[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._data)


class SessionManager:
    def __init__(
        self,
        store: SessionStore,
        ttl_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._ttl = ttl_seconds
        self._clock = clock
        self._last_purge = clock()

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        """Resolve a client-provided id. Unknown or expired ids never get
        adopted: a fresh server-generated id is issued instead."""
        now = self._clock()
        if now - self._last_purge > 60:
            self._store.purge(now - self._ttl)
            self._last_purge = now

        if session_id and SESSION_ID_RE.fullmatch(session_id):
            existing = self._store.get(session_id)
            if existing:
                if now - existing.last_activity <= self._ttl:
                    existing.last_activity = now
                    return existing, False
                self._store.delete(session_id)

        session = Session(id=secrets.token_urlsafe(32), created_at=now, last_activity=now)
        self._store.put(session)
        return session, True

    def peek(self, session_id: str | None) -> Session | None:
        """Read-only lookup: never creates or refreshes a session."""
        if not session_id or not SESSION_ID_RE.fullmatch(session_id):
            return None
        existing = self._store.get(session_id)
        if existing and self._clock() - existing.last_activity <= self._ttl:
            return existing
        return None

    def save(self, session: Session) -> None:
        session.last_activity = self._clock()
        if len(session.history) > MAX_STORED_MESSAGES:
            session.history = session.history[-MAX_STORED_MESSAGES:]
        self._store.put(session)
