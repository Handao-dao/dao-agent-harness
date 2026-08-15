"""In-memory SessionStore for tests and the first Runtime milestone."""

from __future__ import annotations

from threading import RLock

from agent_harness.session import Session


class InMemorySessionStore:
    """Keep live Session aggregates in process memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> Session:
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._lock:
            return self._sessions.setdefault(session_id, Session(id=session_id))

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            session.mark_events_persisted(session.unpersisted_events())

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
