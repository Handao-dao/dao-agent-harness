"""Persistence contract for Session aggregates."""

from __future__ import annotations

from typing import Protocol

from agent_harness.session import Session


class SessionStore(Protocol):
    def get_or_create(self, session_id: str) -> Session: ...

    def save(self, session: Session) -> None: ...

    def delete(self, session_id: str) -> bool: ...
