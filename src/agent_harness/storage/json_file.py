"""Atomic UTF-8 JSON persistence for Session aggregates."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

from agent_harness.session import Session
from agent_harness.storage.codec import SessionCodec, SessionCodecError


class SessionStorageError(RuntimeError):
    """Raised when a Session file cannot be read or written safely."""


class JsonFileSessionStore:
    """Persist one versioned Session snapshot per file using atomic replacement.

    Loaded Sessions are cached as live aggregates so this store preserves the same
    single-process object semantics as ``InMemorySessionStore``. Multi-process
    coordination is intentionally outside the first persistence milestone.
    """

    def __init__(self, directory: str | Path, *, codec: SessionCodec | None = None) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        if not self._directory.is_dir():
            raise ValueError("Session storage path must be a directory")
        self._codec = codec or SessionCodec()
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> Session:
        self._validate_session_id(session_id)
        with self._lock:
            cached = self._sessions.get(session_id)
            if cached is not None:
                return cached

            path = self._session_path(session_id)
            if not path.exists():
                session = Session(id=session_id)
            else:
                session = self._read(path)
                if session.id != session_id:
                    raise SessionStorageError(
                        f"Session file identity mismatch: expected {session_id!r}, "
                        f"found {session.id!r}"
                    )
            self._sessions[session_id] = session
            return session

    def save(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise TypeError("session must be a Session")
        self._validate_session_id(session.id)
        with self._lock:
            try:
                document = self._codec.encode(session)
                serialized = json.dumps(
                    document,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            except (SessionCodecError, TypeError, ValueError) as exc:
                raise SessionStorageError(f"Session {session.id!r} cannot be encoded") from exc

            self._write_atomic(self._session_path(session.id), f"{serialized}\n")
            self._sessions[session.id] = session
            session.mark_events_persisted(session.unpersisted_events())

    def delete(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            was_cached = self._sessions.pop(session_id, None) is not None
            path = self._session_path(session_id)
            try:
                path.unlink()
            except FileNotFoundError:
                return was_cached
            except OSError as exc:
                raise SessionStorageError(f"Cannot delete Session {session_id!r}") from exc
            return True

    def _read(self, path: Path) -> Session:
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            return self._codec.decode(document)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            SessionCodecError,
            TypeError,
            ValueError,
        ) as exc:
            raise SessionStorageError(f"Cannot load Session file: {path.name}") from exc

    def _write_atomic(self, target: Path, content: str) -> None:
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._sync_directory()
        except OSError as exc:
            raise SessionStorageError(f"Cannot save Session file: {target.name}") from exc
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _session_path(self, session_id: str) -> Path:
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.json"

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty text")


__all__ = ["JsonFileSessionStore", "SessionStorageError"]
