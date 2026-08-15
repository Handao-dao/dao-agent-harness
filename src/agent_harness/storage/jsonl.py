"""Append-only JSONL Session event storage."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from agent_harness.session import Session, SessionEventConflictError
from agent_harness.storage.codec import SessionCodec, SessionCodecError
from agent_harness.storage.event_codec import SessionEventCodec
from agent_harness.storage.json_file import SessionStorageError

SESSION_EVENT_LOG_VERSION = 1


class JsonlSessionStore:
    """Persist accepted Session mutations as a versioned append-only event log."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        if not self._directory.is_dir():
            raise ValueError("Session storage path must be a directory")
        self._session_codec = SessionCodec()
        self._event_codec = SessionEventCodec(self._session_codec)
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> Session:
        self._validate_session_id(session_id)
        with self._lock:
            cached = self._sessions.get(session_id)
            if cached is not None:
                return cached
            path = self._session_path(session_id)
            session = self._read(path) if path.exists() else Session(id=session_id)
            if session.id != session_id:
                raise SessionStorageError(
                    f"Session log identity mismatch: expected {session_id!r}, found {session.id!r}"
                )
            self._sessions[session_id] = session
            return session

    def save(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise TypeError("session must be a Session")
        self._validate_session_id(session.id)
        with self._lock:
            events = session.unpersisted_events()
            path = self._session_path(session.id)
            cached = self._sessions.get(session.id)
            if path.exists() and cached is not session:
                raise SessionStorageError(
                    "An existing JSONL Session must be loaded before it can be saved"
                )
            if not path.exists():
                if (
                    session.entries
                    or session.pending_inputs
                    or session.context_summaries
                ) and not events:
                    raise SessionStorageError(
                        "A new JSONL Session must be created through durable Session events"
                    )
                records = [self._encode_header(session)]
                records.extend(self._event_codec.encode(event) for event in events)
                self._write_new(path, records)
            elif events:
                self._append(path, [self._event_codec.encode(event) for event in events])
            session.mark_events_persisted(events)
            self._sessions[session.id] = session

    def delete(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            was_cached = self._sessions.pop(session_id, None) is not None
            try:
                self._session_path(session_id).unlink()
            except FileNotFoundError:
                return was_cached
            except OSError as exc:
                raise SessionStorageError(f"Cannot delete Session {session_id!r}") from exc
            return True

    def _read(self, path: Path) -> Session:
        try:
            content = self._read_complete_content(path)
            lines = content.splitlines()
            if not lines:
                raise SessionStorageError(f"Session log is empty: {path.name}")
            header = json.loads(lines[0])
            session = self._decode_header(header)
            event_ids: set[str] = set()
            for line_number, line in enumerate(lines[1:], start=2):
                try:
                    event = self._event_codec.decode(json.loads(line))
                    if event.id in event_ids:
                        raise SessionCodecError(f"Duplicate SessionEvent ID: {event.id}")
                    event_ids.add(event.id)
                    session.apply_event(event)
                except (json.JSONDecodeError, SessionCodecError, SessionEventConflictError) as exc:
                    raise SessionStorageError(
                        f"Invalid Session event at {path.name}:{line_number}"
                    ) from exc
            return session
        except SessionStorageError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise SessionStorageError(f"Cannot load Session log: {path.name}") from exc

    def _read_complete_content(self, path: Path) -> str:
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            last_newline = raw.rfind(b"\n")
            if last_newline < 0:
                raise SessionStorageError(f"Session log has no complete header: {path.name}")
            complete_length = last_newline + 1
            with path.open("r+b") as handle:
                handle.truncate(complete_length)
                handle.flush()
                os.fsync(handle.fileno())
            raw = raw[:complete_length]
        return raw.decode("utf-8")

    def _encode_header(self, session: Session) -> dict[str, Any]:
        return {
            "type": "session",
            "version": SESSION_EVENT_LOG_VERSION,
            "id": session.id,
            "created_at": self._session_codec._encode_datetime(
                session.created_at, "Session.created_at"
            ),
            "metadata": self._session_codec._copy_json_value(
                session.metadata, "Session.metadata"
            ),
        }

    def _decode_header(self, value: Any) -> Session:
        header = self._session_codec._require_mapping(value, "Session log header")
        version = header.get("version")
        if header.get("type") != "session" or type(version) is not int:
            raise SessionCodecError("Invalid Session log header")
        if version != SESSION_EVENT_LOG_VERSION:
            raise SessionCodecError(f"Unsupported Session event log version: {version!r}")
        metadata = self._session_codec._require_mapping(
            header.get("metadata"), "Session.metadata"
        )
        created_at = self._session_codec._decode_datetime(
            header.get("created_at"), "Session.created_at"
        )
        return Session(
            id=self._session_codec._require_text(header.get("id"), "Session.id"),
            created_at=created_at,
            updated_at=created_at,
            metadata=dict(
                self._session_codec._copy_json_value(metadata, "Session.metadata")
            ),
        )

    def _write_new(self, path: Path, records: list[dict[str, Any]]) -> None:
        content = self._serialize_records(records)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._sync_directory()
        except OSError as exc:
            raise SessionStorageError(f"Cannot create Session log: {path.name}") from exc
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    def _append(self, path: Path, records: list[dict[str, Any]]) -> None:
        content = self._serialize_records(records)
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SessionStorageError(f"Cannot append Session log: {path.name}") from exc

    @staticmethod
    def _serialize_records(records: list[dict[str, Any]]) -> str:
        try:
            return "".join(
                f"{json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}\n"
                for record in records
            )
        except (TypeError, ValueError) as exc:
            raise SessionStorageError("Session event cannot be encoded as JSON") from exc

    def _session_path(self, session_id: str) -> Path:
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.jsonl"

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty text")


__all__ = ["SESSION_EVENT_LOG_VERSION", "JsonlSessionStore"]
