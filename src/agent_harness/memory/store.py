"""In-memory and durable local stores for long-term memory processing."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from agent_harness.memory.codec import MemoryCodec, MemoryCodecError
from agent_harness.memory.models import DreamRunRecord, MemoryInboxEntry
from agent_harness.messages import AgentMessage, utc_now

MEMORY_TEMPLATE = """# Long-term Memory

## User Preferences

## Stable Facts

## Decisions and Conventions

## Reusable Experience
"""


class MemoryStoreError(RuntimeError):
    """Durable memory state could not be read or updated safely."""


class MemoryStore(Protocol):
    def read_memory(self) -> str: ...

    def write_memory(self, content: str) -> None: ...

    def enqueue(
        self,
        *,
        session_id: str,
        source_leaf_id: str,
        context_summary_id: str,
        covered_from_entry_id: str,
        covered_through_entry_id: str,
        source_entry_ids: Sequence[str],
        messages: Sequence[AgentMessage],
        created_at: datetime | None = None,
    ) -> MemoryInboxEntry: ...

    def read_pending(
        self, *, after_cursor: int, limit: int
    ) -> tuple[MemoryInboxEntry, ...]: ...

    def get_dream_cursor(self) -> int: ...

    def advance_dream_cursor(self, cursor: int) -> None: ...

    def append_dream_record(self, record: DreamRunRecord) -> None: ...

    def compact_inbox(self) -> None: ...


class InMemoryMemoryStore:
    def __init__(self, memory: str = "") -> None:
        if not isinstance(memory, str):
            raise TypeError("memory must be text")
        self._memory = memory
        self._entries: list[MemoryInboxEntry] = []
        self._records: list[DreamRunRecord] = []
        self._dream_cursor = 0
        self._lock = RLock()

    @property
    def dream_records(self) -> tuple[DreamRunRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def read_memory(self) -> str:
        with self._lock:
            return self._memory

    def write_memory(self, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        with self._lock:
            self._memory = content

    def enqueue(
        self,
        *,
        session_id: str,
        source_leaf_id: str,
        context_summary_id: str,
        covered_from_entry_id: str,
        covered_through_entry_id: str,
        source_entry_ids: Sequence[str],
        messages: Sequence[AgentMessage],
        created_at: datetime | None = None,
    ) -> MemoryInboxEntry:
        with self._lock:
            duplicate = next(
                (
                    entry
                    for entry in self._entries
                    if entry.context_summary_id == context_summary_id
                ),
                None,
            )
            if duplicate is not None:
                return duplicate
            entry = _build_entry(
                cursor=max(
                    self._dream_cursor,
                    self._entries[-1].cursor if self._entries else 0,
                )
                + 1,
                session_id=session_id,
                source_leaf_id=source_leaf_id,
                context_summary_id=context_summary_id,
                covered_from_entry_id=covered_from_entry_id,
                covered_through_entry_id=covered_through_entry_id,
                source_entry_ids=source_entry_ids,
                messages=messages,
                created_at=created_at,
            )
            self._entries.append(entry)
            return entry

    def read_pending(
        self, *, after_cursor: int, limit: int
    ) -> tuple[MemoryInboxEntry, ...]:
        _validate_read(after_cursor, limit)
        with self._lock:
            return tuple(
                entry for entry in self._entries if entry.cursor > after_cursor
            )[:limit]

    def get_dream_cursor(self) -> int:
        with self._lock:
            return self._dream_cursor

    def advance_dream_cursor(self, cursor: int) -> None:
        with self._lock:
            _validate_cursor_advance(cursor, self._dream_cursor, self._entries)
            self._dream_cursor = cursor

    def append_dream_record(self, record: DreamRunRecord) -> None:
        if not isinstance(record, DreamRunRecord):
            raise TypeError("record must be a DreamRunRecord")
        with self._lock:
            self._records.append(record)

    def compact_inbox(self) -> None:
        # v1 retains processed receipts so enqueue stays globally idempotent.
        return None


class LocalMemoryStore:
    """Atomic MEMORY.md plus append-only Inbox and Dream logs."""

    def __init__(self, directory: str | Path, *, codec: MemoryCodec | None = None) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.directory / "MEMORY.md"
        self.inbox_path = self.directory / "inbox.jsonl"
        self.dream_log_path = self.directory / "dream-log.jsonl"
        self.cursor_path = self.directory / ".dream_cursor"
        self._codec = codec or MemoryCodec()
        self._lock = RLock()

    def read_memory(self) -> str:
        with self._lock:
            try:
                return self.memory_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return ""
            except OSError as exc:
                raise MemoryStoreError(f"Could not read {self.memory_path}: {exc}") from exc

    def write_memory(self, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        with self._lock:
            self._atomic_write_text(self.memory_path, content)

    def enqueue(
        self,
        *,
        session_id: str,
        source_leaf_id: str,
        context_summary_id: str,
        covered_from_entry_id: str,
        covered_through_entry_id: str,
        source_entry_ids: Sequence[str],
        messages: Sequence[AgentMessage],
        created_at: datetime | None = None,
    ) -> MemoryInboxEntry:
        with self._lock:
            entries = self._read_inbox()
            duplicate = next(
                (
                    entry
                    for entry in entries
                    if entry.context_summary_id == context_summary_id
                ),
                None,
            )
            if duplicate is not None:
                return duplicate
            entry = _build_entry(
                cursor=max(
                    self.get_dream_cursor(),
                    entries[-1].cursor if entries else 0,
                )
                + 1,
                session_id=session_id,
                source_leaf_id=source_leaf_id,
                context_summary_id=context_summary_id,
                covered_from_entry_id=covered_from_entry_id,
                covered_through_entry_id=covered_through_entry_id,
                source_entry_ids=source_entry_ids,
                messages=messages,
                created_at=created_at,
            )
            self._append_json(self.inbox_path, self._codec.encode_inbox_entry(entry))
            return entry

    def read_pending(
        self, *, after_cursor: int, limit: int
    ) -> tuple[MemoryInboxEntry, ...]:
        _validate_read(after_cursor, limit)
        with self._lock:
            return tuple(
                entry for entry in self._read_inbox() if entry.cursor > after_cursor
            )[:limit]

    def get_dream_cursor(self) -> int:
        with self._lock:
            try:
                raw = self.cursor_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                return 0
            except OSError as exc:
                raise MemoryStoreError(f"Could not read {self.cursor_path}: {exc}") from exc
            try:
                cursor = int(raw)
            except ValueError as exc:
                raise MemoryStoreError("Dream cursor is not an integer") from exc
            if cursor < 0:
                raise MemoryStoreError("Dream cursor cannot be negative")
            return cursor

    def advance_dream_cursor(self, cursor: int) -> None:
        with self._lock:
            current = self.get_dream_cursor()
            entries = self._read_inbox()
            _validate_cursor_advance(cursor, current, entries)
            self._atomic_write_text(self.cursor_path, str(cursor))

    def append_dream_record(self, record: DreamRunRecord) -> None:
        if not isinstance(record, DreamRunRecord):
            raise TypeError("record must be a DreamRunRecord")
        with self._lock:
            self._append_json(
                self.dream_log_path, self._codec.encode_dream_record(record)
            )

    def read_dream_records(self) -> tuple[DreamRunRecord, ...]:
        with self._lock:
            return tuple(
                self._codec.decode_dream_record(value)
                for value in self._read_jsonl(self.dream_log_path)
            )

    def compact_inbox(self) -> None:
        # v1 retains processed receipts so context_summary_id remains an
        # idempotency key even after restarts. Payload tombstones can replace
        # old records later without changing the public Store contract.
        return None

    def _read_inbox(self) -> list[MemoryInboxEntry]:
        try:
            entries = [
                self._codec.decode_inbox_entry(value)
                for value in self._read_jsonl(self.inbox_path)
            ]
        except MemoryCodecError as exc:
            raise MemoryStoreError(f"Invalid memory Inbox: {exc}") from exc
        cursors = [entry.cursor for entry in entries]
        if cursors != sorted(set(cursors)):
            raise MemoryStoreError("Memory Inbox cursors must be unique and increasing")
        return entries

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise MemoryStoreError(f"Could not read {path}: {exc}") from exc
        if not data:
            return []
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            lines = lines[:-1]
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MemoryStoreError(f"Invalid JSONL record {path}:{index}") from exc
            if not isinstance(value, dict):
                raise MemoryStoreError(f"JSONL record {path}:{index} must be an object")
            records.append(value)
        return records

    def _append_json(self, path: Path, value: dict[str, Any]) -> None:
        encoded = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        try:
            with path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._sync_directory()
        except OSError as exc:
            raise MemoryStoreError(f"Could not append {path}: {exc}") from exc

    def _atomic_write_jsonl(self, path: Path, values: Sequence[dict[str, Any]]) -> None:
        content = "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
            for value in values
        )
        self._atomic_write_text(path, content)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            self._sync_directory()
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise MemoryStoreError(f"Could not atomically write {path}: {exc}") from exc

    def _sync_directory(self) -> None:
        try:
            descriptor = os.open(str(self.directory), os.O_RDONLY)
        except (OSError, PermissionError):
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _build_entry(
    *,
    cursor: int,
    session_id: str,
    source_leaf_id: str,
    context_summary_id: str,
    covered_from_entry_id: str,
    covered_through_entry_id: str,
    source_entry_ids: Sequence[str],
    messages: Sequence[AgentMessage],
    created_at: datetime | None,
) -> MemoryInboxEntry:
    return MemoryInboxEntry(
        cursor=cursor,
        session_id=session_id,
        source_leaf_id=source_leaf_id,
        context_summary_id=context_summary_id,
        covered_from_entry_id=covered_from_entry_id,
        covered_through_entry_id=covered_through_entry_id,
        source_entry_ids=tuple(source_entry_ids),
        messages=tuple(messages),
        created_at=created_at or utc_now(),
    )


def _validate_read(after_cursor: int, limit: int) -> None:
    if type(after_cursor) is not int or after_cursor < 0:
        raise ValueError("after_cursor must be a non-negative integer")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")


def _validate_cursor_advance(
    cursor: int,
    current: int,
    entries: Sequence[MemoryInboxEntry],
) -> None:
    if type(cursor) is not int or cursor < 0:
        raise ValueError("cursor must be a non-negative integer")
    if cursor < current:
        raise MemoryStoreError("Dream cursor cannot move backwards")
    if cursor == current:
        return
    known = {entry.cursor for entry in entries}
    if cursor not in known:
        raise MemoryStoreError("Dream cursor must reference an Inbox entry")


__all__ = [
    "MEMORY_TEMPLATE",
    "InMemoryMemoryStore",
    "LocalMemoryStore",
    "MemoryStore",
    "MemoryStoreError",
]
