"""Strongly typed contracts for durable long-term memory processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Literal, TypeAlias
from uuid import uuid4

from agent_harness.messages import AgentMessage, RuntimeStatusMessage, utc_now

MemoryAction: TypeAlias = Literal["add", "replace", "remove"]
MemorySection: TypeAlias = Literal[
    "user_preferences",
    "stable_facts",
    "decisions_and_conventions",
    "reusable_experience",
]
DreamStopReason: TypeAlias = Literal[
    "completed",
    "analysis_failed",
    "validation_failed",
    "execution_failed",
    "cancelled",
    "limit_reached",
]

MEMORY_ACTIONS = frozenset({"add", "replace", "remove"})
MEMORY_SECTIONS = frozenset(
    {
        "user_preferences",
        "stable_facts",
        "decisions_and_conventions",
        "reusable_experience",
    }
)
DREAM_STOP_REASONS = frozenset(
    {
        "completed",
        "analysis_failed",
        "validation_failed",
        "execution_failed",
        "cancelled",
        "limit_reached",
    }
)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _require_aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def memory_inbox_id(context_summary_id: str) -> str:
    """Return a deterministic idempotency key for one accepted summary."""

    _require_text(context_summary_id, "context_summary_id")
    digest = sha256(context_summary_id.encode("utf-8")).hexdigest()[:32]
    return f"memory_inbox_{digest}"


def new_dream_run_id() -> str:
    return f"dream_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class MemoryInboxEntry:
    """A durable snapshot of exactly one newly archived conversation range."""

    cursor: int
    session_id: str
    source_leaf_id: str
    context_summary_id: str
    covered_from_entry_id: str
    covered_through_entry_id: str
    source_entry_ids: tuple[str, ...]
    messages: tuple[AgentMessage, ...]
    id: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if type(self.cursor) is not int or self.cursor <= 0:
            raise ValueError("MemoryInboxEntry.cursor must be a positive integer")
        for field_name in (
            "session_id",
            "source_leaf_id",
            "context_summary_id",
            "covered_from_entry_id",
            "covered_through_entry_id",
        ):
            _require_text(getattr(self, field_name), f"MemoryInboxEntry.{field_name}")
        expected_id = memory_inbox_id(self.context_summary_id)
        if self.id:
            _require_text(self.id, "MemoryInboxEntry.id")
            if self.id != expected_id:
                raise ValueError("MemoryInboxEntry.id must match context_summary_id")
        else:
            object.__setattr__(self, "id", expected_id)

        source_ids = tuple(self.source_entry_ids)
        messages = tuple(self.messages)
        if not source_ids or len(source_ids) != len(messages):
            raise ValueError(
                "MemoryInboxEntry source_entry_ids and messages must be non-empty and equal length"
            )
        for index, source_id in enumerate(source_ids):
            _require_text(source_id, f"MemoryInboxEntry.source_entry_ids[{index}]")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("MemoryInboxEntry.source_entry_ids must be unique")
        if source_ids[0] != self.covered_from_entry_id:
            raise ValueError("covered_from_entry_id must equal the first source entry id")
        if source_ids[-1] != self.covered_through_entry_id:
            raise ValueError("covered_through_entry_id must equal the last source entry id")
        if any(isinstance(message, RuntimeStatusMessage) for message in messages):
            raise ValueError("RuntimeStatusMessage cannot enter the Memory inbox")
        object.__setattr__(self, "source_entry_ids", source_ids)
        object.__setattr__(self, "messages", messages)
        _require_aware(self.created_at, "MemoryInboxEntry.created_at")


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    action: MemoryAction
    section: MemorySection
    statement: str
    match: str | None
    reason: str
    source_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in MEMORY_ACTIONS:
            raise ValueError(f"Unsupported MemoryOperation.action: {self.action!r}")
        if self.section not in MEMORY_SECTIONS:
            raise ValueError(f"Unsupported MemoryOperation.section: {self.section!r}")
        _require_text(self.statement, "MemoryOperation.statement")
        _require_text(self.reason, "MemoryOperation.reason")
        if self.action == "add" and self.match is not None:
            raise ValueError("add MemoryOperation.match must be None")
        if self.action in {"replace", "remove"}:
            if self.match is None:
                raise ValueError(f"{self.action} MemoryOperation requires match")
            _require_text(self.match, "MemoryOperation.match")
        source_ids = tuple(self.source_entry_ids)
        if not source_ids:
            raise ValueError("MemoryOperation.source_entry_ids must not be empty")
        for index, source_id in enumerate(source_ids):
            _require_text(source_id, f"MemoryOperation.source_entry_ids[{index}]")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("MemoryOperation.source_entry_ids must be unique")
        object.__setattr__(self, "source_entry_ids", source_ids)


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    schema_version: int
    operations: tuple[MemoryOperation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("MemoryPlan.schema_version must equal 1")
        operations = tuple(self.operations)
        if any(not isinstance(item, MemoryOperation) for item in operations):
            raise TypeError("MemoryPlan.operations must contain MemoryOperation values")
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True, slots=True)
class DreamRunRecord:
    first_cursor: int
    last_cursor: int
    source_inbox_ids: tuple[str, ...]
    plan: MemoryPlan | None
    stop_reason: DreamStopReason
    changes: tuple[str, ...] = ()
    error: str | None = None
    id: str = field(default_factory=new_dream_run_id)
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "DreamRunRecord.id")
        if type(self.first_cursor) is not int or self.first_cursor <= 0:
            raise ValueError("DreamRunRecord.first_cursor must be positive")
        if type(self.last_cursor) is not int or self.last_cursor < self.first_cursor:
            raise ValueError("DreamRunRecord.last_cursor must not precede first_cursor")
        source_ids = tuple(self.source_inbox_ids)
        if not source_ids:
            raise ValueError("DreamRunRecord.source_inbox_ids must not be empty")
        for index, source_id in enumerate(source_ids):
            _require_text(source_id, f"DreamRunRecord.source_inbox_ids[{index}]")
        object.__setattr__(self, "source_inbox_ids", source_ids)
        if self.plan is not None and not isinstance(self.plan, MemoryPlan):
            raise TypeError("DreamRunRecord.plan must be a MemoryPlan or None")
        if self.stop_reason not in DREAM_STOP_REASONS:
            raise ValueError(f"Unsupported DreamRunRecord.stop_reason: {self.stop_reason!r}")
        changes = tuple(self.changes)
        for index, change in enumerate(changes):
            _require_text(change, f"DreamRunRecord.changes[{index}]")
        object.__setattr__(self, "changes", changes)
        if self.error is not None:
            _require_text(self.error, "DreamRunRecord.error")
        if self.stop_reason == "completed" and self.error is not None:
            raise ValueError("completed DreamRunRecord cannot contain an error")
        _require_aware(self.started_at, "DreamRunRecord.started_at")
        _require_aware(self.completed_at, "DreamRunRecord.completed_at")
        if self.completed_at < self.started_at:
            raise ValueError("DreamRunRecord.completed_at cannot precede started_at")


__all__ = [
    "DREAM_STOP_REASONS",
    "MEMORY_ACTIONS",
    "MEMORY_SECTIONS",
    "DreamRunRecord",
    "DreamStopReason",
    "MemoryAction",
    "MemoryInboxEntry",
    "MemoryOperation",
    "MemoryPlan",
    "MemorySection",
    "memory_inbox_id",
    "new_dream_run_id",
]
