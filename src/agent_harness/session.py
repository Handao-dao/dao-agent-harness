"""Session projection, pending-input queue, and message Entry Tree events."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterator, TypeAlias
from uuid import uuid4

from agent_harness.messages import AgentMessage, UserMessage, utc_now
from agent_harness.summary import ContextSummary


def new_input_id() -> str:
    return f"input_{uuid4().hex}"


def new_entry_id() -> str:
    return f"entry_{uuid4().hex}"


def new_event_id() -> str:
    return f"event_{uuid4().hex}"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


class SessionError(RuntimeError):
    """Base class for invalid Session operations."""


class PendingInputNotFoundError(SessionError):
    """A requested pending input does not exist in the Session."""


class PendingInputOrderError(SessionError):
    """A commit attempted to skip or reorder queued input."""


class SessionHistoryConflictError(SessionError):
    """The Runner result does not retain the active Session branch prefix."""


class SessionEventConflictError(SessionError):
    """A persisted event cannot be applied to the current Session projection."""


@dataclass(frozen=True, slots=True)
class PendingInput:
    """A durable user input that has not joined the message tree yet."""

    source_message_id: str
    content: str
    id: str = field(default_factory=new_input_id)
    created_at: datetime = field(default_factory=utc_now)
    edited_at: datetime | None = None
    revision: int = 1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("id", self.id),
            ("source_message_id", self.source_message_id),
            ("content", self.content),
        ):
            _require_text(value, f"PendingInput.{field_name}")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("PendingInput.revision must be a positive integer")

    def to_user_message(self) -> UserMessage:
        return UserMessage(id=self.id, content=self.content, created_at=self.created_at)

    def edit(self, content: str, *, edited_at: datetime | None = None) -> PendingInput:
        _require_text(content, "PendingInput.content")
        return replace(
            self,
            content=content,
            edited_at=edited_at or utc_now(),
            revision=self.revision + 1,
        )


@dataclass(frozen=True, slots=True)
class MessageEntry:
    """A message node whose parent link places it in the Session tree."""

    message: AgentMessage
    parent_id: str | None
    id: str = field(default_factory=new_entry_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "MessageEntry.id")
        if self.parent_id is not None:
            _require_text(self.parent_id, "MessageEntry.parent_id")


@dataclass(frozen=True, slots=True)
class InputEnqueued:
    input: PendingInput
    id: str = field(default_factory=new_event_id)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class InputEdited:
    input_id: str
    expected_revision: int
    content: str
    edited_at: datetime
    id: str = field(default_factory=new_event_id)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ConsumedInput:
    id: str
    revision: int


@dataclass(frozen=True, slots=True)
class TurnCommitted:
    base_leaf_id: str | None
    consumed_inputs: tuple[ConsumedInput, ...]
    entries: tuple[MessageEntry, ...]
    new_leaf_id: str
    id: str = field(default_factory=new_event_id)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class LeafChanged:
    from_leaf_id: str | None
    target_leaf_id: str | None
    id: str = field(default_factory=new_event_id)
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ContextSummaryCreated:
    summary: ContextSummary
    id: str = field(default_factory=new_event_id)
    timestamp: datetime = field(default_factory=utc_now)


SessionEvent: TypeAlias = (
    InputEnqueued | InputEdited | TurnCommitted | LeafChanged | ContextSummaryCreated
)


@dataclass(slots=True)
class Session:
    """Materialized Session state reduced from an append-only event log."""

    id: str
    entries: list[MessageEntry] = field(default_factory=list)
    active_leaf_id: str | None = None
    pending_inputs: list[PendingInput] = field(default_factory=list)
    context_summaries: list[ContextSummary] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    _unpersisted_events: list[SessionEvent] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _require_text(self.id, "Session.id")
        self._validate_projection()

    @classmethod
    def from_messages(
        cls,
        *,
        id: str,
        messages: Sequence[AgentMessage],
        pending_inputs: Sequence[PendingInput] = (),
        context_summaries: Sequence[ContextSummary] = (),
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Build a linear tree for legacy snapshots and test fixtures."""

        entries: list[MessageEntry] = []
        parent_id: str | None = None
        for message in messages:
            entry = MessageEntry(
                id=f"entry_{message.id}",
                parent_id=parent_id,
                message=message,
                created_at=message.created_at,
            )
            entries.append(entry)
            parent_id = entry.id
        return cls(
            id=id,
            entries=entries,
            active_leaf_id=parent_id,
            pending_inputs=list(pending_inputs),
            context_summaries=list(context_summaries),
            created_at=created_at or utc_now(),
            updated_at=updated_at or utc_now(),
            metadata=dict(metadata or {}),
        )

    @property
    def messages(self) -> list[AgentMessage]:
        """Return the materialized messages on the active branch."""

        return self.active_messages()

    def active_entries(self) -> list[MessageEntry]:
        if self.active_leaf_id is None:
            return []
        return list(self.branch_entries(self.active_leaf_id))

    def branch_entries(self, leaf_id: str) -> tuple[MessageEntry, ...]:
        """Return one immutable root-to-leaf Entry path for external resolvers."""

        _require_text(leaf_id, "leaf_id")
        by_id = {entry.id: entry for entry in self.entries}
        path: list[MessageEntry] = []
        visited: set[str] = set()
        current_id: str | None = leaf_id
        while current_id is not None:
            if current_id in visited:
                raise SessionEventConflictError("Message Entry Tree contains a cycle")
            visited.add(current_id)
            entry = by_id.get(current_id)
            if entry is None:
                raise SessionEventConflictError(f"Message Entry not found: {current_id}")
            path.append(entry)
            current_id = entry.parent_id
        path.reverse()
        return tuple(path)

    def active_messages(self) -> list[AgentMessage]:
        return [entry.message for entry in self.active_entries()]

    def copy_history(self) -> list[AgentMessage]:
        """Return the active branch copy used to begin an Agent Loop execution."""

        return self.active_messages()

    @contextmanager
    def rollback_on_error(self) -> Iterator[None]:
        """Restore the in-memory projection if a surrounding persistence step fails."""

        entries = list(self.entries)
        active_leaf_id = self.active_leaf_id
        pending_inputs = list(self.pending_inputs)
        updated_at = self.updated_at
        unpersisted_events = list(self._unpersisted_events)
        try:
            yield
        except BaseException:
            self.entries[:] = entries
            self.active_leaf_id = active_leaf_id
            self.pending_inputs[:] = pending_inputs
            self.updated_at = updated_at
            self._unpersisted_events[:] = unpersisted_events
            raise

    def enqueue(self, item: PendingInput) -> PendingInput:
        existing = next(
            (
                pending
                for pending in self.pending_inputs
                if pending.source_message_id == item.source_message_id
            ),
            None,
        )
        if existing is not None:
            return existing
        self.apply_event(InputEnqueued(input=item), track=True)
        return item

    def edit_pending(
        self,
        input_id: str,
        content: str,
        *,
        edited_at: datetime | None = None,
    ) -> PendingInput:
        pending = self._find_pending(input_id)
        event_time = edited_at or utc_now()
        self.apply_event(
            InputEdited(
                input_id=input_id,
                expected_revision=pending.revision,
                content=content,
                edited_at=event_time,
                timestamp=event_time,
            ),
            track=True,
        )
        return self._find_pending(input_id)

    def checkout(self, entry_id: str | None) -> None:
        """Move the active leaf without deleting either branch."""

        if entry_id is not None and not any(entry.id == entry_id for entry in self.entries):
            raise SessionEventConflictError(f"Message Entry not found: {entry_id}")
        if entry_id == self.active_leaf_id:
            return
        self.apply_event(
            LeafChanged(from_leaf_id=self.active_leaf_id, target_leaf_id=entry_id),
            track=True,
        )

    def record_context_summary(self, summary: ContextSummary) -> ContextSummary:
        """Add one durable tree-external summary without changing the active branch."""

        self.apply_event(ContextSummaryCreated(summary=summary), track=True)
        return summary

    def commit_working_messages(
        self,
        *,
        working_messages: Sequence[AgentMessage],
        save_cursor: int,
        base_leaf_id: str | None,
        consumed_input_ids: Sequence[str],
    ) -> tuple[AgentMessage, ...]:
        """Atomically project one completed turn into the active message branch."""

        active_messages = self.active_messages()
        if self.active_leaf_id != base_leaf_id:
            raise SessionHistoryConflictError("Active Session leaf changed during execution")
        if save_cursor != len(active_messages):
            raise SessionHistoryConflictError(
                "save_cursor must equal the active branch message count"
            )
        if tuple(working_messages[:save_cursor]) != tuple(active_messages):
            raise SessionHistoryConflictError(
                "Working messages changed the active Session branch prefix"
            )

        consumed_ids = tuple(consumed_input_ids)
        if not consumed_ids:
            raise PendingInputOrderError("At least one pending input must be consumed")
        queue_prefix = tuple(item.id for item in self.pending_inputs[: len(consumed_ids)])
        if queue_prefix != consumed_ids:
            raise PendingInputOrderError("Consumed inputs must match the pending queue prefix")

        tail = tuple(working_messages[save_cursor:])
        user_messages = tuple(message for message in tail if isinstance(message, UserMessage))
        consumed = tuple(self.pending_inputs[: len(consumed_ids)])
        for pending in consumed:
            matches = tuple(message for message in user_messages if message.id == pending.id)
            if len(matches) != 1:
                raise SessionError("Each consumed input must appear exactly once as a UserMessage")
            if matches[0] != pending.to_user_message():
                raise SessionHistoryConflictError(
                    "Working messages contain stale or changed pending input content"
                )

        existing_message_ids = {entry.message.id for entry in self.entries}
        tail_ids = [message.id for message in tail]
        if existing_message_ids.intersection(tail_ids) or len(tail_ids) != len(set(tail_ids)):
            raise SessionError("Committed message IDs must be unique across all branches")

        parent_id = base_leaf_id
        new_entries: list[MessageEntry] = []
        for message in tail:
            entry = MessageEntry(parent_id=parent_id, message=message)
            new_entries.append(entry)
            parent_id = entry.id
        if not new_entries or parent_id is None:
            raise SessionError("A committed turn must contain at least one new message")

        event = TurnCommitted(
            base_leaf_id=base_leaf_id,
            consumed_inputs=tuple(
                ConsumedInput(id=item.id, revision=item.revision) for item in consumed
            ),
            entries=tuple(new_entries),
            new_leaf_id=parent_id,
        )
        self.apply_event(event, track=True)
        return tail

    def apply_event(self, event: SessionEvent, *, track: bool = False) -> None:
        """Reduce one durable event into this Session projection."""

        if isinstance(event, InputEnqueued):
            self._apply_input_enqueued(event)
        elif isinstance(event, InputEdited):
            self._apply_input_edited(event)
        elif isinstance(event, TurnCommitted):
            self._apply_turn_committed(event)
        elif isinstance(event, LeafChanged):
            self._apply_leaf_changed(event)
        elif isinstance(event, ContextSummaryCreated):
            self._apply_context_summary_created(event)
        else:
            raise TypeError(f"Unsupported SessionEvent: {type(event).__name__}")
        self.updated_at = event.timestamp
        if track:
            self._unpersisted_events.append(event)

    def unpersisted_events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._unpersisted_events)

    def mark_events_persisted(self, events: Sequence[SessionEvent]) -> None:
        expected = tuple(events)
        if tuple(self._unpersisted_events[: len(expected)]) != expected:
            raise SessionEventConflictError("Persisted events are not the pending event prefix")
        del self._unpersisted_events[: len(expected)]

    def _apply_input_enqueued(self, event: InputEnqueued) -> None:
        if any(item.id == event.input.id for item in self.pending_inputs):
            raise SessionEventConflictError(f"Pending input ID already exists: {event.input.id}")
        if any(
            item.source_message_id == event.input.source_message_id
            for item in self.pending_inputs
        ):
            raise SessionEventConflictError(
                f"Pending source message already exists: {event.input.source_message_id}"
            )
        if any(entry.message.id == event.input.id for entry in self.entries):
            raise SessionEventConflictError(
                f"Pending input is already committed: {event.input.id}"
            )
        self.pending_inputs.append(event.input)

    def _apply_input_edited(self, event: InputEdited) -> None:
        pending = self._find_pending(event.input_id)
        if pending.revision != event.expected_revision:
            raise SessionEventConflictError(
                f"Pending input revision changed: {event.input_id}"
            )
        edited = pending.edit(event.content, edited_at=event.edited_at)
        index = self.pending_inputs.index(pending)
        self.pending_inputs[index] = edited

    def _apply_turn_committed(self, event: TurnCommitted) -> None:
        if self.active_leaf_id != event.base_leaf_id:
            raise SessionEventConflictError("Turn base leaf does not match active leaf")
        consumed_prefix = tuple(
            ConsumedInput(id=item.id, revision=item.revision)
            for item in self.pending_inputs[: len(event.consumed_inputs)]
        )
        if consumed_prefix != event.consumed_inputs:
            raise SessionEventConflictError("Turn does not consume the pending queue prefix")
        if not event.entries or event.new_leaf_id != event.entries[-1].id:
            raise SessionEventConflictError("Turn has an invalid new leaf")

        existing_entry_ids = {entry.id for entry in self.entries}
        existing_message_ids = {entry.message.id for entry in self.entries}
        parent_id = event.base_leaf_id
        for entry in event.entries:
            if entry.parent_id != parent_id:
                raise SessionEventConflictError("Turn message entries are not one ordered chain")
            if entry.id in existing_entry_ids or entry.message.id in existing_message_ids:
                raise SessionEventConflictError("Turn contains a duplicate Entry or message ID")
            existing_entry_ids.add(entry.id)
            existing_message_ids.add(entry.message.id)
            parent_id = entry.id

        user_messages = tuple(
            entry.message for entry in event.entries if isinstance(entry.message, UserMessage)
        )
        for pending in self.pending_inputs[: len(event.consumed_inputs)]:
            matches = tuple(message for message in user_messages if message.id == pending.id)
            if matches != (pending.to_user_message(),):
                raise SessionEventConflictError(
                    "Committed turn does not contain the current pending input"
                )

        self.entries.extend(event.entries)
        del self.pending_inputs[: len(event.consumed_inputs)]
        self.active_leaf_id = event.new_leaf_id

    def _apply_leaf_changed(self, event: LeafChanged) -> None:
        if self.active_leaf_id != event.from_leaf_id:
            raise SessionEventConflictError("Leaf change does not start at the active leaf")
        if event.target_leaf_id is not None and not any(
            entry.id == event.target_leaf_id for entry in self.entries
        ):
            raise SessionEventConflictError(
                f"Leaf target does not exist: {event.target_leaf_id}"
            )
        self.active_leaf_id = event.target_leaf_id

    def _apply_context_summary_created(self, event: ContextSummaryCreated) -> None:
        summary = event.summary
        if summary.session_id != self.id:
            raise SessionEventConflictError("ContextSummary belongs to another Session")
        if any(item.id == summary.id for item in self.context_summaries):
            raise SessionEventConflictError(f"ContextSummary ID already exists: {summary.id}")

        path_ids = self._path_ids(summary.source_leaf_id)
        if summary.covered_through_entry_id not in path_ids:
            raise SessionEventConflictError(
                "ContextSummary coverage boundary is not on its source branch"
            )

        if summary.previous_summary_id is not None:
            previous = next(
                (
                    item
                    for item in self.context_summaries
                    if item.id == summary.previous_summary_id
                ),
                None,
            )
            if previous is None:
                raise SessionEventConflictError(
                    f"Previous ContextSummary not found: {summary.previous_summary_id}"
                )
            if previous.covered_through_entry_id not in path_ids:
                raise SessionEventConflictError(
                    "Previous ContextSummary does not apply to the source branch"
                )
            previous_index = path_ids.index(previous.covered_through_entry_id)
            current_index = path_ids.index(summary.covered_through_entry_id)
            if previous_index > current_index:
                raise SessionEventConflictError(
                    "ContextSummary cannot cover less history than its predecessor"
                )

        self.context_summaries.append(summary)

    def _path_ids(self, leaf_id: str) -> list[str]:
        by_id = {entry.id: entry for entry in self.entries}
        if leaf_id not in by_id:
            raise SessionEventConflictError(f"Message Entry not found: {leaf_id}")
        path: list[str] = []
        visited: set[str] = set()
        current_id: str | None = leaf_id
        while current_id is not None:
            if current_id in visited:
                raise SessionEventConflictError("Message Entry Tree contains a cycle")
            visited.add(current_id)
            entry = by_id.get(current_id)
            if entry is None:
                raise SessionEventConflictError(f"Message Entry not found: {current_id}")
            path.append(current_id)
            current_id = entry.parent_id
        path.reverse()
        return path

    def _find_pending(self, input_id: str) -> PendingInput:
        for pending in self.pending_inputs:
            if pending.id == input_id:
                return pending
        raise PendingInputNotFoundError(f"Pending input not found: {input_id}")

    def _validate_projection(self) -> None:
        entry_ids: set[str] = set()
        message_ids: set[str] = set()
        for entry in self.entries:
            if entry.id in entry_ids:
                raise SessionEventConflictError(f"Duplicate Message Entry ID: {entry.id}")
            if entry.parent_id is not None and entry.parent_id not in entry_ids:
                raise SessionEventConflictError(
                    f"Message Entry parent must appear first: {entry.parent_id}"
                )
            if entry.message.id in message_ids:
                raise SessionEventConflictError(
                    f"Duplicate AgentMessage ID: {entry.message.id}"
                )
            entry_ids.add(entry.id)
            message_ids.add(entry.message.id)
        if self.active_leaf_id is not None and self.active_leaf_id not in entry_ids:
            raise SessionEventConflictError(
                f"Active leaf does not exist: {self.active_leaf_id}"
            )
        pending_ids = [item.id for item in self.pending_inputs]
        source_ids = [item.source_message_id for item in self.pending_inputs]
        if len(pending_ids) != len(set(pending_ids)):
            raise SessionEventConflictError("Pending input IDs must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise SessionEventConflictError("Pending source-message IDs must be unique")
        if message_ids.intersection(pending_ids):
            raise SessionEventConflictError("Committed and pending message IDs must be disjoint")

        summary_ids: set[str] = set()
        for summary in self.context_summaries:
            if summary.id in summary_ids:
                raise SessionEventConflictError(f"Duplicate ContextSummary ID: {summary.id}")
            if summary.session_id != self.id:
                raise SessionEventConflictError("ContextSummary belongs to another Session")
            path_ids = self._path_ids(summary.source_leaf_id)
            if summary.covered_through_entry_id not in path_ids:
                raise SessionEventConflictError(
                    "ContextSummary coverage boundary is not on its source branch"
                )
            if summary.previous_summary_id is not None:
                previous = next(
                    (
                        item
                        for item in self.context_summaries
                        if item.id == summary.previous_summary_id
                        and item.id in summary_ids
                    ),
                    None,
                )
                if previous is None:
                    raise SessionEventConflictError(
                        f"Previous ContextSummary not found: {summary.previous_summary_id}"
                    )
                if previous.covered_through_entry_id not in path_ids:
                    raise SessionEventConflictError(
                        "Previous ContextSummary does not apply to the source branch"
                    )
                if path_ids.index(previous.covered_through_entry_id) > path_ids.index(
                    summary.covered_through_entry_id
                ):
                    raise SessionEventConflictError(
                        "ContextSummary cannot cover less history than its predecessor"
                    )
            summary_ids.add(summary.id)


__all__ = [
    "ConsumedInput",
    "ContextSummaryCreated",
    "InputEdited",
    "InputEnqueued",
    "LeafChanged",
    "MessageEntry",
    "PendingInput",
    "PendingInputNotFoundError",
    "PendingInputOrderError",
    "Session",
    "SessionError",
    "SessionEvent",
    "SessionEventConflictError",
    "SessionHistoryConflictError",
    "TurnCommitted",
    "new_entry_id",
    "new_event_id",
    "new_input_id",
]
