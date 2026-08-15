"""Strong checkpoint types and persistence contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Protocol

from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
    UserMessage,
    utc_now,
)

CheckpointPhase = Literal["awaiting_tools", "tools_completed", "final_response"]
CheckpointTerminalStatus = Literal["completed", "limit_reached"]


class CheckpointError(RuntimeError):
    """Base error for checkpoint validation, persistence, and recovery."""


class CheckpointConflictError(CheckpointError):
    """Raised when durable Session state no longer matches a checkpoint."""


class CheckpointCorruptError(CheckpointError):
    """Raised when a persisted checkpoint cannot be trusted."""


class CheckpointStorageError(CheckpointError):
    """Raised when checkpoint storage cannot complete safely."""


@dataclass(frozen=True, slots=True)
class IncorporatedInput:
    """PendingInput identity already incorporated into an uncommitted turn."""

    id: str
    revision: int

    def __post_init__(self) -> None:
        _require_text(self.id, "IncorporatedInput.id")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("IncorporatedInput.revision must be a positive integer")


@dataclass(frozen=True, slots=True)
class RunnerCheckpoint:
    """Runner-owned progress snapshot before Runtime identity is attached."""

    phase: CheckpointPhase
    model: str
    next_model_turn: int
    messages: tuple[AgentMessage, ...]
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    terminal_status: CheckpointTerminalStatus | None = None
    stop_reason: str | None = None
    final_content: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.model, "RunnerCheckpoint.model")
        _validate_common(
            phase=self.phase,
            next_model_turn=self.next_model_turn,
            messages=self.messages,
            tools_used=self.tools_used,
            usage=self.usage,
            terminal_status=self.terminal_status,
            stop_reason=self.stop_reason,
            final_content=self.final_content,
            owner="RunnerCheckpoint",
        )
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools_used", tuple(self.tools_used))
        object.__setattr__(self, "usage", _freeze_usage(self.usage))


@dataclass(frozen=True, slots=True)
class ContextCheckpoint:
    """Latest durable state for one uncommitted PendingInput turn."""

    session_id: str
    input_id: str
    input_revision: int
    base_leaf_id: str | None
    save_cursor: int
    phase: CheckpointPhase
    model: str
    next_model_turn: int
    messages: tuple[AgentMessage, ...]
    incorporated_inputs: tuple[IncorporatedInput, ...] = ()
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    terminal_status: CheckpointTerminalStatus | None = None
    stop_reason: str | None = None
    final_content: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.session_id, "ContextCheckpoint.session_id")
        _require_text(self.input_id, "ContextCheckpoint.input_id")
        _require_text(self.model, "ContextCheckpoint.model")
        if (
            not isinstance(self.input_revision, int)
            or isinstance(self.input_revision, bool)
            or self.input_revision < 1
        ):
            raise ValueError("ContextCheckpoint.input_revision must be a positive integer")
        if self.base_leaf_id is not None:
            _require_text(self.base_leaf_id, "ContextCheckpoint.base_leaf_id")
        if (
            not isinstance(self.save_cursor, int)
            or isinstance(self.save_cursor, bool)
            or self.save_cursor < 0
        ):
            raise ValueError("ContextCheckpoint.save_cursor must be a non-negative integer")
        if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None:
            raise ValueError("ContextCheckpoint.updated_at must be timezone-aware")

        _validate_common(
            phase=self.phase,
            next_model_turn=self.next_model_turn,
            messages=self.messages,
            tools_used=self.tools_used,
            usage=self.usage,
            terminal_status=self.terminal_status,
            stop_reason=self.stop_reason,
            final_content=self.final_content,
            owner="ContextCheckpoint",
        )
        messages = tuple(self.messages)
        if not isinstance(messages[0], UserMessage) or messages[0].id != self.input_id:
            raise ValueError(
                "ContextCheckpoint.messages must start with its PendingInput UserMessage"
            )
        incorporated = tuple(self.incorporated_inputs) or (
            IncorporatedInput(id=self.input_id, revision=self.input_revision),
        )
        if any(not isinstance(item, IncorporatedInput) for item in incorporated):
            raise TypeError(
                "ContextCheckpoint.incorporated_inputs must contain IncorporatedInput values"
            )
        if incorporated[0] != IncorporatedInput(
            id=self.input_id,
            revision=self.input_revision,
        ):
            raise ValueError(
                "ContextCheckpoint.incorporated_inputs must start with its primary input"
            )
        incorporated_ids = [item.id for item in incorporated]
        if len(incorporated_ids) != len(set(incorporated_ids)):
            raise ValueError("ContextCheckpoint.incorporated_inputs IDs must be unique")
        user_ids = [message.id for message in messages if isinstance(message, UserMessage)]
        if user_ids != incorporated_ids:
            raise ValueError(
                "ContextCheckpoint UserMessages must match incorporated_inputs in order"
            )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "incorporated_inputs", incorporated)
        object.__setattr__(self, "tools_used", tuple(self.tools_used))
        object.__setattr__(self, "usage", _freeze_usage(self.usage))


class CheckpointStore(Protocol):
    def load(self, session_id: str) -> ContextCheckpoint | None: ...

    def save(self, checkpoint: ContextCheckpoint) -> None: ...

    def delete(self, session_id: str) -> bool: ...


def _validate_common(
    *,
    phase: CheckpointPhase,
    next_model_turn: int,
    messages: Sequence[AgentMessage],
    tools_used: Sequence[str],
    usage: Mapping[str, int],
    terminal_status: CheckpointTerminalStatus | None,
    stop_reason: str | None,
    final_content: str | None,
    owner: str,
) -> None:
    if phase not in {"awaiting_tools", "tools_completed", "final_response"}:
        raise ValueError(f"Invalid checkpoint phase: {phase}")
    if (
        not isinstance(next_model_turn, int)
        or isinstance(next_model_turn, bool)
        or next_model_turn < 1
    ):
        raise ValueError(f"{owner}.next_model_turn must be a positive integer")
    normalized_messages = tuple(messages)
    if not normalized_messages:
        raise ValueError(f"{owner}.messages cannot be empty")
    message_types = (
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        RuntimeStatusMessage,
    )
    if any(not isinstance(message, message_types) for message in normalized_messages):
        raise TypeError(f"{owner}.messages must contain AgentMessage values")
    if any(not isinstance(name, str) or not name.strip() for name in tools_used):
        raise ValueError(f"{owner}.tools_used must contain non-empty names")
    _freeze_usage(usage)
    _validate_phase_shape(
        phase,
        normalized_messages,
        terminal_status=terminal_status,
        stop_reason=stop_reason,
        final_content=final_content,
        owner=owner,
    )


def _validate_phase_shape(
    phase: CheckpointPhase,
    messages: tuple[AgentMessage, ...],
    *,
    terminal_status: CheckpointTerminalStatus | None,
    stop_reason: str | None,
    final_content: str | None,
    owner: str,
) -> None:
    if phase == "awaiting_tools":
        latest = messages[-1]
        if not isinstance(latest, AssistantMessage) or not latest.tool_calls:
            raise ValueError(f"{owner} awaiting_tools must end with Assistant tool calls")
    elif phase == "tools_completed":
        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], AssistantMessage)
            ),
            None,
        )
        if assistant_index is None:
            raise ValueError(f"{owner} tools_completed has no Assistant tool calls")
        assistant = messages[assistant_index]
        if not isinstance(assistant, AssistantMessage) or not assistant.tool_calls:
            raise ValueError(f"{owner} tools_completed must follow Assistant tool calls")
        results = messages[assistant_index + 1 :]
        if not results or any(not isinstance(item, ToolResultMessage) for item in results):
            raise ValueError(f"{owner} tools_completed must end with ToolResult messages")
        expected_ids = [call.id for call in assistant.tool_calls]
        result_ids = [item.tool_call_id for item in results if isinstance(item, ToolResultMessage)]
        if len(expected_ids) != len(set(expected_ids)) or sorted(result_ids) != sorted(expected_ids):
            raise ValueError(f"{owner} tools_completed must fulfill each latest ToolCall once")

    if phase == "final_response":
        latest = messages[-1]
        if (
            not isinstance(latest, AssistantMessage)
            or latest.tool_calls
            or not latest.content
        ):
            raise ValueError(f"{owner} final_response must end with final Assistant text")
        if terminal_status not in {"completed", "limit_reached"}:
            raise ValueError(f"{owner} final_response requires terminal_status")
        _require_text(stop_reason, f"{owner}.stop_reason")
        _require_text(final_content, f"{owner}.final_content")
        if latest.content != final_content:
            raise ValueError(f"{owner}.final_content must match the final Assistant message")
    elif terminal_status is not None or stop_reason is not None or final_content is not None:
        raise ValueError(f"{owner} non-final checkpoint cannot contain terminal fields")


def _freeze_usage(usage: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(usage, Mapping):
        raise TypeError("checkpoint usage must be a mapping")
    copied: dict[str, int] = {}
    for key, value in usage.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("checkpoint usage keys must be non-empty text")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("checkpoint usage values must be non-negative integers")
        copied[key] = value
    return MappingProxyType(copied)


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


__all__ = [
    "CheckpointConflictError",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointPhase",
    "CheckpointStorageError",
    "CheckpointStore",
    "CheckpointTerminalStatus",
    "ContextCheckpoint",
    "IncorporatedInput",
    "RunnerCheckpoint",
]
