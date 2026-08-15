"""Provider-neutral messages persisted by the Agent Harness."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeAlias
from uuid import uuid4

from agent_harness.artifacts import ArtifactRef
from agent_harness.status import RuntimeStatusSnapshot


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_message_id() -> str:
    return f"msg_{uuid4().hex}"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-neutral tool request embedded in an assistant message."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.id, "ToolCall.id")
        _require_text(self.name, "ToolCall.name")
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class UserMessage:
    """A user message that has successfully entered conversation history."""

    content: str
    id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "UserMessage.id")
        _require_text(self.content, "UserMessage.content")


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """A complete assistant response, optionally containing tool calls."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "AssistantMessage.id")
        if not isinstance(self.content, str):
            raise TypeError("AssistantMessage.content must be text")
        if not self.content and not self.tool_calls:
            raise ValueError("AssistantMessage must contain text or tool calls")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """A terminal result associated with a model tool call by ID."""

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    artifact_refs: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "ToolResultMessage.id")
        _require_text(self.tool_call_id, "ToolResultMessage.tool_call_id")
        _require_text(self.tool_name, "ToolResultMessage.tool_name")
        if not isinstance(self.content, str):
            raise TypeError("ToolResultMessage.content must be text")
        refs = tuple(self.artifact_refs)
        if any(not isinstance(ref, ArtifactRef) for ref in refs):
            raise TypeError("ToolResultMessage.artifact_refs must contain ArtifactRef values")
        object.__setattr__(self, "artifact_refs", refs)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ToolResultMessage.metadata must be a mapping")
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RuntimeStatusMessage:
    """Persisted Harness metadata mapped to a standard Provider user message."""

    snapshot: RuntimeStatusSnapshot
    content: str
    render_profile: str = "dao-default-v1"
    display: bool = False
    id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.id, "RuntimeStatusMessage.id")
        if not isinstance(self.snapshot, RuntimeStatusSnapshot):
            raise TypeError("RuntimeStatusMessage.snapshot must be RuntimeStatusSnapshot")
        _require_text(self.content, "RuntimeStatusMessage.content")
        _require_text(self.render_profile, "RuntimeStatusMessage.render_profile")
        if not isinstance(self.display, bool):
            raise TypeError("RuntimeStatusMessage.display must be boolean")


AgentMessage: TypeAlias = (
    UserMessage | AssistantMessage | ToolResultMessage | RuntimeStatusMessage
)


__all__ = [
    "AgentMessage",
    "AssistantMessage",
    "RuntimeStatusMessage",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "new_message_id",
    "utc_now",
]
