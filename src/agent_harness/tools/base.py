"""Tool contracts used by the first Runner milestone."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_harness.artifacts import ArtifactRef

ToolExecutionMode = Literal["parallel_safe", "sequential"]
ToolExecutionStatus = Literal["completed", "failed", "timed_out"]
ToolExecutionErrorCode = Literal[
    "not_found",
    "invalid_arguments",
    "exception",
    "reported_error",
    "timeout",
    "artifact_store",
]


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    """Registry-wide defaults for one tool call."""

    default_timeout_s: float | None = 60.0

    def __post_init__(self) -> None:
        timeout = self.default_timeout_s
        if timeout is None:
            return
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("default_timeout_s must be None or a positive number")
        object.__setattr__(self, "default_timeout_s", float(timeout))


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """A tool-controlled model view with optional complete Artifact content."""

    content: str
    artifact_content: str | None = None
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allow_externalization: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("ToolOutput.content must be text")
        if self.artifact_content is not None and not isinstance(
            self.artifact_content, str
        ):
            raise TypeError("ToolOutput.artifact_content must be None or text")
        if not isinstance(self.is_error, bool):
            raise TypeError("ToolOutput.is_error must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ToolOutput.metadata must be a mapping")
        if not isinstance(self.allow_externalization, bool):
            raise TypeError("ToolOutput.allow_externalization must be boolean")
        if not self.allow_externalization and self.artifact_content is not None:
            raise ValueError(
                "ToolOutput with artifact_content must allow externalization"
            )
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Normalized outcome of exactly one model-requested tool call."""

    tool_call_id: str
    tool_name: str
    content: str
    status: ToolExecutionStatus
    error_code: ToolExecutionErrorCode | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("tool_call_id must be non-empty text")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be non-empty text")
        if not isinstance(self.content, str):
            raise TypeError("content must be text")
        if self.status not in {"completed", "failed", "timed_out"}:
            raise ValueError(f"Invalid tool execution status: {self.status}")
        if self.status == "completed" and self.error_code is not None:
            raise ValueError("completed tool execution cannot have an error_code")
        if self.status != "completed" and self.error_code is None:
            raise ValueError("non-completed tool execution requires an error_code")
        if self.error_code not in {
            None,
            "not_found",
            "invalid_arguments",
            "exception",
            "reported_error",
            "timeout",
            "artifact_store",
        }:
            raise ValueError(f"Invalid tool execution error_code: {self.error_code}")
        if self.status == "timed_out" and self.error_code != "timeout":
            raise ValueError("timed_out tool execution requires error_code='timeout'")
        refs = tuple(self.artifact_refs)
        if any(not isinstance(ref, ArtifactRef) for ref in refs):
            raise TypeError("artifact_refs must contain ArtifactRef values")
        object.__setattr__(self, "artifact_refs", refs)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @property
    def is_error(self) -> bool:
        return self.status != "completed"


class AgentTool(Protocol):
    name: str
    description: str
    parameters: Mapping[str, Any]
    execution_mode: ToolExecutionMode
    timeout_s: float | None

    async def execute(self, arguments: Mapping[str, Any]) -> Any: ...


__all__ = [
    "AgentTool",
    "ToolExecutionErrorCode",
    "ToolExecutionMode",
    "ToolExecutionPolicy",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "ToolOutput",
]
