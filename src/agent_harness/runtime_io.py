"""Public request and ephemeral streaming protocols for AgentRuntime adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    """One external user input submitted to an AgentRuntime."""

    session_id: str
    source_message_id: str
    content: str

    def __post_init__(self) -> None:
        for name, value in (
            ("session_id", self.session_id),
            ("source_message_id", self.source_message_id),
            ("content", self.content),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class OutputTextDelta:
    """Ephemeral text generated for one model-output segment."""

    input_id: str
    segment_index: int
    text: str


@dataclass(frozen=True, slots=True)
class OutputSegmentEnded:
    """End one model-output segment, optionally before the loop resumes."""

    input_id: str
    segment_index: int
    resuming: bool


RuntimeStreamEvent: TypeAlias = OutputTextDelta | OutputSegmentEnded
RuntimeStreamHandler: TypeAlias = Callable[
    [RuntimeStreamEvent], Awaitable[None] | None
]


__all__ = [
    "OutputSegmentEnded",
    "OutputTextDelta",
    "RuntimeRequest",
    "RuntimeStreamEvent",
    "RuntimeStreamHandler",
]
