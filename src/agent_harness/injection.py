"""Strong protocol for transient mid-run user-message injection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_harness.messages import UserMessage


class MessageInjectionPoint(Enum):
    """Runner locations where durable PendingInput may be incorporated."""

    AFTER_TOOLS = "after_tools"
    AFTER_CANDIDATE_RESPONSE = "after_candidate_response"


@dataclass(frozen=True, slots=True)
class MessageInjectionBatch:
    """One stable batch of user messages returned at an injection point."""

    point: MessageInjectionPoint
    messages: tuple[UserMessage, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.point, MessageInjectionPoint):
            raise TypeError("MessageInjectionBatch.point must be a MessageInjectionPoint")
        messages = tuple(self.messages)
        if any(not isinstance(message, UserMessage) for message in messages):
            raise TypeError("MessageInjectionBatch.messages must contain UserMessage values")
        ids = [message.id for message in messages]
        if len(ids) != len(set(ids)):
            raise ValueError("MessageInjectionBatch.messages must have unique IDs")
        object.__setattr__(self, "messages", messages)

    @classmethod
    def empty(cls, point: MessageInjectionPoint) -> MessageInjectionBatch:
        return cls(point=point)


MessageInjectionHandler = Callable[
    [MessageInjectionPoint, int],
    Awaitable[MessageInjectionBatch] | MessageInjectionBatch,
]


def merge_consecutive_user_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Merge adjacent Provider-facing user text without changing typed history."""

    merged: list[dict[str, Any]] = []
    for message in messages:
        current = dict(message)
        if (
            merged
            and current.get("role") == "user"
            and merged[-1].get("role") == "user"
            and isinstance(current.get("content"), str)
            and isinstance(merged[-1].get("content"), str)
        ):
            previous_content = merged[-1]["content"]
            current_content = current["content"]
            if not isinstance(previous_content, str) or not isinstance(current_content, str):
                raise TypeError("User message content must be text")
            merged[-1] = {
                **merged[-1],
                "content": f"{previous_content}\n\n{current_content}",
            }
            continue
        merged.append(current)
    return tuple(merged)


__all__ = [
    "MessageInjectionBatch",
    "MessageInjectionHandler",
    "MessageInjectionPoint",
    "merge_consecutive_user_messages",
]
