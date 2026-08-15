"""Provider-neutral contracts used by the first Runner milestone."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """A tool invocation requested by a model response."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Minimal response understood by the extracted Runner."""

    content: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()
    finish_reason: str = "stop"
    usage: Mapping[str, int] = field(default_factory=dict)

    @property
    def should_execute_tools(self) -> bool:
        return bool(self.tool_calls) and self.finish_reason in {
            "tool_calls",
            "function_call",
            "stop",
        }


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A provider-neutral piece of assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    """A fully assembled tool call, ready for the Runner to execute."""

    tool_call: ToolCallRequest


@dataclass(frozen=True, slots=True)
class ResponseCompleted:
    """Final metadata for a streamed model response."""

    finish_reason: str = "stop"
    usage: Mapping[str, int] = field(default_factory=dict)


LLMStreamEvent = TextDelta | ToolCallCompleted | ResponseCompleted


class LLMProvider(Protocol):
    """Provider interface required by AgentRunner."""

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse: ...


class StreamingLLMProvider(LLMProvider, Protocol):
    """Optional streaming capability implemented by supporting providers."""

    def stream(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[LLMStreamEvent]: ...
