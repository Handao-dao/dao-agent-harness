"""Deterministic test doubles for Harness consumers and the project test suite."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent_harness.providers.base import LLMResponse
from agent_harness.tools.base import ToolExecutionMode


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    system_prompt: str | None
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]


class ScriptedProvider:
    """Return responses or raise exceptions from a predefined script."""

    def __init__(self, script: Iterable[LLMResponse | Exception]):
        self._script = deque(script)
        self.requests: list[ModelRequest] = []

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        self.requests.append(
            ModelRequest(
                model=model,
                system_prompt=system_prompt,
                messages=tuple(deepcopy(dict(message)) for message in messages),
                tools=tuple(deepcopy(dict(tool)) for tool in tools),
            )
        )
        if not self._script:
            raise RuntimeError("ScriptedProvider has no response left")

        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item


@dataclass(slots=True)
class FakeTool:
    name: str
    result: Any = "ok"
    error: Exception | None = None
    description: str = "A deterministic fake tool"
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def execute(self, arguments: Mapping[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        if self.error is not None:
            raise self.error
        return self.result


__all__ = ["FakeTool", "ModelRequest", "ScriptedProvider"]
