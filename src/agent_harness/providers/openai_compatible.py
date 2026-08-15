"""Minimal OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent_harness.providers.base import (
    LLMResponse,
    LLMStreamEvent,
    ResponseCompleted,
    TextDelta,
    ToolCallCompleted,
    ToolCallRequest,
)
from agent_harness.providers.transport import (
    JsonHttpTransport,
    ProviderHTTPError,
    SseHttpTransport,
    UrllibJsonTransport,
)


class ProviderProtocolError(RuntimeError):
    """A successful HTTP response did not match the expected API schema."""


@dataclass(slots=True)
class _ToolCallAccumulator:
    call_id_parts: list[str] = field(default_factory=list)
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        transport: JsonHttpTransport | None = None,
        stream_transport: SseHttpTransport | None = None,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._timeout_s = timeout_s
        default_transport = UrllibJsonTransport()
        self._transport = transport or default_transport
        self._stream_transport = stream_transport or default_transport

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        payload = self._build_payload(model, system_prompt, messages, tools)
        headers = self._headers()

        response = await self._transport.post_json(
            url=self._endpoint,
            headers=headers,
            payload=payload,
            timeout_s=self._timeout_s,
        )
        return self._parse_response(response)

    async def stream(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[LLMStreamEvent]:
        """Translate OpenAI SSE chunks into provider-neutral stream events."""

        payload = self._build_payload(model, system_prompt, messages, tools)
        payload["stream"] = True
        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        tool_calls: dict[int, _ToolCallAccumulator] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        async for data in self._stream_transport.post_sse(
            url=self._endpoint,
            headers=headers,
            payload=payload,
            timeout_s=self._timeout_s,
        ):
            if data.strip() == "[DONE]":
                break
            chunk = self._decode_stream_chunk(data)
            usage.update(self._parse_usage(chunk.get("usage")))

            choices = chunk.get("choices")
            if choices is None:
                continue
            if not isinstance(choices, list):
                raise ProviderProtocolError("Stream choices must be a list")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ProviderProtocolError("Stream choice must be an object")

            raw_finish_reason = choice.get("finish_reason")
            if isinstance(raw_finish_reason, str):
                finish_reason = raw_finish_reason

            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise ProviderProtocolError("Stream choice has no delta object")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ProviderProtocolError("Stream content delta must be text or null")
                if content:
                    yield TextDelta(content)

            self._accumulate_tool_call_deltas(tool_calls, delta.get("tool_calls"))

        for index in sorted(tool_calls):
            yield ToolCallCompleted(self._finish_tool_call(tool_calls[index], index))
        yield ResponseCompleted(finish_reason=finish_reason, usage=usage)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _build_payload(
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_messages = [deepcopy(dict(message)) for message in messages]
        if system_prompt:
            request_messages.insert(0, {"role": "system", "content": system_prompt})
        payload: dict[str, Any] = {"model": model, "messages": request_messages}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": deepcopy(dict(tool))} for tool in tools
            ]
        return payload

    @staticmethod
    def _decode_stream_chunk(data: str) -> Mapping[str, Any]:
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("Provider stream returned invalid JSON") from exc
        if not isinstance(chunk, Mapping):
            raise ProviderProtocolError("Provider stream chunk must be an object")
        if "error" in chunk:
            raise ProviderProtocolError(f"Provider stream returned an error: {chunk['error']}")
        return chunk

    @staticmethod
    def _parse_usage(raw_usage: Any) -> dict[str, int]:
        if not isinstance(raw_usage, Mapping):
            return {}
        return {key: int(value) for key, value in raw_usage.items() if isinstance(value, int)}

    @staticmethod
    def _accumulate_tool_call_deltas(
        target: dict[int, _ToolCallAccumulator],
        raw_tool_calls: Any,
    ) -> None:
        if raw_tool_calls is None:
            return
        if not isinstance(raw_tool_calls, list):
            raise ProviderProtocolError("Stream tool_calls delta must be a list")
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, Mapping):
                raise ProviderProtocolError("Stream tool call delta must be an object")
            index = raw_call.get("index")
            if not isinstance(index, int) or index < 0:
                raise ProviderProtocolError("Stream tool call delta has no valid index")
            accumulator = target.setdefault(index, _ToolCallAccumulator())
            call_id = raw_call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise ProviderProtocolError("Stream tool call id delta must be text")
                accumulator.call_id_parts.append(call_id)
            function = raw_call.get("function")
            if function is None:
                continue
            if not isinstance(function, Mapping):
                raise ProviderProtocolError("Stream tool call function must be an object")
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise ProviderProtocolError("Stream tool name delta must be text")
                accumulator.name_parts.append(name)
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ProviderProtocolError("Stream tool arguments delta must be text")
                accumulator.argument_parts.append(arguments)

    @staticmethod
    def _finish_tool_call(accumulator: _ToolCallAccumulator, index: int) -> ToolCallRequest:
        call_id = "".join(accumulator.call_id_parts)
        name = "".join(accumulator.name_parts)
        if not call_id:
            raise ProviderProtocolError(f"Stream tool call {index} has no id")
        if not name:
            raise ProviderProtocolError(f"Stream tool call {call_id} has no name")
        raw_arguments = "".join(accumulator.argument_parts) or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError(
                f"Stream tool call {call_id} has invalid JSON arguments"
            ) from exc
        if not isinstance(arguments, dict):
            raise ProviderProtocolError(
                f"Stream tool call {call_id} arguments must decode to an object"
            )
        return ToolCallRequest(id=call_id, name=name, arguments=arguments)

    @classmethod
    def _parse_response(cls, payload: Mapping[str, Any]) -> LLMResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderProtocolError("Provider response has no choices")

        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderProtocolError("Provider choice must be an object")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderProtocolError("Provider choice has no message")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderProtocolError("Assistant content must be text or null")

        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ProviderProtocolError("Assistant tool_calls must be a list")
        tool_calls = tuple(cls._parse_tool_call(item) for item in raw_tool_calls)

        usage = payload.get("usage") or {}
        parsed_usage = (
            {key: int(value) for key, value in usage.items() if isinstance(value, int)}
            if isinstance(usage, Mapping)
            else {}
        )
        finish_reason = choice.get("finish_reason")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason if isinstance(finish_reason, str) else "stop",
            usage=parsed_usage,
        )

    @staticmethod
    def _parse_tool_call(raw: Any) -> ToolCallRequest:
        if not isinstance(raw, Mapping):
            raise ProviderProtocolError("Tool call must be an object")
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise ProviderProtocolError("Tool call has no function")

        call_id = raw.get("id")
        name = function.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise ProviderProtocolError("Tool call has no id")
        if not isinstance(name, str) or not name:
            raise ProviderProtocolError("Tool call has no name")

        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ProviderProtocolError(f"Tool call {call_id} has invalid JSON arguments") from exc
        elif isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            raise ProviderProtocolError(f"Tool call {call_id} arguments must be JSON")

        if not isinstance(arguments, dict):
            raise ProviderProtocolError(f"Tool call {call_id} arguments must decode to an object")
        return ToolCallRequest(id=call_id, name=name, arguments=arguments)


__all__ = [
    "OpenAICompatibleProvider",
    "ProviderHTTPError",
    "ProviderProtocolError",
]
