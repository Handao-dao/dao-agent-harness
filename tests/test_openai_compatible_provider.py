from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from agent_harness.providers.base import (
    ResponseCompleted,
    TextDelta,
    ToolCallCompleted,
    ToolCallRequest,
)
from agent_harness.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderProtocolError,
)


class FakeTransport:
    def __init__(self, response: Mapping[str, Any]):
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        return self.response


class FakeStreamTransport:
    def __init__(self, events: list[str]):
        self.events = events
        self.requests: list[dict[str, Any]] = []

    async def post_sse(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[str]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        for event in self.events:
            yield event


def stream_chunk(
    *,
    delta: Mapping[str, Any],
    finish_reason: str | None = None,
    usage: Mapping[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "choices": [{"delta": dict(delta), "finish_reason": finish_reason}]
    }
    if usage is not None:
        payload["usage"] = dict(usage)
    return json.dumps(payload)


async def test_translates_request_and_parses_text_response() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    )
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1/",
        api_key="secret",
        timeout_s=30,
        transport=transport,
    )

    response = await provider.complete(
        model="test-model",
        system_prompt="be concise",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}],
    )

    assert response.content == "hello"
    assert response.usage["total_tokens"] == 6
    request = transport.requests[0]
    assert request["url"] == "https://example.test/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["payload"]["messages"][0] == {"role": "system", "content": "be concise"}
    assert request["payload"]["tools"][0]["type"] == "function"


async def test_parses_tool_call_arguments() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query":"nanobot"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(base_url="http://localhost:8000/v1", transport=transport)

    response = await provider.complete(
        model="test-model",
        system_prompt=None,
        messages=[],
        tools=[],
    )

    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "nanobot"}


async def test_rejects_invalid_tool_call_arguments() -> None:
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": "{invalid"},
                            }
                        ],
                    }
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(base_url="http://localhost:8000/v1", transport=transport)

    with pytest.raises(ProviderProtocolError, match="invalid JSON"):
        await provider.complete(
            model="test-model",
            system_prompt=None,
            messages=[],
            tools=[],
        )


async def test_stream_emits_provider_neutral_text_events() -> None:
    stream_transport = FakeStreamTransport(
        [
            stream_chunk(delta={"role": "assistant", "content": "你"}),
            stream_chunk(delta={"content": "好"}),
            stream_chunk(delta={}, finish_reason="stop", usage={"total_tokens": 7}),
            "[DONE]",
        ]
    )
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="secret",
        stream_transport=stream_transport,
    )

    events = [
        event
        async for event in provider.stream(
            model="test-model",
            system_prompt="be concise",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )
    ]

    assert events == [
        TextDelta("你"),
        TextDelta("好"),
        ResponseCompleted(finish_reason="stop", usage={"total_tokens": 7}),
    ]
    request = stream_transport.requests[0]
    assert request["payload"]["stream"] is True
    assert request["headers"]["Accept"] == "text/event-stream"
    assert request["payload"]["messages"][0] == {
        "role": "system",
        "content": "be concise",
    }


async def test_stream_assembles_fragmented_tool_calls_before_emitting_them() -> None:
    stream_transport = FakeStreamTransport(
        [
            stream_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call-2",
                            "function": {"name": "second", "arguments": '{"n":'},
                        },
                        {
                            "index": 0,
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": '{"query":'},
                        },
                    ]
                }
            ),
            stream_chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '"nanobot"}'}},
                        {"index": 1, "function": {"arguments": "2}"}},
                    ]
                }
            ),
            stream_chunk(delta={}, finish_reason="tool_calls"),
            "[DONE]",
        ]
    )
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",
        stream_transport=stream_transport,
    )

    events = [
        event
        async for event in provider.stream(
            model="test-model",
            system_prompt=None,
            messages=[],
            tools=[],
        )
    ]

    assert events == [
        ToolCallCompleted(
            tool_call=ToolCallRequest("call-1", "lookup", {"query": "nanobot"})
        ),
        ToolCallCompleted(tool_call=ToolCallRequest("call-2", "second", {"n": 2})),
        ResponseCompleted(finish_reason="tool_calls"),
    ]


async def test_stream_rejects_invalid_assembled_tool_arguments() -> None:
    stream_transport = FakeStreamTransport(
        [
            stream_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": "{invalid"},
                        }
                    ]
                },
                finish_reason="tool_calls",
            ),
            "[DONE]",
        ]
    )
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",
        stream_transport=stream_transport,
    )

    with pytest.raises(ProviderProtocolError, match="invalid JSON"):
        _ = [
            event
            async for event in provider.stream(
                model="test-model",
                system_prompt=None,
                messages=[],
                tools=[],
            )
        ]
