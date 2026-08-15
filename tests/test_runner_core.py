from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from hashlib import sha256
from typing import Any

from agent_harness.artifacts import ArtifactRef
from agent_harness.checkpoints import RunnerCheckpoint
from agent_harness.context_governor import ContextGovernor, ContextGovernorConfig
from agent_harness.injection import MessageInjectionBatch, MessageInjectionPoint
from agent_harness.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent_harness.providers import (
    LLMResponse,
    LLMStreamEvent,
    ResponseCompleted,
    TextDelta,
    ToolCallCompleted,
    ToolCallRequest,
)
from agent_harness.runner import (
    EMPTY_RESPONSE_CONTENT,
    MAX_TURNS_CONTENT,
    AgentRunner,
    AgentRunSpec,
)
from agent_harness.testing import FakeTool, ScriptedProvider
from agent_harness.tools import (
    ToolExecutionMode,
    ToolExecutionPolicy,
    ToolExecutionResult,
    ToolOutput,
    ToolRegistry,
)


def make_spec(
    tools: ToolRegistry | None = None,
    *,
    max_turns: int = 5,
    stream: bool = False,
    on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
    on_stream_end: Callable[[bool], Awaitable[None] | None] | None = None,
) -> AgentRunSpec:
    return AgentRunSpec(
        initial_messages=[UserMessage(id="input-1", content="do the task")],
        tools=tools or ToolRegistry(),
        model="fake-model",
        max_turns=max_turns,
        stream=stream,
        on_text_delta=on_text_delta,
        on_stream_end=on_stream_end,
    )


def tool_call(*, call_id: str = "call-1", name: str = "lookup") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"query": "nanobot"})


class ArtifactResultRegistry(ToolRegistry):
    def __init__(self, ref: ArtifactRef) -> None:
        super().__init__()
        self.ref = ref
        self.register(FakeTool(name="lookup"))

    async def execute_call(self, call: Any) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content="externalized preview",
            status="completed",
            artifact_refs=(self.ref,),
            metadata={"kind": "test_output"},
        )


async def test_runner_preserves_artifact_refs_from_execution_result() -> None:
    content = "large tool result"
    digest = sha256(content.encode("utf-8")).hexdigest()
    ref = ArtifactRef(
        id=f"art_{digest}",
        media_type="text/plain; charset=utf-8",
        size_bytes=len(content.encode("utf-8")),
        size_chars=len(content),
        sha256=digest,
    )
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="done"),
        ]
    )

    result = await AgentRunner(provider).run(make_spec(ArtifactResultRegistry(ref)))

    tool_result = next(
        message for message in result.messages if isinstance(message, ToolResultMessage)
    )
    assert tool_result.content == "externalized preview"
    assert tool_result.artifact_refs == (ref,)
    assert tool_result.metadata == {"kind": "test_output"}


async def test_runner_injects_at_both_points_and_caps_total_followups() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="candidate"),
            LLMResponse(content="final"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="found"))
    calls: list[tuple[MessageInjectionPoint, int]] = []

    async def inject(point: MessageInjectionPoint, limit: int) -> MessageInjectionBatch:
        calls.append((point, limit))
        count = 3 if point is MessageInjectionPoint.AFTER_TOOLS else 2
        return MessageInjectionBatch(
            point=point,
            messages=tuple(
                UserMessage(id=f"injected-{len(calls)}-{index}", content=f"follow-up {index}")
                for index in range(count)
            ),
        )

    base = make_spec(registry, max_turns=3)
    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=base.initial_messages,
            tools=base.tools,
            model=base.model,
            max_turns=base.max_turns,
            injection_callback=inject,
            max_injected_inputs_per_run=5,
        )
    )

    assert result.status == "completed"
    assert result.final_content == "final"
    assert calls == [
        (MessageInjectionPoint.AFTER_TOOLS, 5),
        (MessageInjectionPoint.AFTER_CANDIDATE_RESPONSE, 2),
    ]
    assert len([message for message in result.messages if isinstance(message, UserMessage)]) == 6
    assert provider.requests[1].messages[-1] == {
        "role": "user",
        "content": "follow-up 0\n\nfollow-up 1\n\nfollow-up 2",
    }
    assert provider.requests[2].messages[-1] == {
        "role": "user",
        "content": "follow-up 0\n\nfollow-up 1",
    }


async def test_runner_does_not_claim_injections_without_a_next_model_turn() -> None:
    provider = ScriptedProvider(
        [LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls")]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="found"))
    calls: list[MessageInjectionPoint] = []

    def inject(point: MessageInjectionPoint, limit: int) -> MessageInjectionBatch:
        calls.append(point)
        return MessageInjectionBatch(
            point=point,
            messages=(UserMessage(id="late", content="late follow-up"),),
        )

    base = make_spec(registry, max_turns=1)
    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=base.initial_messages,
            tools=base.tools,
            model=base.model,
            max_turns=base.max_turns,
            injection_callback=inject,
        )
    )

    assert result.status == "limit_reached"
    assert calls == []
    assert all(message.id != "late" for message in result.messages)


async def test_runner_emits_three_durable_checkpoint_phases() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="done", usage={"completion_tokens": 2}),
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="found"))
    captured: list[RunnerCheckpoint] = []

    async def save_checkpoint(value: RunnerCheckpoint) -> None:
        captured.append(value)

    spec = make_spec(registry)
    spec = AgentRunSpec(
        initial_messages=spec.initial_messages,
        tools=spec.tools,
        model=spec.model,
        max_turns=spec.max_turns,
        checkpoint_callback=save_checkpoint,
    )

    result = await AgentRunner(provider).run(spec)

    assert result.status == "completed"
    assert [item.phase for item in captured] == [
        "awaiting_tools",
        "tools_completed",
        "final_response",
    ]
    assert [item.next_model_turn for item in captured] == [1, 1, 2]
    assert captured[1].messages[-1].content == "found"
    assert captured[2].terminal_status == "completed"
    assert captured[2].final_content == "done"
    assert captured[2].usage == {"completion_tokens": 2}


async def test_awaiting_tools_checkpoint_failure_prevents_tool_execution() -> None:
    provider = ScriptedProvider(
        [LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls")]
    )
    tool = FakeTool(name="lookup", result="must not run")
    registry = ToolRegistry()
    registry.register(tool)

    def fail_checkpoint(_value: RunnerCheckpoint) -> None:
        raise OSError("disk unavailable")

    base = make_spec(registry)
    spec = AgentRunSpec(
        initial_messages=base.initial_messages,
        tools=registry,
        model=base.model,
        checkpoint_callback=fail_checkpoint,
    )

    result = await AgentRunner(provider).run(spec)

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_error"
    assert result.error == "OSError: disk unavailable"
    assert tool.calls == []


async def test_tools_completed_checkpoint_failure_stops_before_next_model_call() -> None:
    provider = ScriptedProvider(
        [LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls")]
    )
    tool = FakeTool(name="lookup", result="completed once")
    registry = ToolRegistry()
    registry.register(tool)

    def fail_after_tools(value: RunnerCheckpoint) -> None:
        if value.phase == "tools_completed":
            raise OSError("cannot replace checkpoint")

    base = make_spec(registry)
    spec = AgentRunSpec(
        initial_messages=base.initial_messages,
        tools=registry,
        model=base.model,
        checkpoint_callback=fail_after_tools,
    )

    result = await AgentRunner(provider).run(spec)

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_error"
    assert tool.calls == [{"query": "nanobot"}]
    assert len(provider.requests) == 1


async def test_final_checkpoint_failure_prevents_completed_result() -> None:
    provider = ScriptedProvider([LLMResponse(content="candidate answer")])

    def fail_final(value: RunnerCheckpoint) -> None:
        assert value.phase == "final_response"
        raise OSError("checkpoint full")

    base = make_spec()
    spec = AgentRunSpec(
        initial_messages=base.initial_messages,
        tools=base.tools,
        model=base.model,
        checkpoint_callback=fail_final,
    )

    result = await AgentRunner(provider).run(spec)

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_error"
    assert result.final_content is None


async def test_runner_resume_offset_preserves_budget_usage_and_tools() -> None:
    provider = ScriptedProvider([])
    user = UserMessage(id="input-resume", content="resume")
    assistant = AssistantMessage(
        tool_calls=(ToolCall(id="call-resume", name="lookup"),)
    )
    tool_result = ToolResultMessage(
        tool_call_id="call-resume",
        tool_name="lookup",
        content="done",
    )
    captured: list[RunnerCheckpoint] = []
    spec = AgentRunSpec(
        initial_messages=(user, assistant, tool_result),
        tools=ToolRegistry(),
        model="fake-model",
        max_turns=1,
        model_turn_offset=1,
        initial_tools_used=("lookup",),
        initial_usage={"prompt_tokens": 8},
        checkpoint_callback=captured.append,
        current_turn_start=0,
    )

    result = await AgentRunner(provider).run(spec)

    assert result.status == "limit_reached"
    assert result.tools_used == ("lookup",)
    assert result.usage == {"prompt_tokens": 8}
    assert len(provider.requests) == 0
    assert captured[-1].phase == "final_response"
    assert captured[-1].terminal_status == "limit_reached"


class ScriptedStreamingProvider:
    def __init__(self, turns: Sequence[Sequence[LLMStreamEvent]]):
        self._turns = deque(turns)
        self.requests: list[tuple[dict[str, Any], ...]] = []

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        raise AssertionError("streaming Runner should not call complete()")

    async def stream(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(tuple(dict(message) for message in messages))
        for event in self._turns.popleft():
            yield event


class ControlledTool:
    def __init__(
        self,
        name: str,
        *,
        execution_mode: ToolExecutionMode,
        release: asyncio.Event | None = None,
        trace: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.name = name
        self.description = f"Controlled {name} tool"
        self.parameters: Mapping[str, Any] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.execution_mode = execution_mode
        self.release = release
        self.timeout_s = timeout_s
        self.trace = trace if trace is not None else []
        self.started = asyncio.Event()

    async def execute(self, arguments: Mapping[str, Any]) -> str:
        assert arguments == {}
        self.trace.append(f"start:{self.name}")
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        self.trace.append(f"end:{self.name}")
        return self.name


async def test_returns_plain_assistant_response() -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="done", usage={"prompt_tokens": 3, "completion_tokens": 1})]
    )

    result = await AgentRunner(provider).run(make_spec())

    assert result.status == "completed"
    assert result.stop_reason == "model_stop"
    assert result.final_content == "done"
    assert isinstance(result.messages[-1], AssistantMessage)
    assert result.messages[-1].content == "done"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 1}


async def test_executes_tool_and_continues() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="tool complete"),
        ]
    )
    registry = ToolRegistry()
    tool = FakeTool(name="lookup", result={"answer": 42})
    registry.register(tool)

    result = await AgentRunner(provider).run(make_spec(registry))

    assert result.status == "completed"
    assert result.final_content == "tool complete"
    assert tool.calls == [{"query": "nanobot"}]
    assert result.tools_used == ("lookup",)
    assert len(provider.requests) == 2


async def test_passes_tool_result_to_next_model_call() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="observed"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="tool-output"))

    await AgentRunner(provider).run(make_spec(registry))

    second_request = provider.requests[1]
    tool_message = next(message for message in second_request.messages if message["role"] == "tool")
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "lookup",
        "content": "tool-output",
    }


async def test_converts_tool_error_to_structured_tool_result() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="recovered"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", error=ValueError("bad query")))

    result = await AgentRunner(provider).run(make_spec(registry))

    tool_message = next(
        message for message in result.messages if isinstance(message, ToolResultMessage)
    )
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.content == "ValueError: bad query"
    assert tool_message.is_error is True
    assert provider.requests[1].messages[-1]["content"] == "Error: ValueError: bad query"
    assert result.status == "completed"


async def test_invalid_tool_arguments_return_to_model_without_execution() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="lookup", arguments={}),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="I need a query before I can use that tool."),
        ]
    )
    registry = ToolRegistry()
    tool = FakeTool(
        name="lookup",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    registry.register(tool)

    result = await AgentRunner(provider).run(make_spec(registry))

    error_result = next(
        message for message in result.messages if isinstance(message, ToolResultMessage)
    )
    assert error_result.is_error is True
    assert "missing required query" in error_result.content
    assert tool.calls == []
    assert provider.requests[1].messages[-1]["content"].startswith("Error:")
    assert result.status == "completed"


async def test_unknown_tool_returns_to_model_and_loop_continues() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="missing", arguments={}),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="That tool is not available."),
        ]
    )

    result = await AgentRunner(provider).run(make_spec())

    tool_result = next(
        message for message in result.messages if isinstance(message, ToolResultMessage)
    )
    assert tool_result.is_error is True
    assert tool_result.tool_call_id == "call-1"
    assert "Tool 'missing' not found" in tool_result.content
    assert result.status == "completed"


async def test_tool_reported_error_returns_to_model_as_structured_error() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="used another approach"),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        FakeTool(
            name="lookup",
            result=ToolOutput(content="Error: service unavailable", is_error=True),
        )
    )

    result = await AgentRunner(provider).run(make_spec(registry))

    tool_result = next(
        message for message in result.messages if isinstance(message, ToolResultMessage)
    )
    assert tool_result.is_error is True
    assert tool_result.content == "Error: service unavailable"
    assert provider.requests[1].messages[-1]["content"] == "Error: service unavailable"
    assert result.status == "completed"


async def test_parallel_safe_tools_run_concurrently_and_results_use_completion_order() -> None:
    calls = (
        ToolCallRequest(id="slow-call", name="slow", arguments={}),
        ToolCallRequest(id="fast-call", name="fast", arguments={}),
    )
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=calls, finish_reason="tool_calls"),
            LLMResponse(content="both observed"),
        ]
    )
    slow_release = asyncio.Event()
    slow = ControlledTool("slow", execution_mode="parallel_safe", release=slow_release)
    fast = ControlledTool("fast", execution_mode="parallel_safe")
    registry = ToolRegistry()
    registry.register(slow)
    registry.register(fast)

    run_task = asyncio.create_task(AgentRunner(provider).run(make_spec(registry)))
    await asyncio.wait_for(slow.started.wait(), timeout=1)
    await asyncio.wait_for(fast.started.wait(), timeout=1)
    await asyncio.sleep(0)
    slow_release.set()
    result = await run_task

    tool_messages = [
        message
        for message in provider.requests[1].messages
        if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "fast-call",
        "slow-call",
    ]
    assert result.status == "completed"


async def test_one_parallel_tool_timeout_does_not_cancel_its_batch_peers() -> None:
    calls = (
        ToolCallRequest(id="slow-call", name="slow", arguments={}),
        ToolCallRequest(id="fast-call", name="fast", arguments={}),
    )
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=calls, finish_reason="tool_calls"),
            LLMResponse(content="continued after timeout"),
        ]
    )
    slow = ControlledTool(
        "slow",
        execution_mode="parallel_safe",
        release=asyncio.Event(),
        timeout_s=0.01,
    )
    fast = ControlledTool("fast", execution_mode="parallel_safe")
    registry = ToolRegistry(ToolExecutionPolicy(default_timeout_s=1))
    registry.register(slow)
    registry.register(fast)

    result = await AgentRunner(provider).run(make_spec(registry))

    tool_messages = [
        message
        for message in provider.requests[1].messages
        if message["role"] == "tool"
    ]
    assert result.status == "completed"
    assert [message["tool_call_id"] for message in tool_messages] == [
        "fast-call",
        "slow-call",
    ]
    assert tool_messages[0]["content"] == "fast"
    assert "timed out after 0.01 seconds" in tool_messages[1]["content"]


async def test_external_cancel_reclaims_all_parallel_tool_tasks() -> None:
    class CancellationAwareTool(ControlledTool):
        def __init__(self, name: str) -> None:
            super().__init__(
                name,
                execution_mode="parallel_safe",
                release=asyncio.Event(),
                timeout_s=10,
            )
            self.cleaned = asyncio.Event()

        async def execute(self, arguments: Mapping[str, Any]) -> str:
            try:
                return await super().execute(arguments)
            finally:
                self.cleaned.set()

    calls = tuple(
        ToolCallRequest(id=f"call-{name}", name=name, arguments={})
        for name in ("a", "b")
    )
    provider = ScriptedProvider(
        [LLMResponse(tool_calls=calls, finish_reason="tool_calls")]
    )
    tools = [CancellationAwareTool("a"), CancellationAwareTool("b")]
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    run_task = asyncio.create_task(AgentRunner(provider).run(make_spec(registry)))
    await asyncio.wait_for(
        asyncio.gather(*(tool.started.wait() for tool in tools)),
        timeout=1,
    )

    run_task.cancel()
    result = await run_task

    assert result.status == "cancelled"
    assert all(tool.cleaned.is_set() for tool in tools)


async def test_sequential_tool_is_a_barrier_between_parallel_batches() -> None:
    calls = tuple(
        ToolCallRequest(id=f"call-{name}", name=name, arguments={})
        for name in ("a", "b", "c", "d", "e")
    )
    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=calls, finish_reason="tool_calls"),
            LLMResponse(content="finished"),
        ]
    )
    first_release = asyncio.Event()
    sequential_release = asyncio.Event()
    trace: list[str] = []
    tools = {
        "a": ControlledTool(
            "a", execution_mode="parallel_safe", release=first_release, trace=trace
        ),
        "b": ControlledTool(
            "b", execution_mode="parallel_safe", release=first_release, trace=trace
        ),
        "c": ControlledTool(
            "c", execution_mode="sequential", release=sequential_release, trace=trace
        ),
        "d": ControlledTool("d", execution_mode="parallel_safe", trace=trace),
        "e": ControlledTool("e", execution_mode="parallel_safe", trace=trace),
    }
    registry = ToolRegistry()
    for tool in tools.values():
        registry.register(tool)

    run_task = asyncio.create_task(AgentRunner(provider).run(make_spec(registry)))
    await asyncio.wait_for(
        asyncio.gather(tools["a"].started.wait(), tools["b"].started.wait()),
        timeout=1,
    )
    assert not tools["c"].started.is_set()
    first_release.set()

    await asyncio.wait_for(tools["c"].started.wait(), timeout=1)
    assert not tools["d"].started.is_set()
    assert not tools["e"].started.is_set()
    sequential_release.set()

    result = await run_task

    assert result.status == "completed"
    assert tools["d"].started.is_set()
    assert tools["e"].started.is_set()
    assert trace.index("end:c") < trace.index("start:d")
    assert trace.index("end:c") < trace.index("start:e")


async def test_stops_on_provider_exception_without_fake_assistant_message() -> None:
    provider = ScriptedProvider([RuntimeError("provider unavailable")])

    result = await AgentRunner(provider).run(make_spec())

    assert result.status == "failed"
    assert result.stop_reason == "provider_error"
    assert result.error == "RuntimeError: provider unavailable"
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], UserMessage)
    assert result.messages[0].id == "input-1"
    assert result.messages[0].content == "do the task"


async def test_error_finish_reason_does_not_execute_residual_tool_call() -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="quota exceeded", tool_calls=(tool_call(),), finish_reason="error")]
    )
    registry = ToolRegistry()
    tool = FakeTool(name="lookup", result="unused")
    registry.register(tool)

    result = await AgentRunner(provider).run(make_spec(registry))

    assert result.status == "failed"
    assert result.error == "quota exceeded"
    assert tool.calls == []
    assert len(result.messages) == 1


async def test_ignores_residual_tool_call_for_length_finish_reason() -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="partial", tool_calls=(tool_call(),), finish_reason="length")]
    )
    registry = ToolRegistry()
    tool = FakeTool(name="lookup", result="unused")
    registry.register(tool)

    result = await AgentRunner(provider).run(make_spec(registry))

    assert result.status == "completed"
    assert result.final_content == "partial"
    assert tool.calls == []


async def test_empty_normal_response_uses_framework_fallback() -> None:
    result = await AgentRunner(ScriptedProvider([LLMResponse()])).run(make_spec())

    assert result.status == "completed"
    assert result.final_content == EMPTY_RESPONSE_CONTENT
    assert result.messages[-1].content == EMPTY_RESPONSE_CONTENT


async def test_stops_at_max_turns_with_saveable_terminal_message() -> None:
    provider = ScriptedProvider(
        [LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls")]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="still working"))

    result = await AgentRunner(provider).run(make_spec(registry, max_turns=1))

    assert result.status == "limit_reached"
    assert result.stop_reason == "max_turns"
    assert result.final_content == MAX_TURNS_CONTENT
    assert isinstance(result.messages[-1], AssistantMessage)
    assert result.messages[-1].content == MAX_TURNS_CONTENT
    assert len(result.messages) == 4


async def test_streams_text_and_preserves_normal_result_shape() -> None:
    provider = ScriptedStreamingProvider(
        [[TextDelta("hel"), TextDelta("lo"), ResponseCompleted(usage={"total_tokens": 4})]]
    )
    deltas: list[str] = []
    stream_ends: list[bool] = []

    result = await AgentRunner(provider).run(
        make_spec(
            stream=True,
            on_text_delta=deltas.append,
            on_stream_end=stream_ends.append,
        )
    )

    assert deltas == ["hel", "lo"]
    assert stream_ends == [False]
    assert result.status == "completed"
    assert result.final_content == "hello"
    assert result.messages[-1].content == "hello"
    assert result.usage == {"total_tokens": 4}


async def test_streamed_tool_call_uses_existing_tool_loop() -> None:
    provider = ScriptedStreamingProvider(
        [
            [
                ToolCallCompleted(tool_call()),
                ResponseCompleted(finish_reason="tool_calls"),
            ],
            [TextDelta("observed"), ResponseCompleted()],
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="tool-output"))
    stream_ends: list[bool] = []

    result = await AgentRunner(provider).run(
        make_spec(registry, stream=True, on_stream_end=stream_ends.append)
    )

    assert result.status == "completed"
    assert result.final_content == "observed"
    assert result.tools_used == ("lookup",)
    assert stream_ends == [True, False]
    assert len(provider.requests) == 2
    assert provider.requests[1][-1]["tool_call_id"] == "call-1"
    assert provider.requests[1][-1]["content"] == "tool-output"


async def test_streaming_mode_falls_back_to_complete() -> None:
    provider = ScriptedProvider([LLMResponse(content="fallback")])
    deltas: list[str] = []
    stream_ends: list[bool] = []

    result = await AgentRunner(provider).run(
        make_spec(
            stream=True,
            on_text_delta=deltas.append,
            on_stream_end=stream_ends.append,
        )
    )

    assert result.final_content == "fallback"
    assert deltas == []
    assert stream_ends == []


async def test_streamed_max_turns_emits_framework_terminal_segment() -> None:
    provider = ScriptedStreamingProvider(
        [[ToolCallCompleted(tool_call()), ResponseCompleted(finish_reason="tool_calls")]]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="still working"))
    deltas: list[str] = []
    stream_ends: list[bool] = []

    result = await AgentRunner(provider).run(
        make_spec(
            registry,
            max_turns=1,
            stream=True,
            on_text_delta=deltas.append,
            on_stream_end=stream_ends.append,
        )
    )

    assert result.status == "limit_reached"
    assert deltas == [MAX_TURNS_CONTENT]
    assert stream_ends == [True, False]


async def test_stream_must_end_with_response_completed() -> None:
    provider = ScriptedStreamingProvider([[TextDelta("partial")]])

    result = await AgentRunner(provider).run(make_spec(stream=True))

    assert result.status == "failed"
    assert result.stop_reason == "provider_error"
    assert result.error == "RuntimeError: Provider stream ended without ResponseCompleted"


async def test_cancelled_provider_call_returns_cancelled_result() -> None:
    class CancelledProvider:
        async def complete(self, **request: Any) -> LLMResponse:
            raise asyncio.CancelledError

    result = await AgentRunner(CancelledProvider()).run(make_spec())

    assert result.status == "cancelled"
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], UserMessage)
    assert result.messages[0].id == "input-1"


async def test_cancelled_tool_call_returns_cancelled_result() -> None:
    class CancelledTool(FakeTool):
        async def execute(self, arguments: Mapping[str, Any]) -> Any:
            raise asyncio.CancelledError

    provider = ScriptedProvider(
        [LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls")]
    )
    registry = ToolRegistry()
    registry.register(CancelledTool(name="lookup"))

    result = await AgentRunner(provider).run(make_spec(registry))

    assert result.status == "cancelled"
    assert len(result.messages) == 2
    assert isinstance(result.messages[-1], AssistantMessage)
    assert result.messages[-1].tool_calls[0].id == "call-1"


async def test_context_limit_fails_before_provider_is_called() -> None:
    class OversizedEstimator:
        def estimate(self, **request: Any) -> int:
            return 1_000

    provider = ScriptedProvider([LLMResponse(content="must not be called")])
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=100,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
        ),
        token_estimator=OversizedEstimator(),
    )

    result = await AgentRunner(provider, context_governor=governor).run(make_spec())

    assert result.status == "failed"
    assert result.stop_reason == "context_limit"
    assert provider.requests == []


async def test_governed_tool_result_does_not_replace_full_working_result() -> None:
    class SmallEstimator:
        def estimate(self, **request: Any) -> int:
            return 10

    provider = ScriptedProvider(
        [
            LLMResponse(tool_calls=(tool_call(),), finish_reason="tool_calls"),
            LLMResponse(content="done"),
        ]
    )
    full_result = "HEAD-" + ("x" * 100) + "-TAIL"
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result=full_result))
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=60,
        ),
        token_estimator=SmallEstimator(),
    )

    result = await AgentRunner(provider, context_governor=governor).run(
        make_spec(registry)
    )

    model_tool_result = provider.requests[1].messages[-1]["content"]
    saved_tool_result = next(
        message
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    )
    assert len(model_tool_result) <= 60
    assert "chars omitted" in model_tool_result
    assert saved_tool_result.content == full_result
