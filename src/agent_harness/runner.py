"""Minimal typed model-and-tool Runner extracted from nanobot's behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Literal

from agent_harness.checkpoints import RunnerCheckpoint
from agent_harness.context import ContextBuilder
from agent_harness.context_governor import (
    ContextGovernanceError,
    ContextGovernanceReport,
    ContextGovernor,
    ContextWindowExceededError,
)
from agent_harness.injection import (
    MessageInjectionBatch,
    MessageInjectionHandler,
    MessageInjectionPoint,
)
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.providers.base import (
    LLMProvider,
    LLMResponse,
    ResponseCompleted,
    TextDelta,
    ToolCallCompleted,
    ToolCallRequest,
)
from agent_harness.skills import deduplicate_skill_messages
from agent_harness.status_builder import RuntimeStatusBuilder
from agent_harness.tools.registry import ToolRegistry

RunStatus = Literal["completed", "failed", "cancelled", "limit_reached"]
TextDeltaHandler = Callable[[str], Awaitable[None] | None]
StreamEndHandler = Callable[[bool], Awaitable[None] | None]
CheckpointHandler = Callable[[RunnerCheckpoint], Awaitable[None] | None]

EMPTY_RESPONSE_CONTENT = "The model returned an empty response."
MAX_TURNS_CONTENT = "The agent stopped after reaching the model turn limit."


@dataclass(frozen=True, slots=True)
class AgentRunSpec:
    """Configuration for one minimal Runner invocation."""

    initial_messages: Sequence[AgentMessage]
    tools: ToolRegistry
    model: str
    system_prompt: str | None = None
    max_turns: int = 20
    stream: bool = False
    on_text_delta: TextDeltaHandler | None = None
    on_stream_end: StreamEndHandler | None = None
    model_message_start: int = 0
    context_prefix_messages: Sequence[AgentMessage] = ()
    current_turn_start: int | None = None
    checkpoint_callback: CheckpointHandler | None = None
    model_turn_offset: int = 0
    initial_tools_used: Sequence[str] = ()
    initial_usage: Mapping[str, int] = field(default_factory=dict)
    injection_callback: MessageInjectionHandler | None = None
    max_injected_inputs_per_run: int = 5
    initial_injected_input_count: int = 0


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Complete typed working conversation and its terminal status."""

    final_content: str | None
    messages: tuple[AgentMessage, ...]
    status: RunStatus
    stop_reason: str
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    model_seen_user_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedModelContext:
    messages: tuple[AgentMessage, ...]
    governance_report: ContextGovernanceReport | None = None


class AgentRunner:
    """Execute a Provider/tool loop without Session or Channel responsibilities."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        context_governor: ContextGovernor | None = None,
        status_builder: RuntimeStatusBuilder | None = None,
    ):
        self._provider = provider
        self._context_governor = context_governor
        self._status_builder = status_builder

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        if spec.max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if (
            type(spec.model_turn_offset) is not int
            or spec.model_turn_offset < 0
            or spec.model_turn_offset > spec.max_turns
        ):
            raise ValueError("model_turn_offset must be between zero and max_turns")
        if (
            type(spec.max_injected_inputs_per_run) is not int
            or spec.max_injected_inputs_per_run < 0
        ):
            raise ValueError("max_injected_inputs_per_run must be non-negative")
        if (
            type(spec.initial_injected_input_count) is not int
            or spec.initial_injected_input_count < 0
            or spec.initial_injected_input_count > spec.max_injected_inputs_per_run
        ):
            raise ValueError(
                "initial_injected_input_count must be between zero and "
                "max_injected_inputs_per_run"
            )
        if (
            type(spec.model_message_start) is not int
            or spec.model_message_start < 0
            or spec.model_message_start > len(spec.initial_messages)
        ):
            raise ValueError("model_message_start must index initial_messages")
        current_turn_start = (
            len(spec.initial_messages) - 1
            if spec.current_turn_start is None
            else spec.current_turn_start
        )
        if (
            type(current_turn_start) is not int
            or current_turn_start < spec.model_message_start
            or current_turn_start >= len(spec.initial_messages)
        ):
            raise ValueError(
                "current_turn_start must index initial_messages at or after model_message_start"
            )
        if not isinstance(spec.initial_messages[current_turn_start], UserMessage):
            raise ValueError("current_turn_start must point to a UserMessage")

        messages = list(spec.initial_messages)
        if any(
            not isinstance(name, str) or not name.strip()
            for name in spec.initial_tools_used
        ):
            raise ValueError("initial_tools_used must contain non-empty names")
        tools_used = list(spec.initial_tools_used)
        usage = self._validated_initial_usage(spec.initial_usage)
        injected_input_count = spec.initial_injected_input_count
        uses_streaming = spec.stream and callable(getattr(self._provider, "stream", None))

        for turn_index in range(spec.model_turn_offset, spec.max_turns):
            try:
                prepared = await self._prepare_model_context(
                    spec,
                    messages,
                    current_turn_start=current_turn_start,
                )
                model_context = prepared.messages
                if self._status_builder is not None:
                    status_message = self._status_builder.build(
                        messages[current_turn_start:],
                        governance_report=prepared.governance_report,
                    )
                    messages.append(status_message)
                    model_context = (*model_context, status_message)
                model_messages = ContextBuilder.build_messages(model_context)
                response = await self._request_model(spec, model_messages)
            except asyncio.CancelledError:
                return self._result(
                    messages=messages,
                    status="cancelled",
                    stop_reason="cancelled",
                    tools_used=tools_used,
                    usage=usage,
                )
            except ContextWindowExceededError as exc:
                return self._result(
                    messages=messages,
                    status="failed",
                    stop_reason="context_limit",
                    tools_used=tools_used,
                    usage=usage,
                    error=str(exc),
                )
            except ContextGovernanceError as exc:
                return self._result(
                    messages=messages,
                    status="failed",
                    stop_reason="context_governance_error",
                    tools_used=tools_used,
                    usage=usage,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception as exc:
                return self._result(
                    messages=messages,
                    status="failed",
                    stop_reason="provider_error",
                    tools_used=tools_used,
                    usage=usage,
                    error=f"{type(exc).__name__}: {exc}",
                )

            self._accumulate_usage(usage, response.usage)
            if response.finish_reason == "error":
                return self._result(
                    messages=messages,
                    status="failed",
                    stop_reason="provider_error",
                    tools_used=tools_used,
                    usage=usage,
                    error=response.content or "Provider returned an error response",
                )

            if response.should_execute_tools:
                if uses_streaming:
                    await self._notify_stream_end(spec, resuming=True)
                assistant_message = self._assistant_message(
                    response,
                    include_tool_calls=True,
                )
                messages.append(assistant_message)
                tools_used.extend(call.name for call in assistant_message.tool_calls)
                checkpoint_failure = await self._save_checkpoint(
                    spec,
                    phase="awaiting_tools",
                    next_model_turn=turn_index + 1,
                    messages=messages,
                    tools_used=tools_used,
                    usage=usage,
                )
                if checkpoint_failure is not None:
                    return self._checkpoint_failure_result(
                        messages=messages,
                        tools_used=tools_used,
                        usage=usage,
                        failure=checkpoint_failure,
                    )
                try:
                    tool_results = await self._execute_tools(
                        spec.tools, assistant_message.tool_calls
                    )
                except asyncio.CancelledError:
                    return self._result(
                        messages=messages,
                        status="cancelled",
                        stop_reason="cancelled",
                        tools_used=tools_used,
                        usage=usage,
                    )
                messages.extend(tool_results)
                checkpoint_failure = await self._save_checkpoint(
                    spec,
                    phase="tools_completed",
                    next_model_turn=turn_index + 1,
                    messages=messages,
                    tools_used=tools_used,
                    usage=usage,
                )
                if checkpoint_failure is not None:
                    return self._checkpoint_failure_result(
                        messages=messages,
                        tools_used=tools_used,
                        usage=usage,
                        failure=checkpoint_failure,
                    )
                try:
                    injection = await self._collect_injections(
                        spec,
                        MessageInjectionPoint.AFTER_TOOLS,
                        injected_input_count=injected_input_count,
                        has_next_model_turn=turn_index + 1 < spec.max_turns,
                    )
                except asyncio.CancelledError:
                    return self._result(
                        messages=messages,
                        status="cancelled",
                        stop_reason="cancelled",
                        tools_used=tools_used,
                        usage=usage,
                    )
                except Exception as exc:
                    return self._result(
                        messages=messages,
                        status="failed",
                        stop_reason="message_injection_error",
                        tools_used=tools_used,
                        usage=usage,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if injection.messages:
                    messages.extend(injection.messages)
                    injected_input_count += len(injection.messages)
                continue

            final_content = response.content or EMPTY_RESPONSE_CONTENT
            final_message = AssistantMessage(content=final_content)
            try:
                injection = await self._collect_injections(
                    spec,
                    MessageInjectionPoint.AFTER_CANDIDATE_RESPONSE,
                    injected_input_count=injected_input_count,
                    has_next_model_turn=turn_index + 1 < spec.max_turns,
                )
            except asyncio.CancelledError:
                return self._result(
                    messages=messages,
                    status="cancelled",
                    stop_reason="cancelled",
                    tools_used=tools_used,
                    usage=usage,
                )
            except Exception as exc:
                return self._result(
                    messages=messages,
                    status="failed",
                    stop_reason="message_injection_error",
                    tools_used=tools_used,
                    usage=usage,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if injection.messages:
                messages.append(final_message)
                messages.extend(injection.messages)
                injected_input_count += len(injection.messages)
                if uses_streaming:
                    await self._notify_stream_end(spec, resuming=True)
                continue

            messages.append(final_message)
            checkpoint_failure = await self._save_checkpoint(
                spec,
                phase="final_response",
                next_model_turn=turn_index + 1,
                messages=messages,
                tools_used=tools_used,
                usage=usage,
                terminal_status="completed",
                stop_reason="model_stop",
                final_content=final_content,
            )
            if checkpoint_failure is not None:
                return self._checkpoint_failure_result(
                    messages=messages,
                    tools_used=tools_used,
                    usage=usage,
                    failure=checkpoint_failure,
                )
            if uses_streaming:
                await self._notify_stream_end(spec, resuming=False)
            return self._result(
                messages=messages,
                status="completed",
                stop_reason="model_stop",
                tools_used=tools_used,
                usage=usage,
                final_content=final_content,
            )

        messages.append(AssistantMessage(content=MAX_TURNS_CONTENT))
        checkpoint_failure = await self._save_checkpoint(
            spec,
            phase="final_response",
            next_model_turn=spec.max_turns,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            terminal_status="limit_reached",
            stop_reason="max_turns",
            final_content=MAX_TURNS_CONTENT,
        )
        if checkpoint_failure is not None:
            return self._checkpoint_failure_result(
                messages=messages,
                tools_used=tools_used,
                usage=usage,
                failure=checkpoint_failure,
            )
        if uses_streaming:
            await self._notify_text_delta(spec, MAX_TURNS_CONTENT)
            await self._notify_stream_end(spec, resuming=False)
        return self._result(
            messages=messages,
            status="limit_reached",
            stop_reason="max_turns",
            tools_used=tools_used,
            usage=usage,
            final_content=MAX_TURNS_CONTENT,
        )

    @staticmethod
    async def _collect_injections(
        spec: AgentRunSpec,
        point: MessageInjectionPoint,
        *,
        injected_input_count: int,
        has_next_model_turn: bool,
    ) -> MessageInjectionBatch:
        remaining = spec.max_injected_inputs_per_run - injected_input_count
        if spec.injection_callback is None or not has_next_model_turn or remaining <= 0:
            return MessageInjectionBatch.empty(point)
        value = spec.injection_callback(point, remaining)
        if isawaitable(value):
            value = await value
        if not isinstance(value, MessageInjectionBatch):
            raise TypeError("injection_callback must return MessageInjectionBatch")
        if value.point is not point:
            raise ValueError("injection_callback returned a batch for the wrong point")
        if len(value.messages) > remaining:
            raise ValueError("injection_callback exceeded the remaining input limit")
        return value

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        request = {
            "model": spec.model,
            "system_prompt": spec.system_prompt,
            "messages": messages,
            "tools": spec.tools.definitions(),
        }
        stream_method = getattr(self._provider, "stream", None)
        if not spec.stream or not callable(stream_method):
            return await self._provider.complete(**request)

        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        completed: ResponseCompleted | None = None

        async for event in stream_method(**request):
            if isinstance(event, TextDelta):
                content_parts.append(event.text)
                await self._notify_text_delta(spec, event.text)
            elif isinstance(event, ToolCallCompleted):
                tool_calls.append(event.tool_call)
            elif isinstance(event, ResponseCompleted):
                if completed is not None:
                    raise RuntimeError("Provider stream emitted ResponseCompleted more than once")
                completed = event
            else:
                raise TypeError(f"Unsupported provider stream event: {type(event).__name__}")

        if completed is None:
            raise RuntimeError("Provider stream ended without ResponseCompleted")
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tuple(tool_calls),
            finish_reason=completed.finish_reason,
            usage=completed.usage,
        )

    async def _prepare_model_context(
        self,
        spec: AgentRunSpec,
        messages: Sequence[AgentMessage],
        *,
        current_turn_start: int,
    ) -> _PreparedModelContext:
        context_prefix = tuple(spec.context_prefix_messages)
        if any(
            not isinstance(
                message,
                (UserMessage, AssistantMessage, ToolResultMessage, RuntimeStatusMessage),
            )
            for message in context_prefix
        ):
            raise ContextGovernanceError(
                "context_prefix_messages must contain AgentMessage values"
            )
        visible_messages = (*context_prefix, *messages[spec.model_message_start :])
        visible_turn_start = (
            len(context_prefix)
            + current_turn_start
            - spec.model_message_start
        )
        if self._context_governor is None:
            deduplicated, _removed = deduplicate_skill_messages(visible_messages)
            return _PreparedModelContext(messages=deduplicated)

        governed = await self._context_governor.prepare(
            messages=visible_messages,
            current_turn_start=visible_turn_start,
            model=spec.model,
            system_prompt=spec.system_prompt,
            tools=spec.tools.definitions(),
        )
        return _PreparedModelContext(
            messages=governed.messages,
            governance_report=governed.report,
        )

    @staticmethod
    async def _notify_text_delta(spec: AgentRunSpec, text: str) -> None:
        if spec.on_text_delta is None:
            return
        callback_result = spec.on_text_delta(text)
        if isawaitable(callback_result):
            await callback_result

    @staticmethod
    async def _notify_stream_end(spec: AgentRunSpec, *, resuming: bool) -> None:
        if spec.on_stream_end is None:
            return
        callback_result = spec.on_stream_end(resuming)
        if isawaitable(callback_result):
            await callback_result

    @staticmethod
    async def _save_checkpoint(
        spec: AgentRunSpec,
        *,
        phase: Literal["awaiting_tools", "tools_completed", "final_response"],
        next_model_turn: int,
        messages: Sequence[AgentMessage],
        tools_used: Sequence[str],
        usage: Mapping[str, int],
        terminal_status: Literal["completed", "limit_reached"] | None = None,
        stop_reason: str | None = None,
        final_content: str | None = None,
    ) -> BaseException | None:
        if spec.checkpoint_callback is None:
            return None
        try:
            checkpoint = RunnerCheckpoint(
                phase=phase,
                model=spec.model,
                next_model_turn=next_model_turn,
                messages=tuple(messages),
                tools_used=tuple(tools_used),
                usage=usage,
                terminal_status=terminal_status,
                stop_reason=stop_reason,
                final_content=final_content,
            )
            callback_result = spec.checkpoint_callback(checkpoint)
            if isawaitable(callback_result):
                await callback_result
        except asyncio.CancelledError as exc:
            return exc
        except Exception as exc:
            return exc
        return None

    @staticmethod
    def _assistant_message(
        response: LLMResponse,
        *,
        include_tool_calls: bool,
    ) -> AssistantMessage:
        tool_calls = ()
        if include_tool_calls:
            tool_calls = tuple(
                ToolCall(id=call.id, name=call.name, arguments=call.arguments)
                for call in response.tool_calls
            )
        return AssistantMessage(content=response.content or "", tool_calls=tool_calls)

    @classmethod
    async def _execute_tools(
        cls,
        registry: ToolRegistry,
        calls: Sequence[ToolCall],
    ) -> list[ToolResultMessage]:
        """Execute contiguous parallel-safe groups around sequential barriers."""

        results: list[ToolResultMessage] = []
        for batch in cls._partition_tool_batches(registry, calls):
            if len(batch) == 1:
                results.append(await cls._execute_tool(registry, batch[0]))
                continue

            tasks = [
                asyncio.create_task(cls._execute_tool(registry, call))
                for call in batch
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    results.append(await completed)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return results

    @staticmethod
    def _partition_tool_batches(
        registry: ToolRegistry,
        calls: Sequence[ToolCall],
    ) -> list[list[ToolCall]]:
        batches: list[list[ToolCall]] = []
        parallel_batch: list[ToolCall] = []
        for call in calls:
            tool = registry.get(call.name)
            if tool is not None and tool.execution_mode == "parallel_safe":
                parallel_batch.append(call)
                continue
            if parallel_batch:
                batches.append(parallel_batch)
                parallel_batch = []
            batches.append([call])
        if parallel_batch:
            batches.append(parallel_batch)
        return batches

    @staticmethod
    async def _execute_tool(
        registry: ToolRegistry,
        call: ToolCall,
    ) -> ToolResultMessage:
        result = await registry.execute_call(call)
        return ToolResultMessage(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            content=result.content,
            is_error=result.is_error,
            artifact_refs=result.artifact_refs,
            metadata=result.metadata,
        )

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: Mapping[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + int(value)

    @staticmethod
    def _validated_initial_usage(usage: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(usage, Mapping):
            raise TypeError("initial_usage must be a mapping")
        copied: dict[str, int] = {}
        for key, value in usage.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("initial_usage keys must be non-empty text")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("initial_usage values must be non-negative integers")
            copied[key] = value
        return copied

    @classmethod
    def _checkpoint_failure_result(
        cls,
        *,
        messages: list[AgentMessage],
        tools_used: list[str],
        usage: Mapping[str, int],
        failure: BaseException,
    ) -> AgentRunResult:
        if isinstance(failure, asyncio.CancelledError):
            return cls._result(
                messages=messages,
                status="cancelled",
                stop_reason="cancelled",
                tools_used=tools_used,
                usage=usage,
            )
        return cls._result(
            messages=messages,
            status="failed",
            stop_reason="checkpoint_error",
            tools_used=tools_used,
            usage=usage,
            error=f"{type(failure).__name__}: {failure}",
        )

    @staticmethod
    def _result(
        *,
        messages: list[AgentMessage],
        status: RunStatus,
        stop_reason: str,
        tools_used: list[str],
        usage: Mapping[str, int],
        final_content: str | None = None,
        error: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            final_content=final_content,
            messages=tuple(messages),
            status=status,
            stop_reason=stop_reason,
            tools_used=tuple(tools_used),
            usage=dict(usage),
            error=error,
            model_seen_user_message_ids=(
                tuple(message.id for message in messages if isinstance(message, UserMessage))
                if status in {"completed", "limit_reached"}
                else ()
            ),
        )


__all__ = [
    "AgentRunResult",
    "AgentRunSpec",
    "AgentRunner",
    "CheckpointHandler",
    "EMPTY_RESPONSE_CONTENT",
    "MAX_TURNS_CONTENT",
    "MessageInjectionHandler",
    "RunStatus",
    "StreamEndHandler",
    "TextDeltaHandler",
]
