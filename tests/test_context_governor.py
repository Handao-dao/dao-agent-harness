from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from agent_harness.context_governor import (
    ContextGovernor,
    ContextGovernorConfig,
    ContextWindowExceededError,
)
from agent_harness.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


class FixedEstimator:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def estimate(self, **request: Any) -> int:
        return self.tokens


class MessageCountEstimator:
    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        return len(messages) * 10


class CompactionAwareEstimator:
    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        del model, system_prompt, tools
        compacted = sum(
            "omitted from model context" in str(message.get("content", ""))
            for message in messages
        )
        return 50 if compacted >= 2 else 200


def call(call_id: str, name: str = "lookup") -> ToolCall:
    return ToolCall(id=call_id, name=name)


async def test_repairs_tool_chains_without_mutating_source_messages() -> None:
    orphan = ToolResultMessage(
        tool_call_id="orphan",
        tool_name="lookup",
        content="orphan result",
    )
    user = UserMessage(content="old question")
    assistant = AssistantMessage(tool_calls=(call("a"), call("b")))
    result_a = ToolResultMessage(
        tool_call_id="a",
        tool_name="lookup",
        content="result a",
    )
    current = UserMessage(content="current question")
    source = (orphan, user, assistant, result_a, current)
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
        ),
        token_estimator=FixedEstimator(10),
    )

    governed = await governor.prepare(
        messages=source,
        current_turn_start=4,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert source == (orphan, user, assistant, result_a, current)
    assert orphan not in governed.messages
    backfill = next(
        message
        for message in governed.messages
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "b"
    )
    assert backfill.is_error is True
    assert "unavailable" in backfill.content
    assert governed.report.orphan_results_dropped == 1
    assert governed.report.missing_results_backfilled == 1


async def test_normal_path_truncates_results_without_unneeded_microcompaction() -> None:
    messages: list[Any] = [UserMessage(content="old question")]
    original_results: list[ToolResultMessage] = []
    for index in range(3):
        tool_call = call(f"call-{index}", name="search")
        result = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=f"HEAD-{index}-" + ("x" * 80) + f"-TAIL-{index}",
        )
        messages.extend((AssistantMessage(tool_calls=(tool_call,)), result))
        original_results.append(result)
    current = UserMessage(content="current question")
    messages.append(current)
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=70,
            microcompact_keep_recent=1,
            microcompact_min_chars=20,
            compactable_tool_names=frozenset({"search"}),
        ),
        token_estimator=FixedEstimator(10),
    )

    governed = await governor.prepare(
        messages=messages,
        current_turn_start=len(messages) - 1,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    results = [
        message
        for message in governed.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert all("omitted from model context" not in result.content for result in results)
    assert all("chars omitted" in result.content for result in results)
    assert all(len(result.content) <= 70 for result in results)
    assert governed.report.tool_results_compacted == 0
    assert governed.report.tool_results_truncated == 3
    assert [result.content for result in original_results] == [
        "HEAD-0-" + ("x" * 80) + "-TAIL-0",
        "HEAD-1-" + ("x" * 80) + "-TAIL-1",
        "HEAD-2-" + ("x" * 80) + "-TAIL-2",
    ]


async def test_skill_instructions_are_deduplicated_and_not_truncated() -> None:
    first_call = call("first", name="activate_skill")
    second_call = call("second", name="activate_skill")
    metadata = {
        "kind": "skill_instruction",
        "skill_name": "pdf",
        "content_hash": "same",
    }
    latest_content = "latest-" + ("x" * 100)
    current = UserMessage(content="current")
    messages = (
        AssistantMessage(tool_calls=(first_call,)),
        ToolResultMessage(
            tool_call_id=first_call.id,
            tool_name=first_call.name,
            content="old-" + ("x" * 100),
            metadata=metadata,
        ),
        AssistantMessage(tool_calls=(second_call,)),
        ToolResultMessage(
            tool_call_id=second_call.id,
            tool_name=second_call.name,
            content=latest_content,
            metadata=metadata,
        ),
        current,
    )
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=32,
            compactable_tool_names=frozenset({"activate_skill"}),
        ),
        token_estimator=FixedEstimator(10),
    )

    governed = await governor.prepare(
        messages=messages,
        current_turn_start=4,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    results = [
        message
        for message in governed.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 1
    assert results[0].content == latest_content
    assert governed.report.skill_instructions_deduplicated == 1
    assert governed.report.tool_results_truncated == 0


async def test_pressure_path_microcompacts_history_but_not_active_turn() -> None:
    messages: list[Any] = [UserMessage(content="old question")]
    historical_results: list[ToolResultMessage] = []
    for index in range(3):
        tool_call = call(f"history-{index}", name="search")
        result = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content="history-" + ("x" * 80),
        )
        messages.extend((AssistantMessage(tool_calls=(tool_call,)), result))
        historical_results.append(result)
    current_turn_start = len(messages)
    messages.append(UserMessage(content="current question"))
    active_results: list[ToolResultMessage] = []
    for index in range(2):
        tool_call = call(f"active-{index}", name="search")
        result = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content="active-" + ("y" * 80),
        )
        messages.extend((AssistantMessage(tool_calls=(tool_call,)), result))
        active_results.append(result)
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=100,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=0,
            microcompact_keep_recent=1,
            microcompact_min_chars=20,
            compactable_tool_names=frozenset({"search"}),
        ),
        token_estimator=CompactionAwareEstimator(),
    )

    governed = await governor.prepare(
        messages=messages,
        current_turn_start=current_turn_start,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    result_by_id = {
        message.tool_call_id: message
        for message in governed.messages
        if isinstance(message, ToolResultMessage)
    }
    assert "omitted from model context" in result_by_id["history-0"].content
    assert "omitted from model context" in result_by_id["history-1"].content
    assert result_by_id["history-2"].content == historical_results[2].content
    assert result_by_id["active-0"].content == active_results[0].content
    assert result_by_id["active-1"].content == active_results[1].content
    assert governed.report.tool_results_compacted == 2
    assert governed.report.history_messages_snipped == 0
    assert governed.report.active_turn_messages_snipped == 0


async def test_emergency_snip_keeps_longest_user_bounded_suffix_and_current_turn() -> None:
    messages = (
        UserMessage(content="question-1"),
        AssistantMessage(content="answer-1"),
        UserMessage(content="question-2"),
        AssistantMessage(content="answer-2"),
        UserMessage(content="current"),
    )
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=30,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=0,
        ),
        token_estimator=MessageCountEstimator(),
    )

    governed = await governor.prepare(
        messages=messages,
        current_turn_start=4,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert governed.messages == messages[2:]
    assert governed.messages[-1] is messages[-1]
    assert governed.report.history_messages_snipped == 2
    assert governed.report.estimated_tokens_before == 50
    assert governed.report.estimated_tokens_after == 30


async def test_emergency_snip_can_remove_active_trace_but_keeps_task_anchors() -> None:
    first_call = call("call-1")
    second_call = call("call-2")
    latest_user = UserMessage(content="also preserve this correction")
    third_call = call("call-3")
    messages = (
        UserMessage(content="do the task"),
        AssistantMessage(tool_calls=(first_call,)),
        ToolResultMessage(
            tool_call_id=first_call.id,
            tool_name=first_call.name,
            content="result-1",
        ),
        latest_user,
        AssistantMessage(tool_calls=(second_call,)),
        ToolResultMessage(
            tool_call_id=second_call.id,
            tool_name=second_call.name,
            content="result-2",
        ),
        AssistantMessage(tool_calls=(third_call,)),
        ToolResultMessage(
            tool_call_id=third_call.id,
            tool_name=third_call.name,
            content="result-3",
        ),
    )
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=40,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=0,
        ),
        token_estimator=MessageCountEstimator(),
    )

    governed = await governor.prepare(
        messages=messages,
        current_turn_start=0,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert governed.messages == (
        messages[0],
        latest_user,
        messages[6],
        messages[7],
    )
    assert governed.report.history_messages_snipped == 0
    assert governed.report.active_turn_messages_snipped == 4
    assert governed.report.estimated_tokens_after == 30


async def test_unchanged_normal_context_reuses_the_initial_estimate() -> None:
    class CountingEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, **request: Any) -> int:
            self.calls += 1
            return 10

    estimator = CountingEstimator()
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=100,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
        ),
        token_estimator=estimator,
    )

    governed = await governor.prepare(
        messages=(UserMessage(content="current"),),
        current_turn_start=0,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert governed.report.estimated_tokens_before == 10
    assert governed.report.estimated_tokens_after == 10
    assert estimator.calls == 1


async def test_current_turn_over_budget_raises_before_returning_a_context() -> None:
    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=20,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
        ),
        token_estimator=FixedEstimator(100),
    )

    with pytest.raises(ContextWindowExceededError, match="most recent message block"):
        await governor.prepare(
            messages=(UserMessage(content="current"),),
            current_turn_start=0,
            model="fake-model",
            system_prompt=None,
            tools=(),
        )


async def test_estimator_failure_uses_character_heuristic() -> None:
    class BrokenEstimator:
        def estimate(self, **request: Any) -> int:
            raise RuntimeError("counter unavailable")

    governor = ContextGovernor(
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
        ),
        token_estimator=BrokenEstimator(),
    )

    governed = await governor.prepare(
        messages=(UserMessage(content="current"),),
        current_turn_start=0,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert governed.report.estimation_source == "character_heuristic"
    assert governed.report.estimated_tokens_after is not None


def test_config_rejects_an_unusable_input_budget() -> None:
    with pytest.raises(ValueError, match="no usable input token budget"):
        ContextGovernorConfig(
            context_window_tokens=100,
            max_completion_tokens=100,
        )


def test_config_rejects_impossibly_small_tool_result_limit() -> None:
    with pytest.raises(ValueError, match="zero or at least 32"):
        ContextGovernorConfig(
            context_window_tokens=1_000,
            max_completion_tokens=0,
            safety_buffer_tokens=0,
            max_tool_result_chars=10,
        )
