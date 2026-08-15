from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from agent_harness.context import ContextBuilder
from agent_harness.context_governor import (
    ContextGovernanceReport,
    ContextGovernor,
    ContextGovernorConfig,
)
from agent_harness.messages import (
    AssistantMessage,
    RuntimeStatusMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.providers import LLMResponse
from agent_harness.runner import AgentRunner, AgentRunSpec
from agent_harness.status import DefaultRuntimeStatusRenderer
from agent_harness.status_builder import RuntimeStatusBuilder
from agent_harness.testing import ScriptedProvider
from agent_harness.tools import ToolRegistry

FIXED_TIME = datetime(
    2026,
    8,
    14,
    16,
    30,
    tzinfo=timezone(timedelta(hours=8), name="Asia/Shanghai"),
)


def status_builder() -> RuntimeStatusBuilder:
    return RuntimeStatusBuilder(now=lambda: FIXED_TIME)


def status_message() -> RuntimeStatusMessage:
    return status_builder().build((UserMessage(content="task"),))


def test_builder_renders_only_actionable_runtime_facts() -> None:
    call_a = ToolCall(id="call-a", name="search", arguments={"query": "same"})
    call_b = ToolCall(id="call-b", name="search", arguments={"query": "same"})
    messages = (
        UserMessage(content="task"),
        AssistantMessage(tool_calls=(call_a,)),
        ToolResultMessage(
            tool_call_id=call_a.id,
            tool_name=call_a.name,
            content="failed once",
            is_error=True,
        ),
        AssistantMessage(tool_calls=(call_b,)),
        ToolResultMessage(
            tool_call_id=call_b.id,
            tool_name=call_b.name,
            content="failed twice",
            is_error=True,
        ),
    )
    report = ContextGovernanceReport(
        estimated_tokens_before=200,
        estimated_tokens_after=90,
        tool_results_compacted=2,
        runtime_statuses_dropped=1,
    )

    message = status_builder().build(messages, governance_report=report)

    assert message.snapshot.environment.current_time == FIXED_TIME
    assert message.snapshot.environment.timezone == "Asia/Shanghai"
    assert [item.kind for item in message.snapshot.tool_anomalies] == [
        "repeated_identical_call",
        "repeated_failure",
    ]
    assert message.snapshot.context_visibility is not None
    assert message.snapshot.context_visibility.mode == "pressure"
    assert '<dao_runtime_status version="1"' in message.content
    assert 'kind="repeated_identical_call" tool="search" occurrences="2"' in message.content
    assert "model_turn" not in message.content
    assert "injected" not in message.content
    assert message.render_profile == DefaultRuntimeStatusRenderer.profile
    assert message.display is False


def test_context_builder_maps_status_to_the_standard_user_role() -> None:
    status = status_message()

    projected = ContextBuilder.build_messages((UserMessage(content="task"), status))

    assert projected == (
        {
            "role": "user",
            "content": f"task\n\n{status.content}",
        },
    )


async def test_runner_appends_one_status_before_each_real_model_decision() -> None:
    provider = ScriptedProvider([LLMResponse(content="done")])
    runner = AgentRunner(provider, status_builder=status_builder())

    result = await runner.run(
        AgentRunSpec(
            initial_messages=(UserMessage(id="input-1", content="task"),),
            tools=ToolRegistry(),
            model="fake-model",
        )
    )

    statuses = [
        message for message in result.messages if isinstance(message, RuntimeStatusMessage)
    ]
    assert len(statuses) == 1
    assert result.messages[-2] is statuses[0]
    assert provider.requests[0].messages[-1]["role"] == "user"
    assert statuses[0].content in provider.requests[0].messages[-1]["content"]


class MessageCountEstimator:
    def estimate(self, **request: object) -> int:
        messages = request["messages"]
        assert isinstance(messages, tuple)
        has_status = any(
            "<dao_runtime_status" in str(message.get("content", ""))
            for message in messages
        )
        return 40 if has_status else 30


async def test_context_governor_drops_old_statuses_only_under_pressure() -> None:
    old_status = status_message()
    current = UserMessage(content="current")
    messages = (
        UserMessage(content="old"),
        old_status,
        AssistantMessage(content="answer"),
        current,
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
        current_turn_start=3,
        model="fake-model",
        system_prompt=None,
        tools=(),
    )

    assert old_status not in governed.messages
    assert governed.report.runtime_statuses_dropped == 1
    assert governed.report.history_messages_snipped == 0


def test_builder_requires_timezone_aware_current_time() -> None:
    builder = RuntimeStatusBuilder(now=lambda: datetime(2026, 8, 14))

    try:
        builder.build((UserMessage(content="task"),))
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("Expected a timezone validation failure")


def test_utc_timezone_is_preserved() -> None:
    message = RuntimeStatusBuilder(
        now=lambda: datetime(2026, 8, 14, tzinfo=UTC)
    ).build((UserMessage(content="task"),))

    assert message.snapshot.environment.timezone == "UTC"
