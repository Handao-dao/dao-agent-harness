from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_harness.consolidation import (
    ConsolidationConfig,
    ContextConsolidator,
    ContextSummaryGenerationError,
    ContextSummaryGenerator,
)
from agent_harness.context import ContextBuilder
from agent_harness.memory import InMemoryMemoryStore
from agent_harness.messages import AssistantMessage, UserMessage
from agent_harness.providers import LLMResponse
from agent_harness.runner import AgentRunner
from agent_harness.runtime import AgentRuntime
from agent_harness.runtime_io import RuntimeRequest
from agent_harness.session import MessageEntry, Session
from agent_harness.status_builder import RuntimeStatusBuilder
from agent_harness.storage import InMemorySessionStore
from agent_harness.testing import ScriptedProvider
from agent_harness.tools import ToolRegistry


def summary_json(*, objective: str = "继续任务") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "objective": objective,
            "status": "active",
            "user_constraints": [],
            "established_facts": [],
            "decisions": [],
            "completed_work": ["完成第一轮"],
            "current_work": [],
            "next_steps": [],
            "artifacts": [],
            "unresolved_questions": [],
            "known_issues": [],
            "continuation_note": None,
        },
        ensure_ascii=False,
    )


class MessageCountingEstimator:
    """Deterministic test counter: every non-empty message costs twenty tokens."""

    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        del model, system_prompt, tools
        return max(1, len(messages) * 20)


class BlockingEstimator:
    """Block one selected estimate so tests can observe consolidation coordination."""

    def __init__(self, block_call: int) -> None:
        self.block_call = block_call
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        del model, system_prompt, messages, tools
        self.calls += 1
        if self.calls == self.block_call:
            self.started.set()
            await self.release.wait()
        return 1


class FixedEstimator:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def estimate(self, **_request: Any) -> int:
        return self.tokens


def linear_session() -> Session:
    return Session.from_messages(
        id="session-1",
        messages=(
            UserMessage(id="user-1", content="question-1"),
            AssistantMessage(id="assistant-1", content="answer-1"),
            UserMessage(id="user-2", content="question-2"),
            AssistantMessage(id="assistant-2", content="answer-2"),
        ),
    )


async def test_generator_requests_json_content_without_tools() -> None:
    provider = ScriptedProvider([LLMResponse(content=summary_json())])
    generator = ContextSummaryGenerator(provider, model="fake-model")
    entry = MessageEntry(
        id="entry-1",
        parent_id=None,
        message=UserMessage(id="user-1", content="开始任务"),
    )

    content = await generator.generate(previous_summary=None, entries=(entry,))

    assert content.objective == "继续任务"
    assert provider.requests[0].tools == ()
    payload = json.loads(provider.requests[0].messages[0]["content"])
    assert payload["previous_summary"] is None
    assert payload["new_messages"][0]["entry_id"] == "entry-1"
    assert payload["new_messages"][0]["type"] == "user"


async def test_generator_excludes_expired_runtime_status_entries() -> None:
    provider = ScriptedProvider([LLMResponse(content=summary_json())])
    generator = ContextSummaryGenerator(provider, model="fake-model")
    status = RuntimeStatusBuilder(
        now=lambda: datetime(2026, 8, 14, tzinfo=UTC)
    ).build((UserMessage(content="task"),))
    entries = (
        MessageEntry(id="status-entry", parent_id=None, message=status),
        MessageEntry(
            id="user-entry",
            parent_id="status-entry",
            message=UserMessage(content="retain me"),
        ),
    )

    await generator.generate(previous_summary=None, entries=entries)

    payload = json.loads(provider.requests[0].messages[0]["content"])
    assert [item["type"] for item in payload["new_messages"]] == ["user"]
    assert payload["new_messages"][0]["content"] == "retain me"


async def test_generator_repairs_one_invalid_schema_response() -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content='{"schema_version":1}'),
            LLMResponse(content=summary_json(objective="修复成功")),
        ]
    )
    generator = ContextSummaryGenerator(provider, model="fake-model")
    entry = MessageEntry(parent_id=None, message=UserMessage(content="task"))

    content = await generator.generate(previous_summary=None, entries=(entry,))

    assert content.objective == "修复成功"
    assert len(provider.requests) == 2
    assert "missing required fields" in provider.requests[1].messages[-1]["content"]


async def test_generator_fails_after_the_single_repair_attempt() -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="not json"), LLMResponse(content="still not json")]
    )
    generator = ContextSummaryGenerator(provider, model="fake-model")
    entry = MessageEntry(parent_id=None, message=UserMessage(content="task"))

    with pytest.raises(ContextSummaryGenerationError, match="repair failed"):
        await generator.generate(previous_summary=None, entries=(entry,))

    assert len(provider.requests) == 2


async def test_consolidator_creates_one_durable_summary_at_a_user_turn_boundary(
    tmp_path,
) -> None:
    provider = ScriptedProvider([LLMResponse(content=summary_json())])
    store = InMemorySessionStore()
    session = linear_session()
    store.save(session)
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=MessageCountingEstimator(),
        session_store=store,
        model="fake-model",
        config=ConsolidationConfig(
            context_window_tokens=100,
            max_completion_tokens=10,
            safety_buffer_tokens=0,
        ),
    )

    result = await consolidator.maybe_consolidate(
        session,
        pending_message=UserMessage(content="question-3"),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
    )

    assert len(result.summaries_created) == 1
    assert result.stop_reason == "no_safe_boundary"
    summary = result.summaries_created[0]
    assert summary.covered_through_entry_id == session.entries[1].id
    assert summary.previous_summary_id is None
    assert store.get_or_create(session.id).context_summaries == [summary]
    assert len(session.entries) == 4


async def test_consolidator_enqueues_only_the_newly_covered_memory_range(
    tmp_path,
) -> None:
    provider = ScriptedProvider([LLMResponse(content=summary_json())])
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    session = linear_session()
    session_store.save(session)
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=MessageCountingEstimator(),
        session_store=session_store,
        memory_store=memory_store,
        model="fake-model",
        config=ConsolidationConfig(
            context_window_tokens=100,
            max_completion_tokens=10,
            safety_buffer_tokens=0,
        ),
    )

    result = await consolidator.maybe_consolidate(
        session,
        pending_message=UserMessage(content="question-3"),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
    )

    summary = result.summaries_created[0]
    pending = memory_store.read_pending(after_cursor=0, limit=10)
    assert len(pending) == 1
    assert pending[0].context_summary_id == summary.id
    assert pending[0].source_entry_ids == tuple(entry.id for entry in session.entries[:2])
    assert pending[0].messages == tuple(entry.message for entry in session.entries[:2])


async def test_consolidator_reconciles_a_saved_summary_missing_from_memory_inbox(
    tmp_path,
) -> None:
    provider = ScriptedProvider([LLMResponse(content=summary_json())])
    session_store = InMemorySessionStore()
    session = linear_session()
    session_store.save(session)
    first = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=MessageCountingEstimator(),
        session_store=session_store,
        model="fake-model",
        config=ConsolidationConfig(
            context_window_tokens=100,
            max_completion_tokens=10,
            safety_buffer_tokens=0,
        ),
    )
    result = await first.maybe_consolidate(
        session,
        pending_message=UserMessage(content="question-3"),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
    )
    assert len(result.summaries_created) == 1

    memory_store = InMemoryMemoryStore()
    restored = session_store.get_or_create(session.id)
    reconciler = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=FixedEstimator(1),
        session_store=session_store,
        memory_store=memory_store,
        model="fake-model",
        config=ConsolidationConfig(context_window_tokens=10_000),
    )

    await reconciler.maybe_consolidate(
        restored,
        pending_message=UserMessage(content="question-3"),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
    )

    pending = memory_store.read_pending(after_cursor=0, limit=10)
    assert len(pending) == 1
    assert pending[0].context_summary_id == result.summaries_created[0].id


async def test_proactive_probe_reserves_input_tokens_but_real_input_recheck_does_not(
    tmp_path,
) -> None:
    provider = ScriptedProvider([])
    store = InMemorySessionStore()
    session = Session.from_messages(
        id="session-1",
        messages=(
            UserMessage(content="question-1"),
            AssistantMessage(content="answer-1"),
        ),
    )
    store.save(session)
    config = ConsolidationConfig(
        context_window_tokens=10_000,
        max_completion_tokens=1_000,
        safety_buffer_tokens=0,
    )
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=FixedEstimator(7_500),
        session_store=store,
        model="fake-model",
        config=config,
    )
    kwargs = {
        "context_builder": ContextBuilder(tmp_path),
        "tools": ToolRegistry(),
    }

    proactive = await consolidator.maybe_consolidate(session, **kwargs)
    actual = await consolidator.maybe_consolidate(
        session,
        pending_message=UserMessage(content="question-2"),
        **kwargs,
    )

    assert config.proactive_input_reserve_tokens == 2_048
    assert config.effective_proactive_input_reserve_tokens == 2_048
    assert proactive.stop_reason == "no_safe_boundary"
    assert actual.stop_reason == "within_budget"
    assert provider.requests == []


async def test_runtime_sends_only_uncovered_tail_but_saves_the_full_history(tmp_path) -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content=summary_json()),
            LLMResponse(content="answer-3"),
        ]
    )
    store = InMemorySessionStore()
    session = linear_session()
    store.save(session)
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=MessageCountingEstimator(),
        session_store=store,
        model="fake-model",
        config=ConsolidationConfig(
            context_window_tokens=100,
            max_completion_tokens=10,
            safety_buffer_tokens=0,
        ),
    )
    runtime = AgentRuntime(
        runner=AgentRunner(provider),
        session_store=store,
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
        model="fake-model",
        consolidator=consolidator,
    )

    result = await runtime.submit(RuntimeRequest("session-1", "external-3", "question-3"))

    assert result.status == "completed"
    assert len(provider.requests) == 2
    model_request = provider.requests[1]
    assert [message["content"] for message in model_request.messages] == [
        "question-2",
        "answer-2",
        "question-3",
    ]
    assert "# Archived Conversation Context" in (model_request.system_prompt or "")
    restored = store.get_or_create("session-1")
    assert [message.content for message in restored.messages] == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
        "question-3",
        "answer-3",
    ]
    assert len(restored.context_summaries) == 1


async def test_consolidator_serializes_checks_for_the_same_session(tmp_path) -> None:
    provider = ScriptedProvider([])
    estimator = BlockingEstimator(block_call=1)
    store = InMemorySessionStore()
    session = linear_session()
    store.save(session)
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=estimator,
        session_store=store,
        model="fake-model",
        config=ConsolidationConfig(context_window_tokens=10_000),
    )
    kwargs = {
        "context_builder": ContextBuilder(tmp_path),
        "tools": ToolRegistry(),
    }

    first = asyncio.create_task(consolidator.maybe_consolidate(session, **kwargs))
    await asyncio.wait_for(estimator.started.wait(), timeout=1)
    second = asyncio.create_task(consolidator.maybe_consolidate(session, **kwargs))
    await asyncio.sleep(0)

    assert estimator.calls == 1
    assert not second.done()

    estimator.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert estimator.calls == 2


async def test_prepare_waits_for_post_save_probe_then_rechecks_real_input(tmp_path) -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="answer-1"), LLMResponse(content="answer-2")]
    )
    estimator = BlockingEstimator(block_call=2)
    store = InMemorySessionStore()
    consolidator = ContextConsolidator(
        generator=ContextSummaryGenerator(provider, model="fake-model"),
        token_estimator=estimator,
        session_store=store,
        model="fake-model",
        config=ConsolidationConfig(context_window_tokens=10_000),
    )
    runtime = AgentRuntime(
        runner=AgentRunner(provider),
        session_store=store,
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
        model="fake-model",
        consolidator=consolidator,
    )

    first = await runtime.submit(RuntimeRequest("session-1", "external-1", "question-1"))
    await asyncio.wait_for(estimator.started.wait(), timeout=1)

    second_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-2", "question-2"))
    )
    await asyncio.sleep(0)

    assert first.status == "completed"
    assert len(provider.requests) == 1
    assert not second_task.done()

    estimator.release.set()
    second = await asyncio.wait_for(second_task, timeout=1)
    await runtime.wait_for_background_tasks()

    assert second.status == "completed"
    assert len(provider.requests) == 2
    assert estimator.calls == 4
