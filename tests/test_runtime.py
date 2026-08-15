from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_harness.checkpoints import (
    CheckpointCorruptError,
    CheckpointStorageError,
    ContextCheckpoint,
    IncorporatedInput,
    InMemoryCheckpointStore,
)
from agent_harness.context import ContextBuilder
from agent_harness.context_governor import ContextGovernor, ContextGovernorConfig
from agent_harness.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.providers import (
    LLMResponse,
    LLMStreamEvent,
    ResponseCompleted,
    TextDelta,
    ToolCallRequest,
)
from agent_harness.runner import MAX_TURNS_CONTENT, AgentRunner
from agent_harness.runtime import AgentRuntime, ExecutionPhase
from agent_harness.runtime_io import OutputSegmentEnded, OutputTextDelta, RuntimeRequest
from agent_harness.session import Session
from agent_harness.storage import InMemorySessionStore
from agent_harness.summary import ContextSummary, ContextSummaryContent
from agent_harness.testing import FakeTool, ScriptedProvider
from agent_harness.tools import ToolRegistry


def make_runtime(
    tmp_path: Path,
    provider: Any,
    *,
    store: InMemorySessionStore | None = None,
    tools: ToolRegistry | None = None,
    max_turns: int = 5,
    checkpoint_store: InMemoryCheckpointStore | None = None,
    max_injected_inputs_per_run: int = 5,
    max_input_tokens: int | None = None,
    input_token_estimator: Any = None,
) -> tuple[AgentRuntime, InMemorySessionStore]:
    session_store = store or InMemorySessionStore()
    runtime = AgentRuntime(
        runner=AgentRunner(provider),
        session_store=session_store,
        context_builder=ContextBuilder(tmp_path),
        tools=tools or ToolRegistry(),
        model="fake-model",
        max_turns=max_turns,
        checkpoint_store=checkpoint_store,
        max_injected_inputs_per_run=max_injected_inputs_per_run,
        max_input_tokens=max_input_tokens,
        input_token_estimator=input_token_estimator,
    )
    return runtime, session_store


def make_checkpoint(
    pending: Any,
    *,
    phase: str,
    messages: tuple[Any, ...],
    terminal: bool = False,
    base_leaf_id: str | None = None,
    save_cursor: int = 0,
) -> ContextCheckpoint:
    tools_used = tuple(
        call.name
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    )
    return ContextCheckpoint(
        session_id="session-1",
        input_id=pending.id,
        input_revision=pending.revision,
        base_leaf_id=base_leaf_id,
        save_cursor=save_cursor,
        phase=phase,  # type: ignore[arg-type]
        model="fake-model",
        next_model_turn=1,
        messages=messages,
        tools_used=tools_used,
        usage={"prompt_tokens": 4},
        terminal_status="completed" if terminal else None,
        stop_reason="model_stop" if terminal else None,
        final_content=messages[-1].content if terminal else None,
    )


async def test_run_next_returns_idle_without_pending_input(tmp_path) -> None:
    runtime, _store = make_runtime(tmp_path, ScriptedProvider([]))

    result = await runtime.run_next("session-1")

    assert result.status == "idle"
    assert result.input_id is None
    assert result.stop_reason == "no_pending_input"


async def test_submit_persists_completed_typed_conversation(tmp_path) -> None:
    provider = ScriptedProvider([LLMResponse(content="answer", usage={"total_tokens": 4})])
    runtime, store = make_runtime(tmp_path, provider)

    result = await runtime.submit(RuntimeRequest("session-1", "external-1", "question"))

    session = store.get_or_create("session-1")
    assert result.status == "completed"
    assert result.final_content == "answer"
    assert result.input_id == session.messages[0].id
    assert session.pending_inputs == []
    assert isinstance(session.messages[0], UserMessage)
    assert session.messages[0].content == "question"
    assert isinstance(session.messages[1], AssistantMessage)
    assert session.messages[1].content == "answer"
    assert provider.requests[0].messages[0] == {"role": "user", "content": "question"}


async def test_summary_covered_skill_is_reinjected_as_a_legal_tool_pair(tmp_path) -> None:
    provider = ScriptedProvider([LLMResponse(content="continued")])
    runtime, store = make_runtime(tmp_path, provider)
    skill_call = ToolCall(
        id="skill-call",
        name="activate_skill",
        arguments={"name": "pdf"},
    )
    session = Session.from_messages(
        id="session-1",
        messages=(
            UserMessage(content="old task"),
            AssistantMessage(tool_calls=(skill_call,)),
            ToolResultMessage(
                tool_call_id=skill_call.id,
                tool_name=skill_call.name,
                content="<skill name=\"pdf\">instructions</skill>",
                metadata={
                    "kind": "skill_instruction",
                    "skill_name": "pdf",
                    "content_hash": "hash",
                    "retention": "session",
                },
            ),
            AssistantMessage(content="old answer"),
        ),
    )
    assert session.active_leaf_id is not None
    session.record_context_summary(
        ContextSummary(
            session_id=session.id,
            covered_through_entry_id=session.entries[-1].id,
            source_leaf_id=session.active_leaf_id,
            tokens_before=10_000,
            content=ContextSummaryContent(
                schema_version=1,
                objective="continue the PDF task",
                status="active",
            ),
        )
    )
    store.save(session)

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-current", "continue")
    )

    request = provider.requests[0]
    assert result.status == "completed"
    assert [message["role"] for message in request.messages] == [
        "assistant",
        "tool",
        "user",
    ]
    assert request.messages[0]["tool_calls"][0]["id"] == skill_call.id
    assert request.messages[1]["tool_call_id"] == skill_call.id
    assert "instructions" in request.messages[1]["content"]
    assert request.messages[2]["content"] == "continue"
    assert "Archived Conversation Context" in (request.system_prompt or "")


async def test_provider_failure_keeps_pending_and_history_unchanged(tmp_path) -> None:
    runtime, store = make_runtime(
        tmp_path,
        ScriptedProvider([RuntimeError("offline")]),
    )
    queued = runtime.enqueue_input("session-1", "external-1", "question")

    result = await runtime.run_next("session-1")

    session = store.get_or_create("session-1")
    assert result.status == "failed"
    assert result.error == "RuntimeError: offline"
    assert session.messages == []
    assert session.pending_inputs == [queued]


async def test_failed_pending_input_retries_without_resubmission(tmp_path) -> None:
    provider = ScriptedProvider([RuntimeError("offline"), LLMResponse(content="recovered")])
    runtime, store = make_runtime(tmp_path, provider)
    request = RuntimeRequest("session-1", "external-1", "question")

    failed = await runtime.submit(request)
    recovered = await runtime.run_next("session-1")

    session = store.get_or_create("session-1")
    assert failed.status == "failed"
    assert recovered.status == "completed"
    assert recovered.input_id == failed.input_id
    assert session.pending_inputs == []
    assert [message.content for message in session.messages] == ["question", "recovered"]


async def test_runtime_records_and_clears_all_checkpoint_phases(tmp_path) -> None:
    class RecordingCheckpointStore(InMemoryCheckpointStore):
        def __init__(self) -> None:
            super().__init__()
            self.saved: list[ContextCheckpoint] = []

        def save(self, checkpoint: ContextCheckpoint) -> None:
            self.saved.append(checkpoint)
            super().save(checkpoint)

    checkpoints = RecordingCheckpointStore()
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="lookup", arguments={}),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="final"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FakeTool(name="lookup", result="found"))
    runtime, store = make_runtime(
        tmp_path,
        provider,
        tools=registry,
        checkpoint_store=checkpoints,
    )

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "question")
    )

    assert result.status == "completed"
    assert [item.phase for item in checkpoints.saved] == [
        "awaiting_tools",
        "tools_completed",
        "final_response",
    ]
    assert all(item.messages[0].id == result.input_id for item in checkpoints.saved)
    assert checkpoints.load("session-1") is None
    assert store.get_or_create("session-1").pending_inputs == []


async def test_checkpoint_tracks_followup_incorporated_after_tools(tmp_path) -> None:
    class RecordingCheckpointStore(InMemoryCheckpointStore):
        def __init__(self) -> None:
            super().__init__()
            self.saved: list[ContextCheckpoint] = []

        def save(self, value: ContextCheckpoint) -> None:
            self.saved.append(value)
            super().save(value)

    checkpoints = RecordingCheckpointStore()
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="lookup", arguments={}),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    tools = ToolRegistry()
    tools.register(FakeTool(name="lookup", result="found"))
    runtime, _store = make_runtime(
        tmp_path,
        provider,
        tools=tools,
        checkpoint_store=checkpoints,
    )
    first = runtime.enqueue_input("session-1", "external-1", "first")
    second = runtime.enqueue_input("session-1", "external-2", "second")

    result = await runtime.run_next("session-1")

    assert result.status == "completed"
    assert [item.phase for item in checkpoints.saved] == [
        "awaiting_tools",
        "tools_completed",
        "final_response",
    ]
    assert checkpoints.saved[1].incorporated_inputs == (
        IncorporatedInput(id=first.id, revision=1),
    )
    assert checkpoints.saved[2].incorporated_inputs == (
        IncorporatedInput(id=first.id, revision=1),
        IncorporatedInput(id=second.id, revision=1),
    )


async def test_checkpoint_conflicts_when_incorporated_followup_is_edited(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=checkpoints,
    )
    first = runtime.enqueue_input("session-1", "external-1", "first")
    second = runtime.enqueue_input("session-1", "external-2", "second")
    checkpoints.save(
        ContextCheckpoint(
            session_id="session-1",
            input_id=first.id,
            input_revision=first.revision,
            base_leaf_id=None,
            save_cursor=0,
            phase="final_response",
            model="fake-model",
            next_model_turn=2,
            messages=(
                first.to_user_message(),
                AssistantMessage(content="candidate"),
                second.to_user_message(),
                AssistantMessage(content="final"),
            ),
            incorporated_inputs=(
                IncorporatedInput(id=first.id, revision=first.revision),
                IncorporatedInput(id=second.id, revision=second.revision),
            ),
            terminal_status="completed",
            stop_reason="model_stop",
            final_content="final",
        )
    )
    session = store.get_or_create("session-1")
    session.edit_pending(second.id, "edited")
    store.save(session)

    result = await runtime.run_next("session-1")

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_conflict"
    assert provider.requests == []
    assert len(store.get_or_create("session-1").pending_inputs) == 2


async def test_awaiting_tools_checkpoint_resumes_without_replaying_tool(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="recovered safely")])
    registry = ToolRegistry()
    tool = FakeTool(name="lookup", result="must not execute")
    registry.register(tool)
    runtime, store = make_runtime(
        tmp_path,
        provider,
        tools=registry,
        checkpoint_store=checkpoints,
    )
    pending = runtime.enqueue_input("session-1", "external-1", "question")
    assistant = AssistantMessage(
        id="assistant-tools",
        tool_calls=(ToolCall(id="call-1", name="lookup"),),
    )
    checkpoints.save(
        make_checkpoint(
            pending,
            phase="awaiting_tools",
            messages=(pending.to_user_message(), assistant),
        )
    )

    result = await runtime.run_next("session-1")

    assert result.status == "completed"
    assert tool.calls == []
    request_tool_result = provider.requests[0].messages[-1]
    assert request_tool_result["role"] == "tool"
    assert request_tool_result["tool_call_id"] == "call-1"
    assert "completion state is unknown" in request_tool_result["content"]
    messages = store.get_or_create("session-1").messages
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].is_error is True
    assert checkpoints.load("session-1") is None


async def test_tools_completed_checkpoint_reuses_results_before_model(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="used saved result")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=checkpoints,
    )
    pending = runtime.enqueue_input("session-1", "external-1", "question")
    assistant = AssistantMessage(
        id="assistant-tools",
        tool_calls=(ToolCall(id="call-1", name="lookup"),),
    )
    tool_result = ToolResultMessage(
        id="tool-result",
        tool_call_id="call-1",
        tool_name="lookup",
        content="saved output",
    )
    checkpoints.save(
        make_checkpoint(
            pending,
            phase="tools_completed",
            messages=(pending.to_user_message(), assistant, tool_result),
        )
    )

    result = await runtime.run_next("session-1")

    assert result.status == "completed"
    assert provider.requests[0].messages[-1]["content"] == "saved output"
    assert [message.content for message in store.get_or_create("session-1").messages] == [
        "question",
        "",
        "saved output",
        "used saved result",
    ]


async def test_final_response_checkpoint_skips_provider_and_commits(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=checkpoints,
    )
    pending = runtime.enqueue_input("session-1", "external-1", "question")
    final = AssistantMessage(id="assistant-final", content="already generated")
    checkpoints.save(
        make_checkpoint(
            pending,
            phase="final_response",
            messages=(pending.to_user_message(), final),
            terminal=True,
        )
    )

    result = await runtime.run_next("session-1")

    assert result.status == "completed"
    assert result.final_content == "already generated"
    assert provider.requests == []
    assert checkpoints.load("session-1") is None
    assert [message.content for message in store.get_or_create("session-1").messages] == [
        "question",
        "already generated",
    ]


async def test_checkpoint_revision_change_discards_old_progress(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="new answer")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=checkpoints,
    )
    pending = runtime.enqueue_input("session-1", "external-1", "old question")
    checkpoints.save(
        make_checkpoint(
            pending,
            phase="final_response",
            messages=(
                pending.to_user_message(),
                AssistantMessage(content="stale answer"),
            ),
            terminal=True,
        )
    )
    session = store.get_or_create("session-1")
    session.edit_pending(pending.id, "edited question")
    store.save(session)

    result = await runtime.run_next("session-1")

    assert result.final_content == "new answer"
    assert provider.requests[0].messages[-1] == {
        "role": "user",
        "content": "edited question",
    }


async def test_checkpoint_leaf_conflict_fails_without_provider_call(tmp_path) -> None:
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="must not run")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=checkpoints,
    )
    pending = runtime.enqueue_input("session-1", "external-1", "question")
    checkpoints.save(
        make_checkpoint(
            pending,
            phase="final_response",
            messages=(pending.to_user_message(), AssistantMessage(content="answer")),
            terminal=True,
            base_leaf_id="different-leaf",
        )
    )

    result = await runtime.run_next("session-1")

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_conflict"
    assert provider.requests == []
    assert store.get_or_create("session-1").pending_inputs == [pending]


async def test_corrupt_checkpoint_fails_closed(tmp_path) -> None:
    class CorruptCheckpointStore(InMemoryCheckpointStore):
        def load(self, session_id: str) -> ContextCheckpoint | None:
            raise CheckpointCorruptError("invalid checkpoint bytes")

    provider = ScriptedProvider([LLMResponse(content="must not run")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=CorruptCheckpointStore(),
    )
    pending = runtime.enqueue_input("session-1", "external-1", "question")

    result = await runtime.run_next("session-1")

    assert result.status == "failed"
    assert result.stop_reason == "checkpoint_corrupt"
    assert provider.requests == []
    assert store.get_or_create("session-1").pending_inputs == [pending]


async def test_checkpoint_delete_failure_after_commit_is_nonfatal(tmp_path) -> None:
    class DeleteFailingStore(InMemoryCheckpointStore):
        def delete(self, session_id: str) -> bool:
            raise CheckpointStorageError("cannot delete stale checkpoint")

    provider = ScriptedProvider([LLMResponse(content="answer")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        checkpoint_store=DeleteFailingStore(),
    )

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "question")
    )

    assert result.status == "completed"
    assert store.get_or_create("session-1").pending_inputs == []


async def test_final_checkpoint_retries_save_without_repeating_provider(tmp_path) -> None:
    class FailFirstTurnSaveStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.save_calls = 0

        def save(self, session: Any) -> None:
            self.save_calls += 1
            if self.save_calls == 2:
                raise OSError("simulated Session SAVE failure")
            super().save(session)

    session_store = FailFirstTurnSaveStore()
    checkpoints = InMemoryCheckpointStore()
    provider = ScriptedProvider([LLMResponse(content="generated once")])
    runtime, _store = make_runtime(
        tmp_path,
        provider,
        store=session_store,
        checkpoint_store=checkpoints,
    )

    with pytest.raises(OSError, match="Session SAVE failure"):
        await runtime.submit(
            RuntimeRequest("session-1", "external-1", "question")
        )

    interrupted = session_store.get_or_create("session-1")
    assert len(interrupted.pending_inputs) == 1
    assert interrupted.messages == []
    assert checkpoints.load("session-1").phase == "final_response"  # type: ignore[union-attr]

    recovered = await runtime.run_next("session-1")

    assert recovered.status == "completed"
    assert recovered.final_content == "generated once"
    assert len(provider.requests) == 1
    assert session_store.get_or_create("session-1").pending_inputs == []
    assert checkpoints.load("session-1") is None


async def test_two_submitted_turns_use_runtime_session_history(tmp_path) -> None:
    provider = ScriptedProvider(
        [LLMResponse(content="answer-1"), LLMResponse(content="answer-2")]
    )
    runtime, store = make_runtime(tmp_path, provider)

    await runtime.submit(RuntimeRequest("session-1", "external-1", "question-1"))
    await runtime.submit(RuntimeRequest("session-1", "external-2", "question-2"))

    assert provider.requests[1].messages == (
        {"role": "user", "content": "question-1"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "question-2"},
    )
    assert [message.content for message in store.get_or_create("session-1").messages] == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
    ]


async def test_limit_reached_is_committed_with_terminal_message(tmp_path) -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="lookup", arguments={"query": "x"}),
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    tools = ToolRegistry()
    tools.register(FakeTool(name="lookup", result="still working"))
    runtime, store = make_runtime(tmp_path, provider, tools=tools, max_turns=1)

    result = await runtime.submit(RuntimeRequest("session-1", "external-1", "question"))

    session = store.get_or_create("session-1")
    assert result.status == "limit_reached"
    assert result.final_content == MAX_TURNS_CONTENT
    assert session.pending_inputs == []
    assert isinstance(session.messages[-1], AssistantMessage)
    assert session.messages[-1].content == MAX_TURNS_CONTENT


class BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return LLMResponse(content=f"answer-{self.calls}")


class StepBlockingProvider:
    def __init__(self) -> None:
        self.started = [asyncio.Event() for _ in range(4)]
        self.release = [asyncio.Event() for _ in range(4)]
        self.requests: list[tuple[Mapping[str, Any], ...]] = []

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(tuple(dict(message) for message in messages))
        self.started[index].set()
        await self.release[index].wait()
        return LLMResponse(content=f"answer-{index + 1}", usage={"total_tokens": 3})


async def test_input_enqueued_during_run_is_injected_at_candidate_response(tmp_path) -> None:
    provider = BlockingProvider()
    runtime, store = make_runtime(tmp_path, provider)
    first = runtime.enqueue_input("session-1", "external-1", "first")
    task = asyncio.create_task(runtime.run_next("session-1"))
    await provider.started.wait()

    second = runtime.enqueue_input("session-1", "external-2", "second")
    provider.release.set()
    result = await task

    session = store.get_or_create("session-1")
    assert result.input_id == first.id
    assert session.pending_inputs == []
    assert session.messages[0].id == first.id
    assert session.messages[2].id == second.id
    assert provider.calls == 2
    assert result.final_content == "answer-2"


async def test_pause_revises_initial_input_and_restarts_from_committed_history(tmp_path) -> None:
    provider = StepBlockingProvider()
    runtime, store = make_runtime(tmp_path, provider)
    queued = runtime.enqueue_input("session-1", "external-1", "wrong question")
    running = asyncio.create_task(runtime.run_next("session-1"))
    await provider.started[0].wait()

    paused = await runtime.pause_for_revision("session-1")
    original_result = await running

    assert paused.status == "paused"
    assert original_result == paused
    assert paused.revision_target_input_id == queued.id
    assert paused.discarded_message_count == 1
    assert store.get_or_create("session-1").messages == []

    edited = runtime.revise_paused_input("session-1", queued.id, "correct question")
    provider.release[1].set()
    restarted = await runtime.restart_pending("session-1")

    assert edited.revision == 2
    assert restarted.status == "completed"
    assert provider.requests[1][-1] == {"role": "user", "content": "correct question"}
    assert [message.content for message in store.get_or_create("session-1").messages] == [
        "correct question",
        "answer-2",
    ]


async def test_pause_revises_seen_injection_and_preserves_earlier_model_progress(
    tmp_path,
) -> None:
    provider = StepBlockingProvider()
    runtime, store = make_runtime(tmp_path, provider)
    first = runtime.enqueue_input("session-1", "external-1", "first")
    running = asyncio.create_task(runtime.run_next("session-1"))
    await provider.started[0].wait()
    second = runtime.enqueue_input("session-1", "external-2", "old supplement")
    provider.release[0].set()
    await provider.started[1].wait()

    paused = await runtime.pause_for_revision("session-1")

    assert (await running).status == "paused"
    assert paused.revision_target_input_id == second.id
    assert paused.discarded_message_count == 1
    runtime.revise_paused_input("session-1", second.id, "new supplement")
    provider.release[2].set()
    restarted = await runtime.restart_pending("session-1")

    session = store.get_or_create("session-1")
    assert restarted.status == "completed"
    assert provider.requests[2][-1] == {
        "role": "user",
        "content": "new supplement",
    }
    assert [message.id for message in session.messages if isinstance(message, UserMessage)] == [
        first.id,
        second.id,
    ]
    assert [message.content for message in session.messages] == [
        "first",
        "answer-1",
        "new supplement",
        "answer-3",
    ]


async def test_pause_targets_latest_unseen_supplement_and_blocks_waiting_submit(
    tmp_path,
) -> None:
    provider = StepBlockingProvider()
    runtime, store = make_runtime(tmp_path, provider)
    first_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-1", "first"))
    )
    await provider.started[0].wait()
    second_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-2", "supplement"))
    )
    await asyncio.sleep(0)
    second = store.get_or_create("session-1").pending_inputs[-1]

    paused = await runtime.pause_for_revision("session-1")
    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert paused.revision_target_input_id == second.id
    assert first_result.status == "paused"
    assert second_result.status == "paused"
    assert provider.requests == [({"role": "user", "content": "first"},)]


async def test_pause_marks_in_flight_tool_side_effect_as_uncertain(tmp_path) -> None:
    class BlockingTool:
        name = "mutate"
        description = "A blocking mutating tool"
        parameters: Mapping[str, Any] = {"type": "object", "properties": {}}
        execution_mode = "sequential"
        timeout_s = None

        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def execute(self, arguments: Mapping[str, Any]) -> str:
            self.started.set()
            await asyncio.Event().wait()
            return "done"

    tool = BlockingTool()
    tools = ToolRegistry()
    tools.register(tool)
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(ToolCallRequest(id="call-1", name="mutate", arguments={}),),
                finish_reason="tool_calls",
            )
        ]
    )
    runtime, _store = make_runtime(tmp_path, provider, tools=tools)
    runtime.enqueue_input("session-1", "external-1", "change something")
    running = asyncio.create_task(runtime.run_next("session-1"))
    await tool.started.wait()

    paused = await runtime.pause_for_revision("session-1")
    await running

    assert paused.side_effect_status == "uncertain"
    assert paused.discarded_tool_call_ids == ("call-1",)


async def test_only_selected_paused_input_can_be_revised(tmp_path) -> None:
    runtime, store = make_runtime(tmp_path, ScriptedProvider([]))
    first = runtime.enqueue_input("session-1", "external-1", "first")
    second = runtime.enqueue_input("session-1", "external-2", "second")

    paused = await runtime.pause_for_revision("session-1")

    assert paused.revision_target_input_id == second.id
    with pytest.raises(RuntimeError, match="selected revision target"):
        runtime.revise_paused_input("session-1", first.id, "changed first")
    assert store.get_or_create("session-1").pending_inputs == [first, second]


async def test_same_session_runs_are_serialized(tmp_path) -> None:
    provider = BlockingProvider()
    runtime, store = make_runtime(tmp_path, provider)

    first_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-1", "first"))
    )
    await provider.started.wait()
    second_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-2", "second"))
    )
    await asyncio.sleep(0)
    assert provider.calls == 1

    provider.release.set()
    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result.status == "completed"
    assert second_result.status == "injected"
    assert second_result.stop_reason == "injected_into_active_run"
    assert provider.calls == 2
    assert store.get_or_create("session-1").pending_inputs == []


async def test_sixth_concurrent_followup_starts_a_new_runner(tmp_path) -> None:
    provider = BlockingProvider()
    runtime, store = make_runtime(tmp_path, provider, max_injected_inputs_per_run=5)
    first_task = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-0", "question-0"))
    )
    await provider.started.wait()
    followup_tasks = [
        asyncio.create_task(
            runtime.submit(
                RuntimeRequest(
                    "session-1",
                    f"external-{index}",
                    f"question-{index}",
                )
            )
        )
        for index in range(1, 7)
    ]

    provider.release.set()
    first_result, *followup_results = await asyncio.gather(first_task, *followup_tasks)

    assert first_result.status == "completed"
    assert [result.status for result in followup_results].count("injected") == 5
    assert [result.status for result in followup_results].count("completed") == 1
    assert provider.calls == 3
    assert store.get_or_create("session-1").pending_inputs == []


async def test_injection_quota_leaves_sixth_followup_for_a_new_runner(tmp_path) -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(content="candidate-1"),
            LLMResponse(content="answer-to-batch"),
            LLMResponse(content="answer-to-remainder"),
        ]
    )
    runtime, store = make_runtime(
        tmp_path,
        provider,
        max_injected_inputs_per_run=5,
    )
    queued = [
        runtime.enqueue_input("session-1", f"external-{index}", f"question-{index}")
        for index in range(7)
    ]

    first = await runtime.run_next("session-1")

    assert first.status == "completed"
    assert first.has_pending_continuation is True
    assert first.remaining_pending_count == 1
    assert store.get_or_create("session-1").pending_inputs == [queued[-1]]
    assert provider.requests[0].messages[-1] == {
        "role": "user",
        "content": "question-0",
    }
    assert provider.requests[1].messages[-1] == {
        "role": "user",
        "content": "question-1\n\nquestion-2\n\nquestion-3\n\nquestion-4\n\nquestion-5",
    }

    second = await runtime.run_next("session-1")

    assert second.status == "completed"
    assert second.has_pending_continuation is False
    assert second.remaining_pending_count == 0
    assert store.get_or_create("session-1").pending_inputs == []
    assert provider.requests[2].messages[-1] == {
        "role": "user",
        "content": "question-6",
    }


async def test_last_model_turn_leaves_followup_pending(tmp_path) -> None:
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(id="call-1", name="lookup", arguments={}),
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    tools = ToolRegistry()
    tools.register(FakeTool(name="lookup", result="done"))
    runtime, store = make_runtime(tmp_path, provider, tools=tools, max_turns=1)
    first = runtime.enqueue_input("session-1", "external-1", "first")
    second = runtime.enqueue_input("session-1", "external-2", "second")

    result = await runtime.run_next("session-1")

    assert result.status == "limit_reached"
    assert result.has_pending_continuation is True
    assert result.remaining_pending_count == 1
    assert store.get_or_create("session-1").pending_inputs == [second]
    assert [
        message.id
        for message in store.get_or_create("session-1").messages
        if isinstance(message, UserMessage)
    ] == [first.id]


class ConcurrentProvider:
    def __init__(self) -> None:
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, **request: Any) -> LLMResponse:
        self.calls += 1
        if self.calls == 2:
            self.both_started.set()
        await self.release.wait()
        return LLMResponse(content="done")


async def test_different_sessions_can_run_concurrently(tmp_path) -> None:
    provider = ConcurrentProvider()
    runtime, _store = make_runtime(tmp_path, provider)
    runtime.enqueue_input("session-1", "external-1", "first")
    runtime.enqueue_input("session-2", "external-2", "second")

    first = asyncio.create_task(runtime.run_next("session-1"))
    second = asyncio.create_task(runtime.run_next("session-2"))
    await asyncio.wait_for(provider.both_started.wait(), timeout=1)
    provider.release.set()
    results = await asyncio.gather(first, second)

    assert [result.status for result in results] == ["completed", "completed"]


def test_runtime_has_only_the_agreed_execution_phases() -> None:
    assert tuple(ExecutionPhase) == (
        ExecutionPhase.LOAD,
        ExecutionPhase.PREPARE,
        ExecutionPhase.RUN,
        ExecutionPhase.SAVE,
        ExecutionPhase.RESPOND,
        ExecutionPhase.DONE,
    )


class RuntimeStreamingProvider:
    def __init__(self, turns: Sequence[Sequence[LLMStreamEvent]]) -> None:
        self._turns = deque(turns)

    async def complete(self, **request: Any) -> LLMResponse:
        raise AssertionError("Runtime should request the streaming Provider path")

    async def stream(self, **request: Any) -> AsyncIterator[LLMStreamEvent]:
        for event in self._turns.popleft():
            yield event


async def test_runtime_maps_runner_stream_to_typed_input_scoped_events(tmp_path) -> None:
    provider = RuntimeStreamingProvider(
        [[TextDelta("答"), TextDelta("案"), ResponseCompleted()]]
    )
    runtime, store = make_runtime(tmp_path, provider)
    events: list[OutputTextDelta | OutputSegmentEnded] = []

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "问题"),
        on_stream=events.append,
    )

    assert result.status == "completed"
    assert [type(event) for event in events] == [
        OutputTextDelta,
        OutputTextDelta,
        OutputSegmentEnded,
    ]
    assert all(event.input_id == result.input_id for event in events)
    assert [event.segment_index for event in events] == [0, 0, 0]
    assert events[-1] == OutputSegmentEnded(
        input_id=result.input_id,
        segment_index=0,
        resuming=False,
    )
    assert store.get_or_create("session-1").pending_inputs == []


def test_runtime_request_rejects_blank_external_fields() -> None:
    for values in (
        ("", "external-1", "question"),
        ("session-1", "", "question"),
        ("session-1", "external-1", ""),
    ):
        try:
            RuntimeRequest(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"RuntimeRequest accepted invalid values: {values!r}")


async def test_context_limit_keeps_runtime_pending_input_and_skips_provider(tmp_path) -> None:
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
    store = InMemorySessionStore()
    runtime = AgentRuntime(
        runner=AgentRunner(provider, context_governor=governor),
        session_store=store,
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
        model="fake-model",
    )

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "question")
    )

    session = store.get_or_create("session-1")
    assert result.status == "failed"
    assert result.stop_reason == "context_limit"
    assert len(session.pending_inputs) == 1
    assert session.messages == []
    assert provider.requests == []


class ContentLengthEstimator:
    def estimate(self, **request: Any) -> int:
        content = request["messages"][-1]["content"]
        if not isinstance(content, str):
            raise TypeError("test estimator requires text content")
        return len(content)


async def test_oversized_input_is_retained_for_revision_without_calling_provider(
    tmp_path,
) -> None:
    provider = ScriptedProvider([LLMResponse(content="recovered")])
    runtime, store = make_runtime(
        tmp_path,
        provider,
        max_input_tokens=5,
        input_token_estimator=ContentLengthEstimator(),
    )

    rejected = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "too long input")
    )

    session = store.get_or_create("session-1")
    assert rejected.status == "failed"
    assert rejected.stop_reason == "input_too_large"
    assert rejected.revision_target_input_id == session.pending_inputs[0].id
    assert provider.requests == []
    assert session.messages == []

    paused = await runtime.pause_for_revision("session-1")
    runtime.revise_paused_input(
        "session-1",
        paused.revision_target_input_id or "",
        "short",
    )
    recovered = await runtime.restart_pending("session-1")

    assert recovered.status == "completed"
    assert store.get_or_create("session-1").pending_inputs == []
    assert len(provider.requests) == 1


async def test_oversized_followup_is_not_injected_and_remains_pending(tmp_path) -> None:
    provider = BlockingProvider()
    runtime, store = make_runtime(
        tmp_path,
        provider,
        max_input_tokens=5,
        input_token_estimator=ContentLengthEstimator(),
    )
    running = asyncio.create_task(
        runtime.submit(RuntimeRequest("session-1", "external-1", "short"))
    )
    await provider.started.wait()
    oversized = runtime.enqueue_input(
        "session-1",
        "external-2",
        "oversized followup",
    )

    provider.release.set()
    first = await running

    assert first.status == "completed"
    assert first.has_pending_continuation is True
    assert store.get_or_create("session-1").pending_inputs == [oversized]
    assert provider.calls == 1

    rejected = await runtime.run_next("session-1")

    assert rejected.status == "failed"
    assert rejected.stop_reason == "input_too_large"
    assert store.get_or_create("session-1").pending_inputs == [oversized]
    assert provider.calls == 1
