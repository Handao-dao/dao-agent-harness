from __future__ import annotations

import json
from hashlib import sha256

from agent_harness.context import ContextBuilder
from agent_harness.messages import AssistantMessage
from agent_harness.providers import LLMResponse
from agent_harness.runner import AgentRunner
from agent_harness.runtime import AgentRuntime
from agent_harness.runtime_io import RuntimeRequest
from agent_harness.session import PendingInput
from agent_harness.storage import JsonlSessionStore
from agent_harness.summary import ContextSummary, ContextSummaryContent
from agent_harness.testing import ScriptedProvider
from agent_harness.tools import ToolRegistry


def log_path(directory, session_id: str):
    return directory / f"{sha256(session_id.encode('utf-8')).hexdigest()}.jsonl"


def commit_pending(session, answer: str) -> None:
    pending = session.pending_inputs[0]
    working = session.copy_history()
    save_cursor = len(working)
    base_leaf_id = session.active_leaf_id
    working.extend([pending.to_user_message(), AssistantMessage(content=answer)])
    session.commit_working_messages(
        working_messages=working,
        save_cursor=save_cursor,
        base_leaf_id=base_leaf_id,
        consumed_input_ids=(pending.id,),
    )


def test_jsonl_replays_pending_edits_turns_and_leaf_changes(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="before")
    )
    store.save(session)
    session.edit_pending("input-1", "after")
    store.save(session)
    commit_pending(session, "answer")
    store.save(session)
    committed_leaf = session.active_leaf_id
    session.checkout(session.entries[0].id)
    store.save(session)

    records = [
        json.loads(line)
        for line in log_path(tmp_path, "session-1").read_text(encoding="utf-8").splitlines()
    ]
    restored = JsonlSessionStore(tmp_path).get_or_create("session-1")

    assert [record["type"] for record in records] == [
        "session",
        "input_enqueued",
        "input_edited",
        "turn_committed",
        "leaf_changed",
    ]
    assert len(records[3]["entries"]) == 2
    assert records[3]["consumed_inputs"] == [{"id": "input-1", "revision": 2}]
    assert restored.pending_inputs == []
    assert restored.active_leaf_id == session.entries[0].id
    assert [message.content for message in restored.messages] == ["after"]
    assert committed_leaf != restored.active_leaf_id
    assert restored.unpersisted_events() == ()


def test_message_enqueued_during_previous_turn_remains_pending_after_replay(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    first = session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="first")
    )
    store.save(session)
    working = session.copy_history()
    working.extend([first.to_user_message(), AssistantMessage(content="first-answer")])

    second = session.enqueue(
        PendingInput(id="input-2", source_message_id="external-2", content="second")
    )
    store.save(session)
    session.commit_working_messages(
        working_messages=working,
        save_cursor=0,
        base_leaf_id=None,
        consumed_input_ids=(first.id,),
    )
    store.save(session)

    restored = JsonlSessionStore(tmp_path).get_or_create("session-1")

    assert [message.content for message in restored.messages] == ["first", "first-answer"]
    assert restored.pending_inputs == [second]


def test_jsonl_replays_context_summary_without_adding_a_message_entry(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="question")
    )
    store.save(session)
    commit_pending(session, "answer")
    store.save(session)
    leaf_before = session.active_leaf_id
    summary = ContextSummary(
        id="summary-1",
        session_id=session.id,
        covered_through_entry_id=session.entries[0].id,
        source_leaf_id=leaf_before or "missing",
        content=ContextSummaryContent(
            schema_version=1,
            objective="continue",
            status="active",
        ),
        tokens_before=8_000,
    )
    session.record_context_summary(summary)
    store.save(session)

    restored = JsonlSessionStore(tmp_path).get_or_create("session-1")
    records = [
        json.loads(line)
        for line in log_path(tmp_path, "session-1").read_text(encoding="utf-8").splitlines()
    ]

    assert records[-1]["type"] == "context_summary_created"
    assert restored.context_summaries == [summary]
    assert restored.active_leaf_id == leaf_before
    assert len(restored.entries) == 2


def test_recovery_truncates_only_an_incomplete_final_record(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="question")
    )
    store.save(session)
    path = log_path(tmp_path, "session-1")
    with path.open("ab") as handle:
        handle.write(b'{"type":"input_edited"')

    restored_store = JsonlSessionStore(tmp_path)
    restored = restored_store.get_or_create("session-1")
    restored.edit_pending("input-1", "recovered")
    restored_store.save(restored)
    reopened = JsonlSessionStore(tmp_path).get_or_create("session-1")

    assert reopened.pending_inputs[0].content == "recovered"
    assert path.read_bytes().endswith(b"\n")


async def test_runtime_commits_to_jsonl_and_reopens_the_active_branch(tmp_path) -> None:
    session_directory = tmp_path / "sessions"
    runtime = AgentRuntime(
        runner=AgentRunner(ScriptedProvider([LLMResponse(content="durable-answer")])),
        session_store=JsonlSessionStore(session_directory),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
        model="fake-model",
    )

    result = await runtime.submit(
        RuntimeRequest("session-1", "external-1", "durable-question")
    )
    restored = JsonlSessionStore(session_directory).get_or_create("session-1")

    assert result.status == "completed"
    assert restored.pending_inputs == []
    assert [message.content for message in restored.messages] == [
        "durable-question",
        "durable-answer",
    ]
    assert restored.active_leaf_id == restored.entries[-1].id
