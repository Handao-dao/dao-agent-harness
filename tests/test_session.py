from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_harness.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from agent_harness.session import (
    PendingInput,
    PendingInputNotFoundError,
    PendingInputOrderError,
    Session,
    SessionHistoryConflictError,
)
from agent_harness.storage import InMemorySessionStore


def pending(input_id: str, source_id: str, content: str) -> PendingInput:
    return PendingInput(id=input_id, source_message_id=source_id, content=content)


def with_history(*messages, pending_inputs=()) -> Session:
    return Session.from_messages(
        id="session-1",
        messages=messages,
        pending_inputs=pending_inputs,
    )


def test_pending_input_becomes_user_message_with_the_same_identity() -> None:
    item = pending("input-1", "external-1", "original question")

    message = item.to_user_message()

    assert message == UserMessage(
        id="input-1",
        content="original question",
        created_at=item.created_at,
    )


def test_enqueue_deduplicates_a_still_pending_source_message() -> None:
    session = Session(id="session-1")
    original = pending("input-1", "external-1", "question")
    duplicate = pending("input-2", "external-1", "duplicate delivery")

    first = session.enqueue(original)
    second = session.enqueue(duplicate)

    assert first is original
    assert second is original
    assert session.pending_inputs == [original]


def test_edit_pending_does_not_change_committed_history() -> None:
    history = UserMessage(id="old-user", content="old question")
    session = with_history(history)
    session.enqueue(pending("input-1", "external-1", "before"))
    edited_at = datetime(2026, 8, 6, tzinfo=UTC)

    edited = session.edit_pending("input-1", "after", edited_at=edited_at)

    assert edited.id == "input-1"
    assert edited.content == "after"
    assert edited.edited_at == edited_at
    assert session.messages == [history]


def test_edit_rejects_unknown_pending_input() -> None:
    session = Session(id="session-1")

    with pytest.raises(PendingInputNotFoundError):
        session.edit_pending("missing", "new text")


def test_working_history_is_a_list_copy() -> None:
    history = UserMessage(id="old-user", content="old question")
    session = with_history(history)

    working = session.copy_history()
    working.append(AssistantMessage(content="temporary"))

    assert working is not session.messages
    assert session.messages == [history]


def test_commit_appends_only_working_tail_and_removes_consumed_input() -> None:
    history = [
        UserMessage(id="old-user", content="old question"),
        AssistantMessage(id="old-assistant", content="old answer"),
    ]
    first = pending("input-1", "external-1", "new question")
    second = pending("input-2", "external-2", "queued question")
    session = with_history(*history)
    session.enqueue(first)
    session.enqueue(second)

    working = session.copy_history()
    save_cursor = len(working)
    working.extend(
        [
            first.to_user_message(),
            AssistantMessage(
                id="assistant-tool",
                tool_calls=(ToolCall(id="call-1", name="lookup"),),
            ),
            ToolResultMessage(
                id="tool-result",
                tool_call_id="call-1",
                tool_name="lookup",
                content="found",
            ),
            AssistantMessage(id="assistant-final", content="done"),
        ]
    )

    committed = session.commit_working_messages(
        working_messages=working,
        save_cursor=save_cursor,
        base_leaf_id=session.active_leaf_id,
        consumed_input_ids=(first.id,),
    )

    assert session.messages == working
    assert committed == tuple(working[save_cursor:])
    assert session.pending_inputs == [second]


def test_commit_rejects_a_changed_history_prefix_without_mutating_session() -> None:
    history = UserMessage(id="old-user", content="old question")
    item = pending("input-1", "external-1", "new question")
    session = with_history(history, pending_inputs=[item])
    working = [UserMessage(id="different", content="changed"), item.to_user_message()]

    with pytest.raises(SessionHistoryConflictError):
        session.commit_working_messages(
            working_messages=working,
            save_cursor=1,
            base_leaf_id=session.active_leaf_id,
            consumed_input_ids=(item.id,),
        )

    assert session.messages == [history]
    assert session.pending_inputs == [item]


def test_commit_cannot_skip_the_pending_queue_head() -> None:
    first = pending("input-1", "external-1", "first")
    second = pending("input-2", "external-2", "second")
    session = Session(id="session-1", pending_inputs=[first, second])
    working = [second.to_user_message(), AssistantMessage(content="answer")]

    with pytest.raises(PendingInputOrderError):
        session.commit_working_messages(
            working_messages=working,
            save_cursor=0,
            base_leaf_id=session.active_leaf_id,
            consumed_input_ids=(second.id,),
        )

    assert session.messages == []
    assert session.pending_inputs == [first, second]


def test_commit_rejects_work_created_before_pending_input_was_edited() -> None:
    item = pending("input-1", "external-1", "before")
    session = Session(id="session-1", pending_inputs=[item])
    stale_working = [item.to_user_message(), AssistantMessage(content="stale answer")]
    session.edit_pending("input-1", "after")

    with pytest.raises(SessionHistoryConflictError, match="stale"):
        session.commit_working_messages(
            working_messages=stale_working,
            save_cursor=0,
            base_leaf_id=session.active_leaf_id,
            consumed_input_ids=(item.id,),
        )

    assert session.messages == []
    assert session.pending_inputs[0].content == "after"


def test_failed_execution_needs_no_session_mutation() -> None:
    item = pending("input-1", "external-1", "question")
    session = Session(id="session-1", pending_inputs=[item])
    working = session.copy_history()
    working.extend([item.to_user_message(), AssistantMessage(content="partial")])

    assert session.messages == []
    assert session.pending_inputs == [item]


def test_in_memory_store_manages_live_sessions() -> None:
    store = InMemorySessionStore()
    session = store.get_or_create("session-1")
    session.enqueue(pending("input-1", "external-1", "question"))
    store.save(session)

    assert store.get_or_create("session-1") is session
    assert store.delete("session-1") is True
    assert store.delete("session-1") is False
    assert store.get_or_create("session-1") is not session
