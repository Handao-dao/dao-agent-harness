from __future__ import annotations

import pytest

from agent_harness.messages import AssistantMessage
from agent_harness.session import (
    ContextSummaryCreated,
    PendingInput,
    Session,
    SessionEventConflictError,
    SessionHistoryConflictError,
)
from agent_harness.summary import ContextSummary, ContextSummaryContent


def commit_turn(
    session: Session,
    *,
    input_id: str,
    source_id: str,
    question: str,
    answer: str,
) -> str:
    pending = session.enqueue(
        PendingInput(id=input_id, source_message_id=source_id, content=question)
    )
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
    assert session.active_leaf_id is not None
    return session.active_leaf_id


def test_checkout_past_entry_creates_a_new_inherited_branch() -> None:
    session = Session(id="session-1")
    first_leaf = commit_turn(
        session,
        input_id="input-1",
        source_id="external-1",
        question="question-1",
        answer="answer-1",
    )
    original_leaf = commit_turn(
        session,
        input_id="input-2",
        source_id="external-2",
        question="question-2",
        answer="answer-2",
    )

    session.checkout(first_leaf)
    alternate_leaf = commit_turn(
        session,
        input_id="input-3",
        source_id="external-3",
        question="alternate-question",
        answer="alternate-answer",
    )

    assert [message.content for message in session.messages] == [
        "question-1",
        "answer-1",
        "alternate-question",
        "alternate-answer",
    ]
    assert len(session.entries) == 6
    assert alternate_leaf != original_leaf

    session.checkout(original_leaf)

    assert [message.content for message in session.messages] == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
    ]


def test_turn_is_recorded_as_one_event_containing_the_ordered_message_segment() -> None:
    session = Session(id="session-1")
    session.enqueue(PendingInput(id="input-1", source_message_id="external-1", content="q"))
    session.mark_events_persisted(session.unpersisted_events())
    working = session.copy_history()
    working.extend(
        [
            session.pending_inputs[0].to_user_message(),
            AssistantMessage(id="assistant-1", content="a"),
        ]
    )

    session.commit_working_messages(
        working_messages=working,
        save_cursor=0,
        base_leaf_id=None,
        consumed_input_ids=("input-1",),
    )

    events = session.unpersisted_events()
    assert len(events) == 1
    event = events[0]
    assert tuple(entry.message for entry in event.entries) == tuple(working)
    assert event.entries[0].parent_id is None
    assert event.entries[1].parent_id == event.entries[0].id
    assert event.new_leaf_id == event.entries[-1].id


def test_edit_increments_revision_and_stale_work_cannot_commit() -> None:
    session = Session(id="session-1")
    pending = session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="before")
    )
    stale_working = [pending.to_user_message(), AssistantMessage(content="stale")]

    edited = session.edit_pending("input-1", "after")

    assert edited.revision == 2
    assert session.pending_inputs[0].revision == 2
    with pytest.raises(SessionHistoryConflictError, match="stale"):
        session.commit_working_messages(
            working_messages=stale_working,
            save_cursor=0,
            base_leaf_id=None,
            consumed_input_ids=("input-1",),
        )


def test_context_summary_is_a_tree_external_event_and_does_not_move_the_leaf() -> None:
    session = Session(id="session-1")
    source_leaf = commit_turn(
        session,
        input_id="input-1",
        source_id="external-1",
        question="question",
        answer="answer",
    )
    session.mark_events_persisted(session.unpersisted_events())
    summary = ContextSummary(
        id="summary-1",
        session_id=session.id,
        covered_through_entry_id=session.entries[0].id,
        source_leaf_id=source_leaf,
        content=ContextSummaryContent(
            schema_version=1,
            objective="continue the task",
            status="active",
        ),
        tokens_before=10_000,
    )

    session.record_context_summary(summary)

    assert session.context_summaries == [summary]
    assert session.active_leaf_id == source_leaf
    assert len(session.entries) == 2
    events = session.unpersisted_events()
    assert len(events) == 1
    assert isinstance(events[0], ContextSummaryCreated)
    assert events[0].summary == summary


def test_context_summary_must_cover_an_entry_on_its_source_branch() -> None:
    session = Session(id="session-1")
    source_leaf = commit_turn(
        session,
        input_id="input-1",
        source_id="external-1",
        question="question",
        answer="answer",
    )
    summary = ContextSummary(
        session_id=session.id,
        covered_through_entry_id="entry-not-on-branch",
        source_leaf_id=source_leaf,
        content=ContextSummaryContent(
            schema_version=1,
            objective=None,
            status="unclear",
        ),
        tokens_before=10_000,
    )

    with pytest.raises(SessionEventConflictError, match="coverage boundary"):
        session.record_context_summary(summary)
