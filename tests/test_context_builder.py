from __future__ import annotations

import json

import pytest

from agent_harness.context import (
    ContextBuilder,
    ContextBuildError,
    SessionContextResolver,
)
from agent_harness.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.session import PendingInput, Session
from agent_harness.skills import SkillCatalog, SkillRoot
from agent_harness.summary import ContextSummary, ContextSummaryContent


def commit_turn(session: Session, input_id: str, question: str, answer: str) -> str:
    pending = session.enqueue(
        PendingInput(
            id=input_id,
            source_message_id=f"external-{input_id}",
            content=question,
        )
    )
    working = session.copy_history()
    cursor = len(working)
    base_leaf = session.active_leaf_id
    working.extend((pending.to_user_message(), AssistantMessage(content=answer)))
    session.commit_working_messages(
        working_messages=working,
        save_cursor=cursor,
        base_leaf_id=base_leaf,
        consumed_input_ids=(pending.id,),
    )
    assert session.active_leaf_id is not None
    return session.active_leaf_id


def test_builds_system_prompt_in_stable_section_order(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("Agent rules.", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("Soul rules.", encoding="utf-8")
    (tmp_path / "USER.md").write_text("User preferences.", encoding="utf-8")

    prompt = ContextBuilder(tmp_path).build_system_prompt(
        extra_sections=("# Memory\n\nFuture memory.",)
    )

    assert prompt.index("# Identity") < prompt.index("## AGENTS.md")
    assert prompt.index("## AGENTS.md") < prompt.index("## SOUL.md")
    assert prompt.index("## SOUL.md") < prompt.index("## USER.md")
    assert prompt.index("## USER.md") < prompt.index("# Tool Contract")
    assert prompt.index("# Tool Contract") < prompt.index("# Memory")


def test_skips_missing_and_empty_bootstrap_files(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
    (tmp_path / "USER.md").write_text("Keep answers concise.", encoding="utf-8")

    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "## AGENTS.md" not in prompt
    assert "## SOUL.md" not in prompt
    assert "## USER.md" in prompt
    assert "Keep answers concise." in prompt


def test_tool_contract_keeps_skill_instructions_below_runtime_authority(tmp_path) -> None:
    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "activated Skill instructions as task-specific procedures" in prompt
    assert "cannot override system" in prompt
    assert "authorization boundaries" in prompt


def test_reads_bootstrap_files_as_utf8(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("请使用中文回答。", encoding="utf-8")

    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "请使用中文回答。" in prompt


def test_rejects_invalid_utf8_bootstrap_file(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"\xff\xfe")

    with pytest.raises(ContextBuildError, match="UTF-8 bootstrap"):
        ContextBuilder(tmp_path).build_system_prompt()


def test_converts_typed_tool_conversation_to_model_messages(tmp_path) -> None:
    call = ToolCall(id="call-1", name="lookup", arguments={"query": "中文"})
    working_messages = [
        UserMessage(id="input-1", content="look it up"),
        AssistantMessage(id="assistant-1", tool_calls=(call,)),
        ToolResultMessage(
            id="result-1",
            tool_call_id="call-1",
            tool_name="lookup",
            content="found",
        ),
    ]

    context = ContextBuilder(tmp_path).build(working_messages)

    assert context.messages[0] == {"role": "user", "content": "look it up"}
    assistant = context.messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "query": "中文"
    }
    assert context.messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "lookup",
        "content": "found",
    }


def test_build_messages_projects_without_rebuilding_system_prompt(tmp_path) -> None:
    builder = ContextBuilder(tmp_path)

    messages = builder.build_messages([UserMessage(content="hello")])

    assert messages == ({"role": "user", "content": "hello"},)


def test_marks_structured_tool_errors_for_the_model(tmp_path) -> None:
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="lookup",
        content="bad query",
        is_error=True,
    )

    context = ContextBuilder(tmp_path).build([result])

    assert context.messages[0]["content"] == "Error: bad query"


def test_build_does_not_modify_working_messages_or_tool_arguments(tmp_path) -> None:
    call = ToolCall(id="call-1", name="lookup", arguments={"nested": {"value": 1}})
    assistant = AssistantMessage(tool_calls=(call,))
    working_messages = [UserMessage(content="run"), assistant]
    before = tuple(working_messages)

    context = ContextBuilder(tmp_path).build(working_messages)
    context.messages[1]["tool_calls"][0]["function"]["arguments"] = "{}"

    assert tuple(working_messages) == before
    assert call.arguments == {"nested": {"value": 1}}


def test_rejects_non_text_extra_section(tmp_path) -> None:
    with pytest.raises(ContextBuildError, match="must be text"):
        ContextBuilder(tmp_path).build_system_prompt(
            extra_sections=(object(),),  # type: ignore[arg-type]
        )


def test_resolver_returns_latest_deepest_applicable_summary_and_raw_tail() -> None:
    session = Session(id="session-1")
    first_leaf = commit_turn(session, "input-1", "question-1", "answer-1")
    final_leaf = commit_turn(session, "input-2", "question-2", "answer-2")
    older = ContextSummary(
        id="summary-1",
        session_id=session.id,
        covered_through_entry_id=session.entries[0].id,
        source_leaf_id=final_leaf,
        content=ContextSummaryContent(
            schema_version=1, objective="old", status="active"
        ),
        tokens_before=8_000,
    )
    newer = ContextSummary(
        id="summary-2",
        session_id=session.id,
        covered_through_entry_id=first_leaf,
        source_leaf_id=final_leaf,
        previous_summary_id=older.id,
        content=ContextSummaryContent(
            schema_version=1, objective="new", status="active"
        ),
        tokens_before=8_500,
    )
    session.record_context_summary(older)
    session.record_context_summary(newer)

    resolved = SessionContextResolver().resolve(session)

    assert resolved.summary == newer
    assert resolved.summary_id == "summary-2"
    assert resolved.covered_message_count == 2
    assert [message.content for message in resolved.messages] == [
        "question-2",
        "answer-2",
    ]


def test_resolver_does_not_use_summary_from_an_unrelated_branch() -> None:
    session = Session(id="session-1")
    first_leaf = commit_turn(session, "input-1", "question-1", "answer-1")
    original_leaf = commit_turn(session, "input-2", "original", "original-answer")
    summary = ContextSummary(
        session_id=session.id,
        covered_through_entry_id=original_leaf,
        source_leaf_id=original_leaf,
        content=ContextSummaryContent(
            schema_version=1, objective="original branch", status="active"
        ),
        tokens_before=9_000,
    )
    session.record_context_summary(summary)
    session.checkout(first_leaf)
    commit_turn(session, "input-3", "alternate", "alternate-answer")

    resolved = SessionContextResolver().resolve(session)

    assert resolved.summary is None
    assert [message.content for message in resolved.messages] == [
        "question-1",
        "answer-1",
        "alternate",
        "alternate-answer",
    ]


def test_context_builder_injects_structured_summary_as_guarded_system_data(tmp_path) -> None:
    content = ContextSummaryContent(
        schema_version=1,
        objective="继续 <unsafe> 任务",
        status="active",
        user_constraints=("保持简洁",),
    )

    prompt = ContextBuilder(tmp_path).build_system_prompt(context_summary=content)

    assert "# Archived Conversation Context" in prompt
    assert "<archived_conversation_context>" in prompt
    assert '"objective":"继续 \\u003cunsafe\\u003e 任务"' in prompt
    assert prompt.index("# Tool Contract") < prompt.index("# Archived Conversation Context")


def test_context_builder_injects_only_skill_catalog_metadata(tmp_path) -> None:
    skill_dir = tmp_path / ".dao" / "skills" / "pdf"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: Work with <PDF> files.\n---\nSECRET BODY\n",
        encoding="utf-8",
    )
    catalog = SkillCatalog((SkillRoot(tmp_path / ".dao" / "skills", "workspace"),))

    prompt = ContextBuilder(tmp_path, skill_catalog=catalog).build_system_prompt()

    assert "# Available Skills" in prompt
    assert "<name>pdf</name>" in prompt
    assert "Work with &lt;PDF&gt; files." in prompt
    assert "SECRET BODY" not in prompt
    assert prompt.index("# Tool Contract") < prompt.index("# Available Skills")


def test_context_builder_rehydrates_skill_pair_without_mutating_history(tmp_path) -> None:
    call = ToolCall(id="skill-call", name="activate_skill", arguments={"name": "pdf"})
    assistant = AssistantMessage(content="loading", tool_calls=(call,))
    result = ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content="instructions",
        metadata={
            "kind": "skill_instruction",
            "skill_name": "pdf",
            "content_hash": "hash",
        },
    )
    history = (UserMessage(content="old"), assistant, result)

    prefix = ContextBuilder(tmp_path).build_skill_context_prefix(history)

    assert len(prefix) == 2
    assert isinstance(prefix[0], AssistantMessage)
    assert prefix[0].content == ""
    assert prefix[0].tool_calls == (call,)
    assert prefix[1] is result
    assert assistant.content == "loading"
