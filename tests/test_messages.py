from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from agent_harness.artifacts import ArtifactRef
from agent_harness.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def test_builds_minimal_typed_conversation_messages() -> None:
    call = ToolCall(id="call-1", name="lookup", arguments={"query": "nanobot"})
    user = UserMessage(id="input-1", content="look it up")
    assistant = AssistantMessage(tool_calls=(call,))
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="lookup",
        content="found",
    )

    assert user.id == "input-1"
    assert assistant.content == ""
    assert assistant.tool_calls == (call,)
    assert result.tool_call_id == assistant.tool_calls[0].id
    assert result.is_error is False


def test_messages_are_immutable_records() -> None:
    message = UserMessage(content="hello")

    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


def test_assistant_requires_text_or_tool_calls() -> None:
    with pytest.raises(ValueError, match="text or tool calls"):
        AssistantMessage()


def test_tool_call_copies_argument_mapping() -> None:
    arguments = {"query": "before"}
    call = ToolCall(id="call-1", name="lookup", arguments=arguments)

    arguments["query"] = "after"

    assert call.arguments == {"query": "before"}


def test_tool_result_can_represent_an_error() -> None:
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="lookup",
        content="bad query",
        is_error=True,
    )

    assert result.is_error is True


def test_tool_result_copies_internal_metadata() -> None:
    metadata = {"kind": "skill_instruction", "nested": {"value": 1}}
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="activate_skill",
        content="instructions",
        metadata=metadata,
    )

    metadata["kind"] = "changed"

    assert result.metadata == {
        "kind": "skill_instruction",
        "nested": {"value": 1},
    }


def test_tool_result_normalizes_and_validates_artifact_refs() -> None:
    digest = sha256(b"large result").hexdigest()
    ref = ArtifactRef(
        id=f"art_{digest}",
        media_type="text/plain; charset=utf-8",
        size_bytes=12,
        size_chars=12,
        sha256=digest,
    )
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="lookup",
        content="preview",
        artifact_refs=[ref],  # type: ignore[arg-type]
    )

    assert result.artifact_refs == (ref,)

    with pytest.raises(TypeError, match="ArtifactRef"):
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="lookup",
            content="preview",
            artifact_refs=("bad",),  # type: ignore[arg-type]
        )
