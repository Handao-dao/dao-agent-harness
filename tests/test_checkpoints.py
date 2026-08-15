from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from agent_harness.artifacts import ArtifactRef
from agent_harness.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCodec,
    CheckpointCorruptError,
    ContextCheckpoint,
    IncorporatedInput,
    InMemoryCheckpointStore,
    JsonFileCheckpointStore,
    RunnerCheckpoint,
    UnsupportedCheckpointVersionError,
)
from agent_harness.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def fixed_time() -> datetime:
    return datetime(2026, 8, 10, 12, tzinfo=UTC)


def tool_assistant() -> AssistantMessage:
    return AssistantMessage(
        id="assistant-tools",
        content="checking",
        created_at=fixed_time(),
        tool_calls=(
            ToolCall(id="call-a", name="first"),
            ToolCall(id="call-b", name="second"),
        ),
    )


def artifact_ref() -> ArtifactRef:
    content = "complete externalized result"
    digest = sha256(content.encode("utf-8")).hexdigest()
    return ArtifactRef(
        id=f"art_{digest}",
        media_type="text/plain; charset=utf-8",
        size_bytes=len(content.encode("utf-8")),
        size_chars=len(content),
        sha256=digest,
    )


def checkpoint(
    phase: str = "awaiting_tools",
    *,
    content: str = "answer",
) -> ContextCheckpoint:
    user = UserMessage(id="input-1", content="continue", created_at=fixed_time())
    if phase == "awaiting_tools":
        messages = (user, tool_assistant())
        terminal = {}
    elif phase == "tools_completed":
        messages = (
            user,
            tool_assistant(),
            ToolResultMessage(
                id="result-b",
                tool_call_id="call-b",
                tool_name="second",
                content="preview",
                artifact_refs=(artifact_ref(),),
                created_at=fixed_time(),
            ),
            ToolResultMessage(
                id="result-a",
                tool_call_id="call-a",
                tool_name="first",
                content="Error: unavailable",
                is_error=True,
                created_at=fixed_time(),
            ),
        )
        terminal = {}
    else:
        messages = (
            user,
            AssistantMessage(id="assistant-final", content=content, created_at=fixed_time()),
        )
        terminal = {
            "terminal_status": "completed",
            "stop_reason": "model_stop",
            "final_content": content,
        }
    return ContextCheckpoint(
        session_id="session-中文",
        input_id=user.id,
        input_revision=2,
        base_leaf_id="entry-before",
        save_cursor=3,
        phase=phase,  # type: ignore[arg-type]
        model="test-model",
        next_model_turn=1,
        messages=messages,
        tools_used=("first", "second"),
        usage={"prompt_tokens": 10, "completion_tokens": 2},
        updated_at=fixed_time(),
        **terminal,  # type: ignore[arg-type]
    )


def test_context_checkpoint_is_immutable_and_freezes_collections() -> None:
    usage = {"prompt_tokens": 10}
    value = checkpoint()

    usage["prompt_tokens"] = 99

    assert value.messages[0].id == "input-1"
    assert value.usage == {"prompt_tokens": 10, "completion_tokens": 2}
    with pytest.raises(TypeError):
        value.usage["prompt_tokens"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        value.phase = "final_response"  # type: ignore[misc]


def test_checkpoint_round_trips_multiple_incorporated_inputs() -> None:
    first = UserMessage(id="input-1", content="initial")
    second = UserMessage(id="input-2", content="follow-up")
    value = ContextCheckpoint(
        session_id="session-1",
        input_id="input-1",
        input_revision=1,
        base_leaf_id=None,
        save_cursor=0,
        phase="final_response",
        model="fake-model",
        next_model_turn=2,
        messages=(
            first,
            AssistantMessage(content="candidate"),
            second,
            AssistantMessage(content="final"),
        ),
        incorporated_inputs=(
            IncorporatedInput(id="input-1", revision=1),
            IncorporatedInput(id="input-2", revision=3),
        ),
        terminal_status="completed",
        stop_reason="model_stop",
        final_content="final",
    )

    restored = CheckpointCodec().decode(CheckpointCodec().encode(value))

    assert restored == value
    assert restored.incorporated_inputs == value.incorporated_inputs


def test_checkpoint_rejects_untracked_user_messages() -> None:
    with pytest.raises(ValueError, match="incorporated_inputs"):
        ContextCheckpoint(
            session_id="session-1",
            input_id="input-1",
            input_revision=1,
            base_leaf_id=None,
            save_cursor=0,
            phase="final_response",
            model="fake-model",
            next_model_turn=2,
            messages=(
                UserMessage(id="input-1", content="initial"),
                AssistantMessage(content="candidate"),
                UserMessage(id="input-2", content="untracked"),
                AssistantMessage(content="final"),
            ),
            terminal_status="completed",
            stop_reason="model_stop",
            final_content="final",
        )


def test_checkpoint_codec_migrates_v1_to_primary_incorporated_input() -> None:
    codec = CheckpointCodec()
    document = codec.encode(checkpoint())
    document["schema_version"] = 1
    document.pop("incorporated_inputs")

    restored = codec.decode(document)

    assert restored.incorporated_inputs == (
        IncorporatedInput(id="input-1", revision=restored.input_revision),
    )


def test_checkpoint_phase_shapes_are_strict() -> None:
    user = UserMessage(id="input-1", content="continue")

    with pytest.raises(ValueError, match="awaiting_tools"):
        ContextCheckpoint(
            session_id="session-1",
            input_id=user.id,
            input_revision=1,
            base_leaf_id=None,
            save_cursor=0,
            phase="awaiting_tools",
            model="model",
            next_model_turn=1,
            messages=(user, AssistantMessage(content="not tools")),
        )

    with pytest.raises(ValueError, match="terminal_status"):
        RunnerCheckpoint(
            phase="final_response",
            model="model",
            next_model_turn=1,
            messages=(AssistantMessage(content="answer"),),
            final_content="answer",
            stop_reason="model_stop",
        )

    with pytest.raises(ValueError, match="fulfill each latest ToolCall once"):
        ContextCheckpoint(
            session_id="session-1",
            input_id=user.id,
            input_revision=1,
            base_leaf_id=None,
            save_cursor=0,
            phase="tools_completed",
            model="model",
            next_model_turn=1,
            messages=(
                user,
                tool_assistant(),
                ToolResultMessage(
                    tool_call_id="call-a",
                    tool_name="first",
                    content="ok",
                ),
            ),
        )


@pytest.mark.parametrize(
    "phase",
    ["awaiting_tools", "tools_completed", "final_response"],
)
def test_checkpoint_codec_round_trips_all_phases(phase: str) -> None:
    codec = CheckpointCodec()
    original = checkpoint(phase)

    document = codec.encode(original)
    restored = codec.decode(json.loads(json.dumps(document, ensure_ascii=False)))

    assert document["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert restored == original
    if phase == "tools_completed":
        result = restored.messages[2]
        assert isinstance(result, ToolResultMessage)
        assert result.artifact_refs == (artifact_ref(),)


def test_checkpoint_codec_rejects_unsupported_version() -> None:
    document = CheckpointCodec().encode(checkpoint())
    document["schema_version"] = 99

    with pytest.raises(UnsupportedCheckpointVersionError):
        CheckpointCodec().decode(document)


def test_in_memory_checkpoint_store_replaces_latest_value() -> None:
    store = InMemoryCheckpointStore()
    first = checkpoint()
    final = checkpoint("final_response", content="done")

    assert store.load(first.session_id) is None
    store.save(first)
    store.save(final)

    assert store.load(first.session_id) == final
    assert store.delete(first.session_id) is True
    assert store.delete(first.session_id) is False


def test_json_checkpoint_store_persists_overwrites_and_reopens(tmp_path) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    first = checkpoint()
    final = checkpoint("final_response", content="saved answer")

    store.save(first)
    store.save(final)

    digest = sha256(first.session_id.encode("utf-8")).hexdigest()
    target = tmp_path / f"{digest}.checkpoint.json"
    assert target.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert JsonFileCheckpointStore(tmp_path).load(first.session_id) == final


def test_json_checkpoint_store_rejects_corrupt_and_mismatched_files(tmp_path) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    value = checkpoint()
    store.save(value)
    digest = sha256(value.session_id.encode("utf-8")).hexdigest()
    target = tmp_path / f"{digest}.checkpoint.json"

    target.write_text("{broken", encoding="utf-8")
    with pytest.raises(CheckpointCorruptError, match="Cannot load"):
        store.load(value.session_id)

    document = CheckpointCodec().encode(value)
    document["session_id"] = "another-session"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointCorruptError, match="identity mismatch"):
        store.load(value.session_id)
