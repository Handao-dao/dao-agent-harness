from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from agent_harness.artifacts import ArtifactRef
from agent_harness.context import ContextBuilder
from agent_harness.messages import (
    AssistantMessage,
    RuntimeStatusMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.runner import AgentRunner
from agent_harness.runtime import AgentRuntime
from agent_harness.runtime_io import RuntimeRequest
from agent_harness.session import PendingInput, Session
from agent_harness.status_builder import RuntimeStatusBuilder
from agent_harness.storage import (
    JsonFileSessionStore,
    SessionCodec,
    SessionCodecError,
    SessionStorageError,
    UnsupportedSessionVersionError,
)
from agent_harness.summary import ContextSummary, ContextSummaryContent
from agent_harness.testing import ScriptedProvider
from agent_harness.tools import ToolRegistry


def fixed_time(hour: int) -> datetime:
    return datetime(2026, 8, 8, hour, tzinfo=UTC)


def artifact_ref() -> ArtifactRef:
    content = "完整天气结果"
    digest = sha256(content.encode("utf-8")).hexdigest()
    return ArtifactRef(
        id=f"art_{digest}",
        media_type="text/plain; charset=utf-8",
        size_bytes=len(content.encode("utf-8")),
        size_chars=len(content),
        sha256=digest,
    )


def complete_session() -> Session:
    status = RuntimeStatusBuilder(now=lambda: fixed_time(3)).build(
        (UserMessage(content="查询天气"),)
    )
    return Session.from_messages(
        id="session-中文",
        created_at=fixed_time(1),
        updated_at=fixed_time(5),
        messages=(
            UserMessage(id="user-1", content="查询天气", created_at=fixed_time(2)),
            status,
            AssistantMessage(
                id="assistant-1",
                created_at=fixed_time(3),
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="weather",
                        arguments={"city": "杭州", "days": [1, 2]},
                    ),
                ),
            ),
            ToolResultMessage(
                id="result-1",
                created_at=fixed_time(4),
                tool_call_id="call-1",
                tool_name="weather",
                content="晴",
                is_error=False,
                artifact_refs=(artifact_ref(),),
                metadata={"kind": "skill_resource", "skill_name": "weather"},
            ),
        ),
        pending_inputs=(
            PendingInput(
                id="input-2",
                source_message_id="external-2",
                content="那明天呢？",
                created_at=fixed_time(5),
                edited_at=fixed_time(5),
            ),
        ),
        metadata={"channel": "cli", "flags": ["typed", "durable"]},
    )


def test_codec_round_trip_reconstructs_strong_message_types() -> None:
    codec = SessionCodec()
    original = complete_session()

    document = codec.encode(original)
    restored = codec.decode(json.loads(json.dumps(document, ensure_ascii=False)))

    assert restored == original
    assert document["schema_version"] == 6
    assert [entry["message"]["type"] for entry in document["entries"]] == [
        "user",
        "runtime_status",
        "assistant",
        "tool_result",
    ]
    assert isinstance(restored.messages[0], UserMessage)
    assert isinstance(restored.messages[1], RuntimeStatusMessage)
    assert isinstance(restored.messages[2], AssistantMessage)
    assert isinstance(restored.messages[3], ToolResultMessage)
    assert restored.messages[3].artifact_refs == (artifact_ref(),)
    assert restored.messages[3].metadata == {
        "kind": "skill_resource",
        "skill_name": "weather",
    }


def test_codec_rejects_unknown_schema_and_message_types() -> None:
    codec = SessionCodec()
    document = codec.encode(complete_session())

    unsupported = dict(document, schema_version=999)
    with pytest.raises(UnsupportedSessionVersionError):
        codec.decode(unsupported)

    unknown_message = dict(document)
    first_entry = dict(document["entries"][0])
    first_entry["message"] = dict(first_entry["message"], type="alien")
    unknown_message["entries"] = [first_entry]
    with pytest.raises(SessionCodecError, match="Unknown AgentMessage type"):
        codec.decode(unknown_message)


def test_codec_migrates_v1_linear_messages_to_an_entry_chain() -> None:
    codec = SessionCodec()
    original = complete_session()
    legacy = {
        "schema_version": 1,
        "id": original.id,
        "created_at": original.created_at.isoformat(),
        "updated_at": original.updated_at.isoformat(),
        "messages": [codec.encode_message(message) for message in original.messages],
        "pending_inputs": [
            codec.encode_pending(item) for item in original.pending_inputs
        ],
        "metadata": original.metadata,
    }

    migrated = codec.decode(legacy)

    assert migrated.messages == original.messages
    assert migrated.pending_inputs == original.pending_inputs
    assert migrated.active_leaf_id == migrated.entries[-1].id
    assert all(
        entry.parent_id == migrated.entries[index - 1].id
        for index, entry in enumerate(migrated.entries[1:], start=1)
    )


def test_codec_reads_v2_snapshots_without_context_summaries() -> None:
    codec = SessionCodec()
    original = complete_session()
    document = codec.encode(original)
    document["schema_version"] = 2
    del document["context_summaries"]
    del document["entries"][3]["message"]["artifact_refs"]

    restored = codec.decode(document)

    assert restored.context_summaries == []
    assert restored.messages[3].artifact_refs == ()


def test_codec_reads_v3_tool_results_without_artifact_refs() -> None:
    codec = SessionCodec()
    document = codec.encode(complete_session())
    document["schema_version"] = 3
    del document["entries"][3]["message"]["artifact_refs"]

    restored = codec.decode(document)

    assert isinstance(restored.messages[3], ToolResultMessage)
    assert restored.messages[3].artifact_refs == ()


def test_codec_reads_v4_tool_results_without_metadata() -> None:
    codec = SessionCodec()
    document = codec.encode(complete_session())
    document["schema_version"] = 4
    del document["entries"][3]["message"]["metadata"]

    restored = codec.decode(document)

    assert isinstance(restored.messages[3], ToolResultMessage)
    assert restored.messages[3].metadata == {}


def test_codec_rejects_invalid_artifact_refs() -> None:
    codec = SessionCodec()
    document = codec.encode(complete_session())
    document["entries"][3]["message"]["artifact_refs"][0]["sha256"] = "0" * 64

    with pytest.raises(SessionCodecError, match="ArtifactRef is invalid"):
        codec.decode(document)


def test_codec_round_trips_context_summaries_in_v3_snapshots() -> None:
    codec = SessionCodec()
    original = complete_session()
    summary = ContextSummary(
        id="summary-1",
        session_id=original.id,
        covered_through_entry_id=original.entries[0].id,
        source_leaf_id=original.active_leaf_id or "missing",
        content=ContextSummaryContent(
            schema_version=1,
            objective="查询天气",
            status="active",
            completed_work=("已查询今天的天气",),
        ),
        tokens_before=9_000,
        created_at=fixed_time(5),
    )
    original.record_context_summary(summary)

    restored = codec.decode(codec.encode(original))

    assert restored.context_summaries == [summary]


def test_codec_rejects_non_json_metadata() -> None:
    session = Session(id="session-1", metadata={"bad": object()})

    with pytest.raises(SessionCodecError, match="unsupported value type"):
        SessionCodec().encode(session)


def test_json_store_survives_process_like_reopen(tmp_path) -> None:
    original = complete_session()
    first_store = JsonFileSessionStore(tmp_path)

    first_store.save(original)
    restored = JsonFileSessionStore(tmp_path).get_or_create(original.id)

    assert restored == original
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == f"{sha256(original.id.encode('utf-8')).hexdigest()}.json"
    assert not list(tmp_path.glob("*.tmp"))


def test_pending_input_is_durable_before_conversation_commit(tmp_path) -> None:
    store = JsonFileSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    queued = session.enqueue(
        PendingInput(id="input-1", source_message_id="external-1", content="不要丢失")
    )
    store.save(session)

    restored = JsonFileSessionStore(tmp_path).get_or_create("session-1")

    assert restored.messages == []
    assert restored.pending_inputs == [queued]


async def test_runtime_failure_keeps_pending_input_after_store_reopen(tmp_path) -> None:
    session_directory = tmp_path / "sessions"
    runtime = AgentRuntime(
        runner=AgentRunner(ScriptedProvider([RuntimeError("offline")])),
        session_store=JsonFileSessionStore(session_directory),
        context_builder=ContextBuilder(tmp_path),
        tools=ToolRegistry(),
        model="fake-model",
    )

    result = await runtime.submit(RuntimeRequest("session-1", "external-1", "稍后重试"))
    restored = JsonFileSessionStore(session_directory).get_or_create("session-1")

    assert result.status == "failed"
    assert restored.messages == []
    assert len(restored.pending_inputs) == 1
    assert restored.pending_inputs[0].content == "稍后重试"


def test_json_store_reports_corrupted_session_instead_of_replacing_it(tmp_path) -> None:
    session_id = "session-1"
    filename = f"{sha256(session_id.encode('utf-8')).hexdigest()}.json"
    (tmp_path / filename).write_text("{not-json", encoding="utf-8")

    with pytest.raises(SessionStorageError, match="Cannot load"):
        JsonFileSessionStore(tmp_path).get_or_create(session_id)


def test_json_store_delete_removes_memory_and_disk_state(tmp_path) -> None:
    store = JsonFileSessionStore(tmp_path)
    session = store.get_or_create("session-1")
    store.save(session)

    assert store.delete("session-1") is True
    assert store.delete("session-1") is False
    restored = JsonFileSessionStore(tmp_path).get_or_create("session-1")
    assert restored.id == "session-1"
    assert restored.messages == []
    assert restored.pending_inputs == []
