from __future__ import annotations

import json
from datetime import datetime

import pytest

from agent_harness.artifacts import (
    ArtifactPolicy,
    InMemoryArtifactStore,
    InvalidArtifactIdError,
)
from agent_harness.tools import CurrentTimeTool, ReadArtifactTool, ToolRegistry


async def test_current_time_tool_returns_utc_timestamp() -> None:
    result = await CurrentTimeTool().execute({})

    payload = json.loads(result.content)
    parsed = datetime.fromisoformat(payload["utc"])
    assert parsed.utcoffset() is not None
    assert result.metadata == {"kind": "current_time"}
    assert result.allow_externalization is False


async def test_read_artifact_tool_returns_bounded_structured_slice() -> None:
    store = InMemoryArtifactStore()
    content = "0123456789" * 10
    ref = await store.put_text(content)
    policy = ArtifactPolicy(
        externalize_above_chars=1_000,
        preview_head_chars=100,
        preview_tail_chars=100,
        read_chunk_chars=20,
    )
    tool = ReadArtifactTool(store, policy)

    result = await tool.execute({"artifact_id": ref.id, "offset": 5, "limit": 999})

    assert json.loads(result.content) == {
        "artifact_id": ref.id,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
        "size_chars": ref.size_chars,
        "offset": 5,
        "next_offset": 25,
        "eof": False,
        "content": content[5:25],
    }
    assert result.metadata == {"kind": "artifact_slice", "artifact_id": ref.id}
    assert result.allow_externalization is False


def test_read_artifact_tool_schema_exposes_policy_limit() -> None:
    policy = ArtifactPolicy(read_chunk_chars=123)
    tool = ReadArtifactTool(InMemoryArtifactStore(), policy)
    registry = ToolRegistry()
    registry.register(tool)

    definition = registry.definitions()[0]

    assert definition["name"] == "read_artifact"
    assert definition["parameters"]["properties"]["limit"]["maximum"] == 123
    assert tool.execution_mode == "parallel_safe"


async def test_read_artifact_tool_rejects_noncanonical_id() -> None:
    tool = ReadArtifactTool(InMemoryArtifactStore())

    with pytest.raises(InvalidArtifactIdError):
        await tool.execute({"artifact_id": "../secret"})
