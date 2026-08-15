from __future__ import annotations

import asyncio

import pytest

from agent_harness.artifacts import InMemoryArtifactStore
from agent_harness.messages import ToolCall
from agent_harness.tools import (
    FileMutationQueue,
    LsTool,
    ReadTool,
    ToolPathError,
    ToolPathPolicy,
    ToolRegistry,
)
from agent_harness.tools.truncation import truncate_head, truncate_tail


def test_path_policy_resolves_inside_workspace_and_rejects_escape(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = ToolPathPolicy(workspace)

    assert policy.resolve("src/example.py") == workspace / "src" / "example.py"
    assert policy.display(workspace / "src" / "example.py") == "src/example.py"

    with pytest.raises(ToolPathError, match="outside the workspace"):
        policy.resolve("../secret.txt")


def test_truncation_respects_complete_lines_and_utf8_bytes() -> None:
    by_lines = truncate_head("one\ntwo\nthree", max_lines=2, max_bytes=100)
    by_bytes = truncate_head("甲\n乙\n丙", max_lines=10, max_bytes=7)
    tail = truncate_tail("one\ntwo\nthree", max_lines=2, max_bytes=100)

    assert by_lines.content == "one\ntwo"
    assert by_lines.truncated_by == "lines"
    assert by_bytes.content == "甲\n乙"
    assert by_bytes.truncated_by == "bytes"
    assert by_bytes.output_bytes == 7
    assert tail.content == "two\nthree"


async def test_file_mutation_queue_serializes_same_path_only(tmp_path) -> None:
    queue = FileMutationQueue()
    entered: list[str] = []
    first_release = asyncio.Event()
    other_entered = asyncio.Event()

    async def first() -> None:
        async with queue.hold(tmp_path / "same.txt"):
            entered.append("first")
            await first_release.wait()

    async def second() -> None:
        async with queue.hold(tmp_path / "same.txt"):
            entered.append("second")

    async def other() -> None:
        async with queue.hold(tmp_path / "other.txt"):
            entered.append("other")
            other_entered.set()

    first_task = asyncio.create_task(first())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(second())
    other_task = asyncio.create_task(other())
    await other_entered.wait()

    assert entered == ["first", "other"]
    first_release.set()
    await asyncio.gather(first_task, second_task, other_task)
    assert entered == ["first", "other", "second"]


async def test_read_tool_reads_utf8_text_and_reports_stable_metadata(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("第一行\n第二行\n第三行\n", encoding="utf-8")
    tool = ReadTool(tmp_path)

    result = await tool.execute({"path": "notes.txt"})

    assert result.content == "第一行\n第二行\n第三行"
    assert result.allow_externalization is False
    assert result.metadata["path"] == "notes.txt"
    assert result.metadata["offset"] == 1
    assert result.metadata["next_offset"] is None
    assert result.metadata["total_lines"] == 3


async def test_read_tool_supports_offset_limit_and_continuation(tmp_path) -> None:
    (tmp_path / "lines.txt").write_text(
        "\n".join(f"line {number}" for number in range(1, 6)),
        encoding="utf-8",
    )
    tool = ReadTool(tmp_path)

    result = await tool.execute({"path": "lines.txt", "offset": 2, "limit": 2})

    assert result.content.startswith("line 2\nline 3")
    assert "2 more lines in file" in result.content
    assert "offset=4" in result.content
    assert result.metadata["next_offset"] == 4


async def test_read_tool_truncates_with_actionable_next_offset(tmp_path) -> None:
    (tmp_path / "large.txt").write_text("one\ntwo\nthree\nfour", encoding="utf-8")
    tool = ReadTool(tmp_path, max_lines=2, max_bytes=100)

    result = await tool.execute({"path": "large.txt"})

    assert result.content.startswith("one\ntwo")
    assert "Showing lines 1-2 of 4" in result.content
    assert "offset=3" in result.content
    assert result.metadata["next_offset"] == 3
    assert result.metadata["truncation"]["truncated_by"] == "lines"


async def test_read_tool_rejects_non_utf8_and_workspace_escape(tmp_path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    tool = ReadTool(tmp_path)

    with pytest.raises(ValueError, match="not valid UTF-8"):
        await tool.execute({"path": "binary.bin"})
    with pytest.raises(ToolPathError, match="outside the workspace"):
        await tool.execute({"path": "../secret.txt"})


async def test_read_tool_result_is_not_externalized_again(tmp_path) -> None:
    (tmp_path / "large.txt").write_text("one\ntwo\nthree", encoding="utf-8")
    registry = ToolRegistry(artifact_store=InMemoryArtifactStore())
    registry.register(ReadTool(tmp_path, max_lines=1, max_bytes=100))

    result = await registry.execute_call(
        ToolCall(id="call-1", name="read", arguments={"path": "large.txt"})
    )

    assert result.status == "completed"
    assert result.artifact_refs == ()
    assert "offset=2" in result.content


async def test_ls_tool_lists_hidden_entries_directories_and_stable_order(tmp_path) -> None:
    (tmp_path / "beta.txt").write_text("", encoding="utf-8")
    (tmp_path / "Alpha.txt").write_text("", encoding="utf-8")
    (tmp_path / ".hidden").write_text("", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    tool = LsTool(tmp_path)

    result = await tool.execute({})

    assert result.content == ".hidden\nAlpha.txt\nbeta.txt\nfolder/"
    assert result.metadata["path"] == "."
    assert result.metadata["returned_entries"] == 4
    assert result.metadata["total_entries"] == 4
    assert result.allow_externalization is False


async def test_ls_tool_reports_entry_limit_with_actionable_retry(tmp_path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    tool = LsTool(tmp_path)

    result = await tool.execute({"limit": 2})

    assert result.content.startswith("a.txt\nb.txt")
    assert "2 entry limit reached" in result.content
    assert "limit=4" in result.content
    assert result.metadata["entry_limit_reached"] is True


async def test_ls_tool_applies_byte_limit_without_splitting_entries(tmp_path) -> None:
    for name in ("a", "bbbb", "c"):
        (tmp_path / name).write_text("", encoding="utf-8")
    tool = LsTool(tmp_path, max_bytes=3)

    result = await tool.execute({})

    assert result.content.startswith("a\n")
    assert "output limit reached" in result.content
    assert result.metadata["returned_entries"] == 1
    assert result.metadata["truncation"]["truncated_by"] == "bytes"


async def test_ls_tool_reports_empty_and_rejects_invalid_paths(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    target = tmp_path / "file.txt"
    target.write_text("value", encoding="utf-8")
    tool = LsTool(tmp_path)

    result = await tool.execute({"path": "empty"})
    assert result.content == "(empty directory)"

    with pytest.raises(NotADirectoryError, match="Not a directory"):
        await tool.execute({"path": "file.txt"})
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        await tool.execute({"path": "missing"})
    with pytest.raises(ToolPathError, match="outside the workspace"):
        await tool.execute({"path": "../outside"})


async def test_ls_registry_validation_and_result_protocol(tmp_path) -> None:
    registry = ToolRegistry(artifact_store=InMemoryArtifactStore())
    registry.register(LsTool(tmp_path))

    invalid = await registry.execute_call(ToolCall(id="call-1", name="ls", arguments={"limit": 0}))
    completed = await registry.execute_call(ToolCall(id="call-2", name="ls"))

    assert invalid.status == "failed"
    assert invalid.error_code == "invalid_arguments"
    assert completed.status == "completed"
    assert completed.content == "(empty directory)"
    assert completed.artifact_refs == ()
