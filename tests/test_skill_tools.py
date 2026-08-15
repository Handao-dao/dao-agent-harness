from __future__ import annotations

import json

from agent_harness.artifacts import ArtifactPolicy, InMemoryArtifactStore
from agent_harness.messages import ToolCall
from agent_harness.skills import SkillCatalog, SkillRoot
from agent_harness.tools import ActivateSkillTool, ReadSkillResourceTool, ToolRegistry


def build_catalog(tmp_path) -> SkillCatalog:
    directory = tmp_path / "skills" / "pdf"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: Work with PDFs.\n---\nRead references/guide.md\n",
        encoding="utf-8",
    )
    (directory / "references").mkdir()
    (directory / "references" / "guide.md").write_text(
        "guide content",
        encoding="utf-8",
    )
    return SkillCatalog((SkillRoot(tmp_path / "skills", "workspace"),))


async def test_activate_skill_returns_protected_tool_metadata(tmp_path) -> None:
    catalog = build_catalog(tmp_path)
    registry = ToolRegistry(
        artifact_store=InMemoryArtifactStore(),
        artifact_policy=ArtifactPolicy(
            externalize_above_chars=64,
            preview_head_chars=10,
            preview_tail_chars=10,
            read_chunk_chars=20,
        ),
    )
    registry.register(ActivateSkillTool(catalog))

    result = await registry.execute_call(
        ToolCall(id="call-1", name="activate_skill", arguments={"name": "pdf"})
    )

    assert result.status == "completed"
    assert result.metadata["kind"] == "skill_instruction"
    assert result.metadata["skill_name"] == "pdf"
    assert result.metadata["source"] == "workspace"
    assert result.artifact_refs == ()
    assert '<skill name="pdf" source="workspace"' in result.content
    assert "Read references/guide.md" in result.content


async def test_skill_tools_return_stable_errors_and_bounded_resources(tmp_path) -> None:
    catalog = build_catalog(tmp_path)
    registry = ToolRegistry()
    registry.register(ActivateSkillTool(catalog))
    registry.register(ReadSkillResourceTool(catalog))

    missing = await registry.execute_call(
        ToolCall(id="missing", name="activate_skill", arguments={"name": "pdff"})
    )
    resource = await registry.execute_call(
        ToolCall(
            id="resource",
            name="read_skill_resource",
            arguments={
                "skill_name": "pdf",
                "path": "references/guide.md",
                "limit": 5,
            },
        )
    )

    assert missing.status == "failed"
    assert missing.error_code == "reported_error"
    assert missing.metadata["code"] == "skill_not_found"
    payload = json.loads(resource.content)
    assert payload["content"] == "guide"
    assert payload["next_offset"] == 5
    assert resource.metadata == {
        "kind": "skill_resource",
        "skill_name": "pdf",
        "path": "references/guide.md",
    }
