from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.messages import AssistantMessage, ToolCall, ToolResultMessage
from agent_harness.skills import (
    SkillCatalog,
    SkillResourceError,
    SkillRoot,
    select_active_skill_messages,
)


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    entrypoint = directory / "SKILL.md"
    entrypoint.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return directory


def test_catalog_discovers_valid_skills_and_workspace_overrides_user(tmp_path) -> None:
    user_root = tmp_path / "user"
    workspace_root = tmp_path / "workspace"
    write_skill(user_root, "pdf", "user description", "user body")
    write_skill(workspace_root, "pdf", "workspace description", "workspace body")
    invalid = workspace_root / "Broken"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("missing frontmatter", encoding="utf-8")

    catalog = SkillCatalog(
        (
            SkillRoot(workspace_root, "workspace"),
            SkillRoot(user_root, "user"),
        )
    )

    assert [skill.name for skill in catalog.discover()] == ["pdf"]
    assert catalog.get("pdf").description == "workspace description"  # type: ignore[union-attr]
    assert catalog.load("pdf").instructions == "workspace body"
    assert catalog.diagnostics[0].code == "skill_invalid"


def test_catalog_refreshes_a_changed_skill_hash(tmp_path) -> None:
    root = tmp_path / "skills"
    directory = write_skill(root, "pdf", "PDF work", "first")
    catalog = SkillCatalog((SkillRoot(root, "workspace"),))
    first = catalog.load("pdf")
    (directory / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: PDF work\n---\nsecond\n",
        encoding="utf-8",
    )

    second = catalog.load("pdf")

    assert second.instructions == "second"
    assert second.descriptor.content_hash != first.descriptor.content_hash


def test_catalog_rejects_an_oversized_routing_description(tmp_path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "pdf", "description is too long", "instructions")

    catalog = SkillCatalog(
        (SkillRoot(root, "workspace"),),
        max_description_chars=10,
    )

    assert catalog.discover() == ()
    assert len(catalog.diagnostics) == 1
    assert catalog.diagnostics[0].code == "skill_invalid"
    assert "maximum is 10" in catalog.diagnostics[0].message


def test_resource_reads_are_bounded_and_cannot_escape_skill_root(tmp_path) -> None:
    root = tmp_path / "skills"
    directory = write_skill(root, "pdf", "PDF work", "Use references/guide.md")
    references = directory / "references"
    references.mkdir()
    (references / "guide.md").write_text("0123456789", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    catalog = SkillCatalog(
        (SkillRoot(root, "workspace"),),
        max_resource_chars=4,
    )

    resource = catalog.read_resource("pdf", "references/guide.md", offset=3, limit=99)

    assert resource.content == "3456"
    assert resource.next_offset == 7
    assert resource.eof is False
    with pytest.raises(SkillResourceError, match="escapes"):
        catalog.resolve_resource("pdf", "../../outside.txt")


def test_select_active_skill_messages_keeps_latest_version_as_a_legal_pair() -> None:
    first_call = ToolCall(id="first", name="activate_skill", arguments={"name": "pdf"})
    second_call = ToolCall(id="second", name="activate_skill", arguments={"name": "pdf"})
    messages = (
        AssistantMessage(tool_calls=(first_call,)),
        ToolResultMessage(
            tool_call_id="first",
            tool_name="activate_skill",
            content="old",
            metadata={
                "kind": "skill_instruction",
                "skill_name": "pdf",
                "content_hash": "old",
            },
        ),
        AssistantMessage(tool_calls=(second_call,)),
        ToolResultMessage(
            tool_call_id="second",
            tool_name="activate_skill",
            content="new",
            metadata={
                "kind": "skill_instruction",
                "skill_name": "pdf",
                "content_hash": "new",
            },
        ),
    )

    selected = select_active_skill_messages(messages)

    assert len(selected) == 2
    assert isinstance(selected[0], AssistantMessage)
    assert selected[0].tool_calls == (second_call,)
    assert isinstance(selected[1], ToolResultMessage)
    assert selected[1].content == "new"
