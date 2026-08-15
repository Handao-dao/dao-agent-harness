"""Model-facing tools backed by the internal SkillCatalog."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape
from typing import Any

from agent_harness.skills import SkillCatalog, SkillCatalogError
from agent_harness.tools.base import ToolExecutionMode, ToolOutput


def _error_output(error: SkillCatalogError) -> ToolOutput:
    return ToolOutput(
        content=f"Error: [{error.code}] {error}",
        is_error=True,
        metadata={"kind": "skill_error", "code": error.code},
        allow_externalization=False,
    )


@dataclass(slots=True)
class ActivateSkillTool:
    catalog: SkillCatalog
    name: str = field(init=False, default="activate_skill")
    description: str = field(
        init=False,
        default=(
            "Load the complete instructions for one available Skill before using it. "
            "Referenced files can be read later with read_skill_resource."
        ),
    )
    parameters: Mapping[str, Any] = field(init=False)
    execution_mode: ToolExecutionMode = field(init=False, default="parallel_safe")
    timeout_s: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, SkillCatalog):
            raise TypeError("catalog must be a SkillCatalog")
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact Skill name from the available Skill catalog.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error_output(SkillCatalogError("Skill name must be non-empty text"))
        try:
            activation = self.catalog.load(name)
        except SkillCatalogError as exc:
            return _error_output(exc)
        descriptor = activation.descriptor
        content = (
            f'<skill name="{escape(descriptor.name, quote=True)}" '
            f'source="{escape(descriptor.source, quote=True)}" '
            f'location="skill://{escape(descriptor.name, quote=True)}/SKILL.md">\n'
            "References are relative to this Skill directory. Use read_skill_resource "
            "with the same Skill name to load referenced text files.\n\n"
            f"{activation.instructions}\n"
            "</skill>"
        )
        return ToolOutput(
            content=content,
            metadata={
                "kind": "skill_instruction",
                "skill_name": descriptor.name,
                "source": descriptor.source,
                "content_hash": descriptor.content_hash,
                "retention": "session",
            },
            allow_externalization=False,
        )


@dataclass(slots=True)
class ReadSkillResourceTool:
    catalog: SkillCatalog
    name: str = field(init=False, default="read_skill_resource")
    description: str = field(
        init=False,
        default=(
            "Read a bounded UTF-8 text slice from a file inside an available Skill. "
            "Use paths relative to that Skill directory."
        ),
    )
    parameters: Mapping[str, Any] = field(init=False)
    execution_mode: ToolExecutionMode = field(init=False, default="parallel_safe")
    timeout_s: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, SkillCatalog):
            raise TypeError("catalog must be a SkillCatalog")
        self.parameters = {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "minLength": 1,
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Path relative to the Skill directory.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.catalog.max_resource_chars,
                    "default": self.catalog.max_resource_chars,
                },
            },
            "required": ["skill_name", "path"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        skill_name = arguments.get("skill_name")
        path = arguments.get("path")
        if not isinstance(skill_name, str) or not isinstance(path, str):
            return _error_output(SkillCatalogError("Skill name and path must be text"))
        try:
            resource = self.catalog.read_resource(
                skill_name,
                path,
                offset=arguments.get("offset", 0),
                limit=arguments.get("limit", self.catalog.max_resource_chars),
            )
        except SkillCatalogError as exc:
            return _error_output(exc)
        content = json.dumps(
            {
                "skill_name": resource.skill_name,
                "path": resource.path,
                "media_type": resource.media_type,
                "size_bytes": resource.size_bytes,
                "size_chars": resource.size_chars,
                "offset": resource.offset,
                "next_offset": resource.next_offset,
                "eof": resource.eof,
                "content": resource.content,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ToolOutput(
            content=content,
            metadata={
                "kind": "skill_resource",
                "skill_name": resource.skill_name,
                "path": resource.path,
            },
        )


__all__ = ["ActivateSkillTool", "ReadSkillResourceTool"]
