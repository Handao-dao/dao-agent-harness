"""Build a deterministic model context from persisted Agent messages."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

from agent_harness.injection import merge_consecutive_user_messages
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.session import Session
from agent_harness.skills import (
    DEFAULT_MAX_ACTIVE_SKILL_CHARS,
    SkillCatalog,
    select_active_skill_messages,
)
from agent_harness.summary import ContextSummary, ContextSummaryCodec, ContextSummaryContent


class ContextBuildError(RuntimeError):
    """The model context could not be assembled from its configured sources."""


@dataclass(frozen=True, slots=True)
class ModelContext:
    """A derived, non-persisted view passed to a Runner and Provider."""

    system_prompt: str | None
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ResolvedSessionContext:
    """The applicable summary and uncovered raw tail for one active branch."""

    summary: ContextSummary | None
    messages: tuple[AgentMessage, ...]
    covered_message_count: int

    @property
    def summary_id(self) -> str | None:
        return self.summary.id if self.summary is not None else None


class SessionContextResolver:
    """Resolve a branch-aware ContextSummary without mutating the Session tree."""

    def resolve(self, session: Session) -> ResolvedSessionContext:
        if not isinstance(session, Session):
            raise TypeError("session must be a Session")
        active_entries = session.active_entries()
        if not active_entries:
            return ResolvedSessionContext(
                summary=None,
                messages=(),
                covered_message_count=0,
            )

        positions = {entry.id: index for index, entry in enumerate(active_entries)}
        applicable: list[tuple[int, int, ContextSummary]] = []
        for event_index, summary in enumerate(session.context_summaries):
            position = positions.get(summary.covered_through_entry_id)
            if position is not None:
                applicable.append((position, event_index, summary))

        if not applicable:
            return ResolvedSessionContext(
                summary=None,
                messages=tuple(entry.message for entry in active_entries),
                covered_message_count=0,
            )

        covered_position, _event_index, summary = max(
            applicable, key=lambda item: (item[0], item[1])
        )
        tail = active_entries[covered_position + 1 :]
        return ResolvedSessionContext(
            summary=summary,
            messages=tuple(entry.message for entry in tail),
            covered_message_count=covered_position + 1,
        )


class ContextBuilder:
    """Assemble the initial system prompt and provider-neutral message view."""

    BOOTSTRAP_FILES = ("AGENTS.md", "SOUL.md", "USER.md")
    _SECTION_SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        workspace: Path,
        *,
        summary_codec: ContextSummaryCodec | None = None,
        skill_catalog: SkillCatalog | None = None,
        max_active_skill_chars: int = DEFAULT_MAX_ACTIVE_SKILL_CHARS,
        memory_store: Any | None = None,
    ) -> None:
        if skill_catalog is not None and not isinstance(skill_catalog, SkillCatalog):
            raise TypeError("skill_catalog must be a SkillCatalog")
        if type(max_active_skill_chars) is not int or max_active_skill_chars < 0:
            raise ValueError("max_active_skill_chars must be a non-negative integer")
        self.workspace = Path(workspace)
        self._summary_codec = summary_codec or ContextSummaryCodec()
        self._skill_catalog = skill_catalog
        self._max_active_skill_chars = max_active_skill_chars
        if memory_store is not None and not callable(
            getattr(memory_store, "read_memory", None)
        ):
            raise TypeError("memory_store must provide read_memory()")
        self._memory_store = memory_store

    def build(
        self,
        working_messages: Sequence[AgentMessage],
        *,
        context_summary: ContextSummaryContent | None = None,
        extra_system_sections: Sequence[str] = (),
    ) -> ModelContext:
        """Return a model-only view without modifying the working conversation."""
        return ModelContext(
            system_prompt=self.build_system_prompt(
                context_summary=context_summary,
                extra_sections=extra_system_sections,
            ),
            messages=self.build_messages(working_messages),
        )

    @classmethod
    def build_messages(
        cls,
        messages: Sequence[AgentMessage],
    ) -> tuple[dict[str, Any], ...]:
        """Project typed working messages into a fresh Provider-facing view."""

        projected = tuple(cls._to_model_message(message) for message in messages)
        return merge_consecutive_user_messages(projected)

    def build_system_prompt(
        self,
        *,
        context_summary: ContextSummaryContent | None = None,
        extra_sections: Sequence[str] = (),
    ) -> str:
        """Build stable prompt sections in their defined order."""
        sections = [self._load_template("identity.md")]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            sections.append(bootstrap)

        sections.append(self._load_template("tool_contract.md"))
        memory = self._build_memory_section()
        if memory:
            sections.append(memory)
        skill_catalog = self._build_skill_catalog_section()
        if skill_catalog:
            sections.append(skill_catalog)
        if context_summary is not None:
            sections.append(self._build_summary_section(context_summary))
        sections.extend(self._normalize_extra_sections(extra_sections))
        return self._SECTION_SEPARATOR.join(section.strip() for section in sections)

    def build_skill_context_prefix(
        self,
        messages: Sequence[AgentMessage],
    ) -> tuple[AgentMessage, ...]:
        """Return legal Skill activation blocks for a summarized history prefix."""

        return select_active_skill_messages(
            messages,
            max_chars=self._max_active_skill_chars,
        )

    def _build_skill_catalog_section(self) -> str:
        catalog = self._skill_catalog
        if catalog is None:
            return ""
        skills = catalog.discover()
        if not skills:
            return ""
        entries = "\n".join(
            "  <skill>\n"
            f"    <name>{escape(skill.name)}</name>\n"
            f"    <description>{escape(skill.description)}</description>\n"
            "  </skill>"
            for skill in skills
        )
        return f"{self._load_template('skills_catalog.md')}\n\n<available_skills>\n{entries}\n</available_skills>"

    def _build_memory_section(self) -> str:
        store = self._memory_store
        if store is None:
            return ""
        content = store.read_memory()
        if not isinstance(content, str):
            raise ContextBuildError("memory_store.read_memory() must return text")
        content = content.strip()
        if not content or not self._has_memory_content(content):
            return ""
        escaped = (
            content.replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        return (
            f"{self._load_template('memory_context.md')}\n\n"
            "<long_term_memory>\n"
            f"{escaped}\n"
            "</long_term_memory>"
        )

    @staticmethod
    def _has_memory_content(content: str) -> bool:
        return any(
            line.strip() and not line.lstrip().startswith("#")
            for line in content.splitlines()
        )

    def _build_summary_section(self, content: ContextSummaryContent) -> str:
        if not isinstance(content, ContextSummaryContent):
            raise ContextBuildError("context_summary must be ContextSummaryContent")
        encoded = self._summary_codec.canonical_json(content)
        encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        instructions = self._load_template("archived_context.md")
        return (
            f"{instructions}\n\n"
            "<archived_conversation_context>\n"
            f"{encoded}\n"
            "</archived_conversation_context>"
        )

    def _load_bootstrap_files(self) -> str:
        sections: list[str] = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if not path.exists():
                continue
            if not path.is_file():
                raise ContextBuildError(f"Bootstrap path is not a file: {path}")
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ContextBuildError(f"Could not read UTF-8 bootstrap file: {path}") from exc
            if content.strip():
                sections.append(f"## {filename}\n\n{content.strip()}")
        return "\n\n".join(sections)

    @staticmethod
    def _normalize_extra_sections(sections: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for index, section in enumerate(sections):
            if not isinstance(section, str):
                raise ContextBuildError(f"Extra system section {index} must be text")
            if section.strip():
                normalized.append(section.strip())
        return normalized

    @staticmethod
    def _load_template(filename: str) -> str:
        try:
            content = (
                package_files("agent_harness")
                .joinpath("templates", filename)
                .read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError) as exc:
            raise ContextBuildError(f"Could not load bundled context template: {filename}") from exc
        if not content.strip():
            raise ContextBuildError(f"Bundled context template is empty: {filename}")
        return content.strip()

    @classmethod
    def _to_model_message(cls, message: AgentMessage) -> dict[str, Any]:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}

        if isinstance(message, RuntimeStatusMessage):
            return {"role": "user", "content": message.content}

        if isinstance(message, AssistantMessage):
            result: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": cls._encode_arguments(call.id, call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            return result

        if isinstance(message, ToolResultMessage):
            content = message.content
            if message.is_error and not content.lstrip().lower().startswith("error:"):
                content = f"Error: {content}"
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "name": message.tool_name,
                "content": content,
            }

        raise ContextBuildError(f"Unsupported AgentMessage type: {type(message).__name__}")

    @staticmethod
    def _encode_arguments(call_id: str, arguments: Any) -> str:
        try:
            return json.dumps(
                dict(arguments),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ContextBuildError(
                f"Tool call {call_id} arguments are not JSON serializable"
            ) from exc


__all__ = [
    "ContextBuildError",
    "ContextBuilder",
    "ModelContext",
    "ResolvedSessionContext",
    "SessionContextResolver",
]
