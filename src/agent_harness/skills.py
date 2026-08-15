"""Skill discovery, activation, and bounded resource access."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import get_close_matches
from hashlib import sha256
from pathlib import Path
from typing import Literal

from agent_harness.messages import AgentMessage, AssistantMessage, ToolCall, ToolResultMessage

SkillSource = Literal["builtin", "user", "workspace"]

DEFAULT_MAX_SKILL_INSTRUCTION_CHARS = 20_000
DEFAULT_MAX_ACTIVE_SKILL_CHARS = 100_000
DEFAULT_MAX_SKILL_DESCRIPTION_CHARS = 500
DEFAULT_SKILL_RESOURCE_CHARS = 4_000
_SKILL_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SOURCE_PRIORITY: dict[SkillSource, int] = {
    "builtin": 0,
    "user": 1,
    "workspace": 2,
}


class SkillCatalogError(RuntimeError):
    """Base error for one requested Skill operation."""

    code = "skill_error"


class SkillNotFoundError(SkillCatalogError):
    code = "skill_not_found"


class SkillInvalidError(SkillCatalogError):
    code = "skill_invalid"


class SkillReadError(SkillCatalogError):
    code = "skill_read_failed"


class SkillTooLargeError(SkillCatalogError):
    code = "skill_too_large"


class SkillResourceError(SkillCatalogError):
    code = "skill_resource_error"


@dataclass(frozen=True, slots=True)
class SkillRoot:
    path: Path
    source: SkillSource

    def __post_init__(self) -> None:
        if self.source not in _SOURCE_PRIORITY:
            raise ValueError(f"Unsupported Skill source: {self.source!r}")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    name: str
    description: str
    root: Path
    entrypoint: Path
    content_hash: str
    source: SkillSource


@dataclass(frozen=True, slots=True)
class SkillActivation:
    descriptor: SkillDescriptor
    instructions: str


@dataclass(frozen=True, slots=True)
class SkillCatalogDiagnostic:
    path: Path
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SkillResource:
    skill_name: str
    path: str
    media_type: str
    size_bytes: int
    size_chars: int
    offset: int
    next_offset: int
    eof: bool
    content: str


class SkillCatalog:
    """Discover trusted Skill packages without owning activation state."""

    def __init__(
        self,
        roots: tuple[SkillRoot, ...] | list[SkillRoot],
        *,
        max_instruction_chars: int = DEFAULT_MAX_SKILL_INSTRUCTION_CHARS,
        max_description_chars: int = DEFAULT_MAX_SKILL_DESCRIPTION_CHARS,
        max_resource_chars: int = DEFAULT_SKILL_RESOURCE_CHARS,
    ) -> None:
        if type(max_instruction_chars) is not int or max_instruction_chars <= 0:
            raise ValueError("max_instruction_chars must be a positive integer")
        if type(max_description_chars) is not int or max_description_chars <= 0:
            raise ValueError("max_description_chars must be a positive integer")
        if type(max_resource_chars) is not int or max_resource_chars <= 0:
            raise ValueError("max_resource_chars must be a positive integer")
        normalized = tuple(roots)
        if any(not isinstance(root, SkillRoot) for root in normalized):
            raise TypeError("roots must contain SkillRoot values")
        self._roots = normalized
        self._max_instruction_chars = max_instruction_chars
        self._max_description_chars = max_description_chars
        self._max_resource_chars = max_resource_chars
        self._skills: dict[str, SkillDescriptor] = {}
        self._diagnostics: tuple[SkillCatalogDiagnostic, ...] = ()
        self.refresh()

    @classmethod
    def for_workspace(cls, workspace: Path) -> SkillCatalog:
        workspace = Path(workspace)
        return cls(
            (
                SkillRoot(workspace / ".dao" / "skills", "workspace"),
                SkillRoot(Path.home() / ".dao" / "skills", "user"),
            )
        )

    @property
    def diagnostics(self) -> tuple[SkillCatalogDiagnostic, ...]:
        return self._diagnostics

    @property
    def max_resource_chars(self) -> int:
        return self._max_resource_chars

    def refresh(self) -> None:
        discovered: dict[str, SkillDescriptor] = {}
        diagnostics: list[SkillCatalogDiagnostic] = []
        ordered_roots = sorted(
            self._roots,
            key=lambda item: (_SOURCE_PRIORITY[item.source], str(item.path)),
        )
        for skill_root in ordered_roots:
            if not skill_root.path.exists():
                continue
            if not skill_root.path.is_dir():
                diagnostics.append(
                    SkillCatalogDiagnostic(
                        path=skill_root.path,
                        code="skill_root_invalid",
                        message="Skill root is not a directory",
                    )
                )
                continue
            try:
                directories = sorted(
                    (path for path in skill_root.path.iterdir() if path.is_dir()),
                    key=lambda path: path.name,
                )
            except OSError as exc:
                diagnostics.append(
                    SkillCatalogDiagnostic(
                        path=skill_root.path,
                        code="skill_root_read_failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            for directory in directories:
                entrypoint = directory / "SKILL.md"
                if not entrypoint.is_file():
                    continue
                try:
                    descriptor, _body = self._read_entrypoint(
                        entrypoint,
                        source=skill_root.source,
                    )
                except SkillCatalogError as exc:
                    diagnostics.append(
                        SkillCatalogDiagnostic(
                            path=entrypoint,
                            code=exc.code,
                            message=str(exc),
                        )
                    )
                    continue
                if descriptor.name != directory.name:
                    diagnostics.append(
                        SkillCatalogDiagnostic(
                            path=entrypoint,
                            code=SkillInvalidError.code,
                            message=(
                                f"Skill name {descriptor.name!r} must match directory "
                                f"name {directory.name!r}"
                            ),
                        )
                    )
                    continue
                discovered[descriptor.name] = descriptor
        self._skills = discovered
        self._diagnostics = tuple(diagnostics)

    def discover(self) -> tuple[SkillDescriptor, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> SkillDescriptor | None:
        return self._skills.get(name)

    def suggestions(self, name: str, *, limit: int = 3) -> tuple[str, ...]:
        return tuple(get_close_matches(name, tuple(self._skills), n=limit, cutoff=0.5))

    def load(self, name: str) -> SkillActivation:
        descriptor = self._require_skill(name)
        current, body = self._read_entrypoint(
            descriptor.entrypoint,
            source=descriptor.source,
        )
        if current.name != descriptor.name:
            raise SkillInvalidError(
                f"Skill name changed from {descriptor.name!r} to {current.name!r}"
            )
        if current.content_hash != descriptor.content_hash:
            self.refresh()
            descriptor = self._require_skill(name)
            current, body = self._read_entrypoint(
                descriptor.entrypoint,
                source=descriptor.source,
            )
        if len(body) > self._max_instruction_chars:
            raise SkillTooLargeError(
                f"Skill {name!r} instructions contain {len(body)} characters; "
                f"maximum is {self._max_instruction_chars}. Move details into referenced files."
            )
        return SkillActivation(descriptor=current, instructions=body)

    def resolve_resource(self, name: str, relative_path: str) -> Path:
        descriptor = self._require_skill(name)
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise SkillResourceError("Skill resource path must be non-empty text")
        supplied = Path(relative_path)
        if supplied.is_absolute():
            raise SkillResourceError("Skill resource path must be relative")
        try:
            root = descriptor.root.resolve(strict=True)
            resolved = (root / supplied).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SkillResourceError(
                f"Skill resource {relative_path!r} does not exist"
            ) from exc
        if not resolved.is_relative_to(root):
            raise SkillResourceError("Skill resource path escapes its Skill directory")
        if not resolved.is_file():
            raise SkillResourceError("Skill resource path must identify a file")
        return resolved

    def read_resource(
        self,
        name: str,
        relative_path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> SkillResource:
        if type(offset) is not int or offset < 0:
            raise SkillResourceError("offset must be a non-negative integer")
        if limit is None:
            limit = self._max_resource_chars
        if type(limit) is not int or limit <= 0:
            raise SkillResourceError("limit must be a positive integer")
        limit = min(limit, self._max_resource_chars)
        path = self.resolve_resource(name, relative_path)
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8")
        except UnicodeError as exc:
            raise SkillResourceError(
                "Binary or non-UTF-8 resources require a dedicated media tool"
            ) from exc
        except OSError as exc:
            raise SkillReadError(
                f"Could not read Skill resource {relative_path!r}: {type(exc).__name__}: {exc}"
            ) from exc
        next_offset = min(len(content), offset + limit)
        media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        return SkillResource(
            skill_name=name,
            path=Path(relative_path).as_posix(),
            media_type=media_type,
            size_bytes=len(raw),
            size_chars=len(content),
            offset=min(offset, len(content)),
            next_offset=next_offset,
            eof=next_offset >= len(content),
            content=content[offset:next_offset],
        )

    def _require_skill(self, name: str) -> SkillDescriptor:
        if not isinstance(name, str) or not name.strip():
            raise SkillNotFoundError("Skill name must be non-empty text")
        descriptor = self._skills.get(name)
        if descriptor is None:
            suggestions = self.suggestions(name)
            suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise SkillNotFoundError(f"Skill {name!r} was not found.{suffix}")
        return descriptor

    def _read_entrypoint(
        self,
        entrypoint: Path,
        *,
        source: SkillSource,
    ) -> tuple[SkillDescriptor, str]:
        try:
            raw = entrypoint.read_bytes()
            content = raw.decode("utf-8")
        except UnicodeError as exc:
            raise SkillInvalidError(f"Skill file is not valid UTF-8: {entrypoint}") from exc
        except OSError as exc:
            raise SkillReadError(
                f"Could not read Skill file {entrypoint}: {type(exc).__name__}: {exc}"
            ) from exc
        metadata, body = _parse_frontmatter(content)
        name = metadata.get("name", "")
        description = metadata.get("description", "")
        if not _SKILL_NAME_PATTERN.fullmatch(name):
            raise SkillInvalidError(
                "Skill name must contain lowercase letters, digits, and single hyphens"
            )
        if not description:
            raise SkillInvalidError("Skill description must be non-empty text")
        if len(description) > self._max_description_chars:
            raise SkillInvalidError(
                f"Skill description contains {len(description)} characters; "
                f"maximum is {self._max_description_chars}"
            )
        root = entrypoint.parent
        return (
            SkillDescriptor(
                name=name,
                description=description,
                root=root,
                entrypoint=entrypoint,
                content_hash=sha256(raw).hexdigest(),
                source=source,
            ),
            body,
        )


@dataclass(frozen=True, slots=True)
class _SkillInstructionRecord:
    skill_name: str
    content_hash: str
    assistant_index: int
    call: ToolCall
    result_index: int
    result: ToolResultMessage


def is_skill_instruction(message: AgentMessage) -> bool:
    return (
        isinstance(message, ToolResultMessage)
        and message.metadata.get("kind") == "skill_instruction"
        and isinstance(message.metadata.get("skill_name"), str)
        and isinstance(message.metadata.get("content_hash"), str)
    )


def select_active_skill_messages(
    messages: Sequence[AgentMessage],
    *,
    max_chars: int = DEFAULT_MAX_ACTIVE_SKILL_CHARS,
) -> tuple[AgentMessage, ...]:
    """Build legal, deduplicated activation blocks for ephemeral reinjection."""

    selected = _select_skill_records(messages, max_chars=max_chars)
    if not selected:
        return ()
    by_assistant: dict[int, list[_SkillInstructionRecord]] = {}
    for record in selected:
        by_assistant.setdefault(record.assistant_index, []).append(record)
    projected: list[AgentMessage] = []
    for assistant_index in sorted(by_assistant):
        records = sorted(by_assistant[assistant_index], key=lambda item: item.result_index)
        assistant = messages[assistant_index]
        if not isinstance(assistant, AssistantMessage):
            continue
        selected_ids = {record.call.id for record in records}
        projected.append(
            replace(
                assistant,
                content="",
                tool_calls=tuple(
                    call for call in assistant.tool_calls if call.id in selected_ids
                ),
            )
        )
        projected.extend(record.result for record in records)
    return tuple(projected)


def deduplicate_skill_messages(
    messages: Sequence[AgentMessage],
    *,
    max_chars: int = DEFAULT_MAX_ACTIVE_SKILL_CHARS,
) -> tuple[tuple[AgentMessage, ...], int]:
    """Drop superseded Skill activations while preserving legal tool-call blocks."""

    records = _skill_instruction_records(messages)
    selected = _select_skill_records(messages, max_chars=max_chars)
    selected_results = {record.result_index for record in selected}
    removed_records = [record for record in records if record.result_index not in selected_results]
    if not removed_records:
        return tuple(messages), 0
    removed_result_indices = {record.result_index for record in removed_records}
    removed_call_ids = {record.call.id for record in removed_records}
    updated: list[AgentMessage] = []
    for index, message in enumerate(messages):
        if index in removed_result_indices:
            continue
        if isinstance(message, AssistantMessage) and message.tool_calls:
            calls = tuple(
                call for call in message.tool_calls if call.id not in removed_call_ids
            )
            if calls != message.tool_calls:
                if not calls and not message.content:
                    continue
                message = replace(message, tool_calls=calls)
        updated.append(message)
    return tuple(updated), len(removed_records)


def skill_instruction_indices(messages: Sequence[AgentMessage]) -> frozenset[int]:
    records = _skill_instruction_records(messages)
    return frozenset(
        index
        for record in records
        for index in (record.assistant_index, record.result_index)
    )


def _select_skill_records(
    messages: Sequence[AgentMessage],
    *,
    max_chars: int,
) -> tuple[_SkillInstructionRecord, ...]:
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    latest: dict[str, _SkillInstructionRecord] = {}
    for record in _skill_instruction_records(messages):
        latest[record.skill_name] = record
    remaining = max_chars
    selected: list[_SkillInstructionRecord] = []
    for record in sorted(latest.values(), key=lambda item: item.result_index, reverse=True):
        size = len(record.result.content)
        if size > remaining:
            continue
        selected.append(record)
        remaining -= size
    return tuple(sorted(selected, key=lambda item: item.result_index))


def _skill_instruction_records(
    messages: Sequence[AgentMessage],
) -> tuple[_SkillInstructionRecord, ...]:
    calls: dict[str, tuple[int, ToolCall]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                calls[call.id] = (index, call)
    records: list[_SkillInstructionRecord] = []
    for result_index, message in enumerate(messages):
        if not is_skill_instruction(message):
            continue
        declaration = calls.get(message.tool_call_id)
        if declaration is None:
            continue
        assistant_index, call = declaration
        skill_name = message.metadata["skill_name"]
        content_hash = message.metadata["content_hash"]
        if not isinstance(skill_name, str) or not isinstance(content_hash, str):
            continue
        records.append(
            _SkillInstructionRecord(
                skill_name=skill_name,
                content_hash=content_hash,
                assistant_index=assistant_index,
                call=call,
                result_index=result_index,
                result=message,
            )
        )
    return tuple(records)


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillInvalidError("SKILL.md must start with YAML frontmatter")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise SkillInvalidError("SKILL.md frontmatter is not closed")
    metadata: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:closing], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = stripped.partition(":")
        if not separator or not key.strip():
            raise SkillInvalidError(f"Invalid frontmatter line {line_number}")
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in metadata:
            raise SkillInvalidError(f"Duplicate frontmatter field: {key}")
        metadata[key] = value
    return metadata, "\n".join(lines[closing + 1 :]).strip()


__all__ = [
    "DEFAULT_MAX_ACTIVE_SKILL_CHARS",
    "DEFAULT_MAX_SKILL_DESCRIPTION_CHARS",
    "DEFAULT_MAX_SKILL_INSTRUCTION_CHARS",
    "DEFAULT_SKILL_RESOURCE_CHARS",
    "SkillActivation",
    "SkillCatalog",
    "SkillCatalogDiagnostic",
    "SkillCatalogError",
    "SkillDescriptor",
    "SkillInvalidError",
    "SkillNotFoundError",
    "SkillReadError",
    "SkillResource",
    "SkillResourceError",
    "SkillRoot",
    "SkillSource",
    "SkillTooLargeError",
    "deduplicate_skill_messages",
    "is_skill_instruction",
    "select_active_skill_messages",
    "skill_instruction_indices",
]
