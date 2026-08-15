"""Simple text search tool modeled after pi's grep tool."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.pathing import ToolPathPolicy
from agent_harness.tools.truncation import DEFAULT_MAX_BYTES, truncate_head, truncate_line

DEFAULT_GREP_LIMIT = 100
_IGNORED_DIRECTORIES = {".git", "node_modules", "__pycache__"}


class GrepTool:
    name = "grep"
    description = "Search UTF-8 text files and return matching paths, line numbers, and context."
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1, "default": "."},
            "glob": {"type": "string", "minLength": 1},
            "ignoreCase": {"type": "boolean", "default": False},
            "literal": {"type": "boolean", "default": False},
            "context": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_GREP_LIMIT},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | PathLike[str]) -> None:
        self.path_policy = ToolPathPolicy(workspace)

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        pattern = str(arguments["pattern"])
        raw_path = str(arguments.get("path", "."))
        glob = str(arguments.get("glob", "*"))
        ignore_case = bool(arguments.get("ignoreCase", False))
        literal = bool(arguments.get("literal", False))
        context = int(arguments.get("context", 0))
        limit = int(arguments.get("limit", DEFAULT_GREP_LIMIT))
        target = self.path_policy.resolve(raw_path)
        if not target.exists():
            raise FileNotFoundError(f"Search path not found: {raw_path}")

        expression = re.escape(pattern) if literal else pattern
        regex = re.compile(expression, re.IGNORECASE if ignore_case else 0)
        lines, limit_reached = await asyncio.to_thread(
            _grep,
            target,
            self.path_policy.workspace,
            glob,
            regex,
            context,
            limit,
        )
        truncation = truncate_head(
            "\n".join(lines),
            max_lines=max(len(lines), 1),
            max_bytes=DEFAULT_MAX_BYTES,
        )
        output = truncation.content or "No matches found"
        if limit_reached:
            output += f"\n\n[{limit} match limit reached; refine the pattern or increase limit.]"
        elif truncation.truncated:
            output += "\n\n[Output truncated; refine the pattern or path.]"
        return ToolOutput(content=output, allow_externalization=False)


def _grep(
    target: Path,
    workspace: Path,
    glob: str,
    regex: re.Pattern[str],
    context: int,
    limit: int,
) -> tuple[list[str], bool]:
    files = [target] if target.is_file() else list(target.rglob(glob))
    output: list[str] = []
    matches = 0
    for path in sorted(files, key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for index, line in enumerate(lines):
            if regex.search(line) is None:
                continue
            if matches >= limit:
                return output, True
            matches += 1
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            for shown in range(start, end):
                text, _ = truncate_line(lines[shown])
                separator = ":" if shown == index else "-"
                output.append(f"{relative.as_posix()}{separator}{shown + 1}{separator} {text}")
    return output, False


__all__ = ["DEFAULT_GREP_LIMIT", "GrepTool"]
