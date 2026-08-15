"""Simple workspace file discovery modeled after pi's find tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.pathing import ToolPathPolicy
from agent_harness.tools.truncation import DEFAULT_MAX_BYTES, truncate_head

DEFAULT_FIND_LIMIT = 1_000
_IGNORED_DIRECTORIES = {".git", "node_modules", "__pycache__"}


class FindTool:
    name = "find"
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None

    def __init__(self, workspace: str | PathLike[str]) -> None:
        self.path_policy = ToolPathPolicy(workspace)
        self.description = (
            "Find files by glob pattern. Returns workspace-relative paths. "
            "Use patterns such as '*.py' or 'src/**/*.json'."
        )
        self.parameters: Mapping[str, Any] = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1, "default": "."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": DEFAULT_FIND_LIMIT,
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        pattern = str(arguments["pattern"])
        raw_path = str(arguments.get("path", "."))
        limit = int(arguments.get("limit", DEFAULT_FIND_LIMIT))
        root = self.path_policy.resolve(raw_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {raw_path}")

        matches = await asyncio.to_thread(_find_files, root, pattern)
        limited = matches[:limit]
        truncation = truncate_head(
            "\n".join(limited),
            max_lines=max(len(limited), 1),
            max_bytes=DEFAULT_MAX_BYTES,
        )
        output = truncation.content or "No files found matching pattern"
        if len(matches) > limit:
            output += f"\n\n[{limit} result limit reached; refine the pattern or increase limit.]"
        elif truncation.truncated:
            output += "\n\n[Output truncated; refine the pattern or search path.]"
        return ToolOutput(content=output, allow_externalization=False)


def _find_files(root: Path, pattern: str) -> list[str]:
    matches = [
        path.relative_to(root).as_posix()
        for path in root.rglob(pattern)
        if path.is_file()
        and not any(part in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts)
    ]
    return sorted(matches, key=lambda value: (value.casefold(), value))


__all__ = ["DEFAULT_FIND_LIMIT", "FindTool"]
