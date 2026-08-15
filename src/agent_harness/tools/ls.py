"""Stable, bounded workspace directory listing inspired by pi's ls tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.pathing import ToolPathPolicy
from agent_harness.tools.truncation import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LS_LIMIT = 500


class LsTool:
    """List one directory with deterministic ordering and bounded output."""

    name = "ls"
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None

    def __init__(
        self,
        workspace: str | PathLike[str],
        *,
        allow_outside_workspace: bool = False,
        default_limit: int = DEFAULT_LS_LIMIT,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if (
            not isinstance(default_limit, int)
            or isinstance(default_limit, bool)
            or default_limit <= 0
        ):
            raise ValueError("default_limit must be a positive integer")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.path_policy = ToolPathPolicy(
            workspace,
            allow_outside_workspace=allow_outside_workspace,
        )
        self.default_limit = default_limit
        self.max_bytes = max_bytes
        self.description = (
            "List a directory inside the workspace. Returns case-insensitively sorted "
            "entries, includes hidden entries, and appends '/' to directories. "
            f"Output defaults to {default_limit} entries and is bounded to "
            f"{format_size(max_bytes)}."
        )
        self.parameters: Mapping[str, Any] = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "default": ".",
                    "description": "Directory relative to the workspace. Defaults to '.'.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": default_limit,
                    "description": "Maximum number of entries to return.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        raw_path = arguments.get("path", ".")
        limit = arguments.get("limit", self.default_limit)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("path must be non-empty text")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        path = self.path_policy.resolve(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {raw_path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {raw_path}")

        entries, skipped_entries = await asyncio.to_thread(_scan_directory, path)
        total_entries = len(entries)
        selected = entries[:limit]
        entry_limit_reached = total_entries > limit
        truncation = truncate_head(
            "\n".join(selected),
            max_lines=max(len(selected), 1),
            max_bytes=self.max_bytes,
        )

        if total_entries == 0:
            output = "(empty directory)"
        elif truncation.content:
            output = truncation.content
        else:
            output = "(no complete entry fits within the output byte limit)"
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{limit} entry limit reached; use limit={limit * 2} for more")
        if truncation.truncated:
            notices.append(
                f"{format_size(self.max_bytes)} output limit reached; use a narrower path"
            )
        if skipped_entries:
            notices.append(f"{skipped_entries} unreadable entries skipped")
        if notices:
            output += f"\n\n[{'. '.join(notices)}.]"

        display_path = self.path_policy.display(path)
        if display_path == ".":
            display_path = "."
        return ToolOutput(
            content=output,
            metadata={
                "kind": "directory_listing",
                "path": display_path,
                "returned_entries": truncation.output_lines,
                "total_entries": total_entries,
                "entry_limit_reached": entry_limit_reached,
                "skipped_entries": skipped_entries,
                "truncation": asdict(truncation),
            },
            allow_externalization=False,
        )


def _scan_directory(path: Path) -> tuple[list[str], int]:
    entries: list[str] = []
    skipped = 0
    with os.scandir(path) as iterator:
        for entry in iterator:
            try:
                suffix = "/" if entry.is_dir() else ""
            except OSError:
                skipped += 1
                continue
            entries.append(f"{entry.name}{suffix}")
    entries.sort(key=lambda item: (item.casefold(), item))
    return entries, skipped


__all__ = ["DEFAULT_LS_LIMIT", "LsTool"]
