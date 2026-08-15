"""Bounded workspace file reading inspired by pi's read tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict
from os import PathLike
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.pathing import ToolPathPolicy
from agent_harness.tools.truncation import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)


class ReadTool:
    """Read UTF-8 text with stable line pagination and actionable truncation."""

    name = "read"
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None

    def __init__(
        self,
        workspace: str | PathLike[str],
        *,
        allow_outside_workspace: bool = False,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(max_lines, int) or isinstance(max_lines, bool) or max_lines <= 0:
            raise ValueError("max_lines must be a positive integer")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.path_policy = ToolPathPolicy(
            workspace,
            allow_outside_workspace=allow_outside_workspace,
        )
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.description = (
            "Read a UTF-8 text file inside the workspace. Supports 1-indexed offset "
            f"and line limit. Output is bounded to {max_lines} lines or "
            f"{format_size(max_bytes)}; use the returned next offset to continue."
        )
        self.parameters: Mapping[str, Any] = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "File path relative to the workspace, or an allowed absolute path.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "First line to read, using 1-based line numbers.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of lines to read before output limits apply.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("path must be non-empty text")
        offset = arguments.get("offset", 1)
        limit = arguments.get("limit")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise ValueError("offset must be a positive integer")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")

        path = self.path_policy.resolve(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {raw_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {raw_path}")

        data = await asyncio.to_thread(path.read_bytes)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {raw_path}") from exc

        lines = _split_lines(text)
        total_lines = len(lines)
        start_index = offset - 1
        if total_lines == 0:
            if offset != 1:
                raise ValueError(f"Offset {offset} is beyond end of empty file")
            selected_lines: list[str] = []
        else:
            if start_index >= total_lines:
                raise ValueError(
                    f"Offset {offset} is beyond end of file ({total_lines} lines total)"
                )
            end_index = total_lines if limit is None else min(start_index + limit, total_lines)
            selected_lines = lines[start_index:end_index]

        selected = "\n".join(selected_lines)
        truncation = truncate_head(
            selected,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )
        display_path = self.path_policy.display(path)
        output, next_offset = self._model_view(
            truncation=truncation,
            selected_count=len(selected_lines),
            offset=offset,
            total_lines=total_lines,
            display_path=display_path,
        )
        metadata = {
            "kind": "file_read",
            "path": display_path,
            "offset": offset,
            "next_offset": next_offset,
            "total_lines": total_lines,
            "truncation": asdict(truncation),
        }
        return ToolOutput(
            content=output,
            metadata=metadata,
            allow_externalization=False,
        )

    def _model_view(
        self,
        *,
        truncation: TruncationResult,
        selected_count: int,
        offset: int,
        total_lines: int,
        display_path: str,
    ) -> tuple[str, int | None]:
        if total_lines == 0:
            return "(empty file)", None
        if truncation.first_line_exceeds_limit:
            return (
                f"[Line {offset} in {display_path} exceeds the "
                f"{format_size(self.max_bytes)} per-call byte limit.]",
                None,
            )

        consumed = truncation.output_lines
        next_offset = offset + consumed
        has_more = next_offset <= total_lines and consumed < selected_count
        user_limit_stopped = consumed == selected_count and offset + consumed <= total_lines
        output = truncation.content

        if truncation.truncated:
            end_line = next_offset - 1
            reason = (
                f"{self.max_lines} line limit"
                if truncation.truncated_by == "lines"
                else f"{format_size(self.max_bytes)} limit"
            )
            output += (
                f"\n\n[Showing lines {offset}-{end_line} of {total_lines} "
                f"({reason}). Use offset={next_offset} to continue.]"
            )
            return output, next_offset

        if has_more or user_limit_stopped:
            remaining = total_lines - (next_offset - 1)
            if remaining > 0:
                output += (
                    f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
                return output, next_offset
        return output, None


def _split_lines(content: str) -> list[str]:
    if not content:
        return []
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines.pop()
    return lines


__all__ = ["ReadTool"]
