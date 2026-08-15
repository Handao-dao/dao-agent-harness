"""Exact text replacement tool modeled after pi's edit tool."""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.mutation import DEFAULT_FILE_MUTATION_QUEUE
from agent_harness.tools.pathing import ToolPathPolicy


class EditTool:
    name = "edit"
    description = "Edit one UTF-8 text file using exact replacements. Each oldText must match once."
    execution_mode: ToolExecutionMode = "sequential"
    timeout_s: float | None = None
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string", "minLength": 1},
                        "newText": {"type": "string"},
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["path", "edits"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | PathLike[str]) -> None:
        self.path_policy = ToolPathPolicy(workspace)

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        raw_path = str(arguments["path"])
        edits = arguments["edits"]
        if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes)):
            raise ValueError("edits must be a sequence")
        path = self.path_policy.resolve(raw_path)
        async with DEFAULT_FILE_MUTATION_QUEUE.hold(path):
            patch = await asyncio.to_thread(_edit_file, path, raw_path, edits)
        return ToolOutput(
            content=f"Successfully replaced {len(edits)} block(s) in {raw_path}.",
            metadata={"patch": patch},
            allow_externalization=False,
        )


def _edit_file(
    path: Path,
    display_path: str,
    edits: Sequence[Any],
) -> str:
    raw = path.read_bytes().decode("utf-8")
    bom = "\ufeff" if raw.startswith("\ufeff") else ""
    content = raw[len(bom) :]
    line_ending = "\r\n" if "\r\n" in content else "\n"
    original = content.replace("\r\n", "\n").replace("\r", "\n")
    replacements: list[tuple[int, int, str]] = []

    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise ValueError(f"edits[{index}] must be an object")
        old = str(edit["oldText"]).replace("\r\n", "\n").replace("\r", "\n")
        new = str(edit["newText"]).replace("\r\n", "\n").replace("\r", "\n")
        if not old:
            raise ValueError(f"edits[{index}].oldText must not be empty")
        if original.count(old) != 1:
            raise ValueError(f"edits[{index}].oldText must match exactly once in {display_path}")
        start = original.index(old)
        replacements.append((start, start + len(old), new))

    replacements.sort()
    for previous, current in zip(replacements, replacements[1:]):
        if previous[1] > current[0]:
            raise ValueError("edit ranges must not overlap")

    updated = original
    for start, end, new in reversed(replacements):
        updated = updated[:start] + new + updated[end:]
    if updated == original:
        raise ValueError("replacement produced no changes")

    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=display_path,
            tofile=display_path,
        )
    )
    restored = updated.replace("\n", line_ending)
    path.write_text(bom + restored, encoding="utf-8", newline="")
    return patch


__all__ = ["EditTool"]
