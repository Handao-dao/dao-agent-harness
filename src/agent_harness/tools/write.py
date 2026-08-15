"""Simple file creation and overwrite tool modeled after pi's write tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from os import PathLike
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.mutation import DEFAULT_FILE_MUTATION_QUEUE
from agent_harness.tools.pathing import ToolPathPolicy


class WriteTool:
    name = "write"
    description = "Create or overwrite a UTF-8 text file and create parent directories."
    execution_mode: ToolExecutionMode = "sequential"
    timeout_s: float | None = None
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | PathLike[str]) -> None:
        self.path_policy = ToolPathPolicy(workspace)

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        raw_path = str(arguments["path"])
        content = str(arguments["content"])
        path = self.path_policy.resolve(raw_path)
        async with DEFAULT_FILE_MUTATION_QUEUE.hold(path):
            await asyncio.to_thread(_write_text, path, content)
        return ToolOutput(
            content=f"Successfully wrote {len(content.encode('utf-8'))} bytes to {raw_path}",
            allow_externalization=False,
        )


def _write_text(path: Any, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


__all__ = ["WriteTool"]
