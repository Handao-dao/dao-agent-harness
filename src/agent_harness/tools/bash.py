"""Simple shell command tool modeled after pi's bash tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict
from os import PathLike
from typing import Any

from agent_harness.tools.base import ToolExecutionMode, ToolOutput
from agent_harness.tools.pathing import ToolPathPolicy
from agent_harness.tools.truncation import truncate_tail


class BashTool:
    name = "bash"
    description = (
        "Execute a shell command in the workspace. Returns combined stdout and stderr; "
        "the optional timeout is measured in seconds."
    )
    execution_mode: ToolExecutionMode = "sequential"
    timeout_s: float | None = None
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "timeout": {"type": "number", "minimum": 0.001},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | PathLike[str]) -> None:
        self.path_policy = ToolPathPolicy(workspace)

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        command = str(arguments["command"])
        timeout = arguments.get("timeout")
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.path_policy.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        communicate = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            if timeout is None:
                stdout, _ = await communicate
            else:
                stdout, _ = await asyncio.wait_for(asyncio.shield(communicate), float(timeout))
        except TimeoutError:
            timed_out = True
            process.kill()
            stdout, _ = await communicate
        except asyncio.CancelledError:
            process.kill()
            await communicate
            raise

        full_output = stdout.decode("utf-8", errors="replace")
        truncation = truncate_tail(full_output)
        output = truncation.content or "(no output)"
        is_error = timed_out or process.returncode not in {0, None}
        if timed_out:
            output += f"\n\nCommand timed out after {float(timeout):g} seconds"
        elif process.returncode not in {0, None}:
            output += f"\n\nCommand exited with code {process.returncode}"
        return ToolOutput(
            content=output,
            artifact_content=full_output if truncation.truncated else None,
            is_error=is_error,
            metadata={
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "truncation": asdict(truncation),
            },
        )


__all__ = ["BashTool"]
