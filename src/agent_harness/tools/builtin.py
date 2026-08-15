"""Small built-in tools for the interactive CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_harness.artifacts import ArtifactPolicy, ArtifactStore
from agent_harness.tools.base import ToolExecutionMode, ToolOutput


@dataclass(slots=True)
class CurrentTimeTool:
    name: str = "get_current_time"
    description: str = "Return the current UTC time in ISO 8601 format."
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    execution_mode: ToolExecutionMode = "parallel_safe"
    timeout_s: float | None = None

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        del arguments
        return ToolOutput(
            content=json.dumps(
                {"utc": datetime.now(timezone.utc).isoformat()},
                separators=(",", ":"),
            ),
            metadata={"kind": "current_time"},
            allow_externalization=False,
        )


@dataclass(slots=True)
class ReadArtifactTool:
    store: ArtifactStore
    policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    name: str = field(init=False, default="read_artifact")
    description: str = field(
        init=False,
        default=(
            "Read a bounded text slice from an externalized tool result using its artifact ID."
        ),
    )
    parameters: Mapping[str, Any] = field(init=False)
    execution_mode: ToolExecutionMode = field(init=False, default="parallel_safe")
    timeout_s: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if (
            not callable(getattr(self.store, "put_text", None))
            or not callable(getattr(self.store, "read_text", None))
        ):
            raise TypeError("store must implement ArtifactStore")
        if not isinstance(self.policy, ArtifactPolicy):
            raise TypeError("policy must be an ArtifactPolicy")
        self.parameters = {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The canonical art_<sha256> identifier.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.policy.read_chunk_chars,
                    "default": self.policy.read_chunk_chars,
                },
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolOutput:
        artifact_id = arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id must be non-empty text")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", self.policy.read_chunk_chars)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        bounded_limit = min(limit, self.policy.read_chunk_chars)
        result = await self.store.read_text(
            artifact_id,
            offset=offset,
            limit=bounded_limit,
        )
        return ToolOutput(
            content=json.dumps(
                {
                    "artifact_id": result.ref.id,
                    "media_type": result.ref.media_type,
                    "size_bytes": result.ref.size_bytes,
                    "size_chars": result.ref.size_chars,
                    "offset": result.offset,
                    "next_offset": result.next_offset,
                    "eof": result.eof,
                    "content": result.content,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={
                "kind": "artifact_slice",
                "artifact_id": result.ref.id,
            },
            allow_externalization=False,
        )


__all__ = ["CurrentTimeTool", "ReadArtifactTool"]
