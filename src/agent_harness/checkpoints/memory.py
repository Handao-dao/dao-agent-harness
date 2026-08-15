"""In-memory CheckpointStore."""

from __future__ import annotations

from agent_harness.checkpoints.base import ContextCheckpoint


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, ContextCheckpoint] = {}

    def load(self, session_id: str) -> ContextCheckpoint | None:
        self._validate_session_id(session_id)
        return self._checkpoints.get(session_id)

    def save(self, checkpoint: ContextCheckpoint) -> None:
        if not isinstance(checkpoint, ContextCheckpoint):
            raise TypeError("checkpoint must be a ContextCheckpoint")
        self._checkpoints[checkpoint.session_id] = checkpoint

    def delete(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        return self._checkpoints.pop(session_id, None) is not None

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty text")


__all__ = ["InMemoryCheckpointStore"]
