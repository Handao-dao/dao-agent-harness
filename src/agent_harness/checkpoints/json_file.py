"""Atomic local JSON persistence for the latest ContextCheckpoint per Session."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

from agent_harness.checkpoints.base import (
    CheckpointCorruptError,
    CheckpointStorageError,
    ContextCheckpoint,
)
from agent_harness.checkpoints.codec import CheckpointCodec, CheckpointCodecError


class JsonFileCheckpointStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        codec: CheckpointCodec | None = None,
    ) -> None:
        self._directory = Path(directory)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CheckpointStorageError("Cannot create CheckpointStore directory") from exc
        if not self._directory.is_dir():
            raise ValueError("Checkpoint storage path must be a directory")
        self._codec = codec or CheckpointCodec()
        self._lock = RLock()

    def load(self, session_id: str) -> ContextCheckpoint | None:
        self._validate_session_id(session_id)
        with self._lock:
            path = self._checkpoint_path(session_id)
            if not path.exists():
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    document = json.load(handle)
                checkpoint = self._codec.decode(document)
            except (OSError, UnicodeError, json.JSONDecodeError, CheckpointCodecError) as exc:
                raise CheckpointCorruptError(
                    f"Cannot load Checkpoint file: {path.name}"
                ) from exc
            if checkpoint.session_id != session_id:
                raise CheckpointCorruptError(
                    f"Checkpoint file identity mismatch: {path.name}"
                )
            return checkpoint

    def save(self, checkpoint: ContextCheckpoint) -> None:
        if not isinstance(checkpoint, ContextCheckpoint):
            raise TypeError("checkpoint must be a ContextCheckpoint")
        with self._lock:
            try:
                document = self._codec.encode(checkpoint)
                serialized = json.dumps(
                    document,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            except (CheckpointCodecError, TypeError, ValueError) as exc:
                raise CheckpointStorageError("Checkpoint cannot be encoded") from exc
            self._write_atomic(
                self._checkpoint_path(checkpoint.session_id), f"{serialized}\n"
            )

    def delete(self, session_id: str) -> bool:
        self._validate_session_id(session_id)
        with self._lock:
            try:
                self._checkpoint_path(session_id).unlink()
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise CheckpointStorageError(
                    f"Cannot delete Checkpoint for Session {session_id!r}"
                ) from exc
            return True

    def _write_atomic(self, target: Path, content: str) -> None:
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._sync_directory()
        except OSError as exc:
            raise CheckpointStorageError(
                f"Cannot save Checkpoint file: {target.name}"
            ) from exc
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    def _checkpoint_path(self, session_id: str) -> Path:
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.checkpoint.json"

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty text")


__all__ = ["JsonFileCheckpointStore"]
