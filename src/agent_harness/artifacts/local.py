"""Content-addressed local ArtifactStore with atomic UTF-8 writes."""

from __future__ import annotations

import asyncio
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from agent_harness.artifacts.base import (
    ARTIFACT_ID_PREFIX,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactSlice,
    ArtifactStoreError,
    artifact_digest,
    validate_read_range,
)
from agent_harness.artifacts.memory import build_text_ref, slice_text


class LocalArtifactStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactStoreError("Cannot create ArtifactStore directory") from exc
        if not self._directory.is_dir():
            raise ValueError("Artifact storage path must be a directory")

    async def put_text(self, content: str) -> ArtifactRef:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        return await asyncio.to_thread(self._put_text, content)

    async def read_text(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4_000,
    ) -> ArtifactSlice:
        digest = artifact_digest(artifact_id)
        validate_read_range(offset, limit)
        return await asyncio.to_thread(self._read_text, artifact_id, digest, offset, limit)

    def _put_text(self, content: str) -> ArtifactRef:
        encoded = content.encode("utf-8", errors="strict")
        digest = sha256(encoded).hexdigest()
        artifact_id = f"{ARTIFACT_ID_PREFIX}{digest}"
        target = self._path(digest)

        if target.exists():
            stored = self._read_bytes(target, artifact_id)
            self._verify(stored, digest, artifact_id)
            try:
                stored_content = stored.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ArtifactIntegrityError(f"Artifact is not valid UTF-8: {artifact_id}") from exc
            return build_text_ref(artifact_id, digest, stored, stored_content)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactStoreError(f"Cannot prepare storage for Artifact: {artifact_id}") from exc

        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._sync_directory(target.parent)
        except OSError as exc:
            raise ArtifactStoreError(f"Cannot store Artifact: {artifact_id}") from exc
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

        return build_text_ref(artifact_id, digest, encoded, content)

    def _read_text(
        self,
        artifact_id: str,
        digest: str,
        offset: int,
        limit: int,
    ) -> ArtifactSlice:
        encoded = self._read_bytes(self._path(digest), artifact_id)
        self._verify(encoded, digest, artifact_id)
        try:
            content = encoded.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ArtifactIntegrityError(f"Artifact is not valid UTF-8: {artifact_id}") from exc
        ref = build_text_ref(artifact_id, digest, encoded, content)
        return slice_text(ref, content, offset, limit)

    @staticmethod
    def _read_bytes(path: Path, artifact_id: str) -> bytes:
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}") from exc
        except OSError as exc:
            raise ArtifactStoreError(f"Cannot read Artifact: {artifact_id}") from exc

    @staticmethod
    def _verify(encoded: bytes, expected_digest: str, artifact_id: str) -> None:
        if sha256(encoded).hexdigest() != expected_digest:
            raise ArtifactIntegrityError(f"Artifact content hash mismatch: {artifact_id}")

    def _path(self, digest: str) -> Path:
        return self._directory / digest[:2] / f"{digest}.txt"

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["LocalArtifactStore"]
