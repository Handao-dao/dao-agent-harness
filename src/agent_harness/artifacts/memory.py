"""In-memory ArtifactStore for tests and embedded runtimes."""

from __future__ import annotations

from hashlib import sha256

from agent_harness.artifacts.base import (
    ARTIFACT_ID_PREFIX,
    TEXT_MEDIA_TYPE,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactSlice,
    artifact_digest,
    validate_read_range,
)


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._contents: dict[str, bytes] = {}

    async def put_text(self, content: str) -> ArtifactRef:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        encoded = content.encode("utf-8", errors="strict")
        digest = sha256(encoded).hexdigest()
        artifact_id = f"{ARTIFACT_ID_PREFIX}{digest}"
        self._contents.setdefault(digest, encoded)
        return build_text_ref(artifact_id, digest, encoded, content)

    async def read_text(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4_000,
    ) -> ArtifactSlice:
        digest = artifact_digest(artifact_id)
        validate_read_range(offset, limit)
        try:
            encoded = self._contents[digest]
        except KeyError as exc:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}") from exc
        content = encoded.decode("utf-8", errors="strict")
        return slice_text(build_text_ref(artifact_id, digest, encoded, content), content, offset, limit)


def build_text_ref(artifact_id: str, digest: str, encoded: bytes, content: str) -> ArtifactRef:
    return ArtifactRef(
        id=artifact_id,
        media_type=TEXT_MEDIA_TYPE,
        size_bytes=len(encoded),
        size_chars=len(content),
        sha256=digest,
    )


def slice_text(ref: ArtifactRef, content: str, offset: int, limit: int) -> ArtifactSlice:
    start = min(offset, len(content))
    end = min(start + limit, len(content))
    return ArtifactSlice(
        ref=ref,
        content=content[start:end],
        offset=start,
        next_offset=end,
        eof=end == len(content),
    )


__all__ = ["InMemoryArtifactStore"]
