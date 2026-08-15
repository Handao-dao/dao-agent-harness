"""Strong types and storage contract for externally persisted tool results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

ARTIFACT_ID_PREFIX = "art_"
TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"
_ARTIFACT_ID_PATTERN = re.compile(r"art_([0-9a-f]{64})\Z")


class ArtifactStoreError(RuntimeError):
    """Base error raised when an artifact cannot be stored or read safely."""


class InvalidArtifactIdError(ArtifactStoreError):
    """Raised when a caller supplies a non-canonical artifact identifier."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact identifier has no stored object."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when stored content does not match its content-addressed identity."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    id: str
    media_type: str
    size_bytes: int
    size_chars: int
    sha256: str

    def __post_init__(self) -> None:
        digest = artifact_digest(self.id)
        if self.sha256 != digest:
            raise ValueError("ArtifactRef sha256 must match its id")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("ArtifactRef media_type must be non-empty text")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("ArtifactRef size_bytes must be a non-negative integer")
        if not isinstance(self.size_chars, int) or self.size_chars < 0:
            raise ValueError("ArtifactRef size_chars must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ArtifactSlice:
    ref: ArtifactRef
    content: str
    offset: int
    next_offset: int
    eof: bool

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("ArtifactSlice offset must be non-negative")
        if self.next_offset < self.offset:
            raise ValueError("ArtifactSlice next_offset cannot precede offset")


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    externalize_above_chars: int = 16_000
    preview_head_chars: int = 2_000
    preview_tail_chars: int = 2_000
    read_chunk_chars: int = 4_000

    def __post_init__(self) -> None:
        for name in (
            "externalize_above_chars",
            "preview_head_chars",
            "preview_tail_chars",
            "read_chunk_chars",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.read_chunk_chars > self.externalize_above_chars:
            raise ValueError("read_chunk_chars cannot exceed externalize_above_chars")
        if self.preview_head_chars + self.preview_tail_chars >= self.externalize_above_chars:
            raise ValueError("combined preview size must be below externalize_above_chars")


class ArtifactStore(Protocol):
    async def put_text(self, content: str) -> ArtifactRef: ...

    async def read_text(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4_000,
    ) -> ArtifactSlice: ...


def artifact_digest(artifact_id: str) -> str:
    if not isinstance(artifact_id, str):
        raise InvalidArtifactIdError("artifact_id must be text")
    match = _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
    if match is None:
        raise InvalidArtifactIdError("artifact_id must use the canonical art_<sha256> format")
    return match.group(1)


def validate_read_range(offset: int, limit: int) -> None:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")


__all__ = [
    "ARTIFACT_ID_PREFIX",
    "TEXT_MEDIA_TYPE",
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactPolicy",
    "ArtifactRef",
    "ArtifactSlice",
    "ArtifactStore",
    "ArtifactStoreError",
    "InvalidArtifactIdError",
    "artifact_digest",
    "validate_read_range",
]
