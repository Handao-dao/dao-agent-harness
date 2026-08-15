from __future__ import annotations

from hashlib import sha256

import pytest

from agent_harness.artifacts import (
    TEXT_MEDIA_TYPE,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPolicy,
    InMemoryArtifactStore,
    InvalidArtifactIdError,
    LocalArtifactStore,
)


async def test_in_memory_store_uses_utf8_content_addressing_and_pagination() -> None:
    store = InMemoryArtifactStore()
    content = "甲乙🙂abc"

    ref = await store.put_text(content)

    digest = sha256(content.encode("utf-8")).hexdigest()
    assert ref.id == f"art_{digest}"
    assert ref.sha256 == digest
    assert ref.media_type == TEXT_MEDIA_TYPE
    assert ref.size_chars == len(content)
    assert ref.size_bytes == len(content.encode("utf-8"))

    first = await store.read_text(ref.id, limit=3)
    assert first.content == "甲乙🙂"
    assert first.offset == 0
    assert first.next_offset == 3
    assert first.eof is False

    second = await store.read_text(ref.id, offset=first.next_offset, limit=10)
    assert second.content == "abc"
    assert second.offset == 3
    assert second.next_offset == len(content)
    assert second.eof is True


async def test_in_memory_store_deduplicates_identical_content() -> None:
    store = InMemoryArtifactStore()

    first = await store.put_text("same")
    second = await store.put_text("same")

    assert first == second


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (False, 1), (0, True)],
)
async def test_store_rejects_invalid_read_ranges(offset: int, limit: int) -> None:
    store = InMemoryArtifactStore()
    ref = await store.put_text("value")

    with pytest.raises(ValueError):
        await store.read_text(ref.id, offset=offset, limit=limit)


async def test_store_rejects_invalid_and_missing_ids() -> None:
    store = InMemoryArtifactStore()

    with pytest.raises(InvalidArtifactIdError):
        await store.read_text("../../secret", limit=1)

    missing_id = f"art_{'0' * 64}"
    with pytest.raises(ArtifactNotFoundError, match="Artifact not found"):
        await store.read_text(missing_id, limit=1)


async def test_read_past_end_returns_empty_eof_slice() -> None:
    store = InMemoryArtifactStore()
    ref = await store.put_text("abc")

    result = await store.read_text(ref.id, offset=50, limit=2)

    assert result.content == ""
    assert result.offset == 3
    assert result.next_offset == 3
    assert result.eof is True


async def test_local_store_persists_in_content_addressed_layout_and_reopens(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = "日志第一行\n日志第二行🙂"

    first = await store.put_text(content)
    second = await store.put_text(content)

    digest = first.sha256
    target = tmp_path / digest[:2] / f"{digest}.txt"
    assert first == second
    assert target.read_text(encoding="utf-8") == content
    assert list(tmp_path.rglob("*.txt")) == [target]
    assert list(tmp_path.rglob("*.tmp")) == []

    restored = await LocalArtifactStore(tmp_path).read_text(first.id, offset=2, limit=4)
    assert restored.ref == first
    assert restored.content == content[2:6]


async def test_local_store_detects_content_corruption(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = await store.put_text("trusted at write time")
    target = tmp_path / ref.sha256[:2] / f"{ref.sha256}.txt"
    target.write_text("changed later", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        await store.read_text(ref.id)

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        await store.put_text("trusted at write time")


def test_artifact_policy_rejects_invalid_or_recursive_read_limits() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ArtifactPolicy(preview_head_chars=0)

    with pytest.raises(ValueError, match="cannot exceed"):
        ArtifactPolicy(externalize_above_chars=100, read_chunk_chars=101)

    with pytest.raises(ValueError, match="combined preview"):
        ArtifactPolicy(
            externalize_above_chars=100,
            preview_head_chars=50,
            preview_tail_chars=50,
            read_chunk_chars=50,
        )
