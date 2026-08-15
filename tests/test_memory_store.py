from __future__ import annotations

from collections.abc import Callable

import pytest

from agent_harness.memory import (
    DreamRunRecord,
    InMemoryMemoryStore,
    LocalMemoryStore,
    MemoryStoreError,
)
from agent_harness.messages import UserMessage


@pytest.fixture(params=["memory", "local"])
def store(request: pytest.FixtureRequest, tmp_path):
    factories: dict[str, Callable[[], object]] = {
        "memory": InMemoryMemoryStore,
        "local": lambda: LocalMemoryStore(tmp_path / "memory"),
    }
    return factories[request.param]()


def enqueue(store, *, summary_id: str = "summary-1"):
    return store.enqueue(
        session_id="session-1",
        source_leaf_id="entry-2",
        context_summary_id=summary_id,
        covered_from_entry_id="entry-1",
        covered_through_entry_id="entry-2",
        source_entry_ids=("entry-1", "entry-2"),
        messages=(UserMessage(content="one"), UserMessage(content="two")),
    )


def test_store_enqueue_is_idempotent(store) -> None:
    first = enqueue(store)
    second = enqueue(store)

    assert second == first
    assert store.read_pending(after_cursor=0, limit=10) == (first,)


def test_store_cursor_advances_only_to_known_entry(store) -> None:
    entry = enqueue(store)

    with pytest.raises(MemoryStoreError, match="Inbox entry"):
        store.advance_dream_cursor(entry.cursor + 1)

    store.advance_dream_cursor(entry.cursor)
    assert store.get_dream_cursor() == entry.cursor
    assert store.read_pending(after_cursor=entry.cursor, limit=10) == ()


def test_store_persists_memory_and_dream_log(store) -> None:
    entry = enqueue(store)
    store.write_memory("# Long-term Memory\n\n- durable\n")
    record = DreamRunRecord(
        first_cursor=entry.cursor,
        last_cursor=entry.cursor,
        source_inbox_ids=(entry.id,),
        plan=None,
        stop_reason="analysis_failed",
        error="provider failed",
    )
    store.append_dream_record(record)

    assert store.read_memory().endswith("- durable\n")
    if isinstance(store, InMemoryMemoryStore):
        assert store.dream_records == (record,)
    else:
        assert store.read_dream_records() == (record,)


def test_local_store_recovers_after_restart(tmp_path) -> None:
    directory = tmp_path / "memory"
    first_store = LocalMemoryStore(directory)
    first = enqueue(first_store)
    first_store.advance_dream_cursor(first.cursor)

    restored = LocalMemoryStore(directory)
    duplicate = enqueue(restored)
    second = enqueue(restored, summary_id="summary-2")

    assert duplicate == first
    assert second.cursor == first.cursor + 1
    assert restored.get_dream_cursor() == first.cursor


def test_local_store_ignores_unterminated_tail(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "memory")
    first = enqueue(store)
    with store.inbox_path.open("ab") as handle:
        handle.write(b'{"partial":')

    assert store.read_pending(after_cursor=0, limit=10) == (first,)
