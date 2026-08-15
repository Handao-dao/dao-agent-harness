from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent_harness.memory import (
    DreamRunRecord,
    MemoryCodec,
    MemoryCodecError,
    MemoryInboxEntry,
    MemoryOperation,
    MemoryPlan,
    MemoryPlanParser,
    MemoryPlanValidationError,
    memory_inbox_id,
)
from agent_harness.messages import RuntimeStatusMessage, UserMessage
from agent_harness.status import RuntimeEnvironmentStatus, RuntimeStatusSnapshot


def make_entry() -> MemoryInboxEntry:
    return MemoryInboxEntry(
        cursor=1,
        session_id="session-1",
        source_leaf_id="entry-2",
        context_summary_id="summary-1",
        covered_from_entry_id="entry-1",
        covered_through_entry_id="entry-2",
        source_entry_ids=("entry-1", "entry-2"),
        messages=(UserMessage(content="remember this"), UserMessage(content="and this")),
    )


def make_plan() -> MemoryPlan:
    return MemoryPlan(
        schema_version=1,
        operations=(
            MemoryOperation(
                action="add",
                section="user_preferences",
                statement="The user prefers concise answers.",
                match=None,
                reason="The user explicitly confirmed the preference.",
                source_entry_ids=("entry-1",),
            ),
        ),
    )


def test_memory_inbox_id_is_deterministic() -> None:
    assert memory_inbox_id("summary-1") == memory_inbox_id("summary-1")
    assert memory_inbox_id("summary-1") != memory_inbox_id("summary-2")


def test_memory_inbox_entry_rejects_runtime_status() -> None:
    status = RuntimeStatusMessage(
        snapshot=RuntimeStatusSnapshot(
            schema_version=1,
            environment=RuntimeEnvironmentStatus(
                current_time=datetime(2026, 8, 15, 12, tzinfo=UTC),
                timezone="UTC",
            ),
        ),
        content="status",
    )
    with pytest.raises(ValueError, match="RuntimeStatusMessage"):
        MemoryInboxEntry(
            cursor=1,
            session_id="session-1",
            source_leaf_id="entry-1",
            context_summary_id="summary-1",
            covered_from_entry_id="entry-1",
            covered_through_entry_id="entry-1",
            source_entry_ids=("entry-1",),
            messages=(status,),
        )


def test_memory_codec_round_trips_inbox_entry() -> None:
    codec = MemoryCodec()
    original = make_entry()

    restored = codec.decode_inbox_entry(codec.encode_inbox_entry(original))

    assert restored == original


def test_memory_plan_parser_is_strict_and_checks_batch_sources() -> None:
    codec = MemoryCodec()
    parser = MemoryPlanParser(codec)
    raw = codec.canonical_json(make_plan())

    parsed = parser.parse(raw, allowed_source_entry_ids=frozenset({"entry-1"}))

    assert parsed == make_plan()
    with pytest.raises(MemoryPlanValidationError, match="outside the current batch"):
        parser.parse(raw, allowed_source_entry_ids=frozenset({"entry-2"}))


def test_memory_plan_parser_rejects_extra_fields() -> None:
    payload = MemoryCodec().encode_plan(make_plan())
    payload["unexpected"] = True

    with pytest.raises(MemoryPlanValidationError, match="unexpected unexpected"):
        MemoryPlanParser().parse(json.dumps(payload))


def test_memory_operation_match_contract() -> None:
    with pytest.raises(ValueError, match="requires match"):
        MemoryOperation(
            action="remove",
            section="stable_facts",
            statement="Remove stale fact.",
            match=None,
            reason="It is no longer true.",
            source_entry_ids=("entry-1",),
        )


def test_dream_run_record_codec_round_trip() -> None:
    codec = MemoryCodec()
    original = DreamRunRecord(
        first_cursor=1,
        last_cursor=2,
        source_inbox_ids=("inbox-1", "inbox-2"),
        plan=make_plan(),
        stop_reason="completed",
        changes=("edit: memory/MEMORY.md",),
    )

    restored = codec.decode_dream_record(codec.encode_dream_record(original))

    assert restored == original


def test_memory_codec_rejects_non_monotonic_dream_range() -> None:
    codec = MemoryCodec()
    payload = codec.encode_dream_record(
        DreamRunRecord(
            first_cursor=1,
            last_cursor=2,
            source_inbox_ids=("inbox-1",),
            plan=None,
            stop_reason="analysis_failed",
            error="provider failed",
        )
    )
    payload["first_cursor"] = 3

    with pytest.raises(MemoryCodecError, match="cannot precede|must not precede"):
        codec.decode_dream_record(payload)
