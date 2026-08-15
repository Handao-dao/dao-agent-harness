from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agent_harness.summary import (
    ContextSummary,
    ContextSummaryCodec,
    ContextSummaryContent,
    ContextSummaryOutputError,
    ContextSummaryParser,
    SummaryArtifact,
    SummaryDecision,
)


def valid_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "objective": "构建轻量 Agent Harness",
        "status": "active",
        "user_constraints": ["优先参考 nanobot 的成熟设计"],
        "established_facts": ["Session 使用 Entry Tree"],
        "decisions": [
            {
                "decision": "摘要使用结构化 ContextSummary",
                "rationale": "与任务恢复用 ContextCheckpoint 区分",
            }
        ],
        "completed_work": ["完成消息模型"],
        "current_work": ["设计 Consolidator"],
        "next_steps": ["实现摘要事件"],
        "artifacts": [
            {
                "reference": "docs/session-persistence-design.md",
                "description": "Session 持久化设计",
                "state": "modified",
            }
        ],
        "unresolved_questions": [],
        "known_issues": [],
        "continuation_note": None,
    }


def test_parser_builds_strong_summary_content_from_one_json_object() -> None:
    parser = ContextSummaryParser()

    content = parser.parse(json.dumps(valid_document(), ensure_ascii=False))

    assert content.objective == "构建轻量 Agent Harness"
    assert content.status == "active"
    assert content.decisions == (
        SummaryDecision(
            decision="摘要使用结构化 ContextSummary",
            rationale="与任务恢复用 ContextCheckpoint 区分",
        ),
    )
    assert content.artifacts == (
        SummaryArtifact(
            reference="docs/session-persistence-design.md",
            description="Session 持久化设计",
            state="modified",
        ),
    )


def test_empty_structured_summary_uses_the_same_protocol() -> None:
    document = valid_document()
    document.update(
        objective=None,
        status="unclear",
        user_constraints=[],
        established_facts=[],
        decisions=[],
        completed_work=[],
        current_work=[],
        next_steps=[],
        artifacts=[],
        unresolved_questions=[],
        known_issues=[],
        continuation_note=None,
    )

    content = ContextSummaryParser().parse(json.dumps(document))

    assert content == ContextSummaryContent(schema_version=1, objective=None, status="unclear")


@pytest.mark.parametrize(
    "raw, error",
    [
        ("```json\n{}\n```", "not strict JSON"),
        ('{"schema_version":1,"schema_version":1}', "duplicate key"),
        ('{"schema_version":NaN}', "unsupported constant"),
    ],
)
def test_parser_rejects_non_strict_json(raw: str, error: str) -> None:
    with pytest.raises(ContextSummaryOutputError, match=error):
        ContextSummaryParser().parse(raw)


def test_parser_rejects_missing_and_unknown_fields() -> None:
    missing = valid_document()
    del missing["status"]
    with pytest.raises(ContextSummaryOutputError, match="missing required fields: status"):
        ContextSummaryParser().parse(json.dumps(missing))

    unknown = valid_document()
    unknown["summary"] = "free text"
    with pytest.raises(ContextSummaryOutputError, match="unknown fields: summary"):
        ContextSummaryParser().parse(json.dumps(unknown))


def test_parser_does_not_coerce_types_or_accept_unknown_enum_values() -> None:
    wrong_version = valid_document()
    wrong_version["schema_version"] = "1"
    with pytest.raises(ContextSummaryOutputError, match="schema_version must equal 1"):
        ContextSummaryParser().parse(json.dumps(wrong_version))

    wrong_status = valid_document()
    wrong_status["status"] = "running"
    with pytest.raises(ContextSummaryOutputError, match="status must be one of"):
        ContextSummaryParser().parse(json.dumps(wrong_status))


def test_codec_round_trips_and_emits_canonical_json() -> None:
    codec = ContextSummaryCodec()
    original = ContextSummaryParser(codec).parse(
        json.dumps(valid_document(), ensure_ascii=False)
    )

    encoded = codec.encode_content(original)
    restored = codec.decode_content(encoded)

    assert restored == original
    canonical = codec.canonical_json(original)
    assert '": ' not in canonical
    assert ', "' not in canonical


def test_context_summary_metadata_is_harness_owned_and_validated() -> None:
    content = ContextSummaryParser().parse(json.dumps(valid_document(), ensure_ascii=False))
    summary = ContextSummary(
        id="summary-1",
        session_id="session-1",
        covered_through_entry_id="entry-2",
        source_leaf_id="entry-4",
        previous_summary_id=None,
        content=content,
        tokens_before=12_000,
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
    )

    assert summary.content is content
    with pytest.raises(ValueError, match="positive integer"):
        ContextSummary(
            session_id="session-1",
            covered_through_entry_id="entry-2",
            source_leaf_id="entry-4",
            content=content,
            tokens_before=0,
        )
