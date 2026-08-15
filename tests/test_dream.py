from __future__ import annotations

import json

from agent_harness.memory import Dream, InMemoryMemoryStore, MemoryPlanGenerator
from agent_harness.messages import UserMessage
from agent_harness.providers import LLMResponse, ToolCallRequest
from agent_harness.testing import ScriptedProvider


def enqueue(store: InMemoryMemoryStore):
    return store.enqueue(
        session_id="session-1",
        source_leaf_id="entry-1",
        context_summary_id="summary-1",
        covered_from_entry_id="entry-1",
        covered_through_entry_id="entry-1",
        source_entry_ids=("entry-1",),
        messages=(UserMessage(content="请记住，我偏好简洁的中文回答。"),),
    )


def plan_json(*, operations: list[dict] | None = None) -> str:
    return json.dumps(
        {"schema_version": 1, "operations": operations or []},
        ensure_ascii=False,
    )


def add_preference_operation() -> dict:
    return {
        "action": "add",
        "section": "user_preferences",
        "statement": "The user prefers concise Chinese answers.",
        "match": None,
        "reason": "The user explicitly requested this durable preference.",
        "source_entry_ids": ["entry-1"],
    }


async def test_memory_plan_generator_uses_no_tools_and_includes_sources() -> None:
    store = InMemoryMemoryStore()
    entry = enqueue(store)
    provider = ScriptedProvider(
        [LLMResponse(content=plan_json(operations=[add_preference_operation()]))]
    )

    plan = await MemoryPlanGenerator(provider, model="fake-model").generate(
        current_memory="", entries=(entry,)
    )

    assert len(plan.operations) == 1
    assert provider.requests[0].tools == ()
    payload = json.loads(provider.requests[0].messages[0]["content"])
    assert payload["archived_sources"][0]["messages"][0]["entry_id"] == "entry-1"


async def test_dream_empty_plan_advances_cursor_without_creating_memory() -> None:
    store = InMemoryMemoryStore()
    entry = enqueue(store)
    dream = Dream(
        store=store,
        provider=ScriptedProvider([LLMResponse(content=plan_json())]),
        model="fake-model",
    )

    result = await dream.run()

    assert result.stop_reason == "completed"
    assert store.get_dream_cursor() == entry.cursor
    assert store.read_memory() == ""
    assert store.dream_records[-1].plan is not None
    assert store.dream_records[-1].plan.operations == ()


async def test_dream_applies_plan_through_isolated_edit_tool() -> None:
    store = InMemoryMemoryStore()
    entry = enqueue(store)
    provider = ScriptedProvider(
        [
            LLMResponse(content=plan_json(operations=[add_preference_operation()])),
            LLMResponse(
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCallRequest(
                        id="edit-1",
                        name="edit",
                        arguments={
                            "path": "MEMORY.md",
                            "edits": [
                                {
                                    "oldText": "## User Preferences\n",
                                    "newText": (
                                        "## User Preferences\n\n"
                                        "- The user prefers concise Chinese answers.\n"
                                    ),
                                }
                            ],
                        },
                    ),
                ),
            ),
            LLMResponse(content="Memory update complete."),
        ]
    )
    dream = Dream(store=store, provider=provider, model="fake-model")

    result = await dream.run()

    assert result.stop_reason == "completed"
    assert "prefers concise Chinese answers" in store.read_memory()
    assert store.get_dream_cursor() == entry.cursor
    assert len(provider.requests) == 3
    assert {tool["name"] for tool in provider.requests[1].tools} == {"read", "edit"}
    assert store.dream_records[-1].changes


async def test_dream_failure_does_not_advance_cursor() -> None:
    store = InMemoryMemoryStore()
    enqueue(store)
    dream = Dream(
        store=store,
        provider=ScriptedProvider([RuntimeError("offline")]),
        model="fake-model",
    )

    result = await dream.run()

    assert result.stop_reason == "analysis_failed"
    assert store.get_dream_cursor() == 0
    assert store.dream_records[-1].error is not None


async def test_dream_noops_without_pending_entries() -> None:
    provider = ScriptedProvider([])
    result = await Dream(
        store=InMemoryMemoryStore(), provider=provider, model="fake-model"
    ).run()

    assert result.did_work is False
    assert result.stop_reason == "no_pending_memory"
    assert provider.requests == []
