from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from agent_harness.artifacts import (
    ArtifactPolicy,
    ArtifactStoreError,
    InMemoryArtifactStore,
)
from agent_harness.messages import ToolCall
from agent_harness.testing import FakeTool
from agent_harness.tools import (
    ToolExecutionPolicy,
    ToolOutput,
    ToolRegistry,
)


def validated_tool() -> FakeTool:
    return FakeTool(
        name="search",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "exact": {"type": "boolean"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def small_artifact_policy() -> ArtifactPolicy:
    return ArtifactPolicy(
        externalize_above_chars=10,
        preview_head_chars=3,
        preview_tail_chars=4,
        read_chunk_chars=5,
    )


def test_prepare_call_resolves_casts_and_validates_arguments() -> None:
    registry = ToolRegistry()
    tool = validated_tool()
    registry.register(tool)

    prepared_tool, arguments, error = registry.prepare_call(
        "search",
        {"query": "nanobot", "limit": "3", "exact": "true", "tags": [1, "agent"]},
    )

    assert prepared_tool is tool
    assert arguments == {
        "query": "nanobot",
        "limit": 3,
        "exact": True,
        "tags": ["1", "agent"],
    }
    assert error is None


def test_prepare_call_reports_unknown_tool() -> None:
    registry = ToolRegistry()
    registry.register(validated_tool())

    tool, arguments, error = registry.prepare_call("missing", {"value": 1})

    assert tool is None
    assert arguments == {"value": 1}
    assert error == "Tool 'missing' not found. Available: search"


def test_prepare_call_reports_all_schema_errors() -> None:
    registry = ToolRegistry()
    registry.register(validated_tool())

    tool, arguments, error = registry.prepare_call(
        "search",
        {"limit": 0, "tags": ["a", "b", "c"], "unexpected": True},
    )

    assert tool is not None
    assert arguments["limit"] == 0
    assert error is not None
    assert "missing required query" in error
    assert "unexpected parameter unexpected" in error
    assert "limit must be >= 1" in error
    assert "tags must contain at most 2 items" in error


def test_definitions_do_not_expose_mutable_tool_schema() -> None:
    registry = ToolRegistry()
    tool = validated_tool()
    registry.register(tool)

    definitions = registry.definitions()
    definitions[0]["parameters"]["required"].clear()

    assert tool.parameters["required"] == ["query"]


async def test_execute_call_normalizes_success_and_reported_error() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="json", result={"answer": 42}))
    registry.register(
        FakeTool(
            name="reported",
            result=ToolOutput(content="Error: unavailable", is_error=True),
        )
    )

    completed = await registry.execute_call(ToolCall(id="call-1", name="json"))
    failed = await registry.execute_call(ToolCall(id="call-2", name="reported"))

    assert completed.status == "completed"
    assert completed.content == '{"answer": 42}'
    assert completed.error_code is None
    assert failed.status == "failed"
    assert failed.error_code == "reported_error"
    assert failed.is_error is True


async def test_plain_error_prefixed_text_is_not_inferred_as_failure() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="text", result="Error: appears in source text"))

    result = await registry.execute_call(ToolCall(id="call-1", name="text"))

    assert result.status == "completed"
    assert result.error_code is None


async def test_execute_call_externalizes_only_oversized_successes() -> None:
    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        artifact_store=store,
        artifact_policy=small_artifact_policy(),
    )
    registry.register(FakeTool(name="large", result="HEADmiddleTAIL"))
    registry.register(FakeTool(name="boundary", result="1234567890"))

    externalized = await registry.execute_call(ToolCall(id="call-1", name="large"))
    boundary = await registry.execute_call(ToolCall(id="call-2", name="boundary"))

    assert externalized.status == "completed"
    assert len(externalized.artifact_refs) == 1
    assert "[tool result externalized]" in externalized.content
    assert "--- head preview ---\nHEA" in externalized.content
    assert "--- 7 chars omitted ---" in externalized.content
    assert externalized.content.endswith("TAIL")
    restored = await store.read_text(externalized.artifact_refs[0].id)
    assert restored.content == "HEADmiddleTAIL"

    assert boundary.status == "completed"
    assert boundary.content == "1234567890"
    assert boundary.artifact_refs == ()


async def test_tool_supplied_model_view_keeps_complete_content_as_artifact() -> None:
    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        artifact_store=store,
        artifact_policy=ArtifactPolicy(externalize_above_chars=10_000),
    )
    registry.register(
        FakeTool(
            name="tests",
            result=ToolOutput(
                content="128 tests passed; 2 failed.",
                artifact_content="complete test log",
                metadata={"kind": "test_report"},
            ),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="tests"))

    assert result.status == "completed"
    assert result.content.startswith("128 tests passed; 2 failed.")
    assert "[complete tool result stored as artifact]" in result.content
    assert len(result.artifact_refs) == 1
    assert result.metadata == {"kind": "test_report"}
    restored = await store.read_text(result.artifact_refs[0].id)
    assert restored.content == "complete test log"


def test_tool_output_allows_error_artifacts_but_requires_externalization() -> None:
    output = ToolOutput(
        content="failed",
        artifact_content="complete failure log",
        is_error=True,
    )

    assert output.is_error is True
    assert output.artifact_content == "complete failure log"

    with pytest.raises(ValueError, match="must allow externalization"):
        ToolOutput(
            content="summary",
            artifact_content="complete result",
            allow_externalization=False,
        )


async def test_explicit_artifact_requires_an_artifact_store() -> None:
    registry = ToolRegistry()
    registry.register(
        FakeTool(
            name="report",
            result=ToolOutput(
                content="bounded summary",
                artifact_content="complete result",
            ),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="report"))

    assert result.status == "failed"
    assert result.error_code == "artifact_store"
    assert result.content.startswith("bounded summary")


class FailingArtifactStore:
    def __init__(self, message: str = "sensitive backend detail") -> None:
        self.message = message
        self.put_calls = 0

    async def put_text(self, content: str) -> Any:
        self.put_calls += 1
        raise ArtifactStoreError(self.message)

    async def read_text(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4_000,
    ) -> Any:
        raise AssertionError("not used")


async def test_reported_errors_are_not_externalized() -> None:
    store = FailingArtifactStore()
    registry = ToolRegistry(
        artifact_store=store,
        artifact_policy=small_artifact_policy(),
    )
    content = "Error: service returned a deliberately long failure"
    registry.register(
        FakeTool(
            name="reported",
            result=ToolOutput(content=content, is_error=True),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="reported"))

    assert result.status == "failed"
    assert result.error_code == "reported_error"
    assert result.content == content
    assert result.artifact_refs == ()
    assert store.put_calls == 0


async def test_reported_error_can_preserve_complete_output_as_artifact() -> None:
    store = InMemoryArtifactStore()
    registry = ToolRegistry(
        artifact_store=store,
        artifact_policy=small_artifact_policy(),
    )
    registry.register(
        FakeTool(
            name="command",
            result=ToolOutput(
                content="tests failed; showing final lines",
                artifact_content="complete failure log",
                is_error=True,
                metadata={"exit_code": 1},
            ),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="command"))

    assert result.status == "failed"
    assert result.error_code == "reported_error"
    assert len(result.artifact_refs) == 1
    assert "[complete tool result stored as artifact]" in result.content
    assert result.metadata == {"exit_code": 1}
    restored = await store.read_text(result.artifact_refs[0].id)
    assert restored.content == "complete failure log"


async def test_error_artifact_requires_a_working_store() -> None:
    registry = ToolRegistry()
    registry.register(
        FakeTool(
            name="command",
            result=ToolOutput(
                content="bounded failure",
                artifact_content="complete failure log",
                is_error=True,
            ),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="command"))

    assert result.status == "failed"
    assert result.error_code == "artifact_store"
    assert result.artifact_refs == ()
    assert result.content.startswith("bounded failure")


async def test_artifact_failure_returns_bounded_error_without_backend_details() -> None:
    store = FailingArtifactStore()
    policy = ArtifactPolicy(
        externalize_above_chars=100,
        preview_head_chars=10,
        preview_tail_chars=10,
        read_chunk_chars=50,
    )
    content = "H" * 10 + "M" * 980 + "T" * 10
    registry = ToolRegistry(artifact_store=store, artifact_policy=policy)
    registry.register(FakeTool(name="large", result=content))

    result = await registry.execute_call(ToolCall(id="call-1", name="large"))

    assert result.status == "failed"
    assert result.error_code == "artifact_store"
    assert result.artifact_refs == ()
    assert "[tool result unavailable]" in result.content
    assert "H" * 10 in result.content
    assert result.content.endswith("T" * 10)
    assert "980 chars omitted" in result.content
    assert store.message not in result.content
    assert len(result.content) < len(content)


async def test_explicit_artifact_failure_keeps_only_the_bounded_model_view() -> None:
    store = FailingArtifactStore()
    registry = ToolRegistry(
        artifact_store=store,
        artifact_policy=small_artifact_policy(),
    )
    registry.register(
        FakeTool(
            name="report",
            result=ToolOutput(
                content="bounded summary",
                artifact_content="SECRET FULL OUTPUT",
            ),
        )
    )

    result = await registry.execute_call(ToolCall(id="call-1", name="report"))

    assert result.status == "failed"
    assert result.error_code == "artifact_store"
    assert result.content.startswith("bounded summary")
    assert "complete tool result unavailable" in result.content
    assert "SECRET FULL OUTPUT" not in result.content


async def test_execute_call_classifies_lookup_and_argument_failures() -> None:
    registry = ToolRegistry()
    registry.register(validated_tool())

    missing = await registry.execute_call(ToolCall(id="call-1", name="missing"))
    invalid = await registry.execute_call(
        ToolCall(id="call-2", name="search", arguments={})
    )

    assert (missing.status, missing.error_code) == ("failed", "not_found")
    assert (invalid.status, invalid.error_code) == ("failed", "invalid_arguments")


class WaitingTool:
    name = "wait"
    description = "Wait until released"
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    execution_mode = "parallel_safe"

    def __init__(self, *, timeout_s: float | None) -> None:
        self.timeout_s = timeout_s
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def execute(self, arguments: Mapping[str, Any]) -> str:
        self.started.set()
        try:
            await self.release.wait()
            return "released"
        finally:
            self.cleaned.set()


async def test_tool_timeout_overrides_registry_default_and_cleans_up() -> None:
    tool = WaitingTool(timeout_s=0.01)
    registry = ToolRegistry(ToolExecutionPolicy(default_timeout_s=10))
    registry.register(tool)

    result = await registry.execute_call(ToolCall(id="call-1", name="wait"))

    assert result.status == "timed_out"
    assert result.error_code == "timeout"
    assert "0.01 seconds" in result.content
    assert tool.cleaned.is_set()


async def test_registry_default_timeout_applies_when_tool_has_none() -> None:
    tool = WaitingTool(timeout_s=None)
    registry = ToolRegistry(ToolExecutionPolicy(default_timeout_s=0.01))
    registry.register(tool)

    result = await registry.execute_call(ToolCall(id="call-1", name="wait"))

    assert result.status == "timed_out"
    assert tool.cleaned.is_set()


async def test_tool_raised_timeout_error_is_an_exception_not_harness_timeout() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="raises", error=TimeoutError("remote timeout")))

    result = await registry.execute_call(ToolCall(id="call-1", name="raises"))

    assert result.status == "failed"
    assert result.error_code == "exception"
    assert result.content == "TimeoutError: remote timeout"


async def test_external_cancellation_propagates_and_cleans_up_tool() -> None:
    tool = WaitingTool(timeout_s=10)
    registry = ToolRegistry()
    registry.register(tool)
    task = asyncio.create_task(
        registry.execute_call(ToolCall(id="call-1", name="wait"))
    )
    await tool.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tool.cleaned.is_set()


def test_execution_policy_and_tool_timeout_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ToolExecutionPolicy(default_timeout_s=0)

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Tool timeout"):
        registry.register(WaitingTool(timeout_s=-1))
