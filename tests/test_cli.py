from __future__ import annotations

import asyncio
import json
import threading
from argparse import Namespace
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_harness import cli
from agent_harness.artifacts import LocalArtifactStore
from agent_harness.checkpoints import ContextCheckpoint, JsonFileCheckpointStore
from agent_harness.messages import (
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.providers import LLMResponse, ToolCallRequest
from agent_harness.session import PendingInput
from agent_harness.storage import JsonlSessionStore


class CliProvider:
    script: deque[LLMResponse | Exception]
    requests: list[tuple[dict[str, Any], ...]]
    tool_requests: list[tuple[dict[str, Any], ...]]

    def __init__(self, **config: Any) -> None:
        self.config = config

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        self.requests.append(tuple(dict(message) for message in messages))
        self.tool_requests.append(tuple(dict(tool) for tool in tools))
        item = self.script.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def make_args(tmp_path: Path) -> Namespace:
    return Namespace(
        model="fake-model",
        base_url="http://provider.invalid/v1",
        api_key="test-key",
        system_prompt="test system prompt",
        max_turns=5,
        max_injected_inputs_per_run=5,
        timeout=1.0,
        tool_timeout=1.0,
        context_window_tokens=None,
        max_completion_tokens=4096,
        proactive_input_reserve_tokens=2048,
        max_input_tokens=None,
        max_tool_result_chars=16_000,
        session_id="cli:test",
        session_dir=tmp_path / "sessions",
        artifact_dir=tmp_path / "artifacts",
        checkpoint_dir=tmp_path / "checkpoints",
        workspace=tmp_path,
    )


async def test_cli_routes_two_turns_through_runtime_session(
    tmp_path, monkeypatch, capsys
) -> None:
    CliProvider.script = deque(
        [LLMResponse(content="answer-1"), LLMResponse(content="answer-2")]
    )
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("question-1", "question-2", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(make_args(tmp_path))

    session = JsonlSessionStore(tmp_path / "sessions").get_or_create("cli:test")
    assert result == 0
    conversation = [
        message.content
        for message in session.messages
        if isinstance(message, (UserMessage, AssistantMessage))
    ]
    assert conversation == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
    ]
    assert sum(isinstance(message, RuntimeStatusMessage) for message in session.messages) == 2
    assert CliProvider.requests[1][-1]["role"] == "user"
    assert CliProvider.requests[1][-1]["content"].startswith("question-2\n\n")
    assert "<dao_runtime_status" in CliProvider.requests[1][-1]["content"]
    output = capsys.readouterr().out
    assert "answer-1" in output
    assert "answer-2" in output


async def test_cli_enables_foreground_and_background_consolidation(
    tmp_path, monkeypatch
) -> None:
    class CountingEstimator:
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, **_request: Any) -> int:
            self.calls += 1
            return 1

    estimator = CountingEstimator()
    args = make_args(tmp_path)
    args.context_window_tokens = 10_000
    args.max_completion_tokens = 100
    CliProvider.script = deque([LLMResponse(content="answer")])
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("question", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr(cli, "build_default_token_estimator", lambda _provider: estimator)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(args)

    assert result == 0
    assert estimator.calls == 4


async def test_cli_retry_runs_retained_pending_input(tmp_path, monkeypatch, capsys) -> None:
    CliProvider.script = deque([RuntimeError("offline"), LLMResponse(content="recovered")])
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("question", "/retry", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(make_args(tmp_path))

    session = JsonlSessionStore(tmp_path / "sessions").get_or_create("cli:test")
    assert result == 0
    assert session.pending_inputs == []
    assert [
        message.content
        for message in session.messages
        if isinstance(message, (UserMessage, AssistantMessage))
    ] == ["question", "recovered"]
    assert sum(isinstance(message, RuntimeStatusMessage) for message in session.messages) == 1
    output = capsys.readouterr().out
    assert "Run failed: RuntimeError: offline" in output
    assert "recovered" in output


async def test_cli_registers_read_artifact_and_reads_persisted_content(
    tmp_path, monkeypatch
) -> None:
    args = make_args(tmp_path)
    artifact_store = LocalArtifactStore(args.artifact_dir)
    ref = await artifact_store.put_text("0123456789")
    CliProvider.script = deque(
        [
            LLMResponse(
                tool_calls=(
                    ToolCallRequest(
                        id="call-1",
                        name="read_artifact",
                        arguments={"artifact_id": ref.id, "offset": 2, "limit": 4},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="artifact inspected"),
        ]
    )
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("inspect it", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(args)

    session = JsonlSessionStore(args.session_dir).get_or_create(args.session_id)
    tool_result = next(
        message for message in session.messages if isinstance(message, ToolResultMessage)
    )
    payload = json.loads(tool_result.content)
    assert result == 0
    assert payload["content"] == "2345"
    assert payload["offset"] == 2
    assert payload["next_offset"] == 6
    assert {tool["name"] for tool in CliProvider.tool_requests[0]} == {
        "bash",
        "edit",
        "find",
        "get_current_time",
        "grep",
        "ls",
        "read",
        "read_artifact",
        "activate_skill",
        "read_skill_resource",
        "write",
    }


async def test_cli_retry_restores_final_checkpoint_without_provider_call(
    tmp_path, monkeypatch
) -> None:
    args = make_args(tmp_path)
    session_store = JsonlSessionStore(args.session_dir)
    session = session_store.get_or_create(args.session_id)
    pending = session.enqueue(
        PendingInput(source_message_id="external-checkpoint", content="question")
    )
    session_store.save(session)
    final = AssistantMessage(content="durable candidate")
    checkpoint_store = JsonFileCheckpointStore(args.checkpoint_dir)
    checkpoint_store.save(
        ContextCheckpoint(
            session_id=args.session_id,
            input_id=pending.id,
            input_revision=pending.revision,
            base_leaf_id=None,
            save_cursor=0,
            phase="final_response",
            model=args.model,
            next_model_turn=1,
            messages=(pending.to_user_message(), final),
            terminal_status="completed",
            stop_reason="model_stop",
            final_content=final.content,
        )
    )
    CliProvider.script = deque([])
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("/retry", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(args)

    restored = JsonlSessionStore(args.session_dir).get_or_create(args.session_id)
    assert result == 0
    assert CliProvider.requests == []
    assert [message.content for message in restored.messages] == [
        "question",
        "durable candidate",
    ]
    assert restored.pending_inputs == []
    assert checkpoint_store.load(args.session_id) is None


async def test_cli_clear_deletes_session_and_checkpoint(tmp_path, monkeypatch) -> None:
    args = make_args(tmp_path)
    session_store = JsonlSessionStore(args.session_dir)
    session = session_store.get_or_create(args.session_id)
    pending = session.enqueue(PendingInput(source_message_id="external-1", content="question"))
    session_store.save(session)
    final = AssistantMessage(content="answer")
    checkpoint_store = JsonFileCheckpointStore(args.checkpoint_dir)
    checkpoint_store.save(
        ContextCheckpoint(
            session_id=args.session_id,
            input_id=pending.id,
            input_revision=pending.revision,
            base_leaf_id=None,
            save_cursor=0,
            phase="final_response",
            model=args.model,
            next_model_turn=1,
            messages=(pending.to_user_message(), final),
            terminal_status="completed",
            stop_reason="model_stop",
            final_content="answer",
        )
    )
    CliProvider.script = deque([])
    CliProvider.requests = []
    CliProvider.tool_requests = []
    inputs = iter(("/clear", "/exit"))
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = await cli.run_chat(args)

    cleared = JsonlSessionStore(args.session_dir).get_or_create(args.session_id)
    assert result == 0
    assert cleared.pending_inputs == []
    assert cleared.messages == []
    assert checkpoint_store.load(args.session_id) is None


async def test_cli_pause_edits_latest_input_and_restarts(tmp_path, monkeypatch, capsys) -> None:
    started = threading.Event()

    class PauseCliProvider:
        requests: list[tuple[dict[str, Any], ...]] = []

        def __init__(self, **config: Any) -> None:
            self.calls = 0

        async def complete(
            self,
            *,
            model: str,
            system_prompt: str | None,
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]],
        ) -> LLMResponse:
            self.calls += 1
            self.requests.append(tuple(dict(message) for message in messages))
            if self.calls == 1:
                started.set()
                await asyncio.Event().wait()
            return LLMResponse(content="revised answer")

    values = iter(("wrong question", "/pause", "correct question", "/exit"))

    def next_input(_prompt: str) -> str:
        value = next(values)
        if value == "/pause":
            assert started.wait(timeout=1)
        return value

    monkeypatch.setattr(cli, "OpenAICompatibleProvider", PauseCliProvider)
    monkeypatch.setattr("builtins.input", next_input)

    result = await cli.run_chat(make_args(tmp_path))

    session = JsonlSessionStore(tmp_path / "sessions").get_or_create("cli:test")
    assert result == 0
    assert [
        message.content
        for message in session.messages
        if isinstance(message, (UserMessage, AssistantMessage))
    ] == [
        "correct question",
        "revised answer",
    ]
    assert PauseCliProvider.requests[-1][-1]["role"] == "user"
    assert PauseCliProvider.requests[-1][-1]["content"].startswith(
        "correct question\n\n"
    )
    assert "<dao_runtime_status" in PauseCliProvider.requests[-1][-1]["content"]
    assert "Run paused for message revision." in capsys.readouterr().out
