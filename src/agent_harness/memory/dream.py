"""Two-phase long-term memory consolidation inspired by nanobot Dream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files as package_files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_harness.memory.codec import (
    MemoryCodec,
    MemoryPlanOutputError,
    MemoryPlanParser,
)
from agent_harness.memory.models import DreamRunRecord, MemoryInboxEntry, MemoryPlan
from agent_harness.memory.store import MEMORY_TEMPLATE, MemoryStore
from agent_harness.messages import AssistantMessage, ToolResultMessage, UserMessage, utc_now
from agent_harness.providers.base import LLMProvider, LLMResponse
from agent_harness.runner import AgentRunner, AgentRunResult, AgentRunSpec
from agent_harness.storage.codec import SessionCodec
from agent_harness.tools.edit import EditTool
from agent_harness.tools.read import ReadTool
from agent_harness.tools.registry import ToolRegistry


class MemoryPlanGenerationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DreamConfig:
    max_batch_size: int = 20
    max_turns: int = 10
    memory_preview_chars: int = 32_000
    entry_preview_chars: int = 4_000
    max_memory_chars: int = 128_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_batch_size",
            "max_turns",
            "memory_preview_chars",
            "entry_preview_chars",
            "max_memory_chars",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DreamResult:
    did_work: bool
    stop_reason: str
    record: DreamRunRecord | None = None


class MemoryPlanGenerator:
    """Generate one strict MemoryPlan and allow one protocol repair."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        parser: MemoryPlanParser | None = None,
        codec: MemoryCodec | None = None,
        config: DreamConfig | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty text")
        self._provider = provider
        self._model = model
        self._codec = codec or MemoryCodec()
        self._parser = parser or MemoryPlanParser(self._codec)
        self._config = config or DreamConfig()
        self._session_codec = SessionCodec()
        self._system_prompt = _load_template("memory_plan.md")

    async def generate(
        self,
        *,
        current_memory: str,
        entries: Sequence[MemoryInboxEntry],
    ) -> MemoryPlan:
        entries = tuple(entries)
        if not entries:
            raise ValueError("entries must not be empty")
        allowed_ids = frozenset(
            source_id for entry in entries for source_id in entry.source_entry_ids
        )
        request = UserMessage(content=self._build_input(current_memory, entries))
        messages = (request,)
        response = await self._complete(messages)
        raw = response.content or ""
        try:
            return self._parser.parse(raw, allowed_source_entry_ids=allowed_ids)
        except MemoryPlanOutputError as first_error:
            repair_messages = (
                *messages,
                AssistantMessage(content=raw[:4_000] or "(empty output)"),
                UserMessage(content=self._repair_instruction(first_error)),
            )
            repaired = await self._complete(repair_messages)
            try:
                return self._parser.parse(
                    repaired.content or "", allowed_source_entry_ids=allowed_ids
                )
            except MemoryPlanOutputError as second_error:
                raise MemoryPlanGenerationError(
                    f"MemoryPlan repair failed: {second_error}"
                ) from second_error

    async def _complete(self, messages: Sequence[Any]) -> LLMResponse:
        from agent_harness.context import ContextBuilder

        try:
            response = await self._provider.complete(
                model=self._model,
                system_prompt=self._system_prompt,
                messages=ContextBuilder.build_messages(messages),
                tools=(),
            )
        except Exception as exc:
            raise MemoryPlanGenerationError(
                f"MemoryPlan Provider call failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.finish_reason == "error":
            raise MemoryPlanGenerationError(
                response.content or "MemoryPlan Provider returned an error"
            )
        if response.tool_calls:
            raise MemoryPlanGenerationError("MemoryPlan Provider returned unexpected tool calls")
        return response

    def _build_input(
        self,
        current_memory: str,
        entries: Sequence[MemoryInboxEntry],
    ) -> str:
        memory = _truncate(current_memory or MEMORY_TEMPLATE, self._config.memory_preview_chars)
        sources = []
        for entry in entries:
            messages = []
            for source_id, message in zip(entry.source_entry_ids, entry.messages):
                encoded = self._session_codec.encode_message(message)
                content = encoded.get("content")
                if isinstance(content, str):
                    encoded["content"] = _truncate(content, self._config.entry_preview_chars)
                messages.append({"entry_id": source_id, "message": encoded})
            sources.append(
                {
                    "inbox_id": entry.id,
                    "cursor": entry.cursor,
                    "session_id": entry.session_id,
                    "context_summary_id": entry.context_summary_id,
                    "messages": messages,
                }
            )
        return json.dumps(
            {
                "current_date": utc_now().date().isoformat(),
                "current_memory": memory,
                "archived_sources": sources,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _repair_instruction(error: MemoryPlanOutputError) -> str:
        return (
            "Your previous response violated the MemoryPlan JSON protocol: "
            f"{error}. Return only one corrected complete JSON object."
        )


class Dream:
    """Consume durable Inbox entries and atomically update Workspace memory."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        config: DreamConfig | None = None,
        generator: MemoryPlanGenerator | None = None,
        runner: AgentRunner | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty text")
        self.store = store
        self.provider = provider
        self.model = model
        self.config = config or DreamConfig()
        self._generator = generator or MemoryPlanGenerator(
            provider, model=model, config=self.config
        )
        self._runner = runner or AgentRunner(provider)
        self._lock = asyncio.Lock()
        self._apply_prompt = _load_template("dream_apply.md")

    async def run(self) -> DreamResult:
        async with self._lock:
            cursor = self.store.get_dream_cursor()
            batch = self.store.read_pending(
                after_cursor=cursor, limit=self.config.max_batch_size
            )
            if not batch:
                return DreamResult(did_work=False, stop_reason="no_pending_memory")
            return await self._run_batch(batch)

    async def _run_batch(
        self, batch: Sequence[MemoryInboxEntry]
    ) -> DreamResult:
        started_at = utc_now()
        try:
            plan = await self._generator.generate(
                current_memory=self.store.read_memory(), entries=batch
            )
        except MemoryPlanGenerationError as exc:
            return self._failed(
                batch,
                started_at=started_at,
                stop_reason="analysis_failed",
                error=str(exc),
            )

        if not plan.operations:
            return self._complete_batch(
                batch, plan=plan, changes=(), started_at=started_at
            )

        try:
            result, updated = await self._apply_plan(plan)
        except asyncio.CancelledError:
            return self._failed(
                batch,
                plan=plan,
                started_at=started_at,
                stop_reason="cancelled",
                error="Dream was cancelled",
            )
        except Exception as exc:
            return self._failed(
                batch,
                plan=plan,
                started_at=started_at,
                stop_reason="execution_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

        if result.status != "completed":
            stop_reason = "limit_reached" if result.status == "limit_reached" else (
                "cancelled" if result.status == "cancelled" else "execution_failed"
            )
            return self._failed(
                batch,
                plan=plan,
                started_at=started_at,
                stop_reason=stop_reason,
                error=result.error or result.stop_reason,
            )

        current = self.store.read_memory()
        if updated != (current or MEMORY_TEMPLATE):
            if len(updated) > self.config.max_memory_chars:
                return self._failed(
                    batch,
                    plan=plan,
                    started_at=started_at,
                    stop_reason="execution_failed",
                    error="Updated MEMORY.md exceeds max_memory_chars",
                )
            self.store.write_memory(updated)
        changes = tuple(
            message.metadata.get("patch", message.content)
            for message in result.messages
            if isinstance(message, ToolResultMessage)
            and message.tool_name == "edit"
            and not message.is_error
        )
        return self._complete_batch(
            batch, plan=plan, changes=changes, started_at=started_at
        )

    async def _apply_plan(self, plan: MemoryPlan) -> tuple[AgentRunResult, str]:
        current = self.store.read_memory() or MEMORY_TEMPLATE
        with TemporaryDirectory(prefix="dao-dream-") as temporary:
            root = Path(temporary)
            path = root / "MEMORY.md"
            path.write_text(current, encoding="utf-8", newline="")
            tools = ToolRegistry()
            tools.register(ReadTool(root))
            tools.register(EditTool(root))
            payload = json.dumps(
                MemoryCodec().encode_plan(plan),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            result = await self._runner.run(
                AgentRunSpec(
                    initial_messages=(UserMessage(content=payload),),
                    tools=tools,
                    model=self.model,
                    system_prompt=self._apply_prompt,
                    max_turns=self.config.max_turns,
                )
            )
            return result, path.read_text(encoding="utf-8")

    def _complete_batch(
        self,
        batch: Sequence[MemoryInboxEntry],
        *,
        plan: MemoryPlan,
        changes: Sequence[str],
        started_at: datetime,
    ) -> DreamResult:
        record = self._record(
            batch,
            plan=plan,
            stop_reason="completed",
            changes=changes,
            started_at=started_at,
        )
        self.store.append_dream_record(record)
        self.store.advance_dream_cursor(batch[-1].cursor)
        self.store.compact_inbox()
        return DreamResult(did_work=True, stop_reason="completed", record=record)

    def _failed(
        self,
        batch: Sequence[MemoryInboxEntry],
        *,
        started_at: datetime,
        stop_reason: str,
        error: str,
        plan: MemoryPlan | None = None,
    ) -> DreamResult:
        record = self._record(
            batch,
            plan=plan,
            stop_reason=stop_reason,
            changes=(),
            error=error,
            started_at=started_at,
        )
        self.store.append_dream_record(record)
        return DreamResult(did_work=True, stop_reason=stop_reason, record=record)

    @staticmethod
    def _record(
        batch: Sequence[MemoryInboxEntry],
        *,
        plan: MemoryPlan | None,
        stop_reason: Any,
        changes: Sequence[str],
        started_at: datetime,
        error: str | None = None,
    ) -> DreamRunRecord:
        return DreamRunRecord(
            first_cursor=batch[0].cursor,
            last_cursor=batch[-1].cursor,
            source_inbox_ids=tuple(entry.id for entry in batch),
            plan=plan,
            stop_reason=stop_reason,
            changes=tuple(change for change in changes if change.strip()),
            error=error,
            started_at=started_at,
            completed_at=utc_now(),
        )


def _load_template(filename: str) -> str:
    return (
        package_files("agent_harness")
        .joinpath("templates", filename)
        .read_text(encoding="utf-8")
        .strip()
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = max(1, limit // 2)
    tail = max(1, limit - head - 32)
    return value[:head] + "\n... [truncated] ...\n" + value[-tail:]


__all__ = [
    "Dream",
    "DreamConfig",
    "DreamResult",
    "MemoryPlanGenerationError",
    "MemoryPlanGenerator",
]
