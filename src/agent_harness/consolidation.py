"""Token-budget ContextSummary generation and durable branch consolidation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files as package_files
from inspect import isawaitable
from typing import Any

from agent_harness.context import ContextBuilder, SessionContextResolver
from agent_harness.memory.store import MemoryStore, MemoryStoreError
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.providers.base import LLMProvider, LLMResponse
from agent_harness.session import MessageEntry, Session
from agent_harness.storage.base import SessionStore
from agent_harness.summary import (
    ContextSummary,
    ContextSummaryCodec,
    ContextSummaryContent,
    ContextSummaryOutputError,
    ContextSummaryParser,
)
from agent_harness.token_estimation import (
    PromptTokenEstimationError,
    PromptTokenEstimator,
)
from agent_harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ContextSummaryGenerationError(RuntimeError):
    """A Provider response could not produce a valid ContextSummaryContent."""


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    context_window_tokens: int
    max_completion_tokens: int = 4096
    safety_buffer_tokens: int = 1024
    proactive_input_reserve_tokens: int = 2048
    consolidation_ratio: float = 0.5
    max_rounds: int = 5

    def __post_init__(self) -> None:
        integer_fields = (
            "context_window_tokens",
            "max_completion_tokens",
            "safety_buffer_tokens",
            "proactive_input_reserve_tokens",
            "max_rounds",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            minimum = 1 if field_name in {"context_window_tokens", "max_rounds"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if not 0 < self.consolidation_ratio < 1:
            raise ValueError("consolidation_ratio must be between zero and one")
        if self.input_budget <= 0:
            raise ValueError("Context window has no usable input token budget")

    @property
    def input_budget(self) -> int:
        return (
            self.context_window_tokens
            - self.max_completion_tokens
            - self.safety_buffer_tokens
        )

    @property
    def target_tokens(self) -> int:
        return int(self.input_budget * self.consolidation_ratio)

    @property
    def effective_proactive_input_reserve_tokens(self) -> int:
        """Keep the proactive threshold usable for deliberately small windows."""

        return min(
            self.proactive_input_reserve_tokens,
            max(0, self.input_budget - 1),
        )


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    summaries_created: tuple[ContextSummary, ...] = ()
    estimated_tokens: int | None = None
    stop_reason: str = "within_budget"
    error: str | None = None


class ContextSummaryGenerator:
    """Ask a text Provider for strict JSON and allow one schema-repair attempt."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        parser: ContextSummaryParser | None = None,
        codec: ContextSummaryCodec | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty text")
        self._provider = provider
        self._model = model
        self._codec = codec or ContextSummaryCodec()
        self._parser = parser or ContextSummaryParser(self._codec)
        self._system_prompt = self._load_prompt()

    async def generate(
        self,
        *,
        previous_summary: ContextSummaryContent | None,
        entries: Sequence[MessageEntry],
    ) -> ContextSummaryContent:
        entries = tuple(
            entry
            for entry in entries
            if not isinstance(entry.message, RuntimeStatusMessage)
        )
        if not entries:
            raise ValueError("entries must not be empty")
        input_json = self._build_input(previous_summary, entries)
        request_messages: list[dict[str, Any]] = [
            {"role": "user", "content": input_json}
        ]

        response = await self._complete(request_messages)
        raw = response.content or ""
        try:
            return self._parser.parse(raw)
        except ContextSummaryOutputError as first_error:
            repair_messages = [
                *request_messages,
                {"role": "assistant", "content": raw[:4000] or "(empty output)"},
                {
                    "role": "user",
                    "content": self._repair_instruction(first_error),
                },
            ]
            repaired = await self._complete(repair_messages)
            try:
                return self._parser.parse(repaired.content or "")
            except ContextSummaryOutputError as second_error:
                raise ContextSummaryGenerationError(
                    f"ContextSummary repair failed: {second_error}"
                ) from second_error

    async def _complete(self, messages: Sequence[Mapping[str, Any]]) -> LLMResponse:
        try:
            response = await self._provider.complete(
                model=self._model,
                system_prompt=self._system_prompt,
                messages=messages,
                tools=(),
            )
        except Exception as exc:
            raise ContextSummaryGenerationError(
                f"ContextSummary Provider call failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.finish_reason == "error":
            raise ContextSummaryGenerationError(
                response.content or "ContextSummary Provider returned an error"
            )
        if response.tool_calls:
            raise ContextSummaryGenerationError(
                "ContextSummary Provider returned unexpected tool calls"
            )
        return response

    def _build_input(
        self,
        previous_summary: ContextSummaryContent | None,
        entries: Sequence[MessageEntry],
    ) -> str:
        payload = {
            "previous_summary": (
                self._codec.encode_content(previous_summary)
                if previous_summary is not None
                else None
            ),
            "new_messages": [self._encode_entry(entry) for entry in entries],
        }
        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ContextSummaryGenerationError(
                "ContextSummary input is not JSON serializable"
            ) from exc

    @staticmethod
    def _encode_entry(entry: MessageEntry) -> dict[str, Any]:
        message = entry.message
        common = {
            "entry_id": entry.id,
            "message_id": message.id,
            "created_at": message.created_at.isoformat(),
        }
        if isinstance(message, UserMessage):
            return {**common, "type": "user", "content": message.content}
        if isinstance(message, AssistantMessage):
            return {
                **common,
                "type": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                    for call in message.tool_calls
                ],
            }
        if isinstance(message, ToolResultMessage):
            return {
                **common,
                "type": "tool_result",
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "content": message.content,
                "is_error": message.is_error,
            }
        raise ContextSummaryGenerationError(
            f"Unsupported summary message type: {type(message).__name__}"
        )

    @staticmethod
    def _repair_instruction(error: ContextSummaryOutputError) -> str:
        return (
            "Your previous output did not satisfy the required schema.\n\n"
            f"Validation error:\n- {error}\n\n"
            "Return the complete corrected JSON object only. Do not include explanations "
            "or Markdown fences."
        )

    @staticmethod
    def _load_prompt() -> str:
        return (
            package_files("agent_harness")
            .joinpath("templates", "context_summary.md")
            .read_text(encoding="utf-8")
            .strip()
        )


class ContextConsolidator:
    """Create durable summaries until the active Provider prompt reaches its target."""

    def __init__(
        self,
        *,
        generator: ContextSummaryGenerator,
        token_estimator: PromptTokenEstimator,
        session_store: SessionStore,
        model: str,
        config: ConsolidationConfig,
        resolver: SessionContextResolver | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty text")
        self._generator = generator
        self._token_estimator = token_estimator
        self._session_store = session_store
        self._model = model
        self._config = config
        self._resolver = resolver or SessionContextResolver()
        self._memory_store = memory_store
        self._locks: dict[str, asyncio.Lock] = {}

    async def maybe_consolidate(
        self,
        session: Session,
        *,
        pending_message: UserMessage | None = None,
        context_builder: ContextBuilder,
        tools: ToolRegistry,
        extra_system_sections: Sequence[str] = (),
    ) -> ConsolidationResult:
        """Consolidate one Session under a shared lock.

        A missing pending message denotes the proactive post-SAVE probe. PREPARE
        supplies the real PendingInput and rechecks the resulting prompt.
        """

        lock = self._locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            current = self._session_store.get_or_create(session.id)
            self._reconcile_memory(current)
            input_reserve_tokens = (
                self._config.effective_proactive_input_reserve_tokens
                if pending_message is None
                else 0
            )
            probe = pending_message or UserMessage(content="[next-turn token probe]")
            return await self._consolidate_locked(
                current,
                pending_message=probe,
                input_reserve_tokens=input_reserve_tokens,
                context_builder=context_builder,
                tools=tools,
                extra_system_sections=extra_system_sections,
            )

    async def _consolidate_locked(
        self,
        session: Session,
        *,
        pending_message: UserMessage,
        input_reserve_tokens: int,
        context_builder: ContextBuilder,
        tools: ToolRegistry,
        extra_system_sections: Sequence[str],
    ) -> ConsolidationResult:
        created: list[ContextSummary] = []
        trigger_budget = self._config.input_budget - input_reserve_tokens
        target_tokens = min(self._config.target_tokens, trigger_budget)
        try:
            estimated = await self._estimate_current_prompt(
                session,
                pending_message=pending_message,
                context_builder=context_builder,
                tools=tools,
                extra_system_sections=extra_system_sections,
            )
        except PromptTokenEstimationError as exc:
            return ConsolidationResult(
                estimated_tokens=None,
                stop_reason="estimation_failed",
                error=str(exc),
            )
        if estimated < trigger_budget:
            return ConsolidationResult(estimated_tokens=estimated)

        for _round in range(self._config.max_rounds):
            if estimated <= target_tokens:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=estimated,
                    stop_reason="target_reached",
                )
            try:
                selection = await self._pick_boundary(
                    session,
                    tokens_to_remove=max(1, estimated - target_tokens),
                )
            except PromptTokenEstimationError as exc:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=estimated,
                    stop_reason="estimation_failed",
                    error=str(exc),
                )
            if selection is None:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=estimated,
                    stop_reason="no_safe_boundary",
                )
            previous, chunk, boundary = selection
            try:
                content = await self._generator.generate(
                    previous_summary=previous.content if previous is not None else None,
                    entries=chunk,
                )
            except ContextSummaryGenerationError as exc:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=estimated,
                    stop_reason="generation_failed",
                    error=str(exc),
                )

            source_leaf_id = session.active_leaf_id
            if source_leaf_id is None:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=estimated,
                    stop_reason="no_safe_boundary",
                )
            summary = ContextSummary(
                session_id=session.id,
                covered_through_entry_id=boundary.id,
                source_leaf_id=source_leaf_id,
                previous_summary_id=previous.id if previous is not None else None,
                content=content,
                tokens_before=estimated,
            )
            session.record_context_summary(summary)
            self._session_store.save(session)
            self._enqueue_memory(summary, chunk)
            created.append(summary)
            try:
                estimated = await self._estimate_current_prompt(
                    session,
                    pending_message=pending_message,
                    context_builder=context_builder,
                    tools=tools,
                    extra_system_sections=extra_system_sections,
                )
            except PromptTokenEstimationError as exc:
                return ConsolidationResult(
                    summaries_created=tuple(created),
                    estimated_tokens=None,
                    stop_reason="estimation_failed",
                    error=str(exc),
                )

        return ConsolidationResult(
            summaries_created=tuple(created),
            estimated_tokens=estimated,
            stop_reason="max_rounds",
        )

    def _enqueue_memory(
        self,
        summary: ContextSummary,
        chunk: Sequence[MessageEntry],
    ) -> None:
        store = self._memory_store
        if store is None:
            return
        source_entries = tuple(
            entry
            for entry in chunk
            if not isinstance(entry.message, RuntimeStatusMessage)
        )
        if not source_entries:
            return
        try:
            store.enqueue(
                session_id=summary.session_id,
                source_leaf_id=summary.source_leaf_id,
                context_summary_id=summary.id,
                covered_from_entry_id=source_entries[0].id,
                covered_through_entry_id=source_entries[-1].id,
                source_entry_ids=tuple(entry.id for entry in source_entries),
                messages=tuple(entry.message for entry in source_entries),
                created_at=summary.created_at,
            )
        except (MemoryStoreError, TypeError, ValueError):
            # ContextSummary is already durable and remains the conversation
            # continuity mechanism. Startup/PREPARE reconciliation can retry
            # the idempotent enqueue without invalidating consolidation.
            logger.exception(
                "Could not enqueue memory source for ContextSummary %s",
                summary.id,
            )

    def _reconcile_memory(self, session: Session) -> None:
        """Repair the Summary-saved / Inbox-not-written crash window."""

        if self._memory_store is None:
            return
        summaries = {summary.id: summary for summary in session.context_summaries}
        for summary in session.context_summaries:
            try:
                branch = session.branch_entries(summary.source_leaf_id)
                positions = {entry.id: index for index, entry in enumerate(branch)}
                end = positions[summary.covered_through_entry_id] + 1
                start = 0
                if summary.previous_summary_id is not None:
                    previous = summaries[summary.previous_summary_id]
                    start = positions[previous.covered_through_entry_id] + 1
                chunk = branch[start:end]
                if chunk:
                    self._enqueue_memory(summary, chunk)
            except (KeyError, ValueError):
                logger.exception(
                    "Could not reconstruct memory source for ContextSummary %s",
                    summary.id,
                )

    async def _pick_boundary(
        self,
        session: Session,
        *,
        tokens_to_remove: int,
    ) -> tuple[ContextSummary | None, tuple[MessageEntry, ...], MessageEntry] | None:
        resolved = self._resolver.resolve(session)
        active_entries = session.active_entries()
        start = resolved.covered_message_count
        if start >= len(active_entries):
            return None
        if not isinstance(active_entries[start].message, UserMessage):
            return None

        removed_tokens = 0
        last_boundary_index: int | None = None
        for index in range(start, len(active_entries)):
            entry = active_entries[index]
            if index > start and isinstance(entry.message, UserMessage):
                last_boundary_index = index
                if removed_tokens >= tokens_to_remove:
                    break
            removed_tokens += await self._estimate_messages((entry.message,))

        if last_boundary_index is None:
            return None
        chunk = tuple(active_entries[start:last_boundary_index])
        if not chunk:
            return None
        return resolved.summary, chunk, chunk[-1]

    async def _estimate_current_prompt(
        self,
        session: Session,
        *,
        pending_message: UserMessage,
        context_builder: ContextBuilder,
        tools: ToolRegistry,
        extra_system_sections: Sequence[str],
    ) -> int:
        resolved = self._resolver.resolve(session)
        active_messages = tuple(entry.message for entry in session.active_entries())
        skill_prefix = context_builder.build_skill_context_prefix(
            active_messages[: resolved.covered_message_count]
        )
        messages: tuple[AgentMessage, ...] = (
            *skill_prefix,
            *resolved.messages,
            pending_message,
        )
        system_prompt = context_builder.build_system_prompt(
            context_summary=(
                resolved.summary.content if resolved.summary is not None else None
            ),
            extra_sections=extra_system_sections,
        )
        try:
            result = self._token_estimator.estimate(
                model=self._model,
                system_prompt=system_prompt,
                messages=ContextBuilder.build_messages(messages),
                tools=tools.definitions(),
            )
            value = await result if isawaitable(result) else result
        except Exception as exc:
            raise PromptTokenEstimationError(
                f"Prompt token estimation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if type(value) is not int or value <= 0:
            raise PromptTokenEstimationError(
                "PromptTokenEstimator must return a positive integer"
            )
        return value

    async def _estimate_messages(self, messages: Sequence[AgentMessage]) -> int:
        try:
            result = self._token_estimator.estimate(
                model=self._model,
                system_prompt=None,
                messages=ContextBuilder.build_messages(messages),
                tools=(),
            )
            value = await result if isawaitable(result) else result
        except Exception as exc:
            raise PromptTokenEstimationError(
                f"Message token estimation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if type(value) is not int or value <= 0:
            raise PromptTokenEstimationError(
                "PromptTokenEstimator must return a positive integer"
            )
        return value


__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "ContextConsolidator",
    "ContextSummaryGenerationError",
    "ContextSummaryGenerator",
    "PromptTokenEstimator",
    "PromptTokenEstimationError",
]
