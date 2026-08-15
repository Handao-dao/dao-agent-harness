"""Ephemeral context repair and budgeting before each Provider request."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from inspect import isawaitable
from typing import Any

from agent_harness.context import ContextBuilder
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.skills import (
    DEFAULT_MAX_ACTIVE_SKILL_CHARS,
    deduplicate_skill_messages,
    is_skill_instruction,
    skill_instruction_indices,
)
from agent_harness.token_estimation import PromptTokenEstimator

_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"


class ContextGovernanceError(RuntimeError):
    """A Provider request context could not be governed safely."""


class ContextWindowExceededError(ContextGovernanceError):
    """The minimum viable task context cannot fit within the input budget."""


@dataclass(frozen=True, slots=True)
class ContextGovernorConfig:
    context_window_tokens: int
    max_completion_tokens: int = 4096
    safety_buffer_tokens: int = 1024
    max_tool_result_chars: int = 16_000
    microcompact_keep_recent: int = 10
    microcompact_min_chars: int = 500
    compactable_tool_names: frozenset[str] = field(default_factory=frozenset)
    max_active_skill_chars: int = DEFAULT_MAX_ACTIVE_SKILL_CHARS

    def __post_init__(self) -> None:
        for name in (
            "context_window_tokens",
            "max_completion_tokens",
            "safety_buffer_tokens",
            "max_tool_result_chars",
            "microcompact_keep_recent",
            "microcompact_min_chars",
            "max_active_skill_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be greater than zero")
        if self.input_budget <= 0:
            raise ValueError("Context window has no usable input token budget")
        if 0 < self.max_tool_result_chars < 32:
            raise ValueError("max_tool_result_chars must be zero or at least 32")
        names = frozenset(self.compactable_tool_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("compactable_tool_names must contain non-empty text")
        object.__setattr__(self, "compactable_tool_names", names)

    @property
    def input_budget(self) -> int:
        return (
            self.context_window_tokens
            - self.max_completion_tokens
            - self.safety_buffer_tokens
        )


@dataclass(frozen=True, slots=True)
class ContextGovernanceReport:
    estimated_tokens_before: int | None
    estimated_tokens_after: int | None
    orphan_results_dropped: int = 0
    missing_results_backfilled: int = 0
    tool_results_compacted: int = 0
    tool_results_truncated: int = 0
    history_messages_snipped: int = 0
    active_turn_messages_snipped: int = 0
    runtime_statuses_dropped: int = 0
    skill_instructions_deduplicated: int = 0
    estimation_source: str | None = None


@dataclass(frozen=True, slots=True)
class GovernedContext:
    messages: tuple[AgentMessage, ...]
    report: ContextGovernanceReport


@dataclass(slots=True)
class _GovernanceStats:
    orphan_results_dropped: int = 0
    missing_results_backfilled: int = 0
    tool_results_compacted: int = 0
    tool_results_truncated: int = 0
    history_messages_snipped: int = 0
    active_turn_messages_snipped: int = 0
    runtime_statuses_dropped: int = 0
    skill_instructions_deduplicated: int = 0


class ContextGovernor:
    """Build one legal, budgeted model view without mutating working history."""

    def __init__(
        self,
        config: ContextGovernorConfig,
        *,
        token_estimator: PromptTokenEstimator | None = None,
    ) -> None:
        if not isinstance(config, ContextGovernorConfig):
            raise TypeError("config must be a ContextGovernorConfig")
        self._config = config
        self._token_estimator = token_estimator

    @property
    def config(self) -> ContextGovernorConfig:
        return self._config

    async def prepare(
        self,
        *,
        messages: Sequence[AgentMessage],
        current_turn_start: int,
        model: str,
        system_prompt: str | None,
        tools: Sequence[Mapping[str, Any]],
    ) -> GovernedContext:
        source = tuple(messages)
        self._validate_request(source, current_turn_start, model)
        stats = _GovernanceStats()

        estimated_before, estimation_source = await self._estimate(
            model=model,
            system_prompt=system_prompt,
            messages=source,
            tools=tools,
        )

        governed, deduplicated = deduplicate_skill_messages(
            source,
            max_chars=self._config.max_active_skill_chars,
        )
        stats.skill_instructions_deduplicated += deduplicated
        governed, dropped = self._drop_orphan_tool_results(governed)
        stats.orphan_results_dropped += dropped
        governed, backfilled = self._backfill_missing_tool_results(governed)
        stats.missing_results_backfilled += backfilled
        governed, truncated = self._truncate_tool_results(governed)
        stats.tool_results_truncated += truncated

        adjusted_turn_start = self._find_current_turn(governed, source[current_turn_start])
        if deduplicated or dropped or backfilled or truncated:
            estimated, estimation_source = await self._estimate(
                model=model,
                system_prompt=system_prompt,
                messages=governed,
                tools=tools,
            )
        else:
            estimated = estimated_before

        if estimated > self._config.input_budget:
            governed, statuses_dropped = self._drop_runtime_statuses(governed)
            stats.runtime_statuses_dropped += statuses_dropped
            if statuses_dropped:
                adjusted_turn_start = self._find_current_turn(
                    governed,
                    source[current_turn_start],
                )
                estimated, estimation_source = await self._estimate(
                    model=model,
                    system_prompt=system_prompt,
                    messages=governed,
                    tools=tools,
                )

        if estimated > self._config.input_budget:
            governed, compacted = self._microcompact(
                governed,
                history_end=adjusted_turn_start,
            )
            stats.tool_results_compacted += compacted
            if compacted:
                adjusted_turn_start = self._find_current_turn(
                    governed,
                    source[current_turn_start],
                )
                estimated, estimation_source = await self._estimate(
                    model=model,
                    system_prompt=system_prompt,
                    messages=governed,
                    tools=tools,
                )

        if estimated > self._config.input_budget:
            (
                governed,
                estimated,
                estimation_source,
                history_snipped,
                active_snipped,
                repair_stats,
            ) = await self._emergency_snip(
                messages=governed,
                current_turn_start=adjusted_turn_start,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
            )
            stats.history_messages_snipped += history_snipped
            stats.active_turn_messages_snipped += active_snipped
            stats.orphan_results_dropped += repair_stats.orphan_results_dropped
            stats.missing_results_backfilled += repair_stats.missing_results_backfilled

        if estimated > self._config.input_budget:
            raise ContextWindowExceededError(
                "Minimum task context exceeds the configured input token budget "
                f"({estimated} > {self._config.input_budget})"
            )

        return GovernedContext(
            messages=governed,
            report=ContextGovernanceReport(
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated,
                orphan_results_dropped=stats.orphan_results_dropped,
                missing_results_backfilled=stats.missing_results_backfilled,
                tool_results_compacted=stats.tool_results_compacted,
                tool_results_truncated=stats.tool_results_truncated,
                history_messages_snipped=stats.history_messages_snipped,
                active_turn_messages_snipped=stats.active_turn_messages_snipped,
                runtime_statuses_dropped=stats.runtime_statuses_dropped,
                skill_instructions_deduplicated=(
                    stats.skill_instructions_deduplicated
                ),
                estimation_source=estimation_source,
            ),
        )

    @staticmethod
    def _validate_request(
        messages: tuple[AgentMessage, ...],
        current_turn_start: int,
        model: str,
    ) -> None:
        if not messages:
            raise ContextGovernanceError("messages must not be empty")
        if (
            type(current_turn_start) is not int
            or current_turn_start < 0
            or current_turn_start >= len(messages)
        ):
            raise ContextGovernanceError("current_turn_start must index messages")
        if not isinstance(messages[current_turn_start], UserMessage):
            raise ContextGovernanceError("current_turn_start must point to a UserMessage")
        if not isinstance(model, str) or not model.strip():
            raise ContextGovernanceError("model must be non-empty text")

    @staticmethod
    def _drop_orphan_tool_results(
        messages: Sequence[AgentMessage],
    ) -> tuple[tuple[AgentMessage, ...], int]:
        declared: set[str] = set()
        updated: list[AgentMessage] = []
        dropped = 0
        for message in messages:
            if isinstance(message, AssistantMessage):
                declared.update(call.id for call in message.tool_calls)
            if (
                isinstance(message, ToolResultMessage)
                and message.tool_call_id not in declared
            ):
                dropped += 1
                continue
            updated.append(message)
        return tuple(updated), dropped

    @staticmethod
    def _backfill_missing_tool_results(
        messages: Sequence[AgentMessage],
    ) -> tuple[tuple[AgentMessage, ...], int]:
        fulfilled = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolResultMessage)
        }
        updated: list[AgentMessage] = []
        backfilled = 0
        for message in messages:
            updated.append(message)
            if not isinstance(message, AssistantMessage):
                continue
            for call in message.tool_calls:
                if call.id in fulfilled:
                    continue
                updated.append(
                    ToolResultMessage(
                        id=f"context_backfill_{call.id}",
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=_BACKFILL_CONTENT,
                        is_error=True,
                    )
                )
                fulfilled.add(call.id)
                backfilled += 1
        return tuple(updated), backfilled

    @staticmethod
    def _drop_runtime_statuses(
        messages: Sequence[AgentMessage],
    ) -> tuple[tuple[AgentMessage, ...], int]:
        """Remove expired status metadata only after the request exceeds its budget."""

        updated = tuple(
            message for message in messages if not isinstance(message, RuntimeStatusMessage)
        )
        return updated, len(messages) - len(updated)

    def _microcompact(
        self,
        messages: Sequence[AgentMessage],
        *,
        history_end: int,
    ) -> tuple[tuple[AgentMessage, ...], int]:
        eligible = [
            index
            for index, message in enumerate(messages[:history_end])
            if isinstance(message, ToolResultMessage)
            and message.tool_name in self._config.compactable_tool_names
            and not is_skill_instruction(message)
        ]
        stale_count = max(0, len(eligible) - self._config.microcompact_keep_recent)
        stale = set(eligible[:stale_count])
        if not stale:
            return tuple(messages), 0

        updated = list(messages)
        compacted = 0
        for index in stale:
            message = updated[index]
            if (
                not isinstance(message, ToolResultMessage)
                or len(message.content) < self._config.microcompact_min_chars
            ):
                continue
            updated[index] = replace(
                message,
                content=f"[{message.tool_name} result omitted from model context]",
            )
            compacted += 1
        return tuple(updated), compacted

    def _truncate_tool_results(
        self,
        messages: Sequence[AgentMessage],
    ) -> tuple[tuple[AgentMessage, ...], int]:
        limit = self._config.max_tool_result_chars
        if limit <= 0:
            return tuple(messages), 0
        updated = list(messages)
        truncated = 0
        for index, message in enumerate(updated):
            if (
                not isinstance(message, ToolResultMessage)
                or is_skill_instruction(message)
                or len(message.content) <= limit
            ):
                continue
            updated[index] = replace(
                message,
                content=self._truncate_head_tail(message.content, limit),
            )
            truncated += 1
        return tuple(updated), truncated

    @staticmethod
    def _truncate_head_tail(content: str, max_chars: int) -> str:
        marker_template = "\n...[{omitted} chars omitted]...\n"
        marker = marker_template.format(omitted=len(content))
        if len(marker) >= max_chars:
            return marker[:max_chars]
        available = max_chars - len(marker)
        head_chars = (available + 1) // 2
        tail_chars = available - head_chars
        omitted = len(content) - head_chars - tail_chars
        marker = marker_template.format(omitted=omitted)
        available = max_chars - len(marker)
        head_chars = (available + 1) // 2
        tail_chars = available - head_chars
        tail = content[-tail_chars:] if tail_chars else ""
        return content[:head_chars] + marker + tail

    async def _emergency_snip(
        self,
        *,
        messages: tuple[AgentMessage, ...],
        current_turn_start: int,
        model: str,
        system_prompt: str | None,
        tools: Sequence[Mapping[str, Any]],
    ) -> tuple[
        tuple[AgentMessage, ...],
        int,
        str,
        int,
        int,
        _GovernanceStats,
    ]:
        spans = self._legal_spans(messages)
        candidate_starts = tuple(
            start
            for start, _end in spans
            if start >= current_turn_start or isinstance(messages[start], UserMessage)
        )
        if not candidate_starts:
            raise ContextGovernanceError("No legal emergency context boundary exists")

        latest_user_index = max(
            index
            for index, message in enumerate(messages[current_turn_start:], current_turn_start)
            if isinstance(message, UserMessage)
        )
        anchor_indices = frozenset(
            (
                current_turn_start,
                latest_user_index,
                *skill_instruction_indices(messages),
            )
        )

        async def estimate_start(start: int) -> tuple[int, str]:
            candidate, _selected = self._build_emergency_candidate(
                messages,
                start=start,
                anchor_indices=anchor_indices,
            )
            return await self._estimate(
                model=model,
                system_prompt=system_prompt,
                messages=candidate,
                tools=tools,
            )

        minimal_estimate, _minimal_source = await estimate_start(candidate_starts[-1])
        if minimal_estimate > self._config.input_budget:
            raise ContextWindowExceededError(
                "System prompt, tool definitions, task anchors, and the most recent "
                "message block exceed the configured input token budget "
                f"({self._config.input_budget})"
            )

        low = 0
        high = len(candidate_starts) - 1
        while low < high:
            middle = (low + high) // 2
            estimate, _source = await estimate_start(candidate_starts[middle])
            if estimate <= self._config.input_budget:
                high = middle
            else:
                low = middle + 1

        selected_start_position = low
        while selected_start_position < len(candidate_starts):
            selected_start = candidate_starts[selected_start_position]
            candidate, selected_indices = self._build_emergency_candidate(
                messages,
                start=selected_start,
                anchor_indices=anchor_indices,
            )
            candidate, dropped = self._drop_orphan_tool_results(candidate)
            candidate, backfilled = self._backfill_missing_tool_results(candidate)
            estimate, source = await self._estimate(
                model=model,
                system_prompt=system_prompt,
                messages=candidate,
                tools=tools,
            )
            if estimate <= self._config.input_budget:
                history_snipped = sum(
                    index not in selected_indices
                    for index in range(current_turn_start)
                )
                active_snipped = sum(
                    index not in selected_indices
                    for index in range(current_turn_start, len(messages))
                )
                return (
                    candidate,
                    estimate,
                    source,
                    history_snipped,
                    active_snipped,
                    _GovernanceStats(
                        orphan_results_dropped=dropped,
                        missing_results_backfilled=backfilled,
                    ),
                )
            selected_start_position += 1

        raise ContextWindowExceededError(
            "Minimum task context exceeds the configured input token budget "
            f"({self._config.input_budget})"
        )

    @staticmethod
    def _legal_spans(
        messages: Sequence[AgentMessage],
    ) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            end = index + 1
            if isinstance(message, AssistantMessage) and message.tool_calls:
                while end < len(messages) and isinstance(
                    messages[end], ToolResultMessage
                ):
                    end += 1
            spans.append((index, end))
            index = end
        return tuple(spans)

    @staticmethod
    def _build_emergency_candidate(
        messages: tuple[AgentMessage, ...],
        *,
        start: int,
        anchor_indices: frozenset[int],
    ) -> tuple[tuple[AgentMessage, ...], frozenset[int]]:
        selected = frozenset((*anchor_indices, *range(start, len(messages))))
        return (
            tuple(message for index, message in enumerate(messages) if index in selected),
            selected,
        )

    @staticmethod
    def _find_current_turn(
        messages: Sequence[AgentMessage],
        current_user: AgentMessage,
    ) -> int:
        for index, message in enumerate(messages):
            if message is current_user or message.id == current_user.id:
                return index
        raise ContextGovernanceError("Context governance removed the protected current turn")

    async def _estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[AgentMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> tuple[int, str]:
        projected = ContextBuilder.build_messages(messages)
        if self._token_estimator is not None:
            try:
                result = self._token_estimator.estimate(
                    model=model,
                    system_prompt=system_prompt,
                    messages=projected,
                    tools=tools,
                )
                value = await result if isawaitable(result) else result
                if type(value) is int and value > 0:
                    return value, type(self._token_estimator).__name__
            except Exception:
                pass
        return self._estimate_characters(system_prompt, projected, tools), "character_heuristic"

    @staticmethod
    def _estimate_characters(
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        try:
            payload = json.dumps(
                {
                    "system_prompt": system_prompt,
                    "messages": [dict(message) for message in messages],
                    "tools": [dict(tool) for tool in tools],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError) as exc:
            raise ContextGovernanceError(
                "Provider request is not serializable for context estimation"
            ) from exc
        ascii_chars = sum(character.isascii() for character in payload)
        non_ascii_chars = len(payload) - ascii_chars
        return max(1, math.ceil(ascii_chars / 4) + non_ascii_chars)


__all__ = [
    "ContextGovernanceError",
    "ContextGovernanceReport",
    "ContextGovernor",
    "ContextGovernorConfig",
    "ContextWindowExceededError",
    "GovernedContext",
]
