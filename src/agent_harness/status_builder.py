"""Build runtime-status messages from execution trace and context governance facts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime

from agent_harness.context_governor import ContextGovernanceReport
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolResultMessage,
)
from agent_harness.status import (
    ContextVisibilityStatus,
    DefaultRuntimeStatusRenderer,
    RuntimeEnvironmentStatus,
    RuntimeStatusRenderer,
    RuntimeStatusSnapshot,
    ToolAnomalyStatus,
)


class RuntimeStatusBuilder:
    """Create one deterministic status message immediately before a model request."""

    def __init__(
        self,
        renderer: RuntimeStatusRenderer | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        identical_call_threshold: int = 2,
        repeated_failure_threshold: int = 2,
    ) -> None:
        for name, value in (
            ("identical_call_threshold", identical_call_threshold),
            ("repeated_failure_threshold", repeated_failure_threshold),
        ):
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        self._renderer = renderer or DefaultRuntimeStatusRenderer()
        self._now = now or (lambda: datetime.now().astimezone())
        self._identical_call_threshold = identical_call_threshold
        self._repeated_failure_threshold = repeated_failure_threshold

    def build(
        self,
        messages: Sequence[AgentMessage],
        *,
        governance_report: ContextGovernanceReport | None = None,
    ) -> RuntimeStatusMessage:
        current_time = self._now()
        if (
            not isinstance(current_time, datetime)
            or current_time.tzinfo is None
            or current_time.utcoffset() is None
        ):
            raise ValueError("RuntimeStatusBuilder now() must return a timezone-aware datetime")
        timezone = current_time.tzname() or self._format_offset(current_time)
        snapshot = RuntimeStatusSnapshot(
            schema_version=1,
            environment=RuntimeEnvironmentStatus(
                current_time=current_time,
                timezone=timezone,
            ),
            tool_anomalies=self._tool_anomalies(messages),
            context_visibility=self._context_visibility(governance_report),
        )
        return RuntimeStatusMessage(
            snapshot=snapshot,
            content=self._renderer.render(snapshot),
            render_profile=self._renderer.profile,
            created_at=current_time,
        )

    def _tool_anomalies(
        self,
        messages: Sequence[AgentMessage],
    ) -> tuple[ToolAnomalyStatus, ...]:
        signatures: Counter[tuple[str, str]] = Counter()
        failures: Counter[str] = Counter()
        for message in messages:
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    signature = json.dumps(
                        dict(call.arguments),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    signatures[(call.name, signature)] += 1
            elif isinstance(message, ToolResultMessage) and message.is_error:
                failures[message.tool_name] += 1

        anomalies = [
            ToolAnomalyStatus(
                kind="repeated_identical_call",
                tool_name=tool_name,
                occurrences=count,
            )
            for (tool_name, _signature), count in sorted(signatures.items())
            if count >= self._identical_call_threshold
        ]
        anomalies.extend(
            ToolAnomalyStatus(
                kind="repeated_failure",
                tool_name=tool_name,
                occurrences=count,
            )
            for tool_name, count in sorted(failures.items())
            if count >= self._repeated_failure_threshold
        )
        return tuple(anomalies)

    @staticmethod
    def _context_visibility(
        report: ContextGovernanceReport | None,
    ) -> ContextVisibilityStatus | None:
        if report is None:
            return None
        emergency = bool(
            report.history_messages_snipped or report.active_turn_messages_snipped
        )
        has_loss = bool(
            report.tool_results_compacted
            or report.history_messages_snipped
            or report.active_turn_messages_snipped
            or report.runtime_statuses_dropped
        )
        if not has_loss:
            return None
        return ContextVisibilityStatus(
            mode="emergency" if emergency else "pressure",
            tool_results_compacted=report.tool_results_compacted,
            history_messages_omitted=report.history_messages_snipped,
            active_turn_messages_omitted=report.active_turn_messages_snipped,
            runtime_statuses_omitted=report.runtime_statuses_dropped,
        )

    @staticmethod
    def _format_offset(value: datetime) -> str:
        offset = value.strftime("%z")
        return f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"


__all__ = ["RuntimeStatusBuilder"]
