"""Strong runtime-status snapshots and deterministic Provider-facing rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Literal, Protocol

ToolAnomalyKind = Literal["repeated_identical_call", "repeated_failure"]
ContextPressureMode = Literal["pressure", "emergency"]


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentStatus:
    """Environment facts that the model cannot reliably infer from the transcript."""

    current_time: datetime
    timezone: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.current_time, datetime)
            or self.current_time.tzinfo is None
            or self.current_time.utcoffset() is None
        ):
            raise ValueError("RuntimeEnvironmentStatus.current_time must be timezone-aware")
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("RuntimeEnvironmentStatus.timezone must be non-empty text")


@dataclass(frozen=True, slots=True)
class ToolAnomalyStatus:
    """One actionable repeated-tool anomaly observed in the current execution trace."""

    kind: ToolAnomalyKind
    tool_name: str
    occurrences: int

    def __post_init__(self) -> None:
        if self.kind not in {"repeated_identical_call", "repeated_failure"}:
            raise ValueError(f"Unsupported tool anomaly kind: {self.kind!r}")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("ToolAnomalyStatus.tool_name must be non-empty text")
        if type(self.occurrences) is not int or self.occurrences < 2:
            raise ValueError("ToolAnomalyStatus.occurrences must be an integer >= 2")


@dataclass(frozen=True, slots=True)
class ContextVisibilityStatus:
    """Losses introduced only in the current Provider-facing context view."""

    mode: ContextPressureMode
    tool_results_compacted: int = 0
    history_messages_omitted: int = 0
    active_turn_messages_omitted: int = 0
    runtime_statuses_omitted: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"pressure", "emergency"}:
            raise ValueError(f"Unsupported context pressure mode: {self.mode!r}")
        fields = (
            "tool_results_compacted",
            "history_messages_omitted",
            "active_turn_messages_omitted",
            "runtime_statuses_omitted",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in fields):
            raise ValueError("Context visibility counts must be non-negative integers")
        if not any(getattr(self, name) for name in fields):
            raise ValueError("ContextVisibilityStatus must describe at least one loss")
        if self.mode == "pressure" and (
            self.history_messages_omitted or self.active_turn_messages_omitted
        ):
            raise ValueError("Omitted conversation messages require emergency mode")


@dataclass(frozen=True, slots=True)
class RuntimeStatusSnapshot:
    """One immutable status view captured immediately before a model decision."""

    schema_version: int
    environment: RuntimeEnvironmentStatus
    tool_anomalies: tuple[ToolAnomalyStatus, ...] = ()
    context_visibility: ContextVisibilityStatus | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RuntimeStatusSnapshot.schema_version must be 1")
        if not isinstance(self.environment, RuntimeEnvironmentStatus):
            raise TypeError("RuntimeStatusSnapshot.environment is invalid")
        anomalies = tuple(self.tool_anomalies)
        if any(not isinstance(item, ToolAnomalyStatus) for item in anomalies):
            raise TypeError("RuntimeStatusSnapshot.tool_anomalies contains invalid values")
        if self.context_visibility is not None and not isinstance(
            self.context_visibility, ContextVisibilityStatus
        ):
            raise TypeError("RuntimeStatusSnapshot.context_visibility is invalid")
        object.__setattr__(self, "tool_anomalies", anomalies)


class RuntimeStatusRenderer(Protocol):
    """Render a stable model-facing representation of a status snapshot."""

    @property
    def profile(self) -> str: ...

    def render(self, snapshot: RuntimeStatusSnapshot) -> str: ...


class DefaultRuntimeStatusRenderer:
    """Render compact XML carried through the Provider's standard user role."""

    profile = "dao-default-v1"

    def render(self, snapshot: RuntimeStatusSnapshot) -> str:
        if not isinstance(snapshot, RuntimeStatusSnapshot):
            raise TypeError("snapshot must be a RuntimeStatusSnapshot")
        environment = snapshot.environment
        lines = [
            '<dao_runtime_status version="1" source="harness" authority="metadata">',
            "  <environment>",
            f"    <current_time>{escape(environment.current_time.isoformat())}</current_time>",
            f"    <timezone>{escape(environment.timezone)}</timezone>",
            "  </environment>",
        ]
        if snapshot.tool_anomalies:
            lines.append("  <tool_anomalies>")
            for anomaly in snapshot.tool_anomalies:
                lines.append(
                    "    <anomaly "
                    f'kind="{escape(anomaly.kind)}" '
                    f'tool="{escape(anomaly.tool_name)}" '
                    f'occurrences="{anomaly.occurrences}" />'
                )
            lines.append("  </tool_anomalies>")
        visibility = snapshot.context_visibility
        if visibility is not None:
            lines.extend(
                (
                    f'  <context_visibility mode="{escape(visibility.mode)}">',
                    "    <tool_results_compacted>"
                    f"{visibility.tool_results_compacted}"
                    "</tool_results_compacted>",
                    "    <history_messages_omitted>"
                    f"{visibility.history_messages_omitted}"
                    "</history_messages_omitted>",
                    "    <active_turn_messages_omitted>"
                    f"{visibility.active_turn_messages_omitted}"
                    "</active_turn_messages_omitted>",
                    "    <runtime_statuses_omitted>"
                    f"{visibility.runtime_statuses_omitted}"
                    "</runtime_statuses_omitted>",
                    "  </context_visibility>",
                )
            )
        lines.append("</dao_runtime_status>")
        return "\n".join(lines)


__all__ = [
    "ContextPressureMode",
    "ContextVisibilityStatus",
    "DefaultRuntimeStatusRenderer",
    "RuntimeEnvironmentStatus",
    "RuntimeStatusRenderer",
    "RuntimeStatusSnapshot",
    "ToolAnomalyKind",
    "ToolAnomalyStatus",
]
