"""Versioned JSON codec for strongly typed Session projections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from agent_harness.artifacts import ArtifactRef, ArtifactStoreError
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    RuntimeStatusMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.session import MessageEntry, PendingInput, Session
from agent_harness.status import (
    ContextVisibilityStatus,
    RuntimeEnvironmentStatus,
    RuntimeStatusSnapshot,
    ToolAnomalyStatus,
)
from agent_harness.summary import (
    ContextSummary,
    ContextSummaryCodec,
    ContextSummaryOutputError,
)

SESSION_SCHEMA_VERSION = 6


class SessionCodecError(ValueError):
    """Raised when persisted Session data violates the storage schema."""


class UnsupportedSessionVersionError(SessionCodecError):
    """Raised when persisted data uses an unsupported schema version."""


class SessionCodec:
    """Encode a Session projection and migrate legacy linear snapshots."""

    def __init__(self, summary_codec: ContextSummaryCodec | None = None) -> None:
        self._summary_codec = summary_codec or ContextSummaryCodec()

    def encode(self, session: Session) -> dict[str, Any]:
        if not isinstance(session, Session):
            raise TypeError("session must be a Session")
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "id": session.id,
            "created_at": self._encode_datetime(session.created_at, "Session.created_at"),
            "updated_at": self._encode_datetime(session.updated_at, "Session.updated_at"),
            "entries": [self.encode_entry(entry) for entry in session.entries],
            "active_leaf_id": session.active_leaf_id,
            "pending_inputs": [self.encode_pending(item) for item in session.pending_inputs],
            "context_summaries": [
                self.encode_context_summary(summary) for summary in session.context_summaries
            ],
            "metadata": self._copy_json_value(session.metadata, "Session.metadata"),
        }

    def decode(self, data: Mapping[str, Any]) -> Session:
        root = self._require_mapping(data, "Session document")
        version = root.get("schema_version")
        if type(version) is not int or version not in {
            1,
            2,
            3,
            4,
            5,
            SESSION_SCHEMA_VERSION,
        }:
            raise UnsupportedSessionVersionError(
                f"Unsupported Session schema version: {version!r}"
            )

        session_id = self._require_text(root.get("id"), "Session.id")
        created_at = self._decode_datetime(root.get("created_at"), "Session.created_at")
        updated_at = self._decode_datetime(root.get("updated_at"), "Session.updated_at")
        pending_data = self._require_sequence(
            root.get("pending_inputs"), "Session.pending_inputs"
        )
        pending_inputs = [self.decode_pending(item) for item in pending_data]
        metadata = dict(
            self._copy_json_value(
                self._require_mapping(root.get("metadata"), "Session.metadata"),
                "Session.metadata",
            )
        )

        if version == 1:
            messages_data = self._require_sequence(root.get("messages"), "Session.messages")
            return Session.from_messages(
                id=session_id,
                messages=[self.decode_message(item) for item in messages_data],
                pending_inputs=pending_inputs,
                created_at=created_at,
                updated_at=updated_at,
                metadata=metadata,
            )

        entries_data = self._require_sequence(root.get("entries"), "Session.entries")
        active_leaf_id = root.get("active_leaf_id")
        if active_leaf_id is not None:
            active_leaf_id = self._require_text(active_leaf_id, "Session.active_leaf_id")
        summaries_data = self._require_sequence(
            root.get("context_summaries"), "Session.context_summaries"
        ) if version >= 3 else ()
        return Session(
            id=session_id,
            entries=[self.decode_entry(item) for item in entries_data],
            active_leaf_id=active_leaf_id,
            pending_inputs=pending_inputs,
            context_summaries=[
                self.decode_context_summary(item) for item in summaries_data
            ],
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

    def encode_context_summary(self, summary: ContextSummary) -> dict[str, Any]:
        if not isinstance(summary, ContextSummary):
            raise SessionCodecError("summary must be a ContextSummary")
        return {
            "id": summary.id,
            "session_id": summary.session_id,
            "covered_through_entry_id": summary.covered_through_entry_id,
            "source_leaf_id": summary.source_leaf_id,
            "previous_summary_id": summary.previous_summary_id,
            "tokens_before": summary.tokens_before,
            "created_at": self._encode_datetime(
                summary.created_at, "ContextSummary.created_at"
            ),
            "content": self._summary_codec.encode_content(summary.content),
        }

    def decode_context_summary(self, data: Any) -> ContextSummary:
        item = self._require_mapping(data, "ContextSummary")
        previous_summary_id = item.get("previous_summary_id")
        if previous_summary_id is not None:
            previous_summary_id = self._require_text(
                previous_summary_id, "ContextSummary.previous_summary_id"
            )
        tokens_before = item.get("tokens_before")
        if type(tokens_before) is not int or tokens_before <= 0:
            raise SessionCodecError("ContextSummary.tokens_before must be a positive integer")
        try:
            content = self._summary_codec.decode_content(item.get("content"))
        except ContextSummaryOutputError as exc:
            raise SessionCodecError("ContextSummary.content is invalid") from exc
        return ContextSummary(
            id=self._require_text(item.get("id"), "ContextSummary.id"),
            session_id=self._require_text(
                item.get("session_id"), "ContextSummary.session_id"
            ),
            covered_through_entry_id=self._require_text(
                item.get("covered_through_entry_id"),
                "ContextSummary.covered_through_entry_id",
            ),
            source_leaf_id=self._require_text(
                item.get("source_leaf_id"), "ContextSummary.source_leaf_id"
            ),
            previous_summary_id=previous_summary_id,
            tokens_before=tokens_before,
            created_at=self._decode_datetime(
                item.get("created_at"), "ContextSummary.created_at"
            ),
            content=content,
        )

    def encode_entry(self, entry: MessageEntry) -> dict[str, Any]:
        return {
            "type": "message",
            "id": entry.id,
            "parent_id": entry.parent_id,
            "created_at": self._encode_datetime(entry.created_at, "MessageEntry.created_at"),
            "message": self.encode_message(entry.message),
        }

    def decode_entry(self, data: Any) -> MessageEntry:
        item = self._require_mapping(data, "MessageEntry")
        if item.get("type") != "message":
            raise SessionCodecError(f"Unknown Session Entry type: {item.get('type')!r}")
        parent_id = item.get("parent_id")
        if parent_id is not None:
            parent_id = self._require_text(parent_id, "MessageEntry.parent_id")
        return MessageEntry(
            id=self._require_text(item.get("id"), "MessageEntry.id"),
            parent_id=parent_id,
            created_at=self._decode_datetime(
                item.get("created_at"), "MessageEntry.created_at"
            ),
            message=self.decode_message(item.get("message")),
        )

    def encode_message(self, message: AgentMessage) -> dict[str, Any]:
        base = {
            "id": message.id,
            "created_at": self._encode_datetime(message.created_at, "AgentMessage.created_at"),
        }
        if isinstance(message, UserMessage):
            return {"type": "user", **base, "content": message.content}
        if isinstance(message, AssistantMessage):
            return {
                "type": "assistant",
                **base,
                "content": message.content,
                "tool_calls": [self._encode_tool_call(call) for call in message.tool_calls],
            }
        if isinstance(message, ToolResultMessage):
            return {
                "type": "tool_result",
                **base,
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "content": message.content,
                "is_error": message.is_error,
                "artifact_refs": [
                    self._encode_artifact_ref(ref) for ref in message.artifact_refs
                ],
                "metadata": self._copy_json_value(
                    message.metadata, "ToolResultMessage.metadata"
                ),
            }
        if isinstance(message, RuntimeStatusMessage):
            return {
                "type": "runtime_status",
                **base,
                "content": message.content,
                "render_profile": message.render_profile,
                "display": message.display,
                "snapshot": self._encode_runtime_status_snapshot(message.snapshot),
            }
        raise SessionCodecError(f"Unsupported AgentMessage type: {type(message).__name__}")

    def decode_message(self, data: Any) -> AgentMessage:
        item = self._require_mapping(data, "AgentMessage")
        message_type = item.get("type")
        common = {
            "id": self._require_text(item.get("id"), "AgentMessage.id"),
            "created_at": self._decode_datetime(
                item.get("created_at"), "AgentMessage.created_at"
            ),
        }
        if message_type == "user":
            return UserMessage(
                **common,
                content=self._require_text(item.get("content"), "UserMessage.content"),
            )
        if message_type == "assistant":
            calls = self._require_sequence(
                item.get("tool_calls"), "AssistantMessage.tool_calls"
            )
            content = item.get("content")
            if not isinstance(content, str):
                raise SessionCodecError("AssistantMessage.content must be text")
            return AssistantMessage(
                **common,
                content=content,
                tool_calls=tuple(self._decode_tool_call(call) for call in calls),
            )
        if message_type == "tool_result":
            is_error = item.get("is_error")
            if not isinstance(is_error, bool):
                raise SessionCodecError("ToolResultMessage.is_error must be boolean")
            content = item.get("content")
            if not isinstance(content, str):
                raise SessionCodecError("ToolResultMessage.content must be text")
            artifact_refs = self._require_sequence(
                item.get("artifact_refs", ()), "ToolResultMessage.artifact_refs"
            )
            metadata = self._require_mapping(
                item.get("metadata", {}), "ToolResultMessage.metadata"
            )
            return ToolResultMessage(
                **common,
                tool_call_id=self._require_text(
                    item.get("tool_call_id"), "ToolResultMessage.tool_call_id"
                ),
                tool_name=self._require_text(
                    item.get("tool_name"), "ToolResultMessage.tool_name"
                ),
                content=content,
                is_error=is_error,
                artifact_refs=tuple(
                    self._decode_artifact_ref(ref) for ref in artifact_refs
                ),
                metadata=dict(
                    self._copy_json_value(metadata, "ToolResultMessage.metadata")
                ),
            )
        if message_type == "runtime_status":
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise SessionCodecError("RuntimeStatusMessage.content must be non-empty text")
            display = item.get("display")
            if not isinstance(display, bool):
                raise SessionCodecError("RuntimeStatusMessage.display must be boolean")
            return RuntimeStatusMessage(
                **common,
                snapshot=self._decode_runtime_status_snapshot(item.get("snapshot")),
                content=content,
                render_profile=self._require_text(
                    item.get("render_profile"), "RuntimeStatusMessage.render_profile"
                ),
                display=display,
            )
        raise SessionCodecError(f"Unknown AgentMessage type: {message_type!r}")

    def _encode_runtime_status_snapshot(
        self,
        snapshot: RuntimeStatusSnapshot,
    ) -> dict[str, Any]:
        visibility = snapshot.context_visibility
        return {
            "schema_version": snapshot.schema_version,
            "environment": {
                "current_time": self._encode_datetime(
                    snapshot.environment.current_time,
                    "RuntimeEnvironmentStatus.current_time",
                ),
                "timezone": snapshot.environment.timezone,
            },
            "tool_anomalies": [
                {
                    "kind": anomaly.kind,
                    "tool_name": anomaly.tool_name,
                    "occurrences": anomaly.occurrences,
                }
                for anomaly in snapshot.tool_anomalies
            ],
            "context_visibility": (
                {
                    "mode": visibility.mode,
                    "tool_results_compacted": visibility.tool_results_compacted,
                    "history_messages_omitted": visibility.history_messages_omitted,
                    "active_turn_messages_omitted": (
                        visibility.active_turn_messages_omitted
                    ),
                    "runtime_statuses_omitted": visibility.runtime_statuses_omitted,
                }
                if visibility is not None
                else None
            ),
        }

    def _decode_runtime_status_snapshot(self, data: Any) -> RuntimeStatusSnapshot:
        item = self._require_mapping(data, "RuntimeStatusSnapshot")
        version = item.get("schema_version")
        if version != 1:
            raise SessionCodecError("RuntimeStatusSnapshot.schema_version must be 1")
        environment = self._require_mapping(
            item.get("environment"), "RuntimeStatusSnapshot.environment"
        )
        anomalies_data = self._require_sequence(
            item.get("tool_anomalies"), "RuntimeStatusSnapshot.tool_anomalies"
        )
        anomalies: list[ToolAnomalyStatus] = []
        for raw_anomaly in anomalies_data:
            anomaly = self._require_mapping(raw_anomaly, "ToolAnomalyStatus")
            occurrences = anomaly.get("occurrences")
            if type(occurrences) is not int or occurrences < 2:
                raise SessionCodecError(
                    "ToolAnomalyStatus.occurrences must be an integer >= 2"
                )
            try:
                anomalies.append(
                    ToolAnomalyStatus(
                        kind=self._require_text(
                            anomaly.get("kind"), "ToolAnomalyStatus.kind"
                        ),  # type: ignore[arg-type]
                        tool_name=self._require_text(
                            anomaly.get("tool_name"), "ToolAnomalyStatus.tool_name"
                        ),
                        occurrences=occurrences,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise SessionCodecError("ToolAnomalyStatus is invalid") from exc

        visibility_data = item.get("context_visibility")
        visibility = None
        if visibility_data is not None:
            raw_visibility = self._require_mapping(
                visibility_data, "RuntimeStatusSnapshot.context_visibility"
            )
            counts: dict[str, int] = {}
            for name in (
                "tool_results_compacted",
                "history_messages_omitted",
                "active_turn_messages_omitted",
                "runtime_statuses_omitted",
            ):
                value = raw_visibility.get(name)
                if type(value) is not int or value < 0:
                    raise SessionCodecError(
                        f"ContextVisibilityStatus.{name} must be a non-negative integer"
                    )
                counts[name] = value
            try:
                visibility = ContextVisibilityStatus(
                    mode=self._require_text(
                        raw_visibility.get("mode"), "ContextVisibilityStatus.mode"
                    ),  # type: ignore[arg-type]
                    **counts,
                )
            except (TypeError, ValueError) as exc:
                raise SessionCodecError("ContextVisibilityStatus is invalid") from exc

        try:
            return RuntimeStatusSnapshot(
                schema_version=version,
                environment=RuntimeEnvironmentStatus(
                    current_time=self._decode_datetime(
                        environment.get("current_time"),
                        "RuntimeEnvironmentStatus.current_time",
                    ),
                    timezone=self._require_text(
                        environment.get("timezone"),
                        "RuntimeEnvironmentStatus.timezone",
                    ),
                ),
                tool_anomalies=tuple(anomalies),
                context_visibility=visibility,
            )
        except (TypeError, ValueError) as exc:
            raise SessionCodecError("RuntimeStatusSnapshot is invalid") from exc

    def encode_pending(self, item: PendingInput) -> dict[str, Any]:
        return {
            "id": item.id,
            "source_message_id": item.source_message_id,
            "content": item.content,
            "created_at": self._encode_datetime(item.created_at, "PendingInput.created_at"),
            "edited_at": (
                self._encode_datetime(item.edited_at, "PendingInput.edited_at")
                if item.edited_at is not None
                else None
            ),
            "revision": item.revision,
        }

    def decode_pending(self, data: Any) -> PendingInput:
        item = self._require_mapping(data, "PendingInput")
        edited_at = item.get("edited_at")
        revision = item.get("revision", 1)
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise SessionCodecError("PendingInput.revision must be an integer")
        return PendingInput(
            id=self._require_text(item.get("id"), "PendingInput.id"),
            source_message_id=self._require_text(
                item.get("source_message_id"), "PendingInput.source_message_id"
            ),
            content=self._require_text(item.get("content"), "PendingInput.content"),
            created_at=self._decode_datetime(
                item.get("created_at"), "PendingInput.created_at"
            ),
            edited_at=(
                self._decode_datetime(edited_at, "PendingInput.edited_at")
                if edited_at is not None
                else None
            ),
            revision=revision,
        )

    def _encode_tool_call(self, call: ToolCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "name": call.name,
            "arguments": self._copy_json_value(call.arguments, "ToolCall.arguments"),
        }

    def _decode_tool_call(self, data: Any) -> ToolCall:
        item = self._require_mapping(data, "ToolCall")
        arguments = self._require_mapping(item.get("arguments"), "ToolCall.arguments")
        return ToolCall(
            id=self._require_text(item.get("id"), "ToolCall.id"),
            name=self._require_text(item.get("name"), "ToolCall.name"),
            arguments=dict(self._copy_json_value(arguments, "ToolCall.arguments")),
        )

    @staticmethod
    def _encode_artifact_ref(ref: ArtifactRef) -> dict[str, Any]:
        if not isinstance(ref, ArtifactRef):
            raise SessionCodecError("ToolResultMessage.artifact_refs must contain ArtifactRef values")
        return {
            "id": ref.id,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "size_chars": ref.size_chars,
            "sha256": ref.sha256,
        }

    def _decode_artifact_ref(self, data: Any) -> ArtifactRef:
        item = self._require_mapping(data, "ArtifactRef")
        size_bytes = item.get("size_bytes")
        size_chars = item.get("size_chars")
        if type(size_bytes) is not int or size_bytes < 0:
            raise SessionCodecError("ArtifactRef.size_bytes must be a non-negative integer")
        if type(size_chars) is not int or size_chars < 0:
            raise SessionCodecError("ArtifactRef.size_chars must be a non-negative integer")
        try:
            return ArtifactRef(
                id=self._require_text(item.get("id"), "ArtifactRef.id"),
                media_type=self._require_text(
                    item.get("media_type"), "ArtifactRef.media_type"
                ),
                size_bytes=size_bytes,
                size_chars=size_chars,
                sha256=self._require_text(item.get("sha256"), "ArtifactRef.sha256"),
            )
        except (ArtifactStoreError, TypeError, ValueError) as exc:
            raise SessionCodecError("ArtifactRef is invalid") from exc

    @staticmethod
    def _encode_datetime(value: datetime, field_name: str) -> str:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise SessionCodecError(f"{field_name} must be timezone-aware")
        return value.isoformat()

    @staticmethod
    def _decode_datetime(value: Any, field_name: str) -> datetime:
        if not isinstance(value, str):
            raise SessionCodecError(f"{field_name} must be an ISO 8601 string")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SessionCodecError(f"{field_name} is not valid ISO 8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise SessionCodecError(f"{field_name} must include a timezone")
        return parsed

    @staticmethod
    def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise SessionCodecError(f"{field_name} must be an object")
        if any(not isinstance(key, str) for key in value):
            raise SessionCodecError(f"{field_name} keys must be text")
        return value

    @staticmethod
    def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise SessionCodecError(f"{field_name} must be an array")
        return value

    @staticmethod
    def _require_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SessionCodecError(f"{field_name} must be non-empty text")
        return value

    def _copy_json_value(self, value: Any, field_name: str) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SessionCodecError(f"{field_name} contains a non-finite number")
            return value
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise SessionCodecError(f"{field_name} object keys must be text")
            return {
                key: self._copy_json_value(item, f"{field_name}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._copy_json_value(item, f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
        raise SessionCodecError(
            f"{field_name} contains unsupported value type: {type(value).__name__}"
        )


__all__ = [
    "SESSION_SCHEMA_VERSION",
    "SessionCodec",
    "SessionCodecError",
    "UnsupportedSessionVersionError",
]
