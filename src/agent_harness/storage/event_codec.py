"""JSON codec for append-only Session events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_harness.session import (
    ConsumedInput,
    ContextSummaryCreated,
    InputEdited,
    InputEnqueued,
    LeafChanged,
    SessionEvent,
    TurnCommitted,
)
from agent_harness.storage.codec import SessionCodec, SessionCodecError


class SessionEventCodec:
    """Encode and validate the durable operations that materialize a Session."""

    def __init__(self, session_codec: SessionCodec | None = None) -> None:
        self._session_codec = session_codec or SessionCodec()

    def encode(self, event: SessionEvent) -> dict[str, Any]:
        common = {
            "id": event.id,
            "timestamp": self._session_codec._encode_datetime(
                event.timestamp, "SessionEvent.timestamp"
            ),
        }
        if isinstance(event, InputEnqueued):
            return {
                "type": "input_enqueued",
                **common,
                "input": self._session_codec.encode_pending(event.input),
            }
        if isinstance(event, InputEdited):
            return {
                "type": "input_edited",
                **common,
                "input_id": event.input_id,
                "expected_revision": event.expected_revision,
                "content": event.content,
                "edited_at": self._session_codec._encode_datetime(
                    event.edited_at, "InputEdited.edited_at"
                ),
            }
        if isinstance(event, TurnCommitted):
            return {
                "type": "turn_committed",
                **common,
                "base_leaf_id": event.base_leaf_id,
                "consumed_inputs": [
                    {"id": item.id, "revision": item.revision}
                    for item in event.consumed_inputs
                ],
                "entries": [
                    self._session_codec.encode_entry(entry) for entry in event.entries
                ],
                "new_leaf_id": event.new_leaf_id,
            }
        if isinstance(event, LeafChanged):
            return {
                "type": "leaf_changed",
                **common,
                "from_leaf_id": event.from_leaf_id,
                "target_leaf_id": event.target_leaf_id,
            }
        if isinstance(event, ContextSummaryCreated):
            return {
                "type": "context_summary_created",
                **common,
                "summary": self._session_codec.encode_context_summary(event.summary),
            }
        raise SessionCodecError(f"Unsupported SessionEvent type: {type(event).__name__}")

    def decode(self, data: Any) -> SessionEvent:
        item = self._mapping(data, "SessionEvent")
        event_type = item.get("type")
        common = {
            "id": self._text(item.get("id"), "SessionEvent.id"),
            "timestamp": self._session_codec._decode_datetime(
                item.get("timestamp"), "SessionEvent.timestamp"
            ),
        }
        if event_type == "input_enqueued":
            return InputEnqueued(
                **common,
                input=self._session_codec.decode_pending(item.get("input")),
            )
        if event_type == "input_edited":
            return InputEdited(
                **common,
                input_id=self._text(item.get("input_id"), "InputEdited.input_id"),
                expected_revision=self._positive_int(
                    item.get("expected_revision"), "InputEdited.expected_revision"
                ),
                content=self._text(item.get("content"), "InputEdited.content"),
                edited_at=self._session_codec._decode_datetime(
                    item.get("edited_at"), "InputEdited.edited_at"
                ),
            )
        if event_type == "turn_committed":
            consumed_data = self._sequence(
                item.get("consumed_inputs"), "TurnCommitted.consumed_inputs"
            )
            entries_data = self._sequence(item.get("entries"), "TurnCommitted.entries")
            consumed: list[ConsumedInput] = []
            for raw in consumed_data:
                consumed_item = self._mapping(raw, "ConsumedInput")
                consumed.append(
                    ConsumedInput(
                        id=self._text(consumed_item.get("id"), "ConsumedInput.id"),
                        revision=self._positive_int(
                            consumed_item.get("revision"), "ConsumedInput.revision"
                        ),
                    )
                )
            base_leaf_id = self._optional_text(
                item.get("base_leaf_id"), "TurnCommitted.base_leaf_id"
            )
            return TurnCommitted(
                **common,
                base_leaf_id=base_leaf_id,
                consumed_inputs=tuple(consumed),
                entries=tuple(
                    self._session_codec.decode_entry(entry) for entry in entries_data
                ),
                new_leaf_id=self._text(
                    item.get("new_leaf_id"), "TurnCommitted.new_leaf_id"
                ),
            )
        if event_type == "leaf_changed":
            return LeafChanged(
                **common,
                from_leaf_id=self._optional_text(
                    item.get("from_leaf_id"), "LeafChanged.from_leaf_id"
                ),
                target_leaf_id=self._optional_text(
                    item.get("target_leaf_id"), "LeafChanged.target_leaf_id"
                ),
            )
        if event_type == "context_summary_created":
            return ContextSummaryCreated(
                **common,
                summary=self._session_codec.decode_context_summary(item.get("summary")),
            )
        raise SessionCodecError(f"Unknown SessionEvent type: {event_type!r}")

    @staticmethod
    def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise SessionCodecError(f"{field_name} must be an object")
        return value

    @staticmethod
    def _sequence(value: Any, field_name: str) -> Sequence[Any]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise SessionCodecError(f"{field_name} must be an array")
        return value

    @staticmethod
    def _text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SessionCodecError(f"{field_name} must be non-empty text")
        return value

    def _optional_text(self, value: Any, field_name: str) -> str | None:
        return None if value is None else self._text(value, field_name)

    @staticmethod
    def _positive_int(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SessionCodecError(f"{field_name} must be a positive integer")
        return value


__all__ = ["SessionEventCodec"]
