"""Versioned JSON codec for ContextCheckpoint files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_harness.checkpoints.base import (
    CheckpointCorruptError,
    ContextCheckpoint,
    IncorporatedInput,
)
from agent_harness.storage.codec import SessionCodec, SessionCodecError

CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointCodecError(CheckpointCorruptError):
    """Raised when checkpoint JSON violates the strong storage schema."""


class UnsupportedCheckpointVersionError(CheckpointCodecError):
    """Raised when a checkpoint uses an unknown schema version."""


class CheckpointCodec:
    def __init__(self, message_codec: SessionCodec | None = None) -> None:
        self._message_codec = message_codec or SessionCodec()

    def encode(self, checkpoint: ContextCheckpoint) -> dict[str, Any]:
        if not isinstance(checkpoint, ContextCheckpoint):
            raise TypeError("checkpoint must be a ContextCheckpoint")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "session_id": checkpoint.session_id,
            "input_id": checkpoint.input_id,
            "input_revision": checkpoint.input_revision,
            "base_leaf_id": checkpoint.base_leaf_id,
            "save_cursor": checkpoint.save_cursor,
            "phase": checkpoint.phase,
            "model": checkpoint.model,
            "next_model_turn": checkpoint.next_model_turn,
            "messages": [
                self._message_codec.encode_message(message)
                for message in checkpoint.messages
            ],
            "incorporated_inputs": [
                {"id": item.id, "revision": item.revision}
                for item in checkpoint.incorporated_inputs
            ],
            "tools_used": list(checkpoint.tools_used),
            "usage": dict(checkpoint.usage),
            "terminal_status": checkpoint.terminal_status,
            "stop_reason": checkpoint.stop_reason,
            "final_content": checkpoint.final_content,
            "updated_at": self._message_codec._encode_datetime(
                checkpoint.updated_at, "ContextCheckpoint.updated_at"
            ),
        }

    def decode(self, data: Mapping[str, Any]) -> ContextCheckpoint:
        try:
            root = self._message_codec._require_mapping(data, "ContextCheckpoint")
            version = root.get("schema_version")
            if type(version) is not int or version not in {1, CHECKPOINT_SCHEMA_VERSION}:
                raise UnsupportedCheckpointVersionError(
                    f"Unsupported Checkpoint schema version: {version!r}"
                )

            base_leaf_id = root.get("base_leaf_id")
            if base_leaf_id is not None:
                base_leaf_id = self._message_codec._require_text(
                    base_leaf_id, "ContextCheckpoint.base_leaf_id"
                )
            input_revision = self._non_negative_int(
                root.get("input_revision"), "ContextCheckpoint.input_revision"
            )
            if input_revision < 1:
                raise CheckpointCodecError(
                    "ContextCheckpoint.input_revision must be positive"
                )
            save_cursor = self._non_negative_int(
                root.get("save_cursor"), "ContextCheckpoint.save_cursor"
            )
            next_model_turn = self._non_negative_int(
                root.get("next_model_turn"), "ContextCheckpoint.next_model_turn"
            )
            if next_model_turn < 1:
                raise CheckpointCodecError(
                    "ContextCheckpoint.next_model_turn must be positive"
                )
            messages_data = self._message_codec._require_sequence(
                root.get("messages"), "ContextCheckpoint.messages"
            )
            tools_data = self._message_codec._require_sequence(
                root.get("tools_used"), "ContextCheckpoint.tools_used"
            )
            usage_data = self._message_codec._require_mapping(
                root.get("usage"), "ContextCheckpoint.usage"
            )
            incorporated_inputs = (
                (
                    IncorporatedInput(
                        id=self._message_codec._require_text(
                            root.get("input_id"), "ContextCheckpoint.input_id"
                        ),
                        revision=input_revision,
                    ),
                )
                if version == 1
                else self._decode_incorporated_inputs(root.get("incorporated_inputs"))
            )

            terminal_status = self._optional_text(
                root.get("terminal_status"), "ContextCheckpoint.terminal_status"
            )
            stop_reason = self._optional_text(
                root.get("stop_reason"), "ContextCheckpoint.stop_reason"
            )
            final_content = root.get("final_content")
            if final_content is not None and not isinstance(final_content, str):
                raise CheckpointCodecError(
                    "ContextCheckpoint.final_content must be text or null"
                )

            return ContextCheckpoint(
                session_id=self._message_codec._require_text(
                    root.get("session_id"), "ContextCheckpoint.session_id"
                ),
                input_id=self._message_codec._require_text(
                    root.get("input_id"), "ContextCheckpoint.input_id"
                ),
                input_revision=input_revision,
                base_leaf_id=base_leaf_id,
                save_cursor=save_cursor,
                phase=self._message_codec._require_text(
                    root.get("phase"), "ContextCheckpoint.phase"
                ),  # type: ignore[arg-type]
                model=self._message_codec._require_text(
                    root.get("model"), "ContextCheckpoint.model"
                ),
                next_model_turn=next_model_turn,
                messages=tuple(
                    self._message_codec.decode_message(message)
                    for message in messages_data
                ),
                incorporated_inputs=incorporated_inputs,
                tools_used=tuple(
                    self._message_codec._require_text(
                        name, "ContextCheckpoint.tools_used[]"
                    )
                    for name in tools_data
                ),
                usage=self._decode_usage(usage_data),
                terminal_status=terminal_status,  # type: ignore[arg-type]
                stop_reason=stop_reason,
                final_content=final_content,
                updated_at=self._message_codec._decode_datetime(
                    root.get("updated_at"), "ContextCheckpoint.updated_at"
                ),
            )
        except UnsupportedCheckpointVersionError:
            raise
        except CheckpointCodecError:
            raise
        except (SessionCodecError, TypeError, ValueError) as exc:
            raise CheckpointCodecError("ContextCheckpoint document is invalid") from exc

    @staticmethod
    def _non_negative_int(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointCodecError(f"{field_name} must be a non-negative integer")
        return value

    def _decode_usage(self, value: Mapping[str, Any]) -> dict[str, int]:
        usage: dict[str, int] = {}
        for key, count in value.items():
            name = self._message_codec._require_text(key, "ContextCheckpoint.usage key")
            usage[name] = self._non_negative_int(
                count, f"ContextCheckpoint.usage.{name}"
            )
        return usage

    def _decode_incorporated_inputs(self, value: Any) -> tuple[IncorporatedInput, ...]:
        items = self._message_codec._require_sequence(
            value, "ContextCheckpoint.incorporated_inputs"
        )
        return tuple(
            IncorporatedInput(
                id=self._message_codec._require_text(
                    self._message_codec._require_mapping(
                        item, "ContextCheckpoint.incorporated_inputs[]"
                    ).get("id"),
                    "ContextCheckpoint.incorporated_inputs[].id",
                ),
                revision=self._positive_int_from_mapping(item),
            )
            for item in items
        )

    def _positive_int_from_mapping(self, value: Any) -> int:
        item = self._message_codec._require_mapping(
            value, "ContextCheckpoint.incorporated_inputs[]"
        )
        revision = self._non_negative_int(
            item.get("revision"), "ContextCheckpoint.incorporated_inputs[].revision"
        )
        if revision < 1:
            raise CheckpointCodecError(
                "ContextCheckpoint.incorporated_inputs[].revision must be positive"
            )
        return revision

    def _optional_text(self, value: Any, field_name: str) -> str | None:
        return None if value is None else self._message_codec._require_text(value, field_name)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCodec",
    "CheckpointCodecError",
    "UnsupportedCheckpointVersionError",
]
