"""Strict JSON codecs for long-term memory contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from agent_harness.memory.models import (
    DreamRunRecord,
    MemoryInboxEntry,
    MemoryOperation,
    MemoryPlan,
)
from agent_harness.storage.codec import SessionCodec


class MemoryCodecError(ValueError):
    """A persisted memory value does not satisfy the v1 protocol."""


class MemoryPlanOutputError(ValueError):
    """Provider output is not a valid MemoryPlan."""


class MemoryPlanParseError(MemoryPlanOutputError):
    pass


class MemoryPlanValidationError(MemoryPlanOutputError):
    pass


class MemoryCodec:
    INBOX_KEYS = frozenset(
        {
            "version", "cursor", "id", "session_id", "source_leaf_id",
            "context_summary_id", "covered_from_entry_id",
            "covered_through_entry_id", "source_entry_ids", "messages", "created_at",
        }
    )
    PLAN_KEYS = frozenset({"schema_version", "operations"})
    OPERATION_KEYS = frozenset(
        {"action", "section", "statement", "match", "reason", "source_entry_ids"}
    )
    DREAM_RECORD_KEYS = frozenset(
        {
            "version", "id", "first_cursor", "last_cursor", "source_inbox_ids",
            "plan", "stop_reason", "changes", "error", "started_at", "completed_at",
        }
    )
    MAX_OPERATIONS = 64
    MAX_TEXT_CHARS = 4_000

    def __init__(self, session_codec: SessionCodec | None = None) -> None:
        self._session_codec = session_codec or SessionCodec()

    def encode_inbox_entry(self, entry: MemoryInboxEntry) -> dict[str, Any]:
        if not isinstance(entry, MemoryInboxEntry):
            raise TypeError("entry must be a MemoryInboxEntry")
        return {
            "version": 1,
            "cursor": entry.cursor,
            "id": entry.id,
            "session_id": entry.session_id,
            "source_leaf_id": entry.source_leaf_id,
            "context_summary_id": entry.context_summary_id,
            "covered_from_entry_id": entry.covered_from_entry_id,
            "covered_through_entry_id": entry.covered_through_entry_id,
            "source_entry_ids": list(entry.source_entry_ids),
            "messages": [self._session_codec.encode_message(message) for message in entry.messages],
            "created_at": entry.created_at.isoformat(),
        }

    def decode_inbox_entry(self, value: Any) -> MemoryInboxEntry:
        item = self._mapping(value, "MemoryInboxEntry")
        self._exact_keys(item, self.INBOX_KEYS, "MemoryInboxEntry")
        if item.get("version") != 1:
            raise MemoryCodecError("MemoryInboxEntry.version must equal 1")
        source_ids = self._text_sequence(item.get("source_entry_ids"), "MemoryInboxEntry.source_entry_ids")
        messages_data = self._sequence(item.get("messages"), "MemoryInboxEntry.messages")
        try:
            return MemoryInboxEntry(
                cursor=self._positive_int(item.get("cursor"), "MemoryInboxEntry.cursor"),
                id=self._text(item.get("id"), "MemoryInboxEntry.id"),
                session_id=self._text(item.get("session_id"), "MemoryInboxEntry.session_id"),
                source_leaf_id=self._text(item.get("source_leaf_id"), "MemoryInboxEntry.source_leaf_id"),
                context_summary_id=self._text(item.get("context_summary_id"), "MemoryInboxEntry.context_summary_id"),
                covered_from_entry_id=self._text(item.get("covered_from_entry_id"), "MemoryInboxEntry.covered_from_entry_id"),
                covered_through_entry_id=self._text(item.get("covered_through_entry_id"), "MemoryInboxEntry.covered_through_entry_id"),
                source_entry_ids=source_ids,
                messages=tuple(self._session_codec.decode_message(message) for message in messages_data),
                created_at=self._datetime(item.get("created_at"), "MemoryInboxEntry.created_at"),
            )
        except (TypeError, ValueError) as exc:
            raise MemoryCodecError(str(exc)) from exc

    def encode_plan(self, plan: MemoryPlan) -> dict[str, Any]:
        if not isinstance(plan, MemoryPlan):
            raise TypeError("plan must be a MemoryPlan")
        return {
            "schema_version": plan.schema_version,
            "operations": [self._encode_operation(operation) for operation in plan.operations],
        }

    def decode_plan(
        self,
        value: Any,
        *,
        allowed_source_entry_ids: frozenset[str] | None = None,
    ) -> MemoryPlan:
        item = self._mapping(value, "MemoryPlan", output=True)
        self._exact_keys(item, self.PLAN_KEYS, "MemoryPlan", output=True)
        if item.get("schema_version") != 1:
            raise MemoryPlanValidationError("MemoryPlan.schema_version must equal 1")
        raw_operations = self._sequence(item.get("operations"), "MemoryPlan.operations", output=True)
        if len(raw_operations) > self.MAX_OPERATIONS:
            raise MemoryPlanValidationError(f"MemoryPlan.operations exceeds {self.MAX_OPERATIONS} items")
        operations = tuple(
            self._decode_operation(
                operation,
                index=index,
                allowed_source_entry_ids=allowed_source_entry_ids,
            )
            for index, operation in enumerate(raw_operations)
        )
        fingerprints = {
            (item.action, item.section, item.statement, item.match, item.source_entry_ids)
            for item in operations
        }
        if len(fingerprints) != len(operations):
            raise MemoryPlanValidationError("MemoryPlan.operations contains duplicates")
        try:
            return MemoryPlan(schema_version=1, operations=operations)
        except (TypeError, ValueError) as exc:
            raise MemoryPlanValidationError(str(exc)) from exc

    def canonical_json(self, plan: MemoryPlan) -> str:
        return json.dumps(
            self.encode_plan(plan), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )

    def encode_dream_record(self, record: DreamRunRecord) -> dict[str, Any]:
        if not isinstance(record, DreamRunRecord):
            raise TypeError("record must be a DreamRunRecord")
        return {
            "version": 1,
            "id": record.id,
            "first_cursor": record.first_cursor,
            "last_cursor": record.last_cursor,
            "source_inbox_ids": list(record.source_inbox_ids),
            "plan": self.encode_plan(record.plan) if record.plan is not None else None,
            "stop_reason": record.stop_reason,
            "changes": list(record.changes),
            "error": record.error,
            "started_at": record.started_at.isoformat(),
            "completed_at": record.completed_at.isoformat(),
        }

    def decode_dream_record(self, value: Any) -> DreamRunRecord:
        item = self._mapping(value, "DreamRunRecord")
        self._exact_keys(item, self.DREAM_RECORD_KEYS, "DreamRunRecord")
        if item.get("version") != 1:
            raise MemoryCodecError("DreamRunRecord.version must equal 1")
        plan_value = item.get("plan")
        plan = None if plan_value is None else self.decode_plan(plan_value)
        error_value = item.get("error")
        error = None if error_value is None else self._text(error_value, "DreamRunRecord.error")
        try:
            return DreamRunRecord(
                id=self._text(item.get("id"), "DreamRunRecord.id"),
                first_cursor=self._positive_int(item.get("first_cursor"), "DreamRunRecord.first_cursor"),
                last_cursor=self._positive_int(item.get("last_cursor"), "DreamRunRecord.last_cursor"),
                source_inbox_ids=self._text_sequence(
                    item.get("source_inbox_ids"), "DreamRunRecord.source_inbox_ids"
                ),
                plan=plan,
                stop_reason=self._text(item.get("stop_reason"), "DreamRunRecord.stop_reason"),
                changes=self._text_sequence(item.get("changes"), "DreamRunRecord.changes"),
                error=error,
                started_at=self._datetime(item.get("started_at"), "DreamRunRecord.started_at"),
                completed_at=self._datetime(item.get("completed_at"), "DreamRunRecord.completed_at"),
            )
        except (TypeError, ValueError) as exc:
            raise MemoryCodecError(str(exc)) from exc

    @staticmethod
    def _encode_operation(operation: MemoryOperation) -> dict[str, Any]:
        return {
            "action": operation.action,
            "section": operation.section,
            "statement": operation.statement,
            "match": operation.match,
            "reason": operation.reason,
            "source_entry_ids": list(operation.source_entry_ids),
        }

    def _decode_operation(
        self,
        value: Any,
        *,
        index: int,
        allowed_source_entry_ids: frozenset[str] | None,
    ) -> MemoryOperation:
        path = f"MemoryPlan.operations[{index}]"
        item = self._mapping(value, path, output=True)
        self._exact_keys(item, self.OPERATION_KEYS, path, output=True)
        match = None
        if item.get("match") is not None:
            match = self._text(item.get("match"), f"{path}.match", output=True)
        source_ids = self._text_sequence(
            item.get("source_entry_ids"), f"{path}.source_entry_ids", output=True
        )
        if allowed_source_entry_ids is not None:
            unknown = set(source_ids) - allowed_source_entry_ids
            if unknown:
                raise MemoryPlanValidationError(
                    f"{path}.source_entry_ids contains values outside the current batch: "
                    + ", ".join(sorted(unknown))
                )
        try:
            return MemoryOperation(
                action=self._text(item.get("action"), f"{path}.action", output=True),
                section=self._text(item.get("section"), f"{path}.section", output=True),
                statement=self._text(item.get("statement"), f"{path}.statement", output=True),
                match=match,
                reason=self._text(item.get("reason"), f"{path}.reason", output=True),
                source_entry_ids=source_ids,
            )
        except (TypeError, ValueError) as exc:
            raise MemoryPlanValidationError(str(exc)) from exc

    @staticmethod
    def _mapping(value: Any, path: str, *, output: bool = False) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            error = MemoryPlanValidationError if output else MemoryCodecError
            raise error(f"{path} must be an object")
        return value

    @staticmethod
    def _sequence(value: Any, path: str, *, output: bool = False) -> Sequence[Any]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            error = MemoryPlanValidationError if output else MemoryCodecError
            raise error(f"{path} must be an array")
        return value

    @classmethod
    def _text_sequence(cls, value: Any, path: str, *, output: bool = False) -> tuple[str, ...]:
        values = cls._sequence(value, path, output=output)
        return tuple(cls._text(item, f"{path}[{index}]", output=output) for index, item in enumerate(values))

    @classmethod
    def _text(cls, value: Any, path: str, *, output: bool = False) -> str:
        error = MemoryPlanValidationError if output else MemoryCodecError
        if not isinstance(value, str) or not value.strip():
            raise error(f"{path} must be non-empty text")
        if output and len(value) > cls.MAX_TEXT_CHARS:
            raise error(f"{path} exceeds {cls.MAX_TEXT_CHARS} characters")
        return value

    @staticmethod
    def _positive_int(value: Any, path: str) -> int:
        if type(value) is not int or value <= 0:
            raise MemoryCodecError(f"{path} must be a positive integer")
        return value

    @staticmethod
    def _datetime(value: Any, path: str) -> datetime:
        if not isinstance(value, str):
            raise MemoryCodecError(f"{path} must be ISO datetime text")
        try:
            result = datetime.fromisoformat(value)
        except ValueError as exc:
            raise MemoryCodecError(f"{path} must be valid ISO datetime text") from exc
        if result.tzinfo is None or result.utcoffset() is None:
            raise MemoryCodecError(f"{path} must be timezone-aware")
        return result

    @staticmethod
    def _exact_keys(
        value: Mapping[str, Any],
        expected: frozenset[str],
        path: str,
        *,
        output: bool = False,
    ) -> None:
        actual = frozenset(value)
        if actual == expected:
            return
        error = MemoryPlanValidationError if output else MemoryCodecError
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise error(f"{path} has invalid fields: {'; '.join(details)}")


class MemoryPlanParser:
    MAX_RAW_CHARS = 32_000

    def __init__(self, codec: MemoryCodec | None = None) -> None:
        self._codec = codec or MemoryCodec()

    def parse(
        self,
        raw: str,
        *,
        allowed_source_entry_ids: frozenset[str] | None = None,
    ) -> MemoryPlan:
        if not isinstance(raw, str) or not raw.strip():
            raise MemoryPlanParseError("Provider MemoryPlan content must be non-empty text")
        if len(raw) > self.MAX_RAW_CHARS:
            raise MemoryPlanParseError(f"Provider MemoryPlan content exceeds {self.MAX_RAW_CHARS} characters")
        try:
            value = json.loads(
                raw, object_pairs_hook=self._unique_object, parse_constant=self._reject_constant
            )
        except (json.JSONDecodeError, MemoryPlanParseError) as exc:
            if isinstance(exc, MemoryPlanParseError):
                raise
            raise MemoryPlanParseError(
                f"Provider MemoryPlan content is not strict JSON at line {exc.lineno}, column {exc.colno}"
            ) from exc
        return self._codec.decode_plan(value, allowed_source_entry_ids=allowed_source_entry_ids)

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryPlanParseError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise MemoryPlanParseError(f"Non-finite JSON number is not allowed: {value}")


__all__ = [
    "MemoryCodec",
    "MemoryCodecError",
    "MemoryPlanOutputError",
    "MemoryPlanParseError",
    "MemoryPlanParser",
    "MemoryPlanValidationError",
]
