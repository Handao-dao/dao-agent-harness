"""Strong ContextSummary types and strict model-output decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from agent_harness.messages import utc_now

SummaryStatus: TypeAlias = Literal[
    "active",
    "waiting_for_user",
    "blocked",
    "completed",
    "unclear",
]
ArtifactState: TypeAlias = Literal["created", "modified", "inspected", "planned"]

SUMMARY_STATUSES = frozenset(
    {"active", "waiting_for_user", "blocked", "completed", "unclear"}
)
ARTIFACT_STATES = frozenset({"created", "modified", "inspected", "planned"})


def new_summary_id() -> str:
    return f"summary_{uuid4().hex}"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _require_optional_text(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_text(value, field_name)


@dataclass(frozen=True, slots=True)
class SummaryDecision:
    decision: str
    rationale: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.decision, "SummaryDecision.decision")
        _require_optional_text(self.rationale, "SummaryDecision.rationale")


@dataclass(frozen=True, slots=True)
class SummaryArtifact:
    reference: str
    description: str
    state: ArtifactState

    def __post_init__(self) -> None:
        _require_text(self.reference, "SummaryArtifact.reference")
        _require_text(self.description, "SummaryArtifact.description")
        if self.state not in ARTIFACT_STATES:
            raise ValueError(f"Unsupported SummaryArtifact.state: {self.state!r}")


@dataclass(frozen=True, slots=True)
class ContextSummaryContent:
    """The model-produced, provider-neutral semantic summary."""

    schema_version: int
    objective: str | None
    status: SummaryStatus
    user_constraints: tuple[str, ...] = ()
    established_facts: tuple[str, ...] = ()
    decisions: tuple[SummaryDecision, ...] = ()
    completed_work: tuple[str, ...] = ()
    current_work: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    artifacts: tuple[SummaryArtifact, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    known_issues: tuple[str, ...] = ()
    continuation_note: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("ContextSummaryContent.schema_version must equal 1")
        _require_optional_text(self.objective, "ContextSummaryContent.objective")
        _require_optional_text(
            self.continuation_note, "ContextSummaryContent.continuation_note"
        )
        if self.status not in SUMMARY_STATUSES:
            raise ValueError(f"Unsupported ContextSummaryContent.status: {self.status!r}")

        text_fields = (
            "user_constraints",
            "established_facts",
            "completed_work",
            "current_work",
            "next_steps",
            "unresolved_questions",
            "known_issues",
        )
        for field_name in text_fields:
            values = tuple(getattr(self, field_name))
            for index, value in enumerate(values):
                _require_text(value, f"ContextSummaryContent.{field_name}[{index}]")
            object.__setattr__(self, field_name, values)

        decisions = tuple(self.decisions)
        if any(not isinstance(item, SummaryDecision) for item in decisions):
            raise TypeError("ContextSummaryContent.decisions must contain SummaryDecision")
        object.__setattr__(self, "decisions", decisions)

        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, SummaryArtifact) for item in artifacts):
            raise TypeError("ContextSummaryContent.artifacts must contain SummaryArtifact")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True, slots=True)
class ContextSummary:
    """A durable structured summary covering a prefix of one conversation branch."""

    session_id: str
    covered_through_entry_id: str
    source_leaf_id: str
    content: ContextSummaryContent
    tokens_before: int
    previous_summary_id: str | None = None
    id: str = field(default_factory=new_summary_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for field_name in ("id", "session_id", "covered_through_entry_id", "source_leaf_id"):
            _require_text(getattr(self, field_name), f"ContextSummary.{field_name}")
        _require_optional_text(
            self.previous_summary_id, "ContextSummary.previous_summary_id"
        )
        if not isinstance(self.content, ContextSummaryContent):
            raise TypeError("ContextSummary.content must be ContextSummaryContent")
        if type(self.tokens_before) is not int or self.tokens_before <= 0:
            raise ValueError("ContextSummary.tokens_before must be a positive integer")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("ContextSummary.created_at must be timezone-aware")


class ContextSummaryOutputError(ValueError):
    """A model output cannot be accepted as a ContextSummaryContent."""


class ContextSummaryParseError(ContextSummaryOutputError):
    """The model output is not one strict JSON object."""


class ContextSummaryValidationError(ContextSummaryOutputError):
    """The decoded JSON object does not match the summary schema."""


class ContextSummaryCodec:
    """Encode and strictly validate the stable v1 summary content schema."""

    SCHEMA_VERSION = 1
    MAX_LIST_ITEMS = 20
    MAX_TEXT_CHARS = 500
    MAX_REFERENCE_CHARS = 1000
    MAX_CONTINUATION_CHARS = 1000
    MAX_CANONICAL_CHARS = 8000

    _CONTENT_KEYS = frozenset(
        {
            "schema_version",
            "objective",
            "status",
            "user_constraints",
            "established_facts",
            "decisions",
            "completed_work",
            "current_work",
            "next_steps",
            "artifacts",
            "unresolved_questions",
            "known_issues",
            "continuation_note",
        }
    )
    _DECISION_KEYS = frozenset({"decision", "rationale"})
    _ARTIFACT_KEYS = frozenset({"reference", "description", "state"})

    def encode_content(self, content: ContextSummaryContent) -> dict[str, Any]:
        if not isinstance(content, ContextSummaryContent):
            raise TypeError("content must be ContextSummaryContent")
        return {
            "schema_version": content.schema_version,
            "objective": content.objective,
            "status": content.status,
            "user_constraints": list(content.user_constraints),
            "established_facts": list(content.established_facts),
            "decisions": [
                {"decision": item.decision, "rationale": item.rationale}
                for item in content.decisions
            ],
            "completed_work": list(content.completed_work),
            "current_work": list(content.current_work),
            "next_steps": list(content.next_steps),
            "artifacts": [
                {
                    "reference": item.reference,
                    "description": item.description,
                    "state": item.state,
                }
                for item in content.artifacts
            ],
            "unresolved_questions": list(content.unresolved_questions),
            "known_issues": list(content.known_issues),
            "continuation_note": content.continuation_note,
        }

    def decode_content(self, value: Any) -> ContextSummaryContent:
        item = self._mapping(value, "$")
        self._exact_keys(item, self._CONTENT_KEYS, "$")

        version = item["schema_version"]
        if type(version) is not int or version != self.SCHEMA_VERSION:
            raise ContextSummaryValidationError("$.schema_version must equal 1")

        status = item["status"]
        if not isinstance(status, str) or status not in SUMMARY_STATUSES:
            choices = ", ".join(sorted(SUMMARY_STATUSES))
            raise ContextSummaryValidationError(f"$.status must be one of: {choices}")

        decisions_data = self._sequence(item["decisions"], "$.decisions")
        self._list_size(decisions_data, "$.decisions")
        decisions: list[SummaryDecision] = []
        for index, raw in enumerate(decisions_data):
            path = f"$.decisions[{index}]"
            decision = self._mapping(raw, path)
            self._exact_keys(decision, self._DECISION_KEYS, path)
            decisions.append(
                SummaryDecision(
                    decision=self._text(decision["decision"], f"{path}.decision"),
                    rationale=self._optional_text(
                        decision["rationale"], f"{path}.rationale"
                    ),
                )
            )

        artifacts_data = self._sequence(item["artifacts"], "$.artifacts")
        self._list_size(artifacts_data, "$.artifacts")
        artifacts: list[SummaryArtifact] = []
        for index, raw in enumerate(artifacts_data):
            path = f"$.artifacts[{index}]"
            artifact = self._mapping(raw, path)
            self._exact_keys(artifact, self._ARTIFACT_KEYS, path)
            state = artifact["state"]
            if not isinstance(state, str) or state not in ARTIFACT_STATES:
                choices = ", ".join(sorted(ARTIFACT_STATES))
                raise ContextSummaryValidationError(f"{path}.state must be one of: {choices}")
            artifacts.append(
                SummaryArtifact(
                    reference=self._text(
                        artifact["reference"],
                        f"{path}.reference",
                        max_chars=self.MAX_REFERENCE_CHARS,
                    ),
                    description=self._text(
                        artifact["description"], f"{path}.description"
                    ),
                    state=state,
                )
            )

        content = ContextSummaryContent(
            schema_version=version,
            objective=self._optional_text(item["objective"], "$.objective"),
            status=status,
            user_constraints=self._text_list(
                item["user_constraints"], "$.user_constraints"
            ),
            established_facts=self._text_list(
                item["established_facts"], "$.established_facts"
            ),
            decisions=tuple(decisions),
            completed_work=self._text_list(item["completed_work"], "$.completed_work"),
            current_work=self._text_list(item["current_work"], "$.current_work"),
            next_steps=self._text_list(item["next_steps"], "$.next_steps"),
            artifacts=tuple(artifacts),
            unresolved_questions=self._text_list(
                item["unresolved_questions"], "$.unresolved_questions"
            ),
            known_issues=self._text_list(item["known_issues"], "$.known_issues"),
            continuation_note=self._optional_text(
                item["continuation_note"],
                "$.continuation_note",
                max_chars=self.MAX_CONTINUATION_CHARS,
            ),
        )
        self._check_canonical_size(content)
        return content

    def canonical_json(self, content: ContextSummaryContent) -> str:
        return json.dumps(
            self.encode_content(content),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _check_canonical_size(self, content: ContextSummaryContent) -> None:
        length = len(self.canonical_json(content))
        if length > self.MAX_CANONICAL_CHARS:
            raise ContextSummaryValidationError(
                f"$ encoded summary exceeds {self.MAX_CANONICAL_CHARS} characters"
            )

    def _text_list(self, value: Any, path: str) -> tuple[str, ...]:
        items = self._sequence(value, path)
        self._list_size(items, path)
        return tuple(self._text(item, f"{path}[{index}]") for index, item in enumerate(items))

    def _list_size(self, items: Sequence[Any], path: str) -> None:
        if len(items) > self.MAX_LIST_ITEMS:
            raise ContextSummaryValidationError(
                f"{path} must contain at most {self.MAX_LIST_ITEMS} items"
            )

    @staticmethod
    def _mapping(value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ContextSummaryValidationError(f"{path} must be an object")
        if any(not isinstance(key, str) for key in value):
            raise ContextSummaryValidationError(f"{path} keys must be text")
        return value

    @staticmethod
    def _sequence(value: Any, path: str) -> Sequence[Any]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ContextSummaryValidationError(f"{path} must be an array")
        return value

    @staticmethod
    def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
        actual = frozenset(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise ContextSummaryValidationError(
                f"{path} is missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise ContextSummaryValidationError(
                f"{path} has unknown fields: {', '.join(unknown)}"
            )

    def _text(self, value: Any, path: str, *, max_chars: int | None = None) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ContextSummaryValidationError(f"{path} must be non-empty text")
        normalized = value.strip()
        limit = max_chars or self.MAX_TEXT_CHARS
        if len(normalized) > limit:
            raise ContextSummaryValidationError(
                f"{path} must contain at most {limit} characters"
            )
        return normalized

    def _optional_text(
        self,
        value: Any,
        path: str,
        *,
        max_chars: int | None = None,
    ) -> str | None:
        return None if value is None else self._text(value, path, max_chars=max_chars)


class ContextSummaryParser:
    """Parse exactly one JSON object from an unconstrained Provider content string."""

    MAX_RAW_CHARS = 16_000

    def __init__(self, codec: ContextSummaryCodec | None = None) -> None:
        self.codec = codec or ContextSummaryCodec()

    def parse(self, raw: str) -> ContextSummaryContent:
        if not isinstance(raw, str) or not raw.strip():
            raise ContextSummaryParseError("Provider summary content must be non-empty text")
        if len(raw) > self.MAX_RAW_CHARS:
            raise ContextSummaryParseError(
                f"Provider summary content exceeds {self.MAX_RAW_CHARS} characters"
            )
        try:
            value = json.loads(
                raw,
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ContextSummaryParseError(
                f"Provider summary content is not strict JSON at line {exc.lineno}, column {exc.colno}"
            ) from exc
        return self.codec.decode_content(value)

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContextSummaryParseError(f"JSON object contains duplicate key: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ContextSummaryParseError(f"JSON contains unsupported constant: {value}")


__all__ = [
    "ArtifactState",
    "ContextSummary",
    "ContextSummaryCodec",
    "ContextSummaryContent",
    "ContextSummaryOutputError",
    "ContextSummaryParseError",
    "ContextSummaryParser",
    "ContextSummaryValidationError",
    "SummaryArtifact",
    "SummaryDecision",
    "SummaryStatus",
    "new_summary_id",
]
