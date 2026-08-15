"""Tool registration plus schema-driven call preparation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from hashlib import sha256
from typing import Any

from agent_harness.artifacts import (
    ArtifactPolicy,
    ArtifactRef,
    ArtifactStore,
    ArtifactStoreError,
)
from agent_harness.messages import ToolCall
from agent_harness.tools.base import (
    AgentTool,
    ToolExecutionPolicy,
    ToolExecutionResult,
    ToolOutput,
)

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolRegistry:
    """Own tool definitions and prepare model-supplied calls for execution."""

    def __init__(
        self,
        policy: ToolExecutionPolicy | None = None,
        *,
        artifact_store: ArtifactStore | None = None,
        artifact_policy: ArtifactPolicy | None = None,
    ) -> None:
        if policy is not None and not isinstance(policy, ToolExecutionPolicy):
            raise TypeError("policy must be a ToolExecutionPolicy")
        if artifact_store is not None and (
            not callable(getattr(artifact_store, "put_text", None))
            or not callable(getattr(artifact_store, "read_text", None))
        ):
            raise TypeError("artifact_store must implement ArtifactStore")
        if artifact_policy is not None and not isinstance(artifact_policy, ArtifactPolicy):
            raise TypeError("artifact_policy must be an ArtifactPolicy")
        self._tools: dict[str, AgentTool] = {}
        self._policy = policy or ToolExecutionPolicy()
        self._artifact_store = artifact_store
        self._artifact_policy = artifact_policy or ArtifactPolicy()

    def register(self, tool: AgentTool) -> None:
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("Tool name must be non-empty text")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if tool.execution_mode not in {"parallel_safe", "sequential"}:
            raise ValueError(f"Invalid execution mode for tool {tool.name}: {tool.execution_mode}")
        timeout = getattr(tool, "timeout_s", None)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError(f"Tool timeout must be None or positive: {tool.name}")
        if not isinstance(tool.parameters, Mapping):
            raise TypeError(f"Tool parameters must be a mapping: {tool.name}")
        schema = dict(tool.parameters)
        if schema.get("type", "object") != "object":
            raise ValueError(f"Tool parameter schema must have object type: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def policy(self) -> ToolExecutionPolicy:
        return self._policy

    @property
    def artifact_policy(self) -> ArtifactPolicy:
        return self._artifact_policy

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(dict(tool.parameters)),
            }
            for tool in self._tools.values()
        ]

    def prepare_call(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> tuple[AgentTool | None, dict[str, Any], str | None]:
        """Resolve, safely cast, and validate one model-supplied tool call."""

        raw_arguments = dict(arguments)
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "(none)"
            return None, raw_arguments, f"Tool '{name}' not found. Available: {available}"

        schema = dict(tool.parameters)
        prepared = self._cast_object(raw_arguments, schema)
        errors = self._validate_value(prepared, {**schema, "type": "object"})
        if errors:
            return tool, prepared, (
                f"Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, prepared, None

    async def execute_call(self, call: ToolCall) -> ToolExecutionResult:
        """Prepare and execute one call under its effective deadline."""

        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall")
        tool, arguments, preparation_error = self.prepare_call(
            call.name,
            call.arguments,
        )
        if preparation_error is not None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=preparation_error,
                status="failed",
                error_code=("not_found" if tool is None else "invalid_arguments"),
            )
        if tool is None:
            raise RuntimeError("prepare_call returned no tool without an error")

        timeout_s = getattr(tool, "timeout_s", None)
        if timeout_s is None:
            timeout_s = self._policy.default_timeout_s

        timeout_context: asyncio.Timeout | None = None
        try:
            if timeout_s is None:
                raw_result = await tool.execute(arguments)
            else:
                timeout_context = asyncio.timeout(float(timeout_s))
                async with timeout_context:
                    raw_result = await tool.execute(arguments)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            if timeout_context is not None and timeout_context.expired():
                return self._timeout_result(call, float(timeout_s))
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=f"{type(exc).__name__}: {exc}",
                status="failed",
                error_code="exception",
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=f"{type(exc).__name__}: {exc}",
                status="failed",
                error_code="exception",
            )

        if timeout_context is not None and timeout_context.expired():
            return self._timeout_result(call, float(timeout_s))

        output = raw_result if isinstance(raw_result, ToolOutput) else None
        content = self._normalize_result(
            output.content if output is not None else raw_result
        )
        metadata = output.metadata if output is not None else {}
        if output is not None and output.is_error:
            return await self._failed_result(
                call,
                content,
                artifact_content=output.artifact_content,
                metadata=metadata,
            )
        return await self._completed_result(
            call,
            content,
            artifact_content=(output.artifact_content if output is not None else None),
            metadata=metadata,
            allow_externalization=(
                output.allow_externalization if output is not None else True
            ),
        )

    async def _failed_result(
        self,
        call: ToolCall,
        model_content: str,
        *,
        artifact_content: str | None,
        metadata: Mapping[str, Any],
    ) -> ToolExecutionResult:
        if artifact_content is None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=model_content,
                status="failed",
                error_code="reported_error",
                metadata=metadata,
            )

        store = self._artifact_store
        if store is None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=self._explicit_artifact_failure_content(model_content),
                status="failed",
                error_code="artifact_store",
                metadata=metadata,
            )

        try:
            ref = await store.put_text(artifact_content)
            self._validate_artifact_ref(ref, artifact_content)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=self._explicit_artifact_failure_content(model_content),
                status="failed",
                error_code="artifact_store",
                metadata=metadata,
            )

        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=self._attach_artifact_reference(model_content, ref),
            status="failed",
            error_code="reported_error",
            artifact_refs=(ref,),
            metadata=metadata,
        )

    async def _completed_result(
        self,
        call: ToolCall,
        model_content: str,
        *,
        artifact_content: str | None,
        metadata: Mapping[str, Any],
        allow_externalization: bool,
    ) -> ToolExecutionResult:
        store = self._artifact_store
        policy = self._artifact_policy
        if not allow_externalization:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=model_content,
                status="completed",
                metadata=metadata,
            )

        complete_content = artifact_content
        explicit_artifact = complete_content is not None
        if explicit_artifact and store is None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=self._explicit_artifact_failure_content(model_content),
                status="failed",
                error_code="artifact_store",
                metadata=metadata,
            )
        if store is None:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=model_content,
                status="completed",
                metadata=metadata,
            )
        if complete_content is None:
            if len(model_content) <= policy.externalize_above_chars:
                return ToolExecutionResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=model_content,
                    status="completed",
                    metadata=metadata,
                )
            complete_content = model_content

        try:
            ref = await store.put_text(complete_content)
            self._validate_artifact_ref(ref, complete_content)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolExecutionResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=(
                    self._explicit_artifact_failure_content(model_content)
                    if explicit_artifact
                    else self._artifact_failure_content(complete_content)
                ),
                status="failed",
                error_code="artifact_store",
                metadata=metadata,
            )

        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=(
                self._attach_artifact_reference(model_content, ref)
                if explicit_artifact
                else self._externalized_content(ref, complete_content)
            ),
            status="completed",
            artifact_refs=(ref,),
            metadata=metadata,
        )

    @staticmethod
    def _attach_artifact_reference(content: str, ref: ArtifactRef) -> str:
        suffix = (
            "[complete tool result stored as artifact]\n"
            f"artifact_id: {ref.id}\n"
            f"media_type: {ref.media_type}\n"
            f"size: {ref.size_chars} chars / {ref.size_bytes} bytes\n"
            "Use read_artifact with artifact_id and optional offset/limit to inspect more."
        )
        return f"{content}\n\n{suffix}" if content else suffix

    @staticmethod
    def _validate_artifact_ref(ref: ArtifactRef, content: str) -> None:
        if not isinstance(ref, ArtifactRef):
            raise ArtifactStoreError("ArtifactStore returned an invalid reference")
        encoded = content.encode("utf-8", errors="strict")
        if (
            ref.sha256 != sha256(encoded).hexdigest()
            or ref.size_bytes != len(encoded)
            or ref.size_chars != len(content)
        ):
            raise ArtifactStoreError("ArtifactStore returned a mismatched reference")

    def _externalized_content(self, ref: ArtifactRef, content: str) -> str:
        head, tail, omitted = self._preview(content)
        return (
            "[tool result externalized]\n"
            f"artifact_id: {ref.id}\n"
            f"media_type: {ref.media_type}\n"
            f"size: {ref.size_chars} chars / {ref.size_bytes} bytes\n"
            "Use read_artifact with artifact_id and optional offset/limit to inspect more.\n\n"
            "--- head preview ---\n"
            f"{head}\n"
            f"--- {omitted} chars omitted ---\n"
            "--- tail preview ---\n"
            f"{tail}"
        )

    def _artifact_failure_content(self, content: str) -> str:
        head, tail, omitted = self._preview(content)
        return (
            "[tool result unavailable]\n"
            "The tool completed, but its oversized result could not be stored safely.\n\n"
            "--- head preview ---\n"
            f"{head}\n"
            f"--- {omitted} chars omitted ---\n"
            "--- tail preview ---\n"
            f"{tail}"
        )

    @staticmethod
    def _explicit_artifact_failure_content(model_content: str) -> str:
        notice = (
            "[complete tool result unavailable]\n"
            "The tool completed, but its complete result could not be stored safely."
        )
        return f"{model_content}\n\n{notice}" if model_content else notice

    def _preview(self, content: str) -> tuple[str, str, int]:
        head_chars = min(self._artifact_policy.preview_head_chars, len(content))
        remaining = len(content) - head_chars
        tail_chars = min(self._artifact_policy.preview_tail_chars, remaining)
        head = content[:head_chars]
        tail = content[-tail_chars:] if tail_chars else ""
        return head, tail, len(content) - head_chars - tail_chars

    @staticmethod
    def _timeout_result(call: ToolCall, timeout_s: float) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=f"Tool '{call.name}' timed out after {timeout_s:g} seconds",
            status="timed_out",
            error_code="timeout",
        )

    @staticmethod
    def _normalize_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    @classmethod
    def _cast_object(
        cls,
        value: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        return {
            key: cls._cast_value(item, properties[key]) if key in properties else item
            for key, item in value.items()
        }

    @classmethod
    def _cast_value(cls, value: Any, schema: Any) -> Any:
        if not isinstance(schema, Mapping):
            return value
        schema_type = cls._schema_type(schema.get("type"))
        if value is None:
            return None
        if schema_type == "integer" and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        if schema_type == "number" and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if schema_type == "boolean" and isinstance(value, str):
            normalized = value.lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            return value
        if schema_type == "string" and not isinstance(value, (dict, list)):
            return str(value)
        if schema_type == "object" and isinstance(value, Mapping):
            return cls._cast_object(value, schema)
        if schema_type == "array" and isinstance(value, list):
            item_schema = schema.get("items")
            if item_schema is not None:
                return [cls._cast_value(item, item_schema) for item in value]
        return value

    @classmethod
    def _validate_value(
        cls,
        value: Any,
        schema: Mapping[str, Any],
        path: str = "",
    ) -> list[str]:
        schema_type = cls._schema_type(schema.get("type"))
        nullable = cls._is_nullable(schema)
        label = path or "parameters"
        if value is None:
            return [] if nullable else [f"{label} must not be null"]

        expected = _JSON_TYPES.get(schema_type or "")
        if schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return [f"{label} must be an integer"]
        elif schema_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return [f"{label} must be a number"]
        elif expected is not None and not isinstance(value, expected):
            return [f"{label} must be {schema_type}"]

        errors: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")

        if schema_type in {"integer", "number"}:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")

        if schema_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} must contain at least {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} must contain at most {schema['maxLength']} characters")

        if schema_type == "object":
            properties = schema.get("properties", {})
            properties = properties if isinstance(properties, Mapping) else {}
            for key in schema.get("required", ()):
                if key not in value:
                    errors.append(f"missing required {cls._subpath(path, key)}")
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"unexpected parameter {cls._subpath(path, key)}")
            for key, item in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, Mapping):
                    errors.extend(
                        cls._validate_value(item, child_schema, cls._subpath(path, key))
                    )

        if schema_type == "array":
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{label} must contain at least {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{label} must contain at most {schema['maxItems']} items")
            item_schema = schema.get("items")
            if isinstance(item_schema, Mapping):
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]" if path else f"[{index}]"
                    errors.extend(cls._validate_value(item, item_schema, item_path))

        return errors

    @staticmethod
    def _schema_type(raw_type: Any) -> str | None:
        if isinstance(raw_type, list):
            return next((item for item in raw_type if item != "null"), None)
        return raw_type if isinstance(raw_type, str) else None

    @staticmethod
    def _is_nullable(schema: Mapping[str, Any]) -> bool:
        raw_type = schema.get("type")
        return bool(schema.get("nullable")) or (
            isinstance(raw_type, list) and "null" in raw_type
        )

    @staticmethod
    def _subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    def __iter__(self) -> Iterator[AgentTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


__all__ = ["ToolRegistry"]
