"""Composable Provider-first prompt token estimation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping, Sequence
from inspect import isawaitable
from typing import Any, Protocol, TypeAlias

TokenEstimate: TypeAlias = int | Awaitable[int]


class PromptTokenEstimator(Protocol):
    """Count one complete Provider request before it is sent."""

    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> TokenEstimate: ...


class PromptTokenEstimationError(RuntimeError):
    """No configured estimator could produce a valid prompt token count."""


class TokenEstimationUnavailableError(PromptTokenEstimationError):
    """One estimator is unsupported in the current Provider or environment."""


class ProviderPromptTokenEstimator:
    """Adapt an optional Provider ``count_prompt_tokens`` capability."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> TokenEstimate:
        counter = getattr(self._provider, "count_prompt_tokens", None)
        if not callable(counter):
            raise TokenEstimationUnavailableError(
                f"{type(self._provider).__name__} has no count_prompt_tokens capability"
            )
        return counter(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )


class TiktokenPromptEstimator:
    """Estimate a provider-neutral request locally with an optional tiktoken extra."""

    def __init__(self, *, fallback_encoding: str = "cl100k_base") -> None:
        if not isinstance(fallback_encoding, str) or not fallback_encoding.strip():
            raise ValueError("fallback_encoding must be non-empty text")
        self._fallback_encoding = fallback_encoding

    def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenEstimationUnavailableError(
                "tiktoken is not installed; install agent-harness[tokenizers]"
            ) from exc

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding(self._fallback_encoding)

        payload = {
            "system_prompt": system_prompt,
            "messages": [dict(message) for message in messages],
            "tools": [dict(tool) for tool in tools],
        }
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise PromptTokenEstimationError(
                "Provider request is not JSON serializable for local token estimation"
            ) from exc
        return len(encoding.encode(serialized))


class PromptTokenEstimatorChain:
    """Try Provider-native counting first, then configured local fallbacks."""

    def __init__(self, estimators: Sequence[PromptTokenEstimator]) -> None:
        self._estimators = tuple(estimators)
        if not self._estimators:
            raise ValueError("At least one PromptTokenEstimator is required")

    async def estimate(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        failures: list[str] = []
        for estimator in self._estimators:
            try:
                result = estimator.estimate(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                )
                value = await result if isawaitable(result) else result
                if type(value) is not int or value <= 0:
                    raise PromptTokenEstimationError(
                        "estimator must return a positive integer"
                    )
                return value
            except Exception as exc:
                failures.append(f"{type(estimator).__name__}: {type(exc).__name__}: {exc}")
        raise PromptTokenEstimationError(
            "All prompt token estimators failed: " + " | ".join(failures)
        )


def build_default_token_estimator(provider: Any) -> PromptTokenEstimatorChain:
    """Return the agreed Provider counter -> local tokenizer chain."""

    return PromptTokenEstimatorChain(
        (
            ProviderPromptTokenEstimator(provider),
            TiktokenPromptEstimator(),
        )
    )


__all__ = [
    "PromptTokenEstimationError",
    "PromptTokenEstimator",
    "PromptTokenEstimatorChain",
    "ProviderPromptTokenEstimator",
    "TiktokenPromptEstimator",
    "TokenEstimate",
    "TokenEstimationUnavailableError",
    "build_default_token_estimator",
]
