from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness.token_estimation import (
    PromptTokenEstimationError,
    PromptTokenEstimatorChain,
    ProviderPromptTokenEstimator,
    TiktokenPromptEstimator,
    build_default_token_estimator,
)

REQUEST = {
    "model": "fake-model",
    "system_prompt": "system",
    "messages": ({"role": "user", "content": "hello"},),
    "tools": (),
}


class AsyncCountingProvider:
    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    async def count_prompt_tokens(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        del model, system_prompt, messages, tools
        self.calls += 1
        return self.value


class StaticEstimator:
    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def estimate(self, **request: Any) -> int:
        del request
        self.calls += 1
        return self.value


async def test_chain_awaits_provider_native_counter_and_stops_before_fallback() -> None:
    provider = AsyncCountingProvider(37)
    fallback = StaticEstimator(99)
    chain = PromptTokenEstimatorChain(
        (ProviderPromptTokenEstimator(provider), fallback)
    )

    result = await chain.estimate(**REQUEST)

    assert result == 37
    assert provider.calls == 1
    assert fallback.calls == 0


async def test_chain_falls_back_when_provider_has_no_counter() -> None:
    fallback = StaticEstimator(41)
    chain = PromptTokenEstimatorChain(
        (ProviderPromptTokenEstimator(object()), fallback)
    )

    result = await chain.estimate(**REQUEST)

    assert result == 41
    assert fallback.calls == 1


async def test_chain_reports_all_failures_instead_of_guessing() -> None:
    chain = PromptTokenEstimatorChain(
        (ProviderPromptTokenEstimator(object()), StaticEstimator(0))
    )

    with pytest.raises(PromptTokenEstimationError, match="All prompt token estimators failed"):
        await chain.estimate(**REQUEST)


def test_tiktoken_estimator_uses_model_encoding_then_fallback(monkeypatch) -> None:
    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

    fake_tiktoken = SimpleNamespace(
        encoding_for_model=lambda _model: (_ for _ in ()).throw(KeyError()),
        get_encoding=lambda name: FakeEncoding() if name == "cl100k_base" else None,
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    result = TiktokenPromptEstimator().estimate(**REQUEST)

    assert result > len("hello")


def test_default_chain_keeps_provider_counter_before_local_tokenizer() -> None:
    provider = AsyncCountingProvider(12)

    chain = build_default_token_estimator(provider)

    assert isinstance(chain, PromptTokenEstimatorChain)
