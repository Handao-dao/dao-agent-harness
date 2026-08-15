from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agent_harness.providers.transport import UrllibJsonTransport


class FakeSseResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self) -> FakeSseResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)


def test_urllib_transport_decodes_sse_frames(monkeypatch: Any) -> None:
    response = FakeSseResponse(
        [
            b": keep-alive\n",
            b"data: {\"part\":\n",
            b"data: 1}\n",
            b"\n",
            b"data: [DONE]\r\n",
            b"\r\n",
        ]
    )
    monkeypatch.setattr(
        "agent_harness.providers.transport.urllib.request.urlopen",
        lambda request, timeout: response,
    )

    events = list(
        UrllibJsonTransport._post_sse_sync(
            url="https://example.test/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            payload={"stream": True},
            timeout_s=10,
        )
    )

    assert events == ['{"part":\n1}', "[DONE]"]
