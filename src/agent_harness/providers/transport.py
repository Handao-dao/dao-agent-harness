"""Small JSON-over-HTTP transport used by compatible providers."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Protocol


class ProviderHTTPError(RuntimeError):
    """An HTTP or connectivity failure from a model endpoint."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class JsonHttpTransport(Protocol):
    async def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]: ...


class SseHttpTransport(Protocol):
    def post_sse(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[str]: ...


class UrllibJsonTransport:
    """Dependency-free async facade over urllib's blocking request API."""

    async def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            self._post_json_sync,
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )

    async def post_sse(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> AsyncIterator[str]:
        """Yield decoded SSE data fields without blocking the event loop."""

        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                for event in self._post_sse_sync(
                    url=url,
                    headers=headers,
                    payload=payload,
                    timeout_s=timeout_s,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not worker.done():
                worker.cancel()

    @staticmethod
    def _post_json_sync(
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            detail = raw_error.strip() or str(exc.reason)
            raise ProviderHTTPError(
                f"Provider returned HTTP {exc.code}: {detail}",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderHTTPError(f"Provider request failed: {exc.reason}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderHTTPError("Provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ProviderHTTPError("Provider JSON response must be an object")
        return decoded

    @staticmethod
    def _post_sse_sync(
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Iterator[str]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                data_lines: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            yield "\n".join(data_lines)
                            data_lines.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data = line[5:]
                        data_lines.append(data[1:] if data.startswith(" ") else data)
                if data_lines:
                    yield "\n".join(data_lines)
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            detail = raw_error.strip() or str(exc.reason)
            raise ProviderHTTPError(
                f"Provider returned HTTP {exc.code}: {detail}",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderHTTPError(f"Provider request failed: {exc.reason}") from exc
