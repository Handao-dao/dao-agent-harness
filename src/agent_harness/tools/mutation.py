"""Per-file mutation serialization shared by write-oriented tools."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from os import PathLike
from os.path import normcase
from pathlib import Path
from typing import AsyncIterator


@dataclass(slots=True)
class _QueueEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class FileMutationQueue:
    """Serialize mutations to one canonical path while allowing other files."""

    def __init__(self) -> None:
        self._entries: dict[str, _QueueEntry] = {}

    @asynccontextmanager
    async def hold(self, path: str | PathLike[str]) -> AsyncIterator[None]:
        key = normcase(str(Path(path).resolve(strict=False)))
        entry = self._entries.setdefault(key, _QueueEntry())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0 and not entry.lock.locked():
                self._entries.pop(key, None)


DEFAULT_FILE_MUTATION_QUEUE = FileMutationQueue()


__all__ = ["DEFAULT_FILE_MUTATION_QUEUE", "FileMutationQueue"]
