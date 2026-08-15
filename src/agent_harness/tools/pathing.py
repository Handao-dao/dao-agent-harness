"""Workspace-relative path resolution shared by file tools."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path


class ToolPathError(ValueError):
    """Raised when a model-supplied path violates the workspace policy."""


@dataclass(frozen=True, slots=True)
class ToolPathPolicy:
    """Resolve model paths against one workspace and contain them by default."""

    workspace: Path
    allow_outside_workspace: bool = False

    def __init__(
        self,
        workspace: str | PathLike[str],
        *,
        allow_outside_workspace: bool = False,
    ) -> None:
        if not isinstance(allow_outside_workspace, bool):
            raise TypeError("allow_outside_workspace must be boolean")
        root = Path(workspace).expanduser().resolve()
        if not root.exists():
            raise ToolPathError(f"Workspace does not exist: {root}")
        if not root.is_dir():
            raise ToolPathError(f"Workspace is not a directory: {root}")
        object.__setattr__(self, "workspace", root)
        object.__setattr__(self, "allow_outside_workspace", allow_outside_workspace)

    def resolve(self, raw_path: str) -> Path:
        """Return a canonical path or reject a path escaping the workspace."""

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolPathError("path must be non-empty text")
        supplied = Path(raw_path).expanduser()
        candidate = supplied if supplied.is_absolute() else self.workspace / supplied
        resolved = candidate.resolve(strict=False)
        if not self.allow_outside_workspace:
            try:
                resolved.relative_to(self.workspace)
            except ValueError as exc:
                raise ToolPathError(f"Path is outside the workspace: {raw_path}") from exc
        return resolved

    def display(self, path: Path) -> str:
        """Prefer a stable workspace-relative path for model-visible output."""

        resolved = path.resolve(strict=False)
        try:
            return resolved.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(resolved)


__all__ = ["ToolPathError", "ToolPathPolicy"]
