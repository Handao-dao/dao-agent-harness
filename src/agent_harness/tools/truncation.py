"""Line- and UTF-8-byte-aware truncation for model-visible tool results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_CHARS = 500


@dataclass(frozen=True, slots=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    max_lines: int
    max_bytes: int
    first_line_exceeds_limit: bool = False
    boundary_line_partial: bool = False


def _validate_limits(max_lines: int, max_bytes: int) -> None:
    if not isinstance(max_lines, int) or isinstance(max_lines, bool) or max_lines <= 0:
        raise ValueError("max_lines must be a positive integer")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")


def _lines(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _result_without_truncation(
    content: str,
    *,
    lines: list[str],
    total_bytes: int,
    max_lines: int,
    max_bytes: int,
) -> TruncationResult:
    return TruncationResult(
        content=content,
        truncated=False,
        truncated_by=None,
        total_lines=len(lines),
        total_bytes=total_bytes,
        output_lines=len(lines),
        output_bytes=total_bytes,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep complete lines from the beginning, whichever limit is reached first."""

    _validate_limits(max_lines, max_bytes)
    lines = _lines(content)
    total_bytes = len(content.encode("utf-8"))
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return _result_without_truncation(
            content,
            lines=lines,
            total_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=len(lines),
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            max_lines=max_lines,
            max_bytes=max_bytes,
            first_line_exceeds_limit=True,
        )

    output: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for line in lines[:max_lines]:
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output.append(line)
        output_bytes += line_bytes

    return TruncationResult(
        content="\n".join(output),
        truncated=True,
        truncated_by=truncated_by,
        total_lines=len(lines),
        total_bytes=total_bytes,
        output_lines=len(output),
        output_bytes=output_bytes,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _utf8_tail(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and encoded[start] & 0xC0 == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the end of output, allowing one partial boundary line as a fallback."""

    _validate_limits(max_lines, max_bytes)
    lines = _lines(content)
    total_bytes = len(content.encode("utf-8"))
    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return _result_without_truncation(
            content,
            lines=lines,
            total_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    boundary_line_partial = False
    for line in reversed(lines[-max_lines:]):
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output:
                partial = _utf8_tail(line, max_bytes)
                output.append(partial)
                output_bytes = len(partial.encode("utf-8"))
                boundary_line_partial = True
            break
        output.append(line)
        output_bytes += line_bytes
    output.reverse()

    return TruncationResult(
        content="\n".join(output),
        truncated=True,
        truncated_by=truncated_by,
        total_lines=len(lines),
        total_bytes=total_bytes,
        output_lines=len(output),
        output_bytes=output_bytes,
        max_lines=max_lines,
        max_bytes=max_bytes,
        boundary_line_partial=boundary_line_partial,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_CHARS) -> tuple[str, bool]:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True


def format_size(size_bytes: int) -> str:
    if size_bytes < 1_024:
        return f"{size_bytes}B"
    if size_bytes < 1_024 * 1_024:
        return f"{size_bytes / 1_024:.1f}KB"
    return f"{size_bytes / (1_024 * 1_024):.1f}MB"


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_CHARS",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
]
