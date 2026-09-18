"""Source ingestion boundary for future PDF and HTML adapters."""

from __future__ import annotations

from pathlib import Path


def read_utf8_text(path: str | Path) -> str:
    """Read one UTF-8 text source with a clear error message.

    Args:
        path: Source file path.

    Returns:
        Source text.

    Raises:
        UnicodeError: If the source is not valid UTF-8.
        OSError: If the file cannot be read.
    """
    source_path = Path(path)
    return source_path.read_text(encoding="utf-8")
