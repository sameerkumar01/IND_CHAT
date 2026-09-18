"""Text cleaning and PII handling."""

from __future__ import annotations

import re

from .validator import InstructionExample, PiiRedaction, redact_pii


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving Unicode scripts.

    Args:
        text: Raw text to normalize.

    Returns:
        Cleaned UTF-8-compatible text.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_example(example: InstructionExample, redact_sensitive: bool = True) -> tuple[InstructionExample, list[PiiRedaction]]:
    """Clean one validated example and optionally redact PII.

    Args:
        example: Example to clean.
        redact_sensitive: Whether to apply common PII redactions.

    Returns:
        Cleaned example and a list of applied redactions.
    """
    redactions: list[PiiRedaction] = []
    values = {}
    for field in ("instruction", "input", "output", "topic", "source"):
        value = clean_text(getattr(example, field))
        if redact_sensitive:
            value, field_redactions = redact_pii(value)
            redactions.extend(field_redactions)
        values[field] = value
    return InstructionExample(**values), redactions
