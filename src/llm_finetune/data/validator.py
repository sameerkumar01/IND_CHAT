"""Validation and safety checks for instruction-tuning examples."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InstructionExample(BaseModel):
    """A validated instruction-tuning example.

    Args:
        instruction: The user-facing task or question.
        input: Optional supporting context.
        output: The target assistant response.
        topic: Optional topic label used for balance reporting.
        source: Optional source URL or document identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=10)
    topic: str = "unlabelled"
    source: str = "unknown"

    @field_validator("instruction", "output")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain no non-whitespace characters."""
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


@dataclass(frozen=True)
class PiiRedaction:
    """A record of one redaction applied to an example."""

    kind: str
    count: int


_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aadhaar", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)")),
)


def redact_pii(text: str) -> tuple[str, list[PiiRedaction]]:
    """Redact common Indian PII patterns from text.

    Args:
        text: Text to scan and redact.

    Returns:
        A tuple containing redacted text and redaction counts.
    """
    redactions: list[PiiRedaction] = []
    redacted = text
    for kind, pattern in _PII_PATTERNS:
        redacted, count = pattern.subn(f"[{kind.upper()}_REDACTED]", redacted)
        if count:
            redactions.append(PiiRedaction(kind=kind, count=count))
    return redacted, redactions


def normalize_for_comparison(example: InstructionExample) -> str:
    """Create a whitespace-normalized string for duplicate detection."""
    combined = " ".join((example.instruction, example.input, example.output))
    return " ".join(combined.casefold().split())


def find_near_duplicate_indices(
    examples: Iterable[InstructionExample], threshold: float = 0.92
) -> set[int]:
    """Return indices of examples that duplicate an earlier example.

    Args:
        examples: Examples in deterministic input order.
        threshold: Similarity ratio at or above which an example is a duplicate.

    Returns:
        Indices to remove, keeping the first occurrence.
    """
    normalized = [normalize_for_comparison(example) for example in examples]
    token_sets = [set(text.split()) for text in normalized]
    duplicate_indices: set[int] = set()
    for index, candidate_tokens in enumerate(token_sets):
        for previous_tokens in token_sets[:index]:
            union = candidate_tokens | previous_tokens
            similarity = len(candidate_tokens & previous_tokens) / len(union) if union else 1.0
            if similarity >= threshold:
                duplicate_indices.add(index)
                break
    return duplicate_indices
