"""Lightweight text-generation evaluation metrics."""

from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for lexical comparison."""
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def exact_match(prediction: str, reference: str) -> float:
    """Return one when normalized strings match exactly, otherwise zero."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between a prediction and reference."""
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return the length of the longest common subsequence."""
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """Compute the F1-style ROUGE-L score."""
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    lcs = _lcs_length(predicted, expected)
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
