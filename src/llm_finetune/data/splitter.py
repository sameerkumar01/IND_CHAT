"""Deterministic dataset deduplication and splitting."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .validator import InstructionExample, find_near_duplicate_indices, normalize_for_comparison


@dataclass(frozen=True)
class SplitResult:
    """Dataset splits and processing statistics."""

    train: list[InstructionExample]
    validation: list[InstructionExample]
    evaluation: list[InstructionExample]
    duplicate_count: int

    @property
    def topic_counts(self) -> dict[str, int]:
        """Return topic counts across all retained examples."""
        return dict(Counter(example.topic for example in self.all_examples))

    @property
    def all_examples(self) -> list[InstructionExample]:
        """Return retained examples in split order."""
        return [*self.train, *self.validation, *self.evaluation]


def split_examples(
    examples: Sequence[InstructionExample],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
    near_duplicate_threshold: float = 0.92,
) -> SplitResult:
    """Deduplicate and split examples without cross-split leakage.

    Args:
        examples: Validated, cleaned examples.
        train_ratio: Fraction assigned to training.
        validation_ratio: Fraction assigned to validation.
        seed: Deterministic shuffle seed.
        near_duplicate_threshold: Similarity threshold for deduplication.

    Returns:
        A deterministic split result.

    Raises:
        ValueError: If ratios are invalid or a split would contain leakage.
    """
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios must be between zero and one")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than one")

    duplicate_indices = find_near_duplicate_indices(examples, near_duplicate_threshold)
    retained = [example for index, example in enumerate(examples) if index not in duplicate_indices]
    shuffled = list(retained)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    result = SplitResult(
        train=shuffled[:train_end],
        validation=shuffled[train_end:validation_end],
        evaluation=shuffled[validation_end:],
        duplicate_count=len(duplicate_indices),
    )
    split_keys = [set(normalize_for_comparison(example) for example in split) for split in (result.train, result.validation, result.evaluation)]
    if split_keys[0] & split_keys[1] or split_keys[0] & split_keys[2] or split_keys[1] & split_keys[2]:
        raise ValueError("dataset leakage detected across splits")
    return result
