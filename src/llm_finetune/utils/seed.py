"""Reproducibility helpers."""

import random


def set_seed(seed: int) -> None:
    """Set the Python random seed.

    Args:
        seed: Deterministic seed value.
    """
    random.seed(seed)
