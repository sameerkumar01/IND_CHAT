"""Logging configuration for the project."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent timestamped console logging.

    Args:
        level: Logging threshold for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
