"""Creates and reuses standardized application loggers with consistent formatting."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the specified name, configured with a consistent format and INFO level."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
