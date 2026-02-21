"""
Logging utilities for Skill TTRL.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure logging for the Skill TTRL project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs to.

    Returns:
        Root logger configured for the project.
    """
    logger = logging.getLogger("skill_ttrl")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricsTracker:
    """Simple metrics tracker for training runs."""

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}

    def log(self, key: str, value: float) -> None:
        self._metrics.setdefault(key, []).append(value)

    def get_latest(self, key: str) -> float | None:
        values = self._metrics.get(key)
        return values[-1] if values else None

    def get_mean(self, key: str, window: int = 10) -> float | None:
        values = self._metrics.get(key)
        if not values:
            return None
        recent = values[-window:]
        return sum(recent) / len(recent)

    def get_all(self, key: str) -> list[float]:
        return self._metrics.get(key, [])

    def summary(self) -> dict[str, float]:
        return {
            key: values[-1] if values else 0.0
            for key, values in self._metrics.items()
        }

    def reset(self) -> None:
        self._metrics.clear()
