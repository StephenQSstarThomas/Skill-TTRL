"""
Logging utilities for Skill TTRL.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional


class StepLogger:
    """
    Structured step logger for detailed training information.

    Logs detailed per-step information including:
    - All answers before majority voting
    - The voted pseudo ground-truth
    - Rewards for each sample
    - Merged skills from external LLM
    """

    def __init__(self, output_dir: str | Path, prefix: str = "step_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"{prefix}.jsonl"
        self._logger = logging.getLogger("skill_ttrl.step_logger")

    def log_step(
        self,
        epoch: int,
        step: int,
        problem_idx: int,
        problem: str,
        all_answers: list[str],
        pseudo_gt: str,
        agreement_ratio: float,
        rewards: list[float],
        merged_skills: list[dict],
        raw_responses: Optional[list[str]] = None,
    ) -> None:
        """
        Log detailed information for a single problem in a training step.

        Args:
            epoch: Current epoch number.
            step: Current step number.
            problem_idx: Index of the problem in the batch.
            problem: The problem text.
            all_answers: All parsed answers before voting.
            pseudo_gt: The majority-voted pseudo ground-truth.
            agreement_ratio: Fraction of samples agreeing with pseudo-GT.
            rewards: Reward for each sample (1.0 if matches, 0.0 otherwise).
            merged_skills: Skills merged/extracted by external LLM.
            raw_responses: Optional full response texts.
        """
        log_entry = {
            "epoch": epoch,
            "step": step,
            "problem_idx": problem_idx,
            "problem": problem[:500],  # Truncate for readability
            "voting_details": {
                "all_answers": all_answers,
                "pseudo_gt": pseudo_gt,
                "agreement_ratio": agreement_ratio,
                "num_matching": sum(1 for r in rewards if r > 0.5),
                "num_total": len(rewards),
            },
            "rewards": rewards,
            "merged_skills": merged_skills[:3],  # Limit to 3 skills
        }

        if raw_responses:
            # Store truncated responses for debugging
            log_entry["responses_preview"] = [r[:300] for r in raw_responses[:5]]

        # Write to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Also log summary to console
        self._logger.info(
            f"[E{epoch}S{step}P{problem_idx}] "
            f"Pseudo-GT: '{pseudo_gt}' | "
            f"Agreement: {agreement_ratio:.1%} | "
            f"Winners: {sum(1 for r in rewards if r > 0.5)}/{len(rewards)} | "
            f"New skills: {len(merged_skills)}"
        )


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
