"""
Reward computation for Skill TTRL.

Combines majority-voting-based pseudo rewards with optional answer grading.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from skill_ttrl.core.majority_vote import (
    majority_vote,
    compute_voting_rewards,
    normalize_answer,
)
from skill_ttrl.core.output_parser import OutputParser, ParsedOutput
from skill_ttrl.utils.math_grader import compute_reward as grade_reward

logger = logging.getLogger(__name__)


class RewardManager:
    """
    Manages reward computation for Skill TTRL.

    Primary reward: majority voting (TTRL-style pseudo reward).
    Optional: grading against ground truth for metrics.
    """

    def __init__(
        self,
        format_weight: float = 0.0,
    ):
        self.format_weight = format_weight
        self.parser = OutputParser()

    def compute_majority_rewards(
        self,
        responses: list[str],
        prompt_ids: list[int],
        n_per_prompt: int,
    ) -> dict:
        """
        Compute majority-voting rewards for a batch of responses.

        Args:
            responses: All response texts (batch_size * n_per_prompt).
            prompt_ids: Prompt group ID for each response.
            n_per_prompt: Number of samples per prompt.

        Returns:
            Dictionary with:
            - rewards: torch.Tensor of shape (total_responses,)
            - majority_answers: list of majority-voted answers per prompt
            - agreement_ratios: list of agreement ratios per prompt
            - parsed_outputs: list of ParsedOutput for all responses
        """
        # Parse all responses
        parsed = self.parser.parse_batch(responses)

        # Group by prompt_id
        groups: dict[int, list[int]] = {}
        for i, pid in enumerate(prompt_ids):
            groups.setdefault(pid, []).append(i)

        rewards = torch.zeros(len(responses))
        majority_answers = {}
        agreement_ratios = {}

        for pid, indices in groups.items():
            # Extract answers for this group
            group_answers = [parsed[i].answer for i in indices]

            # Majority vote
            maj_answer, agreement, matches = majority_vote(group_answers)
            majority_answers[pid] = maj_answer
            agreement_ratios[pid] = agreement

            # Assign rewards
            for idx, match in zip(indices, matches):
                rewards[idx] = 1.0 if match else 0.0

        return {
            "rewards": rewards,
            "majority_answers": majority_answers,
            "agreement_ratios": agreement_ratios,
            "parsed_outputs": parsed,
        }

    def compute_gt_metrics(
        self,
        responses: list[str],
        ground_truths: list[str],
    ) -> dict:
        """
        Compute metrics against actual ground truth (for evaluation only).

        Args:
            responses: Response texts.
            ground_truths: Ground truth answers.

        Returns:
            Dictionary with accuracy and per-sample scores.
        """
        scores = []
        for resp, gt in zip(responses, ground_truths):
            result = grade_reward(resp, gt, format_weight=self.format_weight)
            scores.append(result["score"])

        accuracy = sum(1 for s in scores if s > 0.5) / max(len(scores), 1)
        return {
            "accuracy": accuracy,
            "scores": scores,
            "mean_score": sum(scores) / max(len(scores), 1),
        }

    def compute_label_accuracy(
        self,
        majority_answers: dict[int, str],
        ground_truths: dict[int, str],
    ) -> float:
        """
        Compute how often majority vote matches ground truth.

        This measures the quality of the pseudo-labeling.
        """
        if not majority_answers:
            return 0.0

        correct = 0
        total = 0
        for pid, maj_ans in majority_answers.items():
            gt = ground_truths.get(pid, "")
            if gt:
                total += 1
                na, nb = normalize_answer(maj_ans), normalize_answer(gt)
                if na == nb:
                    correct += 1

        return correct / max(total, 1)
