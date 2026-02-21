"""Skill TTRL: Training models to autonomously generate, retrieve, and evolve skills."""

__version__ = "0.1.0"

from skill_ttrl.core.skill_bank import SkillBank
from skill_ttrl.core.output_parser import OutputParser
from skill_ttrl.core.majority_vote import majority_vote
from skill_ttrl.core.merge_llm import SkillMerger
from skill_ttrl.core.grpo import compute_grpo_advantage, compute_policy_loss
from skill_ttrl.training.trainer import SkillTTRLTrainer

__all__ = [
    "SkillBank",
    "OutputParser",
    "majority_vote",
    "SkillMerger",
    "compute_grpo_advantage",
    "compute_policy_loss",
    "SkillTTRLTrainer",
]
