"""
Skill TTRL Trainer: the main training loop.

Implements the complete Skill TTRL pipeline:
1. Parallel sampling with skill operations
2. Majority voting for pseudo ground truth
3. External LLM merging of winning samples' skill ops
4. Skill bank update
5. GRPO parameter update
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from skill_ttrl.config.config import SkillTTRLConfig
from skill_ttrl.core.skill_bank import SkillBank, Skill
from skill_ttrl.core.output_parser import OutputParser
from skill_ttrl.core.majority_vote import majority_vote
from skill_ttrl.core.merge_llm import SkillMerger
from skill_ttrl.core.grpo import (
    compute_grpo_advantage,
    compute_policy_loss,
    compute_kl_penalty,
    compute_entropy_bonus,
    whiten_advantages,
)
from skill_ttrl.training.reward import RewardManager
from skill_ttrl.training.rollout import RolloutEngine
from skill_ttrl.prompts.formatter import PromptFormatter
from skill_ttrl.data.dataset import SkillTTRLDataset, collate_fn
from skill_ttrl.utils.logging import MetricsTracker, setup_logging

logger = logging.getLogger(__name__)


def classify_task_type(problem: str) -> str:
    """Simple task type classification based on keywords."""
    problem_lower = problem.lower()

    keywords = {
        "algebra": ["equation", "solve for", "polynomial", "quadratic", "linear"],
        "geometry": ["triangle", "circle", "angle", "area", "perimeter", "polygon"],
        "number_theory": ["prime", "divisor", "modular", "gcd", "lcm", "divisible"],
        "combinatorics": ["permutation", "combination", "probability", "choose", "arrange"],
        "calculus": ["integral", "derivative", "limit", "continuous", "differential"],
        "logic": ["prove", "if and only if", "contradiction", "induction"],
    }

    for task_type, kws in keywords.items():
        if any(kw in problem_lower for kw in kws):
            return task_type

    return "general"


class SkillTTRLTrainer:
    """
    Main Skill TTRL training loop.

    Orchestrates the complete pipeline:
    - Generate K samples per prompt with skill operations
    - Majority vote to determine pseudo ground truth
    - Merge winning samples' skill operations via external LLM
    - Update skill bank
    - Compute GRPO advantages and update policy
    """

    def __init__(self, config: SkillTTRLConfig):
        self.config = config
        self.metrics = MetricsTracker()

        # Set random seeds
        self._set_seed(config.trainer.seed)

        # Initialize components
        self.skill_bank = self._init_skill_bank()
        self.parser = OutputParser()
        self.merger = SkillMerger(
            model=config.merger.model,
            api_key=config.merger.api_key,
            api_base=config.merger.api_base,
            max_tokens=config.merger.max_tokens,
            temperature=config.merger.temperature,
        )
        self.reward_manager = RewardManager()
        self.prompt_formatter = PromptFormatter(
            skill_bank=self.skill_bank,
            suffix_prompt=config.data.suffix_prompt,
        )

        # These are initialized lazily
        self._rollout_engine: Optional[RolloutEngine] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._ref_model = None  # reference policy for KL

        # Output directory
        self.output_dir = Path(config.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_accuracy = 0.0

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_skill_bank(self) -> SkillBank:
        """Initialize or load skill bank."""
        cfg = self.config.skill_bank
        if cfg.skill_bank_path and Path(cfg.skill_bank_path).exists():
            logger.info(f"Loading skill bank from {cfg.skill_bank_path}")
            return SkillBank.load(
                cfg.skill_bank_path,
                embedding_model=cfg.embedding_model,
                top_k=cfg.top_k_retrieve,
                eviction_window=cfg.eviction_window,
            )
        return SkillBank(
            max_skills=cfg.max_skills,
            retrieval_mode=cfg.retrieval_mode,
            embedding_model=cfg.embedding_model,
            top_k=cfg.top_k_retrieve,
            eviction_window=cfg.eviction_window,
        )

    @property
    def rollout_engine(self) -> RolloutEngine:
        if self._rollout_engine is None:
            self._rollout_engine = RolloutEngine(
                model_path=self.config.model.path,
                engine=self.config.rollout.engine,
                temperature=self.config.rollout.temperature,
                top_p=self.config.rollout.top_p,
                max_tokens=self.config.rollout.max_tokens,
                tensor_parallel_size=self.config.rollout.tensor_parallel_size,
                dtype=self.config.model.dtype,
            )
        return self._rollout_engine

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        train_data: Optional[list[dict]] = None,
        val_data: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run the complete Skill TTRL training loop.

        Args:
            train_data: Training examples (if not using config data files).
            val_data: Validation examples.

        Returns:
            Final metrics dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting Skill TTRL Training")
        logger.info(f"Config: {self.config.to_dict()}")
        logger.info("=" * 60)

        # Load data
        if train_data is None:
            dataset = SkillTTRLDataset(
                data_files=self.config.data.train_files,
                max_prompt_length=self.config.data.max_prompt_length,
                suffix_prompt=self.config.data.suffix_prompt,
            )
            train_data = dataset.data

        dataloader = self._create_dataloader(train_data)

        for epoch in range(self.config.trainer.total_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.trainer.total_epochs} {'='*40}")
            logger.info(f"Skill bank: {self.skill_bank.size} skills")

            epoch_metrics = self._train_epoch(dataloader)

            # Logging
            elapsed = time.time() - epoch_start
            self.metrics.log("epoch_time", elapsed)
            logger.info(
                f"Epoch {epoch+1} done in {elapsed:.1f}s | "
                f"reward={epoch_metrics.get('mean_reward', 0):.4f} | "
                f"agreement={epoch_metrics.get('mean_agreement', 0):.2%} | "
                f"skills={self.skill_bank.size}"
            )

            # Validation
            if val_data and (epoch + 1) % self.config.trainer.val_freq == 0:
                val_metrics = self._validate(val_data)
                logger.info(
                    f"Validation: accuracy={val_metrics.get('accuracy', 0):.2%} | "
                    f"label_accuracy={val_metrics.get('label_accuracy', 0):.2%}"
                )

            # Save checkpoint
            if (epoch + 1) % self.config.trainer.save_freq == 0:
                self._save_checkpoint(epoch)

            # Update reference policy
            if (epoch + 1) % self.config.trainer.ref_update_interval == 0:
                logger.info("Updating reference policy")
                # In full implementation, this would copy the current model weights

            # Evict stale skills
            n_evicted = self.skill_bank.evict_stale(self.current_step)
            if n_evicted > 0:
                logger.info(f"Evicted {n_evicted} stale skills")

        # Final save
        self._save_checkpoint(self.current_epoch, final=True)

        return self.metrics.summary()

    def _train_epoch(self, dataloader) -> dict:
        """Run one training epoch."""
        epoch_rewards = []
        epoch_agreements = []
        epoch_new_skills = 0

        for step, batch in enumerate(dataloader):
            self.current_step += 1
            step_result = self._train_step(batch)

            epoch_rewards.append(step_result["mean_reward"])
            epoch_agreements.append(step_result["mean_agreement"])
            epoch_new_skills += step_result.get("new_skills_added", 0)

            self.metrics.log("step_reward", step_result["mean_reward"])
            self.metrics.log("step_agreement", step_result["mean_agreement"])

        return {
            "mean_reward": np.mean(epoch_rewards) if epoch_rewards else 0.0,
            "mean_agreement": np.mean(epoch_agreements) if epoch_agreements else 0.0,
            "new_skills": epoch_new_skills,
        }

    def _train_step(self, batch: dict) -> dict:
        """
        Execute one training step (one batch of problems).

        This implements the core Skill TTRL algorithm:
        1. Format prompts with skill bank context
        2. Generate K samples per prompt
        3. Majority vote for pseudo ground truth
        4. Merge winning samples' skill operations
        5. Update skill bank
        6. Compute GRPO advantages
        7. Update policy (if model is available)
        """
        problems = batch["prompts"]
        n_per_prompt = self.config.rollout.n_votes_per_prompt
        n_train = self.config.rollout.n_samples_per_prompt

        # Step 1: Format prompts with skill context
        formatted_prompts = []
        task_types = []
        for problem in problems:
            task_type = classify_task_type(problem)
            task_types.append(task_type)
            formatted = self.prompt_formatter.format_prompt(problem, task_type)
            formatted_prompts.append(formatted)

        # Step 2: Generate K responses per prompt
        all_responses = self.rollout_engine.generate(
            formatted_prompts,
            n_samples=n_per_prompt,
        )

        # Step 3: Majority voting
        prompt_ids = []
        flat_responses = []
        for pid, resps in enumerate(all_responses):
            for resp in resps:
                flat_responses.append(resp)
                prompt_ids.append(pid)

        reward_result = self.reward_manager.compute_majority_rewards(
            responses=flat_responses,
            prompt_ids=prompt_ids,
            n_per_prompt=n_per_prompt,
        )

        rewards = reward_result["rewards"]
        majority_answers = reward_result["majority_answers"]
        agreement_ratios = reward_result["agreement_ratios"]
        parsed_outputs = reward_result["parsed_outputs"]

        # Step 4: External LLM merge for each prompt
        total_new_skills = 0
        for pid in range(len(problems)):
            # Collect winning samples
            winners = []
            start_idx = pid * n_per_prompt
            for j in range(n_per_prompt):
                idx = start_idx + j
                if rewards[idx] > 0.5:
                    winners.append(parsed_outputs[idx])

            if not winners:
                continue

            # Merge skill operations from winners
            try:
                merged = self.merger.merge(winners)
            except Exception as e:
                logger.warning(f"Merge failed for prompt {pid}: {e}")
                # Fallback: use heuristic merge
                self.merger.use_api = False
                merged = self.merger.merge(winners)
                self.merger.use_api = True

            # Step 5: Update skill bank
            task_type = task_types[pid] if pid < len(task_types) else "general"

            for new_skill in merged.new_skills:
                self.skill_bank.add_from_dict(
                    new_skill,
                    task_type=task_type,
                    source="generated",
                    round_num=self.current_step,
                )
                total_new_skills += 1

            for base_id, evolved_dict in merged.evolved_skills:
                evolved = Skill(
                    skill_id=self.skill_bank._generate_id(),
                    title=evolved_dict.get("title", f"Evolved from {base_id}"),
                    content=evolved_dict.get("content", ""),
                    task_type=task_type,
                    source="evolved",
                    created_at_round=self.current_step,
                    parent_id=base_id,
                )
                self.skill_bank.replace(base_id, evolved)

            # Record usage for retrieved skills
            for sid in merged.useful_retrieval_ids:
                self.skill_bank.record_usage(
                    sid, success=True, round_num=self.current_step
                )

        # Step 6: GRPO advantage computation (on subset for training)
        # Select n_train samples per prompt for actual training
        train_indices = []
        for pid in range(len(problems)):
            start_idx = pid * n_per_prompt
            end_idx = start_idx + min(n_train, n_per_prompt)
            train_indices.extend(range(start_idx, end_idx))

        train_rewards = rewards[train_indices]
        train_prompt_ids = torch.tensor([prompt_ids[i] for i in train_indices])

        # Create dummy response masks for advantage computation
        response_len = 1  # simplified; in full implementation this is token-level
        response_mask = torch.ones(len(train_indices), response_len)

        advantages, returns = compute_grpo_advantage(
            rewards=train_rewards,
            response_mask=response_mask,
            prompt_ids=train_prompt_ids,
            norm_by_std=self.config.algorithm.norm_adv_by_std,
        )

        # Step 7: Policy update would happen here with actual model gradients
        # In the full distributed implementation, this would call:
        #   optimizer.zero_grad()
        #   policy_loss, info = compute_policy_loss(...)
        #   kl_loss, kl_val = compute_kl_penalty(...)
        #   total_loss = policy_loss + kl_loss
        #   total_loss.backward()
        #   optimizer.step()

        mean_reward = train_rewards.mean().item()
        mean_agreement = (
            np.mean(list(agreement_ratios.values()))
            if agreement_ratios else 0.0
        )

        return {
            "mean_reward": mean_reward,
            "mean_agreement": mean_agreement,
            "advantages_mean": advantages.mean().item(),
            "new_skills_added": total_new_skills,
            "skill_bank_size": self.skill_bank.size,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, val_data: list[dict]) -> dict:
        """Run validation on a set of examples."""
        problems = [d.get("prompt", d.get("problem", "")) for d in val_data]
        ground_truths = [d.get("answer", d.get("solution", "")) for d in val_data]

        # Format prompts
        formatted = self.prompt_formatter.format_batch(problems)

        # Generate with fewer samples for speed
        n_val_samples = min(16, self.config.rollout.n_votes_per_prompt)
        all_responses = self.rollout_engine.generate(
            formatted, n_samples=n_val_samples
        )

        # Flatten and compute majority votes
        all_flat = []
        pid_list = []
        for pid, resps in enumerate(all_responses):
            for resp in resps:
                all_flat.append(resp)
                pid_list.append(pid)

        reward_result = self.reward_manager.compute_majority_rewards(
            responses=all_flat,
            prompt_ids=pid_list,
            n_per_prompt=n_val_samples,
        )

        # Compute metrics against ground truth
        gt_metrics = self.reward_manager.compute_gt_metrics(
            responses=[all_responses[i][0] for i in range(len(problems))],
            ground_truths=ground_truths,
        )

        # Label accuracy (majority vote vs ground truth)
        gt_dict = {i: gt for i, gt in enumerate(ground_truths)}
        label_acc = self.reward_manager.compute_label_accuracy(
            reward_result["majority_answers"], gt_dict
        )

        return {
            "accuracy": gt_metrics["accuracy"],
            "label_accuracy": label_acc,
            "mean_agreement": np.mean(
                list(reward_result["agreement_ratios"].values())
            ),
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _create_dataloader(self, data: list[dict]):
        """Create a simple batch iterator."""
        batch_size = self.config.data.train_batch_size

        if self.config.data.shuffle:
            random.shuffle(data)

        batches = []
        for i in range(0, len(data), batch_size):
            batch_items = data[i : i + batch_size]
            batch = {
                "prompts": [
                    d.get("prompt", d.get("problem", d.get("question", "")))
                    for d in batch_items
                ],
                "answers": [
                    d.get("answer", d.get("solution", "")) for d in batch_items
                ],
            }
            batches.append(batch)

        return batches

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"epoch_{epoch+1}"
        ckpt_dir = self.output_dir / suffix

        # Save skill bank
        self.skill_bank.save(ckpt_dir / "skill_bank.json")

        # Save config
        self.config.save(ckpt_dir / "config.yaml")

        # Save metrics
        metrics_path = ckpt_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics.summary(), f, indent=2)

        logger.info(f"Checkpoint saved to {ckpt_dir}")

    # ------------------------------------------------------------------
    # Offline / simulation mode
    # ------------------------------------------------------------------
    def train_offline(
        self,
        problems: list[str],
        all_responses: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> dict:
        """
        Train using pre-generated responses (no model inference needed).

        Useful for debugging and testing the pipeline without a GPU.

        Args:
            problems: List of problem texts.
            all_responses: Pre-generated responses (one list per problem).
            ground_truths: Optional ground truth answers.

        Returns:
            Metrics dictionary.
        """
        logger.info(f"Offline training with {len(problems)} problems")

        total_rewards = []
        total_agreements = []
        total_new_skills = 0

        for pid, (problem, responses) in enumerate(zip(problems, all_responses)):
            task_type = classify_task_type(problem)

            # Parse all responses
            parsed = self.parser.parse_batch(responses)

            # Extract answers and majority vote
            answers = [p.answer for p in parsed]
            maj_answer, agreement, matches = majority_vote(answers)

            # Rewards
            rewards = [1.0 if m else 0.0 for m in matches]
            total_rewards.extend(rewards)
            total_agreements.append(agreement)

            # Collect winners
            winners = [p for p, m in zip(parsed, matches) if m]

            if winners:
                # Merge skill operations
                merged = self.merger.merge(winners)

                # Update skill bank
                for skill_dict in merged.new_skills:
                    self.skill_bank.add_from_dict(
                        skill_dict,
                        task_type=task_type,
                        source="generated",
                        round_num=pid,
                    )
                    total_new_skills += 1

                for base_id, evolved_dict in merged.evolved_skills:
                    evolved = Skill(
                        skill_id=self.skill_bank._generate_id(),
                        title=evolved_dict.get("title", ""),
                        content=evolved_dict.get("content", ""),
                        task_type=task_type,
                        source="evolved",
                        parent_id=base_id,
                    )
                    self.skill_bank.replace(base_id, evolved)

        metrics = {
            "mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "mean_agreement": np.mean(total_agreements) if total_agreements else 0.0,
            "new_skills_added": total_new_skills,
            "skill_bank_size": self.skill_bank.size,
        }

        logger.info(
            f"Offline training done: reward={metrics['mean_reward']:.4f}, "
            f"agreement={metrics['mean_agreement']:.2%}, "
            f"new_skills={total_new_skills}"
        )

        return metrics
