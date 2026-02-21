"""
Integration test: end-to-end pipeline test using offline mode.

Tests the complete Skill TTRL pipeline without requiring a GPU or API:
- Prompt formatting with skill bank
- Output parsing
- Majority voting
- Skill merging (heuristic)
- Skill bank updates
- GRPO advantage computation
"""

import pytest
import torch

from skill_ttrl.config.config import SkillTTRLConfig
from skill_ttrl.core.skill_bank import SkillBank, Skill
from skill_ttrl.core.output_parser import OutputParser
from skill_ttrl.core.majority_vote import majority_vote, compute_voting_rewards
from skill_ttrl.core.merge_llm import SkillMerger
from skill_ttrl.core.grpo import compute_grpo_advantage, compute_policy_loss
from skill_ttrl.prompts.formatter import PromptFormatter
from skill_ttrl.training.reward import RewardManager
from skill_ttrl.training.trainer import SkillTTRLTrainer, classify_task_type
from skill_ttrl.utils.math_grader import compute_reward


class TestEndToEndPipeline:
    """Integration test for the complete Skill TTRL pipeline."""

    def _simulate_responses(self, n_samples: int = 8) -> list[str]:
        """Generate simulated model responses for testing."""
        responses = []

        # 5 correct responses with varying skill operations
        for i in range(5):
            if i == 0:
                # Generate + correct answer
                resp = (
                    "<skill_ops>\n"
                    "  <generate>For modular arithmetic, check patterns in powers.</generate>\n"
                    "</skill_ops>\n"
                    "<solution>2^10 = 1024. 1024 mod 7 = 2.</solution>\n"
                    "<answer>2</answer>"
                )
            elif i == 1:
                # Retrieve + correct answer
                resp = (
                    "<skill_ops>\n"
                    '  <retrieve query="modular arithmetic powers">'
                    "Skill #sk_001: Use Fermat's theorem."
                    "</retrieve>\n"
                    "</skill_ops>\n"
                    "<solution>By Fermat's, 2^6≡1 mod 7. 2^10=2^6*2^4≡2^4=16≡2.</solution>\n"
                    "<answer>2</answer>"
                )
            elif i == 2:
                # Evolve + correct answer
                resp = (
                    "<skill_ops>\n"
                    '  <evolve base="#sk_001">'
                    "Improved: Also handle composite moduli via CRT."
                    "</evolve>\n"
                    "</skill_ops>\n"
                    "<solution>Direct computation: 2^10 mod 7 = 1024 mod 7 = 2.</solution>\n"
                    "<answer>2</answer>"
                )
            elif i == 3:
                # All three ops + correct answer
                resp = (
                    "<skill_ops>\n"
                    "  <generate>Powers of 2 cycle mod 7 with period 3.</generate>\n"
                    '  <retrieve query="number theory cycles">'
                    "Skill #sk_002: Check for cyclic patterns."
                    "</retrieve>\n"
                    '  <evolve base="#sk_002">'
                    "Enhanced: Use order of element to find cycle length."
                    "</evolve>\n"
                    "</skill_ops>\n"
                    "<solution>2^1=2, 2^2=4, 2^3=1 mod 7. Period=3. 10 mod 3=1. Answer=2^1=2.</solution>\n"
                    "<answer>2</answer>"
                )
            else:
                # No ops + correct answer
                resp = (
                    "<solution>2^10 = 1024 = 146*7 + 2, so 1024 mod 7 = 2.</solution>\n"
                    "<answer>2</answer>"
                )
            responses.append(resp)

        # 3 incorrect responses
        for wrong in ["3", "1", "4"]:
            responses.append(
                f"<solution>Some wrong reasoning.</solution>\n"
                f"<answer>{wrong}</answer>"
            )

        return responses[:n_samples]

    def test_full_pipeline(self):
        """Test the complete pipeline from responses to skill bank update."""
        # Step 1: Setup
        skill_bank = SkillBank(max_skills=50, retrieval_mode="keyword", top_k=3)
        skill_bank.add(Skill(
            skill_id="sk_001",
            title="Fermat's Little Theorem",
            content="For prime p, a^(p-1) ≡ 1 mod p.",
            task_type="number_theory",
        ))
        skill_bank.add(Skill(
            skill_id="sk_002",
            title="Cyclic Patterns",
            content="Look for cyclic patterns in sequences.",
            task_type="number_theory",
        ))

        parser = OutputParser()
        merger = SkillMerger(use_api=False)

        # Step 2: Simulate responses
        responses = self._simulate_responses(8)

        # Step 3: Parse all responses
        parsed = parser.parse_batch(responses)
        assert len(parsed) == 8

        # Verify parsing
        assert parsed[0].skill_ops.has_generate
        assert parsed[1].skill_ops.has_retrieve
        assert parsed[2].skill_ops.has_evolve
        assert parsed[3].has_skill_ops  # all three
        assert not parsed[4].has_skill_ops  # no ops

        # Step 4: Majority voting
        answers = [p.answer for p in parsed]
        maj_answer, agreement, matches = majority_vote(answers)

        assert maj_answer == "2"
        assert agreement == 5 / 8
        assert sum(matches) == 5

        # Step 5: Collect winners
        winners = [p for p, m in zip(parsed, matches) if m]
        assert len(winners) == 5

        # Step 6: Merge skill operations
        merged = merger.merge(winners)

        # Should have some new skills from generate ops
        assert len(merged.new_skills) >= 1

        # Should have useful retrievals (sk_001 appears in multiple winners)
        # Note: heuristic merge uses simple text matching

        # Step 7: Update skill bank
        initial_size = skill_bank.size
        for skill_dict in merged.new_skills:
            skill_bank.add_from_dict(
                skill_dict, task_type="number_theory", source="generated",
            )

        assert skill_bank.size >= initial_size

        # Step 8: GRPO advantage computation
        rewards = torch.tensor(
            [1.0 if m else 0.0 for m in matches]
        )
        prompt_ids = torch.zeros(8, dtype=torch.long)  # all same prompt
        response_mask = torch.ones(8, 1)

        advantages, returns = compute_grpo_advantage(
            rewards, response_mask, prompt_ids
        )

        # Winners should have positive advantages
        for i in range(5):
            assert advantages[i, 0].item() > 0
        # Losers should have negative advantages
        for i in range(5, 8):
            assert advantages[i, 0].item() < 0

    def test_offline_trainer(self):
        """Test the SkillTTRLTrainer in offline mode."""
        config = SkillTTRLConfig()
        config.merger.model = "gpt-4o"  # won't be used in offline mode
        config.skill_bank.retrieval_mode = "keyword"

        trainer = SkillTTRLTrainer(config)

        # Use heuristic merge (no API)
        trainer.merger.use_api = False

        problems = [
            "Find 2^10 mod 7.",
            "What is the sum of 1+2+...+100?",
        ]

        all_responses = [
            self._simulate_responses(8),
            [
                "<answer>5050</answer>",
                "<answer>5050</answer>",
                "<answer>5050</answer>",
                "<answer>5050</answer>",
                "<skill_ops><generate>Use Gauss formula n*(n+1)/2.</generate></skill_ops>\n<answer>5050</answer>",
                "<answer>5000</answer>",
                "<answer>5100</answer>",
                "<answer>5050</answer>",
            ],
        ]

        metrics = trainer.train_offline(
            problems=problems,
            all_responses=all_responses,
        )

        assert metrics["mean_reward"] > 0
        assert metrics["mean_agreement"] > 0
        assert metrics["skill_bank_size"] >= 0

    def test_reward_manager(self):
        """Test RewardManager with majority voting."""
        rm = RewardManager()

        responses = [
            "<answer>42</answer>",
            "<answer>42</answer>",
            "<answer>42</answer>",
            "<answer>43</answer>",
        ]

        result = rm.compute_majority_rewards(
            responses=responses,
            prompt_ids=[0, 0, 0, 0],
            n_per_prompt=4,
        )

        assert result["rewards"][0].item() == 1.0
        assert result["rewards"][3].item() == 0.0
        assert result["majority_answers"][0] == "42"
        assert result["agreement_ratios"][0] == 0.75

    def test_classify_task_type(self):
        assert classify_task_type("Solve the quadratic equation") == "algebra"
        assert classify_task_type("Find the area of a triangle") == "geometry"
        assert classify_task_type("Find all prime divisors") == "number_theory"
        assert classify_task_type("How many combinations") == "combinatorics"
        assert classify_task_type("Hello world") == "general"

    def test_config_save_load(self):
        """Test configuration serialization."""
        import tempfile
        from pathlib import Path

        config = SkillTTRLConfig()
        config.trainer.total_epochs = 10
        config.rollout.n_votes_per_prompt = 32

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save(path)

            from skill_ttrl.config.config import load_config
            loaded = load_config(path)
            assert loaded.trainer.total_epochs == 10
            assert loaded.rollout.n_votes_per_prompt == 32

    def test_policy_loss_integration(self):
        """Test policy loss computation with realistic shapes."""
        batch_size = 4
        seq_len = 20

        log_probs = torch.randn(batch_size, seq_len)
        old_log_probs = log_probs + torch.randn_like(log_probs) * 0.1
        advantages = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        # Simulate variable-length responses
        mask[0, 15:] = 0
        mask[1, 10:] = 0
        mask[2, 18:] = 0

        loss, info = compute_policy_loss(
            log_probs, old_log_probs, advantages, mask
        )

        assert loss.requires_grad is False  # no graph since inputs don't require grad
        assert 0 <= info["clip_fraction"] <= 1
        assert info["ratio_mean"] > 0
