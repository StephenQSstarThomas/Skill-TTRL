"""Tests for GRPO algorithm implementation."""

import pytest
import torch

from skill_ttrl.core.grpo import (
    compute_grpo_advantage,
    compute_gae_advantage,
    compute_policy_loss,
    compute_kl_penalty,
    compute_entropy_bonus,
    whiten_advantages,
)


class TestGRPOAdvantage:
    """Tests for GRPO advantage computation."""

    def test_basic_grpo(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        response_mask = torch.ones(4, 5)
        prompt_ids = torch.tensor([0, 0, 1, 1])

        advantages, returns = compute_grpo_advantage(
            rewards, response_mask, prompt_ids
        )

        assert advantages.shape == (4, 5)
        # Within group 0: reward 1.0 should have positive advantage
        assert advantages[0, 0].item() > 0
        assert advantages[1, 0].item() < 0
        # Within group 1: same pattern
        assert advantages[2, 0].item() > 0
        assert advantages[3, 0].item() < 0

    def test_grpo_uniform_rewards(self):
        rewards = torch.tensor([1.0, 1.0])
        response_mask = torch.ones(2, 3)
        prompt_ids = torch.tensor([0, 0])

        advantages, _ = compute_grpo_advantage(
            rewards, response_mask, prompt_ids
        )
        # All same reward → advantages should be ~0
        assert torch.abs(advantages).max().item() < 1e-5

    def test_grpo_mask(self):
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float)
        prompt_ids = torch.tensor([0, 0])

        advantages, _ = compute_grpo_advantage(rewards, mask, prompt_ids)
        # Masked positions should be 0
        assert advantages[0, 2].item() == 0.0
        assert advantages[1, 1].item() == 0.0
        assert advantages[1, 2].item() == 0.0

    def test_grpo_no_std_norm(self):
        rewards = torch.tensor([1.0, 0.0, 0.5])
        response_mask = torch.ones(3, 2)
        prompt_ids = torch.tensor([0, 0, 0])

        advantages, _ = compute_grpo_advantage(
            rewards, response_mask, prompt_ids, norm_by_std=False
        )
        # Without std normalization, advantage = reward - mean
        mean_r = rewards.mean().item()
        expected_0 = 1.0 - mean_r
        assert abs(advantages[0, 0].item() - expected_0) < 1e-5


class TestGAEAdvantage:
    """Tests for GAE advantage computation."""

    def test_basic_gae(self):
        rewards = torch.zeros(2, 5)
        rewards[0, 4] = 1.0  # reward at end
        values = torch.zeros(2, 5)
        mask = torch.ones(2, 5)

        advantages, returns = compute_gae_advantage(
            rewards, values, mask, gamma=1.0, lam=1.0
        )
        assert advantages.shape == (2, 5)
        # First sequence should have positive advantages (got reward)
        assert advantages[0].sum().item() > 0

    def test_gae_with_values(self):
        rewards = torch.zeros(1, 3)
        rewards[0, 2] = 1.0
        values = torch.tensor([[0.5, 0.7, 0.9]])
        mask = torch.ones(1, 3)

        advantages, returns = compute_gae_advantage(
            rewards, values, mask, gamma=0.99, lam=0.95
        )
        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)


class TestPolicyLoss:
    """Tests for PPO-style policy loss."""

    def test_basic_loss(self):
        log_probs = torch.randn(4, 10)
        old_log_probs = log_probs.clone()  # same policy
        advantages = torch.randn(4, 10)
        mask = torch.ones(4, 10)

        loss, info = compute_policy_loss(
            log_probs, old_log_probs, advantages, mask
        )

        assert loss.shape == ()
        assert "policy_loss" in info
        assert "approx_kl" in info
        assert "clip_fraction" in info
        # When policies are identical, KL should be ~0
        assert abs(info["approx_kl"]) < 1e-5

    def test_clipping(self):
        log_probs = torch.zeros(2, 5)
        old_log_probs = torch.ones(2, 5)  # ratio = exp(-1) ≈ 0.37
        advantages = torch.ones(2, 5)
        mask = torch.ones(2, 5)

        loss, info = compute_policy_loss(
            log_probs, old_log_probs, advantages, mask, clip_ratio=0.2
        )
        # Should have non-zero clip fraction since ratio deviates a lot
        assert info["clip_fraction"] > 0

    def test_seq_mean_agg(self):
        log_probs = torch.randn(2, 5)
        old_log_probs = log_probs.clone()
        advantages = torch.randn(2, 5)
        mask = torch.ones(2, 5)

        loss, _ = compute_policy_loss(
            log_probs, old_log_probs, advantages, mask,
            loss_agg_mode="seq-mean-token-mean"
        )
        assert loss.shape == ()


class TestKLPenalty:
    """Tests for KL divergence computation."""

    def test_zero_kl_same_policy(self):
        log_probs = torch.randn(2, 5)
        ref_log_probs = log_probs.clone()
        mask = torch.ones(2, 5)

        kl_loss, kl_val = compute_kl_penalty(log_probs, ref_log_probs, mask)
        assert abs(kl_val) < 1e-5

    def test_positive_kl_different_policy(self):
        log_probs = torch.randn(2, 5)
        ref_log_probs = log_probs + 0.5
        mask = torch.ones(2, 5)

        kl_loss, kl_val = compute_kl_penalty(log_probs, ref_log_probs, mask)
        assert kl_val >= 0  # KL is non-negative


class TestEntropyBonus:
    def test_entropy(self):
        log_probs = -torch.ones(2, 5)  # uniform-ish
        mask = torch.ones(2, 5)

        ent_loss, ent_val = compute_entropy_bonus(log_probs, mask, entropy_coeff=0.01)
        assert ent_val > 0  # entropy should be positive
        assert ent_loss.item() < 0  # loss is negative (we maximize entropy)


class TestWhitenAdvantages:
    def test_whiten(self):
        advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.ones(2, 3)

        whitened = whiten_advantages(advantages, mask)
        valid = whitened[mask.bool()]
        assert abs(valid.mean().item()) < 1e-5
        assert abs(valid.std().item() - 1.0) < 0.1
