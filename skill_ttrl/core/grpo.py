"""
GRPO: Group Relative Policy Optimization.

Implements the core RL algorithm used in both TTRL and SkillRL:
- GRPO advantage estimation (group-relative baseline)
- PPO-style clipped policy loss
- KL penalty computation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdvantageEstimator(str, Enum):
    GRPO = "grpo"
    GAE = "gae"
    REINFORCE_PP = "reinforce_pp"


# ----------------------------------------------------------------------
# Advantage estimation
# ----------------------------------------------------------------------
def compute_grpo_advantage(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    norm_by_std: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO (Group Relative Policy Optimization) advantages.

    Groups samples by prompt_id, normalizes rewards within each group:
        advantage_i = (reward_i - mean_group) / (std_group + eps)

    Args:
        rewards: Per-sample rewards. Shape: (batch_size,)
        response_mask: Binary mask for response tokens. Shape: (batch_size, seq_len)
        prompt_ids: Group identifier for each sample. Shape: (batch_size,)
        norm_by_std: Whether to divide by group std.
        eps: Small constant for numerical stability.

    Returns:
        advantages: Shape (batch_size, seq_len), broadcasted to response tokens.
        returns: Same shape, used for value function targets if needed.
    """
    batch_size = rewards.shape[0]
    device = rewards.device

    # Group rewards by prompt_id
    id_to_indices: dict[int, list[int]] = {}
    for i in range(batch_size):
        pid = int(prompt_ids[i].item())
        id_to_indices.setdefault(pid, []).append(i)

    advantages = torch.zeros(batch_size, device=device)

    for pid, indices in id_to_indices.items():
        group_rewards = rewards[indices]
        mean_r = group_rewards.mean()
        std_r = group_rewards.std()

        for idx in indices:
            if norm_by_std and std_r > eps:
                advantages[idx] = (rewards[idx] - mean_r) / (std_r + eps)
            else:
                advantages[idx] = rewards[idx] - mean_r

    # Broadcast to sequence length (same advantage for all response tokens)
    seq_len = response_mask.shape[1]
    advantages_seq = advantages.unsqueeze(1).expand(-1, seq_len) * response_mask

    return advantages_seq, advantages_seq.clone()


def compute_gae_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Token-level rewards. Shape: (batch_size, seq_len)
        values: Value function estimates. Shape: (batch_size, seq_len)
        response_mask: Binary mask. Shape: (batch_size, seq_len)
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        advantages: Shape (batch_size, seq_len)
        returns: Shape (batch_size, seq_len)
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=device)

    for t in reversed(range(seq_len)):
        mask_t = response_mask[:, t]
        if t == seq_len - 1:
            next_value = torch.zeros(batch_size, device=device)
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae * mask_t

    returns = advantages + values
    return advantages, returns


# ----------------------------------------------------------------------
# Policy loss
# ----------------------------------------------------------------------
def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_high: float = 5.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO-style clipped policy gradient loss.

    Args:
        log_probs: Current policy log probabilities. Shape: (bs, seq_len)
        old_log_probs: Old policy log probabilities. Shape: (bs, seq_len)
        advantages: Advantage estimates. Shape: (bs, seq_len)
        response_mask: Binary mask. Shape: (bs, seq_len)
        clip_ratio: PPO clip parameter epsilon (lower bound).
        clip_ratio_high: Dual-clip upper bound for negative advantages.
        loss_agg_mode: "token-mean" or "seq-mean-token-mean".

    Returns:
        loss: Scalar loss tensor.
        info: Dictionary with diagnostic metrics.
    """
    # Importance sampling ratio
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped surrogate
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Dual-clip: for negative advantages, prevent ratio from being too large
    if clip_ratio_high > 0:
        dual_clip_loss = -clip_ratio_high * advantages
        neg_adv_mask = (advantages < 0).float()
        pg_loss = pg_loss * (1 - neg_adv_mask) + \
            torch.min(pg_loss, dual_clip_loss) * neg_adv_mask

    # Apply mask
    pg_loss = pg_loss * response_mask

    # Aggregate
    if loss_agg_mode == "token-mean":
        n_tokens = response_mask.sum().clamp(min=1)
        loss = pg_loss.sum() / n_tokens
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_lens = response_mask.sum(dim=1).clamp(min=1)
        per_seq_loss = (pg_loss.sum(dim=1)) / seq_lens
        loss = per_seq_loss.mean()
    else:
        loss = pg_loss.mean()

    # Diagnostics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = (
            ((ratio - 1.0).abs() > clip_ratio).float() * response_mask
        ).sum().item() / response_mask.sum().clamp(min=1).item()

    info = {
        "policy_loss": loss.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
        "ratio_mean": ratio.mean().item(),
    }

    return loss, info


# ----------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------
def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coef: float = 0.001,
) -> tuple[torch.Tensor, float]:
    """
    Compute KL divergence penalty between current and reference policy.

    Uses the approximation: KL ≈ exp(log_ref - log_cur) - (log_ref - log_cur) - 1

    Args:
        log_probs: Current policy log probs. Shape: (bs, seq_len)
        ref_log_probs: Reference policy log probs. Shape: (bs, seq_len)
        response_mask: Binary mask. Shape: (bs, seq_len)
        kl_coef: KL penalty coefficient.

    Returns:
        kl_loss: Scalar KL loss (weighted by kl_coef).
        kl_value: Raw KL divergence value.
    """
    log_ratio = ref_log_probs - log_probs
    kl = (torch.exp(log_ratio) - log_ratio - 1.0) * response_mask

    n_tokens = response_mask.sum().clamp(min=1)
    kl_value = (kl.sum() / n_tokens).item()
    kl_loss = kl_coef * kl.sum() / n_tokens

    return kl_loss, kl_value


# ----------------------------------------------------------------------
# Entropy bonus
# ----------------------------------------------------------------------
def compute_entropy_bonus(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_coeff: float = 0.001,
) -> tuple[torch.Tensor, float]:
    """
    Compute entropy bonus to encourage exploration.

    Args:
        log_probs: Log probabilities. Shape: (bs, seq_len)
        response_mask: Binary mask. Shape: (bs, seq_len)
        entropy_coeff: Entropy coefficient.

    Returns:
        entropy_loss: Scalar loss (negative because we maximize entropy).
        entropy_value: Raw entropy value.
    """
    entropy = -log_probs * response_mask
    n_tokens = response_mask.sum().clamp(min=1)
    entropy_value = (entropy.sum() / n_tokens).item()
    entropy_loss = -entropy_coeff * entropy.sum() / n_tokens

    return entropy_loss, entropy_value


# ----------------------------------------------------------------------
# Whiten advantages
# ----------------------------------------------------------------------
def whiten_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Whiten (normalize) advantages across the batch."""
    valid = advantages[response_mask.bool()]
    if valid.numel() == 0:
        return advantages
    mean = valid.mean()
    std = valid.std()
    if std < eps:
        return advantages - mean
    return (advantages - mean) / (std + eps) * response_mask
