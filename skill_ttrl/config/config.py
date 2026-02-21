"""Configuration dataclasses for Skill TTRL."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _get_env_api_key() -> Optional[str]:
    """Get API key from environment variables (supports multiple naming conventions)."""
    for key in ["OPENAI_API_KEY", "SKILL_TTRL_API_KEY", "LLM_API_KEY"]:
        val = os.environ.get(key)
        if val:
            return val
    return None


def _get_env_api_base() -> Optional[str]:
    """Get API base URL from environment variables."""
    for key in ["OPENAI_API_BASE", "OPENAI_BASE_URL", "SKILL_TTRL_API_BASE"]:
        val = os.environ.get(key)
        if val:
            return val
    return None


@dataclass
class DataConfig:
    """Data loading configuration."""
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    max_prompt_length: int = 512
    max_response_length: int = 3072
    train_batch_size: int = 8
    shuffle: bool = True
    suffix_prompt: str = (
        "\nPlease reason step by step, and put your final answer "
        "within \\boxed{}."
    )


@dataclass
class ModelConfig:
    """Model configuration."""
    path: str = "Qwen/Qwen2.5-Math-7B"
    tokenizer_path: Optional[str] = None
    dtype: str = "bfloat16"
    lora_rank: int = 0
    lora_alpha: int = 16


@dataclass
class RolloutConfig:
    """Rollout / generation configuration."""
    engine: str = "vllm"  # "vllm" or "hf"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 3072
    n_votes_per_prompt: int = 64
    n_samples_per_prompt: int = 32
    tensor_parallel_size: int = 1


@dataclass
class AlgorithmConfig:
    """RL algorithm configuration."""
    adv_estimator: str = "grpo"  # "grpo", "gae", "reinforce_pp"
    gamma: float = 1.0
    lam: float = 1.0
    clip_ratio: float = 0.2
    clip_ratio_high: float = 5.0
    norm_adv_by_std: bool = True
    use_kl_loss: bool = True
    kl_coef: float = 0.001
    entropy_coeff: float = 0.001
    loss_agg_mode: str = "token-mean"  # "token-mean" or "seq-mean-token-mean"


@dataclass
class SkillBankConfig:
    """Skill bank configuration."""
    max_skills: int = 200
    retrieval_mode: str = "embedding"  # "embedding" or "keyword"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_retrieve: int = 6
    skill_bank_path: Optional[str] = None
    enable_evolve: bool = True
    enable_generate: bool = True
    enable_retrieve: bool = True
    eviction_window: int = 50  # evict skills not used in last N rounds
    min_winners_for_extraction: int = 2  # min correct solutions to trigger extraction
    max_skills_per_problem: int = 3  # max skills extracted per problem


@dataclass
class MergerConfig:
    """External LLM merger configuration."""
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.3


@dataclass
class TrainerConfig:
    """Training loop configuration."""
    total_epochs: int = 80
    save_freq: int = 5
    val_freq: int = 2
    log_freq: int = 1
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    ref_update_interval: int = 10
    output_dir: str = "outputs"
    seed: int = 42
    n_gpus: int = 1
    gradient_accumulation_steps: int = 1
    micro_batch_size: int = 2
    mini_batch_size: int = 1


@dataclass
class SkillTTRLConfig:
    """Top-level Skill TTRL configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    skill_bank: SkillBankConfig = field(default_factory=SkillBankConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> SkillTTRLConfig:
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            rollout=RolloutConfig(**d.get("rollout", {})),
            algorithm=AlgorithmConfig(**d.get("algorithm", {})),
            skill_bank=SkillBankConfig(**d.get("skill_bank", {})),
            merger=MergerConfig(**d.get("merger", {})),
            trainer=TrainerConfig(**d.get("trainer", {})),
        )


def load_config(path: str | Path) -> SkillTTRLConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Environment variables override YAML settings for sensitive values:
    - OPENAI_API_KEY / SKILL_TTRL_API_KEY / LLM_API_KEY -> merger.api_key
    - OPENAI_API_BASE / OPENAI_BASE_URL / SKILL_TTRL_API_BASE -> merger.api_base
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        config = SkillTTRLConfig.from_dict(raw)
    else:
        config = SkillTTRLConfig()

    # Apply environment variable overrides for API settings
    env_api_key = _get_env_api_key()
    if env_api_key and not config.merger.api_key:
        config.merger.api_key = env_api_key

    env_api_base = _get_env_api_base()
    if env_api_base and not config.merger.api_base:
        config.merger.api_base = env_api_base

    return config
