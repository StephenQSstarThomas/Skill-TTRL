# Skill TTRL

**Training models to autonomously generate, retrieve, and evolve skills via test-time reinforcement learning.**

Skill TTRL fuses [TTRL](https://arxiv.org/abs/2504.16084) (Test-Time Reinforcement Learning) with [SkillRL](https://arxiv.org/abs/2602.08234) (Skill-augmented Recursive RL) into a unified framework where the model autonomously decides when and how to generate, retrieve, and evolve skills -- all trained end-to-end through majority-voting rewards without labeled data.

---

## Architecture Overview

```
                          ┌─────────────────────┐
                          │   Problem / Task x   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Prompt Formatter    │
                          │  (inject skill bank  │
                          │   context into prompt)│
                          └──────────┬──────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Parallel Sampling (K outputs)  │
                    │   Each sample autonomously:      │
                    │   • <generate> new skills        │
                    │   • <retrieve> from skill bank   │
                    │   • <evolve> existing skills     │
                    │   Then solves the problem         │
                    └────────────────┬────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Majority Voting    │
                          │  pseudo ground truth  │
                          │  + binary rewards     │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   External LLM Merger            │
                    │   Merge winning samples' ops:    │
                    │   • Dedup generated skills       │
                    │   • Find useful retrieval subset  │
                    │   • Combine evolution directions  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        Skill Bank Update         │
                    │   • Add new skills               │
                    │   • Replace evolved skills        │
                    │   • Evict stale skills            │
                    └────────────────┬────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   GRPO Update        │
                          │  (policy gradient    │
                          │   with group-relative │
                          │   advantages)         │
                          └──────────────────────┘
```

---

## Project Structure

```
Skill-TTRL/
├── skill_ttrl/                    # Main package
│   ├── __init__.py
│   ├── config/                    # Configuration system
│   │   ├── config.py              # Dataclass configs + YAML I/O
│   │   └── __init__.py
│   ├── core/                      # Core algorithm modules
│   │   ├── skill_bank.py          # Skill storage, retrieval, evolution
│   │   ├── output_parser.py       # Parse <skill_ops>, <solution>, <answer>
│   │   ├── majority_vote.py       # TTRL majority voting
│   │   ├── merge_llm.py           # External LLM skill operation merger
│   │   ├── grpo.py                # GRPO advantage & policy loss
│   │   └── __init__.py
│   ├── training/                  # Training pipeline
│   │   ├── trainer.py             # Main SkillTTRLTrainer
│   │   ├── rollout.py             # vLLM/HF generation engine
│   │   ├── reward.py              # Reward computation manager
│   │   └── __init__.py
│   ├── prompts/                   # Prompt engineering
│   │   ├── templates.py           # System prompts & templates
│   │   ├── formatter.py           # Skill-aware prompt formatting
│   │   └── __init__.py
│   ├── data/                      # Data handling
│   │   ├── dataset.py             # Dataset loading (JSON/JSONL/Parquet)
│   │   └── __init__.py
│   └── utils/                     # Utilities
│       ├── math_grader.py         # Math answer extraction & grading
│       ├── logging.py             # Logging & metrics tracking
│       └── __init__.py
├── configs/
│   └── default.yaml               # Default configuration
├── scripts/
│   └── train.py                   # Training entry point
├── tests/                         # Test suite (98 tests)
│   ├── test_skill_bank.py
│   ├── test_output_parser.py
│   ├── test_majority_vote.py
│   ├── test_merge_llm.py
│   ├── test_grpo.py
│   ├── test_prompt_formatter.py
│   ├── test_math_grader.py
│   └── test_pipeline.py           # End-to-end integration tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Completed Modules

### 1. Skill Bank (`skill_ttrl/core/skill_bank.py`)

The skill bank stores, retrieves, and manages skills organized by task type.

**Key class: `SkillBank`**

| Method | Description |
|--------|-------------|
| `add(skill)` | Add a new skill, auto-evict if over capacity |
| `add_from_dict(d, task_type, source)` | Add from raw dictionary |
| `get(skill_id)` | Retrieve a skill by ID |
| `remove(skill_id)` | Remove a skill |
| `replace(old_id, new_skill)` | Replace with evolved version (preserves usage stats) |
| `retrieve(query, task_type, top_k)` | Retrieve most relevant skills |
| `record_usage(skill_id, success, round)` | Track skill usage and success |
| `evict_stale(current_round)` | Remove skills unused within sliding window |
| `save(path)` / `load(path)` | Persist to/from JSON |

**Retrieval modes:**
- `"embedding"`: Semantic similarity via `sentence-transformers` (cosine similarity)
- `"keyword"`: Token overlap matching (no GPU needed)

```python
from skill_ttrl.core.skill_bank import SkillBank, Skill

bank = SkillBank(max_skills=200, retrieval_mode="keyword", top_k=6)
bank.add(Skill(
    skill_id="sk_001",
    title="Modular Arithmetic",
    content="Use Fermat's little theorem for prime moduli.",
    task_type="number_theory",
))
results = bank.retrieve("Find 2^100 mod 7", task_type="number_theory")
```

---

### 2. Output Parser (`skill_ttrl/core/output_parser.py`)

Parses model outputs containing `<skill_ops>`, `<solution>`, and `<answer>` XML tags.

```python
from skill_ttrl.core.output_parser import OutputParser

parser = OutputParser()
result = parser.parse("""
<skill_ops>
  <generate>Check modular patterns in powers.</generate>
  <retrieve query="number theory">Skill #12: Fermat's theorem</retrieve>
  <evolve base="#12">Add Euler's theorem for composites.</evolve>
</skill_ops>
<solution>2^10 mod 7 = 2</solution>
<answer>2</answer>
""")

print(result.answer)                    # "2"
print(result.skill_ops.has_generate)    # True
print(result.skill_ops.retrieve_query)  # "number theory"
print(result.skill_ops.evolve_base_id)  # "12"
```

Also supports `\boxed{}` answer extraction (with nested braces) as fallback.

---

### 3. Majority Voting (`skill_ttrl/core/majority_vote.py`)

Implements TTRL's core mechanism: K samples vote to determine pseudo ground truth.

```python
from skill_ttrl.core.majority_vote import majority_vote, compute_voting_rewards

answers = ["42", "42", "42", "43", "44", "42", "41", "42"]
majority, agreement, matches = majority_vote(answers)
# majority = "42", agreement = 0.625, matches = [True, True, True, False, ...]

rewards, maj, agr = compute_voting_rewards(answers)
# rewards = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
```

**Features:**
- Numeric normalization (42.0 == 42)
- LaTeX cleanup (`\text{}`, `\mathrm{}`)
- SymPy symbolic comparison for mathematical equivalence
- Batch voting for multiple prompts

---

### 4. Skill Merger (`skill_ttrl/core/merge_llm.py`)

Merges winning samples' skill operations into a single best result.

```python
from skill_ttrl.core.merge_llm import SkillMerger

merger = SkillMerger(use_api=False)  # heuristic mode (no API needed)
# merger = SkillMerger(model="gpt-4o", api_key="...")  # API mode

merged = merger.merge(winning_parsed_outputs)
# merged.new_skills         - deduplicated generated skills
# merged.useful_retrieval_ids - skill IDs used by multiple winners
# merged.evolved_skills     - merged evolution results
```

**Two modes:**
- **API mode**: Calls an external LLM (GPT-4o, etc.) for intelligent merging
- **Heuristic mode**: Uses frequency counting and deduplication (no API needed)

---

### 5. GRPO Algorithm (`skill_ttrl/core/grpo.py`)

Group Relative Policy Optimization -- the RL algorithm from TTRL.

```python
from skill_ttrl.core.grpo import (
    compute_grpo_advantage,
    compute_policy_loss,
    compute_kl_penalty,
)

# GRPO advantage: normalize rewards within each prompt group
advantages, returns = compute_grpo_advantage(
    rewards=per_sample_rewards,         # (batch_size,)
    response_mask=response_mask,        # (batch_size, seq_len)
    prompt_ids=prompt_group_ids,        # (batch_size,)
)

# PPO-style clipped policy loss
loss, info = compute_policy_loss(
    log_probs, old_log_probs, advantages, response_mask,
    clip_ratio=0.2,
)

# KL penalty against reference policy
kl_loss, kl_value = compute_kl_penalty(
    log_probs, ref_log_probs, response_mask, kl_coef=0.001,
)
```

**Also includes:**
- GAE (Generalized Advantage Estimation) for value-based methods
- Entropy bonus for exploration
- Advantage whitening
- Dual-clip PPO for stability

---

### 6. Prompt Formatter (`skill_ttrl/prompts/formatter.py`)

Constructs prompts with skill bank context injection.

```python
from skill_ttrl.prompts.formatter import PromptFormatter

formatter = PromptFormatter(skill_bank=bank, task_type="math")

# For reasoning tasks
prompt = formatter.format_prompt("Find 2^100 mod 7")

# For agent tasks
prompt = formatter.format_agent_prompt(
    task_description="Put apple on counter",
    current_state="Kitchen, apple on table",
    available_actions="take, go, look",
)

# Batch processing
prompts = formatter.format_batch(["Problem 1", "Problem 2"])
```

Handles cold start (empty skill bank) automatically.

---

### 7. Training Pipeline (`skill_ttrl/training/trainer.py`)

The main `SkillTTRLTrainer` orchestrates the complete pipeline.

```python
from skill_ttrl.config import SkillTTRLConfig
from skill_ttrl.training.trainer import SkillTTRLTrainer

config = SkillTTRLConfig()
trainer = SkillTTRLTrainer(config)

# Full training (requires GPU + model)
metrics = trainer.train(train_data=data, val_data=val_data)

# Offline mode (no GPU needed -- for testing/debugging)
metrics = trainer.train_offline(
    problems=["Find 2^10 mod 7"],
    all_responses=[["<answer>2</answer>", "<answer>2</answer>", "<answer>3</answer>"]],
)
```

**Training step implements:**
1. Format prompts with skill bank context
2. Generate K samples per prompt (via vLLM or HF)
3. Majority vote for pseudo ground truth
4. Merge winning samples' skill operations
5. Update skill bank (add new, replace evolved, evict stale)
6. Compute GRPO advantages
7. Policy gradient update

---

### 8. Rollout Engine (`skill_ttrl/training/rollout.py`)

Efficient generation with vLLM or HuggingFace transformers.

```python
from skill_ttrl.training.rollout import RolloutEngine

engine = RolloutEngine(model_path="Qwen/Qwen2.5-Math-7B", engine="vllm")
responses = engine.generate(prompts, n_samples=64)  # 64 samples per prompt
```

---

### 9. Reward Manager (`skill_ttrl/training/reward.py`)

Computes majority-voting rewards and evaluation metrics.

```python
from skill_ttrl.training.reward import RewardManager

rm = RewardManager()
result = rm.compute_majority_rewards(responses, prompt_ids, n_per_prompt=64)
# result["rewards"] - binary rewards (1.0 = matches majority, 0.0 = doesn't)
# result["majority_answers"] - voted answers per prompt
# result["agreement_ratios"] - voting confidence per prompt
```

---

### 10. Math Grader (`skill_ttrl/utils/math_grader.py`)

Answer extraction and grading with multiple comparison strategies.

```python
from skill_ttrl.utils.math_grader import grade_answer, compute_reward

grade_answer("42.0", "42")     # {"score": 1.0, "match_type": "numeric"}
grade_answer("1/2", "0.5")    # {"score": 1.0, "match_type": "numeric"}

compute_reward("The answer is \\boxed{42}.", ground_truth="42")
# {"score": 1.0, "match_type": "numeric", "predicted_answer": "42"}
```

---

### 11. Configuration System (`skill_ttrl/config/config.py`)

Hierarchical dataclass configuration with YAML persistence.

```python
from skill_ttrl.config import SkillTTRLConfig, load_config

# Create with defaults
config = SkillTTRLConfig()
config.rollout.n_votes_per_prompt = 64
config.algorithm.clip_ratio = 0.2

# Save/load YAML
config.save("my_config.yaml")
config = load_config("my_config.yaml")
```

**Config sections:** `data`, `model`, `rollout`, `algorithm`, `skill_bank`, `merger`, `trainer`

---

## Installation

```bash
# Install with all dependencies
pip install -e ".[dev]"

# Or install core dependencies only
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch >= 2.1, transformers, vLLM, sentence-transformers, openai, sympy

---

## Running Tests

```bash
# Run all 98 tests
pytest tests/ -v

# Run specific test module
pytest tests/test_skill_bank.py -v
pytest tests/test_pipeline.py -v    # integration tests
```

---

## Quick Start

### Offline Pipeline Test (no GPU needed)

```python
from skill_ttrl.config import SkillTTRLConfig
from skill_ttrl.training.trainer import SkillTTRLTrainer

config = SkillTTRLConfig()
config.skill_bank.retrieval_mode = "keyword"

trainer = SkillTTRLTrainer(config)
trainer.merger.use_api = False  # use heuristic merge

metrics = trainer.train_offline(
    problems=["What is 2^10 mod 7?"],
    all_responses=[[
        "<skill_ops><generate>Use Fermat's theorem.</generate></skill_ops>"
        "<answer>2</answer>",
        "<answer>2</answer>",
        "<answer>2</answer>",
        "<answer>3</answer>",
    ]],
)
print(f"Reward: {metrics['mean_reward']:.2f}")
print(f"Skills: {metrics['skill_bank_size']}")
```

### Examplar Training (First-Run with 1.5b model && self-curated dataset)
```bash
python scripts/train.py \
    --config configs/qwen_1.5b.yaml \
    --epochs 5 \
    --output_dir outputs/first_run \
    --log_level INFO
```

### Full Training (requires GPU)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 10 \
    --output_dir outputs/experiment_1
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unified skill ops** | Generate, retrieve, evolve in one `<skill_ops>` block. Model learns optimal combination via RL. |
| **Majority voting** | No labeled data needed. "Lucky Hit" effect ensures >75% reward accuracy. |
| **External LLM merger** | Prevents skill bank bloat. Deduplicates and fuses winning samples' operations. |
| **Heuristic fallback** | All components work without API access for testing and development. |
| **Sliding window eviction** | Keeps skill bank fresh and bounded. Stale unused skills are removed. |
| **GRPO over PPO** | No critic network needed. Group-relative baseline reduces compute and memory. |

---

## References

- **TTRL**: [Test-Time Reinforcement Learning](https://arxiv.org/abs/2504.16084) (Tsinghua & Shanghai AI Lab)
- **SkillRL**: [Skill-Augmented Recursive RL](https://arxiv.org/abs/2602.08234) (UNC Chapel Hill)

---

## License

Apache-2.0
