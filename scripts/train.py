#!/usr/bin/env python3
"""
Training entry point for Skill TTRL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --data.train_files '["data/train.json"]'

Environment variables:
    OPENAI_API_KEY: API key for external LLM skill extraction/merging
    OPENAI_API_BASE: Optional custom API endpoint
"""

# IMPORTANT: Set multiprocessing start method BEFORE any other imports
# This fixes "Cannot re-initialize CUDA in forked subprocess" error with vLLM
import os

# Load .env file if it exists (for API keys)
def _load_dotenv():
    """Load environment variables from .env file if present."""
    from pathlib import Path
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    # Don't override existing env vars
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_dotenv()

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skill_ttrl.config.config import load_config, SkillTTRLConfig
from skill_ttrl.training.trainer import SkillTTRLTrainer
from skill_ttrl.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Skill TTRL Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--train_files", type=str, nargs="+", default=None,
        help="Override training data files (e.g., --train_files data/custom.json)"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load config (will pick up env vars for API keys)
    config = load_config(args.config)

    # Log API configuration status
    import logging
    logger = logging.getLogger(__name__)
    if config.merger.api_key:
        logger.info("OpenAI API key configured - LLM skill extraction enabled")
    else:
        logger.warning(
            "No API key found. Set OPENAI_API_KEY env var or create .env file. "
            "Using heuristic skill extraction only."
        )

    # Apply overrides
    if args.output_dir:
        config.trainer.output_dir = args.output_dir
    if args.epochs:
        config.trainer.total_epochs = args.epochs
    if args.seed:
        config.trainer.seed = args.seed
    if args.train_files:
        config.data.train_files = args.train_files

    # Create trainer and run
    trainer = SkillTTRLTrainer(config)

    # Save config
    config.save(Path(config.trainer.output_dir) / "config.yaml")

    # Train
    metrics = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
