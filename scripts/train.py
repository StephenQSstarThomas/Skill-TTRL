#!/usr/bin/env python3
"""
Training entry point for Skill TTRL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --data.train_files '["data/train.json"]'
"""

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
        "--log_level", type=str, default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.output_dir:
        config.trainer.output_dir = args.output_dir
    if args.epochs:
        config.trainer.total_epochs = args.epochs
    if args.seed:
        config.trainer.seed = args.seed

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
