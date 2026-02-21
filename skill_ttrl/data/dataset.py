"""
Dataset: loading and preprocessing for Skill TTRL.

Supports loading from JSON, JSONL, and Parquet formats.
Handles tokenization and chat template application.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SkillTTRLDataset(Dataset):
    """
    Dataset for Skill TTRL training.

    Each item contains:
    - prompt: The problem/question text
    - answer: Ground truth answer (used for metrics, not for TTRL reward)
    - data_source: Origin dataset name
    - extra_info: Additional metadata
    """

    def __init__(
        self,
        data_files: list[str],
        tokenizer=None,
        max_prompt_length: int = 512,
        suffix_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.suffix_prompt = suffix_prompt
        self.data: list[dict] = []

        for file_path in data_files:
            self.data.extend(self._load_file(file_path))

        logger.info(f"Loaded {len(self.data)} examples from {len(data_files)} files")

    def _load_file(self, path: str) -> list[dict]:
        """Load data from a single file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        if path.suffix == ".parquet":
            return self._load_parquet(path)
        elif path.suffix == ".jsonl":
            return self._load_jsonl(path)
        elif path.suffix == ".json":
            return self._load_json(path)
        else:
            logger.warning(f"Unsupported file format: {path.suffix}")
            return []

    def _load_parquet(self, path: Path) -> list[dict]:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict("records")
        except ImportError:
            logger.warning("pandas/pyarrow not installed, cannot load parquet")
            return []

    def _load_json(self, path: Path) -> list[dict]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    def _load_jsonl(self, path: Path) -> list[dict]:
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        prompt = item.get("prompt", item.get("question", item.get("problem", "")))
        if self.suffix_prompt:
            prompt = prompt + self.suffix_prompt

        answer = item.get("answer", item.get("solution", ""))
        data_source = item.get("data_source", item.get("source", "unknown"))

        result = {
            "prompt": prompt,
            "answer": answer,
            "data_source": data_source,
            "index": idx,
        }

        # Tokenize if tokenizer is available
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_prompt_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            result["input_ids"] = encoded["input_ids"].squeeze(0)
            result["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch of dataset items."""
    result = {
        "prompts": [item["prompt"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "data_sources": [item["data_source"] for item in batch],
        "indices": [item["index"] for item in batch],
    }

    # Stack tensors if available
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        result["attention_mask"] = torch.stack(
            [item["attention_mask"] for item in batch]
        )

    return result
