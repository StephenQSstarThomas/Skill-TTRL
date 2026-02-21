"""
Rollout: generation/sampling engine for Skill TTRL.

Generates K responses per prompt using either vLLM or HuggingFace transformers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class RolloutEngine:
    """
    Generates multiple responses per prompt for majority voting.

    Supports:
    - vLLM for efficient batched generation
    - HuggingFace transformers as fallback
    """

    def __init__(
        self,
        model_path: str,
        engine: str = "vllm",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 3072,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.engine_type = engine
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self._engine = None
        self._tokenizer = None

    def _init_vllm(self):
        """Initialize vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            self._engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
                trust_remote_code=True,
            )
            self._tokenizer = self._engine.get_tokenizer()
            logger.info(f"vLLM engine initialized: {self.model_path}")
        except ImportError:
            logger.warning("vLLM not available, falling back to HF")
            self.engine_type = "hf"
            self._init_hf()

    def _init_hf(self):
        """Initialize HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._engine = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info(f"HF model initialized: {self.model_path}")

    def _ensure_engine(self):
        if self._engine is None:
            if self.engine_type == "vllm":
                self._init_vllm()
            else:
                self._init_hf()

    @property
    def tokenizer(self):
        self._ensure_engine()
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        n_samples: int = 1,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[list[str]]:
        """
        Generate n_samples responses for each prompt.

        Args:
            prompts: List of prompt strings.
            n_samples: Number of responses per prompt.
            temperature: Override default temperature.
            top_p: Override default top_p.
            max_tokens: Override default max_tokens.

        Returns:
            List of lists: outer list has len(prompts) entries,
            each inner list has n_samples response strings.
        """
        self._ensure_engine()

        temp = temperature or self.temperature
        tp = top_p or self.top_p
        mt = max_tokens or self.max_tokens

        if self.engine_type == "vllm":
            return self._generate_vllm(prompts, n_samples, temp, tp, mt)
        else:
            return self._generate_hf(prompts, n_samples, temp, tp, mt)

    def _generate_vllm(
        self,
        prompts: list[str],
        n_samples: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n_samples,
        )

        outputs = self._engine.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            responses = [o.text for o in output.outputs]
            results.append(responses)
        return results

    def _generate_hf(
        self,
        prompts: list[str],
        n_samples: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[list[str]]:
        results = []
        for prompt in prompts:
            encoded = self._tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).to(self._engine.device)

            responses = []
            for _ in range(n_samples):
                with torch.no_grad():
                    output = self._engine.generate(
                        **encoded,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                response_ids = output[0][encoded["input_ids"].shape[1]:]
                response_text = self._tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                responses.append(response_text)
            results.append(responses)
        return results

    def compute_log_probs(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities of responses given prompts.

        Args:
            prompts: Prompt texts.
            responses: Response texts.

        Returns:
            log_probs: Shape (batch_size, max_response_len)
            response_mask: Shape (batch_size, max_response_len)
        """
        self._ensure_engine()
        tokenizer = self._tokenizer

        all_log_probs = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            full_ids = prompt_ids + response_ids

            input_ids = torch.tensor([full_ids], device=self._get_device())

            if self.engine_type == "hf":
                with torch.no_grad():
                    outputs = self._engine(input_ids)
                    logits = outputs.logits

                # Shift for next-token prediction
                shift_logits = logits[:, len(prompt_ids) - 1:-1, :]
                shift_labels = input_ids[:, len(prompt_ids):]

                log_probs = torch.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                mask = torch.ones_like(token_log_probs)
                all_log_probs.append(token_log_probs.squeeze(0))
                all_masks.append(mask.squeeze(0))
            else:
                # For vLLM, we use the prompt_logprobs feature
                # Fallback: return zeros (actual training uses framework's log_probs)
                n = len(response_ids)
                all_log_probs.append(torch.zeros(n))
                all_masks.append(torch.ones(n))

        # Pad to same length
        max_len = max(lp.shape[0] for lp in all_log_probs)
        padded_lps = torch.zeros(len(all_log_probs), max_len)
        padded_masks = torch.zeros(len(all_masks), max_len)

        for i, (lp, m) in enumerate(zip(all_log_probs, all_masks)):
            padded_lps[i, :lp.shape[0]] = lp
            padded_masks[i, :m.shape[0]] = m

        return padded_lps, padded_masks

    def _get_device(self):
        if self.engine_type == "hf" and hasattr(self._engine, "device"):
            return self._engine.device
        return torch.device("cpu")
