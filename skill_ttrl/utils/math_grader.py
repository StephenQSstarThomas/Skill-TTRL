"""
Math Grader: answer extraction and grading for mathematical problems.

Supports string matching, numeric comparison, and symbolic comparison via SymPy.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} notation, handling nested braces."""
    # Find all \\boxed{ occurrences and manually match braces
    results = []
    idx = 0
    while True:
        pos = text.find("\\boxed{", idx)
        if pos == -1:
            break
        # Start after the opening brace
        start = pos + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i - 1])
        idx = i
    if results:
        return results[-1].strip()
    return ""


def extract_answer_from_text(text: str) -> str:
    """Extract the final answer from model output, trying multiple patterns."""
    # Try <answer> tag
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try \\boxed{}
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Try "The answer is X" pattern
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize_numeric(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "").replace(" ", "")
    # Remove trailing period
    s = s.rstrip(".")
    # Remove percentage sign
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            pass
    # Handle fractions like 3/4
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s)
    if match:
        try:
            return float(match.group(1)) / float(match.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return None


def grade_answer(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-6,
) -> dict:
    """
    Grade a predicted answer against ground truth.

    Returns:
        Dictionary with:
        - score: 1.0 if correct, 0.0 if incorrect
        - match_type: how the match was determined
        - predicted_normalized: normalized predicted answer
        - ground_truth_normalized: normalized ground truth
    """
    pred_clean = predicted.strip()
    gt_clean = ground_truth.strip()

    # 1. Exact string match
    if pred_clean == gt_clean:
        return {
            "score": 1.0,
            "match_type": "exact",
            "predicted_normalized": pred_clean,
            "ground_truth_normalized": gt_clean,
        }

    # 2. Case-insensitive match
    if pred_clean.lower() == gt_clean.lower():
        return {
            "score": 1.0,
            "match_type": "case_insensitive",
            "predicted_normalized": pred_clean.lower(),
            "ground_truth_normalized": gt_clean.lower(),
        }

    # 3. Numeric comparison
    pred_num = normalize_numeric(pred_clean)
    gt_num = normalize_numeric(gt_clean)
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) <= tolerance:
            return {
                "score": 1.0,
                "match_type": "numeric",
                "predicted_normalized": str(pred_num),
                "ground_truth_normalized": str(gt_num),
            }

    # 4. SymPy symbolic comparison
    try:
        from sympy import simplify, sympify
        try:
            expr_pred = sympify(pred_clean)
            expr_gt = sympify(gt_clean)
            if simplify(expr_pred - expr_gt) == 0:
                return {
                    "score": 1.0,
                    "match_type": "symbolic",
                    "predicted_normalized": str(expr_pred),
                    "ground_truth_normalized": str(expr_gt),
                }
        except Exception:
            pass
    except ImportError:
        pass

    return {
        "score": 0.0,
        "match_type": "no_match",
        "predicted_normalized": pred_clean,
        "ground_truth_normalized": gt_clean,
    }


def compute_reward(
    response: str,
    ground_truth: str,
    format_weight: float = 0.0,
) -> dict:
    """
    Compute reward for a model response.

    Args:
        response: Full model response text.
        ground_truth: Ground truth answer string.
        format_weight: Weight for format compliance reward.

    Returns:
        Dictionary with score and details.
    """
    predicted = extract_answer_from_text(response)
    result = grade_answer(predicted, ground_truth)

    # Optional format reward
    if format_weight > 0:
        has_skill_ops = "<skill_ops>" in response and "</skill_ops>" in response
        has_solution = "<solution>" in response and "</solution>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        format_score = (
            (0.33 if has_skill_ops else 0.0)
            + (0.33 if has_solution else 0.0)
            + (0.34 if has_answer else 0.0)
        )
        result["format_score"] = format_score
        result["score"] = (
            (1 - format_weight) * result["score"]
            + format_weight * format_score
        )

    result["predicted_answer"] = predicted
    return result
