"""
Majority Voting: determine pseudo ground-truth from multiple samples.

Implements the core TTRL mechanism -- given K samples, the majority-voted answer
becomes the pseudo ground-truth, and samples agreeing with it receive reward 1.
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Handles:
    - Whitespace/case normalization
    - LaTeX cleanup (\\text{}, \\mathrm{}, etc.)
    - Common mathematical equivalences
    """
    if not answer:
        return ""

    s = answer.strip()

    # Remove surrounding $ signs
    s = s.strip("$")

    # Remove \\text{...}, \\mathrm{...}, \\textbf{...}
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)

    # Remove \\left, \\right
    s = re.sub(r"\\(?:left|right)", "", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Try numeric normalization
    try:
        val = float(s.replace(",", ""))
        # Normalize to remove trailing zeros
        if val == int(val):
            s = str(int(val))
        else:
            s = str(val)
    except (ValueError, OverflowError):
        # Not numeric, use lowercase
        s = s.lower()

    return s


def _try_sympy_simplify(a: str, b: str) -> bool:
    """Try to check equality using SymPy (for mathematical expressions)."""
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex

        # Try direct sympify first
        try:
            expr_a = sympify(a)
            expr_b = sympify(b)
        except Exception:
            # Try LaTeX parsing
            try:
                expr_a = parse_latex(a)
                expr_b = parse_latex(b)
            except Exception:
                return False

        diff = simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        return False


def answers_match(a: str, b: str) -> bool:
    """
    Check if two answers are equivalent.

    Tries string matching first, then SymPy for mathematical expressions.
    """
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True

    # Try symbolic comparison for math
    return _try_sympy_simplify(a, b)


def majority_vote(
    answers: list[str],
    normalize: bool = True,
) -> tuple[str, float, list[bool]]:
    """
    Perform majority voting on a list of answers.

    Args:
        answers: List of K answer strings from K samples.
        normalize: Whether to normalize answers before comparison.

    Returns:
        Tuple of:
        - majority_answer: The most common answer (original form).
        - agreement_ratio: Fraction of samples agreeing with majority.
        - matches: Boolean list indicating which samples match the majority.
    """
    if not answers:
        return "", 0.0, []

    # Normalize all answers
    if normalize:
        normed = [normalize_answer(a) for a in answers]
    else:
        normed = list(answers)

    # Count occurrences
    counter = Counter(normed)
    majority_normed, majority_count = counter.most_common(1)[0]

    # Find original form of majority answer
    majority_original = ""
    for orig, norm in zip(answers, normed):
        if norm == majority_normed:
            majority_original = orig.strip()
            break

    agreement_ratio = majority_count / len(answers)

    # Build match list
    matches = [n == majority_normed for n in normed]

    logger.debug(
        f"Majority vote: '{majority_original}' with "
        f"{agreement_ratio:.1%} agreement ({majority_count}/{len(answers)})"
    )

    return majority_original, agreement_ratio, matches


def compute_voting_rewards(
    answers: list[str],
    normalize: bool = True,
) -> tuple[list[float], str, float]:
    """
    Compute binary rewards based on majority voting.

    Args:
        answers: List of K answer strings.
        normalize: Whether to normalize answers.

    Returns:
        Tuple of:
        - rewards: List of float rewards (1.0 if matches majority, 0.0 otherwise).
        - majority_answer: The majority-voted answer.
        - agreement_ratio: Fraction agreeing with majority.
    """
    majority_answer, agreement_ratio, matches = majority_vote(
        answers, normalize=normalize
    )
    rewards = [1.0 if m else 0.0 for m in matches]
    return rewards, majority_answer, agreement_ratio


def batch_majority_vote(
    answers_per_prompt: list[list[str]],
    normalize: bool = True,
) -> list[tuple[str, float, list[bool]]]:
    """
    Perform majority voting for a batch of prompts.

    Args:
        answers_per_prompt: List of lists, each inner list has K answers for one prompt.

    Returns:
        List of (majority_answer, agreement_ratio, matches) tuples.
    """
    return [majority_vote(answers, normalize) for answers in answers_per_prompt]
