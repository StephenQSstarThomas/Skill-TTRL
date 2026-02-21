"""Tests for the math grader utility."""

import pytest

from skill_ttrl.utils.math_grader import (
    extract_boxed_answer,
    extract_answer_from_text,
    normalize_numeric,
    grade_answer,
    compute_reward,
)


class TestExtractBoxedAnswer:
    def test_simple_boxed(self):
        assert extract_boxed_answer("\\boxed{42}") == "42"

    def test_nested_boxed(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed(self):
        text = "First \\boxed{1}, then \\boxed{2}"
        assert extract_boxed_answer(text) == "2"  # last one

    def test_no_boxed(self):
        assert extract_boxed_answer("No boxed answer") == ""


class TestExtractAnswer:
    def test_answer_tag(self):
        assert extract_answer_from_text("<answer>42</answer>") == "42"

    def test_boxed_fallback(self):
        assert extract_answer_from_text("Result: \\boxed{7}") == "7"

    def test_answer_is_pattern(self):
        text = "The final answer is 42."
        assert "42" in extract_answer_from_text(text)


class TestNormalizeNumeric:
    def test_integer(self):
        assert normalize_numeric("42") == 42.0

    def test_float(self):
        assert normalize_numeric("3.14") == pytest.approx(3.14)

    def test_comma(self):
        assert normalize_numeric("1,000") == 1000.0

    def test_percentage(self):
        assert normalize_numeric("50%") == pytest.approx(0.5)

    def test_fraction(self):
        assert normalize_numeric("3/4") == pytest.approx(0.75)

    def test_non_numeric(self):
        assert normalize_numeric("abc") is None


class TestGradeAnswer:
    def test_exact_match(self):
        result = grade_answer("42", "42")
        assert result["score"] == 1.0
        assert result["match_type"] == "exact"

    def test_case_insensitive(self):
        result = grade_answer("Yes", "yes")
        assert result["score"] == 1.0

    def test_numeric_match(self):
        result = grade_answer("42.0", "42")
        assert result["score"] == 1.0
        assert result["match_type"] == "numeric"

    def test_no_match(self):
        result = grade_answer("42", "43")
        assert result["score"] == 0.0

    def test_fraction_vs_decimal(self):
        result = grade_answer("0.5", "1/2")
        assert result["score"] == 1.0


class TestComputeReward:
    def test_correct_boxed(self):
        response = "The answer is \\boxed{42}."
        result = compute_reward(response, "42")
        assert result["score"] == 1.0

    def test_correct_tagged(self):
        response = "<answer>42</answer>"
        result = compute_reward(response, "42")
        assert result["score"] == 1.0

    def test_incorrect(self):
        response = "<answer>43</answer>"
        result = compute_reward(response, "42")
        assert result["score"] == 0.0

    def test_format_weight(self):
        response = (
            "<skill_ops><generate>skill</generate></skill_ops>"
            "<solution>steps</solution>"
            "<answer>42</answer>"
        )
        result = compute_reward(response, "42", format_weight=0.1)
        assert result["score"] > 0.9
        assert result["format_score"] > 0.9
