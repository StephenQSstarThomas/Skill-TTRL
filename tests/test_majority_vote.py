"""Tests for majority voting."""

import pytest

from skill_ttrl.core.majority_vote import (
    normalize_answer,
    answers_match,
    majority_vote,
    compute_voting_rewards,
    batch_majority_vote,
)


class TestNormalizeAnswer:
    def test_basic_normalization(self):
        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("$42$") == "42"
        assert normalize_answer("42.0") == "42"

    def test_numeric(self):
        assert normalize_answer("3.14") == "3.14"
        assert normalize_answer("1,000") == "1000"
        assert normalize_answer("100") == "100"

    def test_latex_cleanup(self):
        assert normalize_answer("\\text{hello}") == "hello"
        assert normalize_answer("\\mathrm{x}") == "x"

    def test_non_numeric(self):
        assert normalize_answer("ABC") == "abc"


class TestAnswersMatch:
    def test_exact_match(self):
        assert answers_match("42", "42")

    def test_numeric_match(self):
        assert answers_match("42.0", "42")

    def test_case_insensitive(self):
        assert answers_match("abc", "ABC")

    def test_no_match(self):
        assert not answers_match("42", "43")


class TestMajorityVote:
    def test_clear_majority(self):
        answers = ["42", "42", "42", "43", "44"]
        majority, agreement, matches = majority_vote(answers)
        assert majority == "42"
        assert agreement == pytest.approx(0.6)
        assert matches == [True, True, True, False, False]

    def test_unanimous(self):
        answers = ["7", "7", "7", "7"]
        majority, agreement, matches = majority_vote(answers)
        assert majority == "7"
        assert agreement == 1.0
        assert all(matches)

    def test_no_clear_winner(self):
        answers = ["1", "2", "3"]
        majority, agreement, matches = majority_vote(answers)
        # One of them should win with 1/3
        assert agreement == pytest.approx(1 / 3)
        assert sum(matches) == 1

    def test_numeric_equivalence(self):
        answers = ["42", "42.0", "42", "43"]
        majority, agreement, matches = majority_vote(answers)
        assert agreement == pytest.approx(0.75)

    def test_empty_list(self):
        majority, agreement, matches = majority_vote([])
        assert majority == ""
        assert agreement == 0.0
        assert matches == []

    def test_single_answer(self):
        majority, agreement, matches = majority_vote(["42"])
        assert majority == "42"
        assert agreement == 1.0


class TestComputeVotingRewards:
    def test_binary_rewards(self):
        answers = ["42", "42", "43", "42", "44"]
        rewards, majority, agreement = compute_voting_rewards(answers)
        assert majority == "42"
        assert rewards == [1.0, 1.0, 0.0, 1.0, 0.0]

    def test_all_same(self):
        answers = ["A"] * 5
        rewards, majority, agreement = compute_voting_rewards(answers)
        assert rewards == [1.0] * 5
        assert agreement == 1.0


class TestBatchMajorityVote:
    def test_batch(self):
        batch = [
            ["42", "42", "43"],
            ["7", "8", "7"],
        ]
        results = batch_majority_vote(batch)
        assert len(results) == 2
        assert results[0][0] == "42"
        assert results[1][0] == "7"
