"""Tests for the OutputParser class."""

import pytest

from skill_ttrl.core.output_parser import OutputParser, ParsedOutput, SkillOps


class TestOutputParser:
    """Tests for output parsing."""

    def setup_method(self):
        self.parser = OutputParser()

    def test_parse_full_output(self):
        text = """
<skill_ops>
  <generate>
  New skill: For number theory problems, always check modular patterns.
  </generate>

  <retrieve query="number theory modular">
  Retrieved skill #12: Use Fermat's little theorem for prime moduli.
  </retrieve>

  <evolve base="#12">
  Improved: Use Fermat's little theorem, and also consider
  Euler's theorem for composite moduli.
  </evolve>
</skill_ops>

<solution>
Step 1: We need to find 2^100 mod 7.
Step 2: By Fermat's little theorem, 2^6 ≡ 1 (mod 7).
Step 3: 100 = 6 * 16 + 4, so 2^100 ≡ 2^4 = 16 ≡ 2 (mod 7).
</solution>

<answer>
2
</answer>
"""
        result = self.parser.parse(text)

        assert result.parse_success is True
        assert result.answer == "2"
        assert "Fermat" in result.solution

        ops = result.skill_ops
        assert ops.has_generate
        assert "modular" in ops.generate

        assert ops.has_retrieve
        assert ops.retrieve_query == "number theory modular"

        assert ops.has_evolve
        assert ops.evolve_base_id == "12"
        assert "Euler" in ops.evolve_content

    def test_parse_no_skill_ops(self):
        text = """
<solution>
The answer is straightforward: 2 + 3 = 5.
</solution>

<answer>
5
</answer>
"""
        result = self.parser.parse(text)
        assert result.answer == "5"
        assert not result.has_skill_ops

    def test_parse_only_generate(self):
        text = """
<skill_ops>
  <generate>Always simplify fractions before comparing.</generate>
</skill_ops>

<solution>3/6 = 1/2, which equals 0.5.</solution>
<answer>0.5</answer>
"""
        result = self.parser.parse(text)
        assert result.skill_ops.has_generate
        assert not result.skill_ops.has_retrieve
        assert not result.skill_ops.has_evolve
        assert result.skill_ops.ops_summary == ["generate"]

    def test_parse_boxed_answer(self):
        text = "The answer is \\boxed{42}."
        result = self.parser.parse(text)
        assert result.answer == "42"

    def test_parse_nested_boxed(self):
        text = "Therefore \\boxed{\\frac{1}{2}}."
        result = self.parser.parse(text)
        assert result.answer == "\\frac{1}{2}"

    def test_extract_answer(self):
        assert self.parser.extract_answer("<answer>42</answer>") == "42"
        assert self.parser.extract_answer("\\boxed{7}") == "7"
        assert self.parser.extract_answer("The final answer is 3.") == "The final answer is 3."

    def test_extract_answers_batch(self):
        texts = [
            "<answer>1</answer>",
            "<answer>2</answer>",
            "\\boxed{3}",
        ]
        answers = self.parser.extract_answers_batch(texts)
        assert answers == ["1", "2", "3"]

    def test_parse_batch(self):
        texts = [
            "<answer>A</answer>",
            "<skill_ops><generate>Skill</generate></skill_ops><answer>B</answer>",
        ]
        results = self.parser.parse_batch(texts)
        assert len(results) == 2
        assert results[0].answer == "A"
        assert results[1].answer == "B"
        assert results[1].has_skill_ops

    def test_parse_empty(self):
        result = self.parser.parse("")
        assert result.answer == ""
        assert not result.parse_success

    def test_ops_summary(self):
        ops = SkillOps(
            generate="new skill",
            retrieve_query="query",
            retrieve_results="results",
        )
        assert ops.ops_summary == ["generate", "retrieve"]
