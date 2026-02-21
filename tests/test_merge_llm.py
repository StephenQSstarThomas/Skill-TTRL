"""Tests for the SkillMerger (using heuristic mode, no API needed)."""

import pytest

from skill_ttrl.core.merge_llm import SkillMerger, MergedSkillOps
from skill_ttrl.core.output_parser import OutputParser, ParsedOutput, SkillOps


class TestSkillMerger:
    """Tests for skill merging with heuristic fallback."""

    def setup_method(self):
        # Use heuristic mode (no API calls needed)
        self.merger = SkillMerger(use_api=False)
        self.parser = OutputParser()

    def _make_parsed(
        self,
        answer: str = "42",
        generate: str = None,
        retrieve_query: str = None,
        retrieve_results: str = None,
        evolve_base_id: str = None,
        evolve_content: str = None,
    ) -> ParsedOutput:
        ops = SkillOps(
            generate=generate,
            retrieve_query=retrieve_query,
            retrieve_results=retrieve_results,
            evolve_base_id=evolve_base_id,
            evolve_content=evolve_content,
        )
        return ParsedOutput(skill_ops=ops, answer=answer)

    def test_merge_generates_dedup(self):
        winners = [
            self._make_parsed(generate="Use modular arithmetic for divisibility."),
            self._make_parsed(generate="Use modular arithmetic for divisibility."),
            self._make_parsed(generate="Check boundary conditions carefully."),
        ]
        result = self.merger.merge(winners)
        # Should deduplicate: 2 unique skills
        assert len(result.new_skills) == 2

    def test_merge_retrieves_frequency(self):
        winners = [
            self._make_parsed(
                retrieve_query="number theory",
                retrieve_results="Skill #12: Fermat's theorem. Skill #27: Euler's."
            ),
            self._make_parsed(
                retrieve_query="modular",
                retrieve_results="Skill #12: Fermat's theorem. Skill #33: CRT."
            ),
            self._make_parsed(
                retrieve_query="primes",
                retrieve_results="Skill #12: Fermat's theorem."
            ),
        ]
        result = self.merger.merge(winners)
        # #12 appears in all 3 winners -> should be in useful subset
        assert "12" in result.useful_retrieval_ids

    def test_merge_evolves(self):
        winners = [
            self._make_parsed(
                evolve_base_id="12",
                evolve_content="Improved: Add Euler's theorem case."
            ),
            self._make_parsed(
                evolve_base_id="12",
                evolve_content="Improved: Add Euler's theorem and also handle composite."
            ),
        ]
        result = self.merger.merge(winners)
        assert len(result.evolved_skills) == 1
        base_id, evolved = result.evolved_skills[0]
        assert base_id == "12"
        # Should keep the longer (more detailed) version
        assert "composite" in evolved["content"]

    def test_merge_no_ops(self):
        winners = [
            self._make_parsed(answer="42"),
            self._make_parsed(answer="42"),
        ]
        result = self.merger.merge(winners)
        assert result.new_skills == []
        assert result.useful_retrieval_ids == []
        assert result.evolved_skills == []
        assert result.merge_summary == "no ops"

    def test_merge_summary(self):
        winners = [
            self._make_parsed(
                generate="New skill here",
                evolve_base_id="5",
                evolve_content="Evolved content",
            ),
        ]
        result = self.merger.merge(winners)
        assert "new skill" in result.merge_summary.lower() or len(result.new_skills) > 0

    def test_parse_json_response_direct(self):
        text = '[{"title": "Test", "content": "Content"}]'
        parsed = self.merger._parse_json_response(text)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Test"

    def test_parse_json_response_markdown(self):
        text = '```json\n[{"title": "Test"}]\n```'
        parsed = self.merger._parse_json_response(text)
        assert len(parsed) == 1

    def test_parse_json_response_invalid(self):
        text = "This is not JSON"
        parsed = self.merger._parse_json_response(text)
        assert parsed == []
