"""
Output Parser: parse model outputs containing skill operations and answers.

Expected format:
    <skill_ops>
      <generate>...</generate>
      <retrieve query="...">...</retrieve>
      <evolve base="#id">...</evolve>
    </skill_ops>
    <solution>...</solution>
    <answer>...</answer>
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillOps:
    """Parsed skill operations from a single sample."""
    generate: Optional[str] = None
    retrieve_query: Optional[str] = None
    retrieve_results: Optional[str] = None
    evolve_base_id: Optional[str] = None
    evolve_content: Optional[str] = None
    raw: str = ""

    @property
    def has_generate(self) -> bool:
        return self.generate is not None and len(self.generate.strip()) > 0

    @property
    def has_retrieve(self) -> bool:
        return self.retrieve_query is not None and len(self.retrieve_query.strip()) > 0

    @property
    def has_evolve(self) -> bool:
        return self.evolve_content is not None and len(self.evolve_content.strip()) > 0

    @property
    def ops_summary(self) -> list[str]:
        ops = []
        if self.has_generate:
            ops.append("generate")
        if self.has_retrieve:
            ops.append("retrieve")
        if self.has_evolve:
            ops.append("evolve")
        return ops


@dataclass
class ParsedOutput:
    """A fully parsed model output."""
    skill_ops: SkillOps = field(default_factory=SkillOps)
    solution: str = ""
    answer: str = ""
    raw: str = ""
    parse_success: bool = True

    @property
    def has_skill_ops(self) -> bool:
        return bool(self.skill_ops.ops_summary)


class OutputParser:
    """
    Parses model outputs that contain <skill_ops>, <solution>, and <answer> tags.

    Robust to missing tags -- falls back to extracting answer from \\boxed{} or
    the last line.
    """

    # Pre-compiled patterns
    _SKILL_OPS_RE = re.compile(
        r"<skill_ops>(.*?)</skill_ops>", re.DOTALL
    )
    _GENERATE_RE = re.compile(
        r"<generate>(.*?)</generate>", re.DOTALL
    )
    _RETRIEVE_RE = re.compile(
        r'<retrieve\s+query="([^"]*)">(.*?)</retrieve>', re.DOTALL
    )
    _EVOLVE_RE = re.compile(
        r'<evolve\s+base="([^"]*)">(.*?)</evolve>', re.DOTALL
    )
    _SOLUTION_RE = re.compile(
        r"<solution>(.*?)</solution>", re.DOTALL
    )
    _ANSWER_RE = re.compile(
        r"<answer>(.*?)</answer>", re.DOTALL
    )
    _BOXED_RE = None  # use custom parser instead

    def parse(self, text: str) -> ParsedOutput:
        """Parse a single model output into structured components."""
        result = ParsedOutput(raw=text)

        # Parse skill_ops
        skill_ops_match = self._SKILL_OPS_RE.search(text)
        if skill_ops_match:
            result.skill_ops = self._parse_skill_ops(skill_ops_match.group(1))

        # Parse solution
        solution_match = self._SOLUTION_RE.search(text)
        if solution_match:
            result.solution = solution_match.group(1).strip()

        # Parse answer
        answer_match = self._ANSWER_RE.search(text)
        if answer_match:
            result.answer = answer_match.group(1).strip()
        else:
            # Fallback: try \\boxed{}
            result.answer = self._extract_boxed_answer(text)
            if not result.answer:
                result.parse_success = False

        return result

    def parse_batch(self, texts: list[str]) -> list[ParsedOutput]:
        """Parse a batch of model outputs."""
        return [self.parse(t) for t in texts]

    def extract_answer(self, text: str) -> str:
        """Extract just the answer from a model output (for voting)."""
        # Try <answer> tag first
        match = self._ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        # Try \\boxed{}
        return self._extract_boxed_answer(text)

    def extract_answers_batch(self, texts: list[str]) -> list[str]:
        """Extract answers from a batch of outputs."""
        return [self.extract_answer(t) for t in texts]

    def _parse_skill_ops(self, ops_text: str) -> SkillOps:
        """Parse the content inside <skill_ops>...</skill_ops>."""
        ops = SkillOps(raw=ops_text)

        gen_match = self._GENERATE_RE.search(ops_text)
        if gen_match:
            ops.generate = gen_match.group(1).strip()

        ret_match = self._RETRIEVE_RE.search(ops_text)
        if ret_match:
            ops.retrieve_query = ret_match.group(1).strip()
            ops.retrieve_results = ret_match.group(2).strip()

        evo_match = self._EVOLVE_RE.search(ops_text)
        if evo_match:
            ops.evolve_base_id = evo_match.group(1).strip().lstrip("#")
            ops.evolve_content = evo_match.group(2).strip()

        return ops

    def _extract_boxed_answer(self, text: str) -> str:
        """Extract answer from \\boxed{...} notation, handling nested braces."""
        results = []
        idx = 0
        while True:
            pos = text.find("\\boxed{", idx)
            if pos == -1:
                break
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
        # Last resort: last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""
