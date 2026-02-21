"""
Skill Extractor: automatically extract skills from successful solutions.

Unlike the original approach that relied on models voluntarily outputting
<skill_ops> tags, this module extracts skills from winning solutions using
either an external LLM or a heuristic approach.

Inspired by the reference SkillRL project's SkillUpdater, but adapted to
extract reusable strategies from *successful* solutions rather than analyzing
failures.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt for external LLM skill extraction
_EXTRACT_SKILLS_PROMPT = """\
You are a math education expert. Analyze these {n} correct solutions to the same problem \
and extract reusable problem-solving skills/strategies.

## Problem:
{problem}

## Correct Solutions:
{solutions_text}

## Instructions:
- Identify 1-3 reusable mathematical strategies or techniques used across solutions
- Each skill should be a general principle, NOT specific to this exact problem
- Skills should be actionable: describe WHEN and HOW to apply the technique
- Skip trivially obvious strategies (like "read the problem carefully")

Output ONLY a JSON array. Each element:
{{"title": "3-6 word title", "content": "1-3 sentence description of the strategy and when to apply it"}}

Output ONLY the JSON array, no other text."""


# Patterns that indicate mathematical techniques/strategies
_TECHNIQUE_PATTERNS = {
    "substitution": [
        r"(?:let|set|denote|define)\s+\w+\s*=",
        r"substitut",
    ],
    "factoring": [
        r"factor(?:iz|is)",
        r"(?:can be written as|rewrite as)\s*\(",
    ],
    "quadratic_formula": [
        r"quadratic formula",
        r"[-+]?\s*\\?sqrt\{?\s*b\^?2",
        r"discriminant",
    ],
    "trigonometric_identities": [
        r"(?:sin|cos|tan)\^?2?\s*[+\-]\s*(?:sin|cos|tan)\^?2?\s*=",
        r"double angle",
        r"sum[- ]to[- ]product",
        r"identity",
    ],
    "completing_the_square": [
        r"complet(?:e|ing)\s+the\s+square",
    ],
    "induction": [
        r"(?:mathematical\s+)?induction",
        r"base\s+case",
        r"inductive\s+(?:step|hypothesis)",
    ],
    "cases_analysis": [
        r"case\s+[12345ivIV]",
        r"consider\s+(?:the\s+)?(?:two|three|following)\s+cases",
    ],
    "derivative_analysis": [
        r"(?:take|find|compute)\s+(?:the\s+)?derivative",
        r"f['\u2032]\s*\(",
        r"critical\s+point",
        r"(?:increasing|decreasing)\s+(?:on|in)",
    ],
    "integration_technique": [
        r"integrat(?:e|ion|ing)",
        r"by\s+parts",
        r"partial\s+fractions",
    ],
    "geometric_reasoning": [
        r"(?:law of|cosine rule|sine rule)",
        r"(?:similar|congruent)\s+triangles",
        r"area\s*=\s*(?:\\frac)?",
        r"Pythagorean",
    ],
    "sequence_pattern": [
        r"(?:arithmetic|geometric)\s+(?:sequence|progression|series)",
        r"(?:common|constant)\s+(?:ratio|difference)",
        r"telescop",
        r"recurrence",
    ],
    "probability_reasoning": [
        r"(?:binomial|Poisson|geometric)\s+distribution",
        r"(?:expected|expectation|E\[|E\()",
        r"(?:independent|conditional|Bayes)",
        r"(?:combinat|choose|binom)",
    ],
    "inequality_technique": [
        r"(?:AM-GM|Cauchy|Jensen|Chebyshev)",
        r"(?:triangle\s+)?inequality",
        r"(?:sign|number)\s+line",
    ],
    "coordinate_geometry": [
        r"(?:equation of|slope|intercept|midpoint|distance formula)",
        r"(?:ellipse|parabola|hyperbola)",
        r"(?:focus|foci|directrix|vertex)",
    ],
}

# Mapping from technique to a skill description
_TECHNIQUE_SKILLS = {
    "substitution": {
        "title": "Variable Substitution Strategy",
        "content": "When an expression is complex, introduce a substitution variable to simplify. Let u = (complex expression), solve in terms of u, then substitute back. Apply when you see repeated sub-expressions or when the problem becomes simpler with a change of variable.",
    },
    "factoring": {
        "title": "Factor Before Solving",
        "content": "Before attempting to solve equations, look for factoring opportunities. Factor common terms, use difference of squares, or group terms. This often reveals solutions more directly than expanding.",
    },
    "quadratic_formula": {
        "title": "Discriminant-First Approach",
        "content": "For quadratic equations, compute the discriminant b^2-4ac first to determine the number and nature of solutions. This guides whether to factor, use the formula, or recognize no real solutions exist.",
    },
    "trigonometric_identities": {
        "title": "Trig Identity Transformation",
        "content": "When solving trigonometric problems, identify which identity to apply: Pythagorean (sin^2+cos^2=1), double angle, or sum-to-product. Transform the expression to involve a single trig function when possible.",
    },
    "completing_the_square": {
        "title": "Complete the Square Method",
        "content": "Transform quadratic expressions into perfect square form (x+a)^2+b to find extrema, solve equations, or simplify expressions. Especially useful for optimization and conic sections.",
    },
    "induction": {
        "title": "Mathematical Induction Framework",
        "content": "For proving statements about all positive integers: (1) verify base case, (2) assume true for n=k, (3) prove for n=k+1. Structure the inductive step by expressing the k+1 case in terms of the k case.",
    },
    "cases_analysis": {
        "title": "Systematic Case Analysis",
        "content": "When a problem has different behaviors in different regions, systematically enumerate all cases. Ensure cases are exhaustive and mutually exclusive. Solve each case independently, then combine results.",
    },
    "derivative_analysis": {
        "title": "Derivative Sign Analysis",
        "content": "To find extrema: compute f'(x), find critical points where f'(x)=0 or undefined, then test intervals. Use second derivative or sign change to classify as max/min. Always check endpoints for closed intervals.",
    },
    "geometric_reasoning": {
        "title": "Law of Cosines Strategy",
        "content": "In triangle problems with mixed side/angle information, the Law of Cosines (c^2=a^2+b^2-2ab*cos(C)) connects sides and angles. Use it to find unknown sides or angles, then apply area formula (1/2)ab*sin(C).",
    },
    "sequence_pattern": {
        "title": "Recurrence to Closed Form",
        "content": "For recursive sequences a_{n+1}=f(a_n): (1) compute first few terms to spot patterns, (2) if linear recurrence a_{n+1}=pa_n+q, transform to geometric by letting b_n=a_n+q/(p-1), (3) derive closed form from geometric sequence.",
    },
    "probability_reasoning": {
        "title": "Distribution Recognition",
        "content": "Identify the probability distribution type: geometric (trials until first success), binomial (fixed trials, count successes), or hypergeometric (sampling without replacement). Apply the corresponding formula for E(X) and P(X=k).",
    },
    "inequality_technique": {
        "title": "Solve Inequalities Systematically",
        "content": "For systems of inequalities: solve each inequality independently, then find the intersection of solution sets. Use a number line to visualize overlapping regions. Check boundary points carefully.",
    },
    "coordinate_geometry": {
        "title": "Conic Section Focal Properties",
        "content": "For ellipse/hyperbola problems: identify foci from the equation, use focal chord properties, and connect midpoint conditions with slope relationships. The relationship between midpoint and slope often yields the line equation directly.",
    },
}


class SkillExtractor:
    """
    Extracts reusable skills from successful solutions.

    Two modes:
    - API mode: uses an external LLM to analyze solutions and extract skills
    - Heuristic mode: uses pattern matching to detect mathematical techniques

    The heuristic mode is always available as a fallback and doesn't require
    any API configuration.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        min_winners_for_extraction: int = 2,
        max_skills_per_problem: int = 3,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_winners_for_extraction = min_winners_for_extraction
        self.max_skills_per_problem = max_skills_per_problem
        self._client = None
        self.use_api = bool(api_key)

    def _get_client(self):
        if self._client is None and self.use_api:
            try:
                from openai import OpenAI
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                self._client = OpenAI(**kwargs)
            except ImportError:
                logger.warning("openai package not installed, using heuristic extraction")
                self.use_api = False
        return self._client

    def extract_from_solutions(
        self,
        problem: str,
        winning_solutions: list[str],
        existing_skill_titles: Optional[set[str]] = None,
    ) -> list[dict]:
        """
        Extract skills from winning (correct) solutions to a problem.

        Args:
            problem: The original problem text.
            winning_solutions: List of solution texts that gave the correct answer.
            existing_skill_titles: Set of already-known skill titles (for dedup).

        Returns:
            List of skill dicts with 'title' and 'content' keys.
        """
        if len(winning_solutions) < self.min_winners_for_extraction:
            return []

        existing_skill_titles = existing_skill_titles or set()

        # Try API extraction first
        if self.use_api:
            try:
                skills = self._api_extract(problem, winning_solutions)
                return self._dedup_skills(skills, existing_skill_titles)
            except Exception as e:
                logger.warning(f"API extraction failed: {e}, using heuristic")

        # Heuristic extraction
        skills = self._heuristic_extract(problem, winning_solutions)
        return self._dedup_skills(skills, existing_skill_titles)

    def _api_extract(
        self,
        problem: str,
        solutions: list[str],
    ) -> list[dict]:
        """Extract skills using an external LLM."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("No LLM client available")

        solutions_text = "\n\n".join(
            f"### Solution {i+1}:\n{s[:1500]}"
            for i, s in enumerate(solutions[:5])
        )

        prompt = _EXTRACT_SKILLS_PROMPT.format(
            n=len(solutions),
            problem=problem[:500],
            solutions_text=solutions_text,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        text = response.choices[0].message.content.strip()
        return self._parse_json_skills(text)

    def _heuristic_extract(
        self,
        problem: str,
        solutions: list[str],
    ) -> list[dict]:
        """
        Extract skills using pattern matching on solution text.

        Detects mathematical techniques by regex patterns in the solutions.
        Skills found in multiple solutions are prioritized.
        """
        combined_text = problem + " " + " ".join(solutions)
        combined_lower = combined_text.lower()

        # Count technique occurrences across solutions
        technique_counts: Counter = Counter()
        technique_in_solutions: dict[str, int] = defaultdict(int)

        for technique, patterns in _TECHNIQUE_PATTERNS.items():
            for solution in solutions:
                sol_lower = solution.lower()
                for pattern in patterns:
                    if re.search(pattern, sol_lower, re.IGNORECASE):
                        technique_in_solutions[technique] += 1
                        break  # count once per solution per technique

            # Also check the problem text
            for pattern in patterns:
                if re.search(pattern, combined_lower, re.IGNORECASE):
                    technique_counts[technique] += 1

        # Prioritize techniques found in multiple solutions
        scored_techniques = []
        for technique, count in technique_in_solutions.items():
            # Score: number of solutions using this technique
            score = count + technique_counts.get(technique, 0)
            scored_techniques.append((score, technique))

        scored_techniques.sort(reverse=True)

        # Convert top techniques to skills
        skills = []
        for score, technique in scored_techniques[:self.max_skills_per_problem]:
            if score < 1:
                continue
            if technique in _TECHNIQUE_SKILLS:
                skills.append(dict(_TECHNIQUE_SKILLS[technique]))

        return skills

    def _parse_json_skills(self, text: str) -> list[dict]:
        """Parse JSON array of skills from LLM response."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    s for s in parsed
                    if isinstance(s, dict) and "title" in s and "content" in s
                ]
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return [
                        s for s in parsed
                        if isinstance(s, dict) and "title" in s and "content" in s
                    ]
            except json.JSONDecodeError:
                pass

        # Try finding array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [
                        s for s in parsed
                        if isinstance(s, dict) and "title" in s and "content" in s
                    ]
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse skills JSON from response: {text[:200]}")
        return []

    def _dedup_skills(
        self,
        skills: list[dict],
        existing_titles: set[str],
    ) -> list[dict]:
        """Remove duplicate skills based on title similarity."""
        result = []
        seen_titles = set()

        for skill in skills:
            title = skill.get("title", "")
            if not title:
                continue
            title = title.strip()
            title_lower = title.lower()

            # Skip if too similar to existing skills
            if any(self._title_similar(title_lower, et.lower()) for et in existing_titles):
                continue

            # Skip if duplicate within batch
            if any(self._title_similar(title_lower, st) for st in seen_titles):
                continue

            seen_titles.add(title_lower)
            result.append(skill)

        return result[:self.max_skills_per_problem]

    @staticmethod
    def _title_similar(a: str, b: str) -> bool:
        """Check if two titles are similar (simple word overlap)."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        return overlap / min(len(words_a), len(words_b)) > 0.6
