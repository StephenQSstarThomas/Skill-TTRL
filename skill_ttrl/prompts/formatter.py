"""
Prompt Formatter: constructs prompts with skill bank context injection.

Takes a problem/task and the skill bank state, formats them into the
model prompt with optional skill retrieval results.
"""

from __future__ import annotations

import logging
from typing import Optional

from skill_ttrl.core.skill_bank import SkillBank, Skill
from skill_ttrl.prompts.templates import (
    SKILL_TTRL_SYSTEM_PROMPT,
    REASONING_PROMPT_TEMPLATE,
    AGENT_PROMPT_TEMPLATE,
    COLD_START_TEMPLATE,
    MATH_SUFFIX,
)

logger = logging.getLogger(__name__)


class PromptFormatter:
    """
    Formats prompts for Skill TTRL by injecting skill bank context.

    For each problem, retrieves relevant skills and constructs a prompt
    that instructs the model on available skill operations.
    """

    def __init__(
        self,
        skill_bank: SkillBank,
        task_type: str = "math",
        suffix_prompt: str = MATH_SUFFIX,
        max_skills_in_prompt: int = 10,
    ):
        self.skill_bank = skill_bank
        self.task_type = task_type
        self.suffix_prompt = suffix_prompt
        self.max_skills_in_prompt = max_skills_in_prompt

    def format_skill_context(
        self,
        query: str,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Retrieve skills and format them for prompt injection.

        Args:
            query: The problem/task text used for skill retrieval.
            task_type: Optional task type to filter skills.

        Returns:
            Formatted skill bank context string.
        """
        if self.skill_bank.size == 0:
            return "## Skill Bank\n\n(No skills available yet. Consider generating new ones.)"

        skills = self.skill_bank.retrieve(
            query=query,
            task_type=task_type or self.task_type,
            top_k=self.max_skills_in_prompt,
        )

        if not skills:
            return "## Skill Bank\n\n(No relevant skills found for this problem.)"

        lines = [
            f"## Skill Bank ({len(skills)} relevant skills)\n",
            "The following skills from previous experience may be useful:\n",
        ]

        for i, skill in enumerate(skills, 1):
            lines.append(
                f"### Skill #{skill.skill_id} — {skill.title}\n"
                f"{skill.content}\n"
                f"*(Type: {skill.task_type}, "
                f"Success rate: {skill.success_rate:.0%}, "
                f"Used: {skill.usage_count} times)*\n"
            )

        return "\n".join(lines)

    def format_prompt(
        self,
        problem: str,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Format a complete prompt for a reasoning/math problem.

        Args:
            problem: The problem text.
            task_type: Optional task type for skill retrieval.

        Returns:
            Complete formatted prompt string.
        """
        if self.skill_bank.size == 0:
            return COLD_START_TEMPLATE.format(
                system_prompt=SKILL_TTRL_SYSTEM_PROMPT,
                problem=problem,
                suffix_prompt=self.suffix_prompt,
            )

        skill_context = self.format_skill_context(problem, task_type)

        return REASONING_PROMPT_TEMPLATE.format(
            system_prompt=SKILL_TTRL_SYSTEM_PROMPT,
            skill_bank_context=skill_context,
            problem=problem,
            suffix_prompt=self.suffix_prompt,
        )

    def format_agent_prompt(
        self,
        task_description: str,
        current_state: str,
        available_actions: str,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Format a prompt for agent tasks (ALFWorld, WebShop).

        Args:
            task_description: Description of the agent task.
            current_state: Current environment observation.
            available_actions: Available actions in current state.
            task_type: Task type for skill retrieval.

        Returns:
            Complete formatted agent prompt.
        """
        skill_context = self.format_skill_context(task_description, task_type)

        return AGENT_PROMPT_TEMPLATE.format(
            system_prompt=SKILL_TTRL_SYSTEM_PROMPT,
            skill_bank_context=skill_context,
            task_description=task_description,
            current_state=current_state,
            available_actions=available_actions,
        )

    def format_batch(
        self,
        problems: list[str],
        task_type: Optional[str] = None,
    ) -> list[str]:
        """Format a batch of problems into prompts."""
        return [self.format_prompt(p, task_type) for p in problems]
