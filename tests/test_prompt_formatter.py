"""Tests for prompt formatting."""

import pytest

from skill_ttrl.core.skill_bank import SkillBank, Skill
from skill_ttrl.prompts.formatter import PromptFormatter
from skill_ttrl.prompts.templates import SKILL_TTRL_SYSTEM_PROMPT


class TestPromptFormatter:
    """Tests for prompt formatting with skill injection."""

    def setup_method(self):
        self.bank = SkillBank(max_skills=50, retrieval_mode="keyword", top_k=3)
        self.formatter = PromptFormatter(
            skill_bank=self.bank,
            task_type="math",
            suffix_prompt="\nPut answer in \\boxed{}.",
        )

    def test_cold_start_prompt(self):
        """When skill bank is empty, should use cold start template."""
        prompt = self.formatter.format_prompt("What is 2+2?")
        assert "skill bank is currently empty" in prompt.lower()
        assert "2+2" in prompt
        assert "\\boxed{}" in prompt

    def test_prompt_with_skills(self):
        """When skills exist, should inject them."""
        self.bank.add(Skill(
            skill_id="sk_001",
            title="Basic Addition",
            content="Add numbers by combining values.",
            task_type="math",
        ))
        self.bank.add(Skill(
            skill_id="sk_002",
            title="Number Patterns",
            content="Look for patterns in number sequences.",
            task_type="math",
        ))

        prompt = self.formatter.format_prompt("What is 2+2?")
        assert "Skill Bank" in prompt
        assert "Basic Addition" in prompt
        assert "skill_ops" in prompt.lower() or "generate" in prompt.lower()

    def test_skill_context_format(self):
        self.bank.add(Skill(
            skill_id="sk_001",
            title="Test Skill",
            content="Test content.",
            task_type="math",
            usage_count=5,
            success_count=3,
        ))

        context = self.formatter.format_skill_context("test query")
        assert "Test Skill" in context
        assert "sk_001" in context
        assert "60%" in context  # 3/5 = 60%

    def test_format_batch(self):
        prompts = self.formatter.format_batch(["Problem 1", "Problem 2"])
        assert len(prompts) == 2
        assert "Problem 1" in prompts[0]
        assert "Problem 2" in prompts[1]

    def test_agent_prompt(self):
        self.bank.add(Skill(
            skill_id="sk_001",
            title="Search Strategy",
            content="Search systematically.",
            task_type="agent",
        ))

        prompt = self.formatter.format_agent_prompt(
            task_description="Put apple on counter",
            current_state="Kitchen, apple on table",
            available_actions="take, go, look",
            task_type="agent",
        )
        assert "Put apple on counter" in prompt
        assert "Kitchen" in prompt
        assert "take, go, look" in prompt

    def test_system_prompt_included(self):
        prompt = self.formatter.format_prompt("Test problem")
        assert "skill operations" in prompt.lower() or "skill_ops" in prompt.lower()
