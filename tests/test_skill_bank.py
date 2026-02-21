"""Tests for the SkillBank class."""

import json
import tempfile
from pathlib import Path

import pytest

from skill_ttrl.core.skill_bank import SkillBank, Skill


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_create_skill(self):
        skill = Skill(
            skill_id="sk_001",
            title="Test Skill",
            content="This is a test skill.",
            task_type="math",
        )
        assert skill.skill_id == "sk_001"
        assert skill.title == "Test Skill"
        assert skill.task_type == "math"
        assert skill.success_rate == 0.0

    def test_success_rate(self):
        skill = Skill(skill_id="sk_001", title="Test", content="Test")
        skill.usage_count = 10
        skill.success_count = 7
        assert skill.success_rate == pytest.approx(0.7)

    def test_to_dict(self):
        skill = Skill(skill_id="sk_001", title="Test", content="Content")
        d = skill.to_dict()
        assert d["skill_id"] == "sk_001"
        assert "embedding" not in d

    def test_from_dict(self):
        d = {
            "skill_id": "sk_002",
            "title": "From Dict",
            "content": "Test content",
            "task_type": "algebra",
        }
        skill = Skill.from_dict(d)
        assert skill.skill_id == "sk_002"
        assert skill.task_type == "algebra"


class TestSkillBank:
    """Tests for the SkillBank class."""

    def setup_method(self):
        self.bank = SkillBank(
            max_skills=10,
            retrieval_mode="keyword",
            top_k=3,
        )

    def test_empty_bank(self):
        assert self.bank.size == 0
        assert len(self.bank) == 0
        assert self.bank.all_skills == []

    def test_add_skill(self):
        skill = Skill(
            skill_id="sk_001",
            title="Modular Arithmetic",
            content="Use modular arithmetic for divisibility problems.",
            task_type="number_theory",
        )
        sid = self.bank.add(skill)
        assert sid == "sk_001"
        assert self.bank.size == 1
        assert self.bank.get("sk_001") is not None

    def test_add_from_dict(self):
        sid = self.bank.add_from_dict(
            {"title": "Test Skill", "content": "Test content"},
            task_type="math",
        )
        assert self.bank.size == 1
        assert self.bank.get(sid) is not None

    def test_remove_skill(self):
        self.bank.add(Skill(skill_id="sk_001", title="T", content="C"))
        assert self.bank.remove("sk_001") is True
        assert self.bank.size == 0
        assert self.bank.remove("nonexistent") is False

    def test_replace_skill(self):
        old = Skill(skill_id="sk_old", title="Old", content="Old content",
                     usage_count=5, success_count=3)
        self.bank.add(old)

        new = Skill(skill_id="sk_new", title="New", content="New content")
        self.bank.replace("sk_old", new)

        assert self.bank.get("sk_old") is None
        assert self.bank.get("sk_new") is not None
        assert self.bank.get("sk_new").parent_id == "sk_old"
        assert self.bank.get("sk_new").usage_count == 5

    def test_record_usage(self):
        self.bank.add(Skill(skill_id="sk_001", title="T", content="C"))
        self.bank.record_usage("sk_001", success=True, round_num=1)
        self.bank.record_usage("sk_001", success=False, round_num=2)

        skill = self.bank.get("sk_001")
        assert skill.usage_count == 2
        assert skill.success_count == 1
        assert skill.last_used_round == 2

    def test_keyword_retrieve(self):
        self.bank.add(Skill(
            skill_id="sk_001", title="Modular Arithmetic",
            content="Use modular arithmetic for divisibility.",
            task_type="number_theory",
        ))
        self.bank.add(Skill(
            skill_id="sk_002", title="Triangle Area",
            content="Calculate triangle area using base and height.",
            task_type="geometry",
        ))
        self.bank.add(Skill(
            skill_id="sk_003", title="Prime Factorization",
            content="Decompose numbers into prime factors.",
            task_type="number_theory",
        ))

        results = self.bank.retrieve("find prime factors of a number", top_k=2)
        assert len(results) <= 2
        # Should prefer number theory related skills
        ids = [r.skill_id for r in results]
        assert "sk_003" in ids or "sk_001" in ids

    def test_max_skills_eviction(self):
        bank = SkillBank(max_skills=3, retrieval_mode="keyword")
        for i in range(5):
            bank.add(Skill(
                skill_id=f"sk_{i:03d}",
                title=f"Skill {i}",
                content=f"Content {i}",
            ))
        assert bank.size <= 3

    def test_evict_stale(self):
        self.bank.add(Skill(
            skill_id="sk_old", title="Old", content="Old",
            last_used_round=0, source="generated",
        ))
        self.bank.add(Skill(
            skill_id="sk_new", title="New", content="New",
            last_used_round=100, source="generated",
        ))

        n_evicted = self.bank.evict_stale(current_round=100)
        assert n_evicted == 1
        assert self.bank.get("sk_old") is None
        assert self.bank.get("sk_new") is not None

    def test_save_and_load(self):
        self.bank.add(Skill(
            skill_id="sk_001", title="Test Skill",
            content="Persistence test.", task_type="math",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skills.json"
            self.bank.save(path)

            loaded = SkillBank.load(path)
            assert loaded.size == 1
            assert loaded.get("sk_001").title == "Test Skill"

    def test_summary(self):
        self.bank.add(Skill(skill_id="sk_001", title="Test", content="C"))
        summary = self.bank.summary()
        assert "sk_001" in summary
        assert "Test" in summary

    def test_task_type_index(self):
        self.bank.add(Skill(
            skill_id="sk_001", title="A", content="A", task_type="math",
        ))
        self.bank.add(Skill(
            skill_id="sk_002", title="B", content="B", task_type="code",
        ))
        assert "math" in self.bank.task_types
        assert "code" in self.bank.task_types
