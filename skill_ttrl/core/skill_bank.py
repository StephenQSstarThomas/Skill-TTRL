"""
Skill Bank: storage, retrieval, evolution, and management of skills.

The skill bank stores skills organized by task type, supports semantic retrieval
via embeddings or keyword matching, and manages the lifecycle of skills including
addition, replacement, and eviction.
"""

from __future__ import annotations

import json
import uuid
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A single skill entry in the skill bank."""
    skill_id: str
    title: str
    content: str
    task_type: str = "general"
    source: str = "generated"  # "generated", "evolved", "seeded"
    usage_count: int = 0
    success_count: int = 0
    created_at_round: int = 0
    last_used_round: int = 0
    parent_id: Optional[str] = None  # for evolved skills
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        d = dict(d)
        d.pop("embedding", None)
        return cls(**d)

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class SkillBank:
    """
    Manages a collection of skills organized by task type.

    Supports:
    - Adding new skills (from generation or evolution)
    - Retrieving relevant skills (via embedding similarity or keyword)
    - Replacing/evolving existing skills
    - Evicting stale skills based on sliding window
    - Persistence to/from JSON
    """

    def __init__(
        self,
        max_skills: int = 200,
        retrieval_mode: str = "embedding",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 6,
        eviction_window: int = 50,
    ):
        self.max_skills = max_skills
        self.retrieval_mode = retrieval_mode
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.eviction_window = eviction_window

        self._skills: dict[str, Skill] = {}  # skill_id -> Skill
        self._task_type_index: dict[str, set[str]] = {}  # task_type -> {skill_ids}
        self._embedding_model = None
        self._embeddings_dirty = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._skills)

    @property
    def task_types(self) -> list[str]:
        return sorted(self._task_type_index.keys())

    @property
    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add(self, skill: Skill) -> str:
        """Add a skill to the bank. Returns the skill_id."""
        if not skill.skill_id:
            skill.skill_id = self._generate_id()

        self._skills[skill.skill_id] = skill
        self._task_type_index.setdefault(skill.task_type, set()).add(skill.skill_id)
        self._embeddings_dirty = True

        # Enforce capacity
        if self.size > self.max_skills:
            self._evict_one()

        logger.debug(f"Added skill {skill.skill_id}: {skill.title}")
        return skill.skill_id

    def add_from_dict(self, d: dict, task_type: str = "general",
                      source: str = "generated", round_num: int = 0) -> str:
        """Convenience: add a skill from a raw dictionary."""
        skill = Skill(
            skill_id=d.get("skill_id", self._generate_id()),
            title=d.get("title", "Untitled"),
            content=d.get("content", d.get("principle", "")),
            task_type=task_type,
            source=source,
            created_at_round=round_num,
        )
        return self.add(skill)

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def remove(self, skill_id: str) -> bool:
        skill = self._skills.pop(skill_id, None)
        if skill is None:
            return False
        self._task_type_index.get(skill.task_type, set()).discard(skill_id)
        self._embeddings_dirty = True
        return True

    def replace(self, old_id: str, new_skill: Skill) -> str:
        """Replace an existing skill with an evolved version."""
        old = self._skills.get(old_id)
        if old is not None:
            new_skill.parent_id = old_id
            new_skill.usage_count = old.usage_count
            new_skill.success_count = old.success_count
            self.remove(old_id)
        return self.add(new_skill)

    def record_usage(self, skill_id: str, success: bool, round_num: int) -> None:
        """Record that a skill was used (and whether it led to success)."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return
        skill.usage_count += 1
        skill.last_used_round = round_num
        if success:
            skill.success_count += 1

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        task_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[Skill]:
        """
        Retrieve the most relevant skills for a query.

        Args:
            query: Natural language query describing the task/problem.
            task_type: If provided, restrict to this task type.
            top_k: Number of skills to return (defaults to self.top_k).

        Returns:
            List of the most relevant Skill objects.
        """
        top_k = top_k or self.top_k
        if self.size == 0:
            return []

        candidates = self._get_candidates(task_type)
        if not candidates:
            return []

        if self.retrieval_mode == "embedding":
            return self._embedding_retrieve(query, candidates, top_k)
        else:
            return self._keyword_retrieve(query, candidates, top_k)

    def summary(self, task_type: Optional[str] = None) -> str:
        """Return a text summary of the skill bank for prompt injection."""
        skills = self._get_candidates(task_type)
        if not skills:
            return "(Skill bank is empty)"

        lines = [f"Skill Bank ({len(skills)} skills):"]
        for s in skills[:20]:  # cap at 20 for prompt length
            lines.append(f"  [{s.skill_id}] {s.title} (type={s.task_type})")
        if len(skills) > 20:
            lines.append(f"  ... and {len(skills) - 20} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Embedding-based retrieval
    # ------------------------------------------------------------------
    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, falling back to keyword retrieval"
                )
                self.retrieval_mode = "keyword"
                return None
        return self._embedding_model

    def _compute_embedding(self, text: str) -> np.ndarray:
        model = self._get_embedding_model()
        if model is None:
            return np.zeros(384)
        return model.encode(text, normalize_embeddings=True)

    def _ensure_embeddings(self) -> None:
        """Compute embeddings for all skills that don't have them yet."""
        if not self._embeddings_dirty:
            return
        model = self._get_embedding_model()
        if model is None:
            return

        needs_embedding = [
            s for s in self._skills.values() if s.embedding is None
        ]
        if needs_embedding:
            texts = [f"{s.title}: {s.content}" for s in needs_embedding]
            embeddings = model.encode(texts, normalize_embeddings=True)
            for skill, emb in zip(needs_embedding, embeddings):
                skill.embedding = emb.tolist()

        self._embeddings_dirty = False

    def _embedding_retrieve(
        self, query: str, candidates: list[Skill], top_k: int
    ) -> list[Skill]:
        self._ensure_embeddings()
        query_emb = self._compute_embedding(query)

        scored = []
        for s in candidates:
            if s.embedding is not None:
                sim = float(np.dot(query_emb, np.array(s.embedding)))
            else:
                sim = 0.0
            scored.append((sim, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def _keyword_retrieve(
        self, query: str, candidates: list[Skill], top_k: int
    ) -> list[Skill]:
        query_tokens = set(query.lower().split())

        scored = []
        for s in candidates:
            skill_tokens = set(
                (s.title + " " + s.content + " " + s.task_type).lower().split()
            )
            overlap = len(query_tokens & skill_tokens)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_candidates(self, task_type: Optional[str] = None) -> list[Skill]:
        if task_type is not None:
            ids = self._task_type_index.get(task_type, set())
            general_ids = self._task_type_index.get("general", set())
            all_ids = ids | general_ids
            return [self._skills[sid] for sid in all_ids if sid in self._skills]
        return list(self._skills.values())

    def _evict_one(self) -> None:
        """Evict the least useful skill to maintain capacity."""
        if not self._skills:
            return
        # Score: lower is worse (evict first)
        worst_id = min(
            self._skills,
            key=lambda sid: (
                self._skills[sid].success_rate,
                self._skills[sid].last_used_round,
            ),
        )
        logger.debug(f"Evicting skill {worst_id}")
        self.remove(worst_id)

    def evict_stale(self, current_round: int) -> int:
        """Evict skills not used within the eviction window."""
        threshold = current_round - self.eviction_window
        stale = [
            sid for sid, s in self._skills.items()
            if s.last_used_round < threshold and s.source != "seeded"
        ]
        for sid in stale:
            self.remove(sid)
        return len(stale)

    @staticmethod
    def _generate_id() -> str:
        return f"sk_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_skills": self.max_skills,
            "retrieval_mode": self.retrieval_mode,
            "skills": [s.to_dict() for s in self._skills.values()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {self.size} skills to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 6,
        eviction_window: int = 50,
    ) -> SkillBank:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        bank = cls(
            max_skills=data.get("max_skills", 200),
            retrieval_mode=data.get("retrieval_mode", "embedding"),
            embedding_model=embedding_model,
            top_k=top_k,
            eviction_window=eviction_window,
        )
        for sd in data.get("skills", []):
            bank.add(Skill.from_dict(sd))
        return bank

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"SkillBank(size={self.size}, max={self.max_skills}, "
            f"types={self.task_types})"
        )
