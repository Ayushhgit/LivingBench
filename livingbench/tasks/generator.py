"""
Task Generation Engine.

Orchestrates task generation from multiple sources:
- Fetches from each enabled source
- Validates and deduplicates
- Labels with skills and difficulty
- Maintains task pool with freshness guarantees

The engine is designed to ensure the benchmark never freezes:
- Daily generation of new tasks
- Version tracking for reproducibility
- Cache management for efficiency
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterator
import random

from livingbench.core.types import Task, TaskSource, DifficultyLevel
from livingbench.core.config import LivingBenchConfig, TaskGenerationConfig
from livingbench.tasks.base import TaskGeneratorBase
from livingbench.tasks.labeling import SkillLabeler, DifficultyEstimator, TaskValidator
from livingbench.tasks.sources.synthetic import (
    SyntheticReasoningGenerator,
    SyntheticMathGenerator,
)
from livingbench.tasks.sources.github import GitHubTaskGenerator
from livingbench.tasks.sources.arxiv import ArxivTaskGenerator
from livingbench.tasks.sources.tool_use import ToolUseTaskGenerator


class TaskGenerationEngine:
    """
    Central engine for task generation and management.

    Responsibilities:
    - Coordinate multiple task generators
    - Ensure quality and diversity
    - Manage task versioning and caching
    - Support reproducible generation
    """

    def __init__(
        self,
        config: TaskGenerationConfig | None = None,
        seed: int = 42,
        output_dir: str = "data/tasks",
    ):
        self.config = config or TaskGenerationConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.skill_labeler = SkillLabeler()
        self.difficulty_estimator = DifficultyEstimator()
        self.validator = TaskValidator()

        # Initialize generators
        self._generators: dict[TaskSource, TaskGeneratorBase] = {}
        self._init_generators()

        # Task pool
        self._task_pool: list[Task] = []
        self._seen_hashes: set[str] = set()

    def _init_generators(self) -> None:
        """Initialize enabled task generators."""
        if self.config.synthetic_reasoning_enabled:
            self._generators[TaskSource.SYNTHETIC_REASONING] = SyntheticReasoningGenerator(
                seed=self.seed
            )

        if self.config.synthetic_math_enabled:
            self._generators[TaskSource.SYNTHETIC_MATH] = SyntheticMathGenerator(
                seed=self.seed
            )

        if self.config.github_enabled:
            self._generators[TaskSource.GITHUB_ISSUE] = GitHubTaskGenerator(
                repos=self.config.github_repos,
                cache_dir=self.config.cache_dir,
            )

        if self.config.arxiv_enabled:
            self._generators[TaskSource.ARXIV_ABSTRACT] = ArxivTaskGenerator(
                categories=self.config.arxiv_categories,
                cache_dir=self.config.cache_dir,
                seed=self.seed,
            )

        if self.config.tool_use_enabled:
            self._generators[TaskSource.TOOL_USE_SCENARIO] = ToolUseTaskGenerator(
                seed=self.seed
            )

    def generate(
        self,
        n_tasks: int,
        sources: list[TaskSource] | None = None,
        difficulty_distribution: dict[DifficultyLevel, float] | None = None,
    ) -> list[Task]:
        """
        Generate a batch of tasks.

        Args:
            n_tasks: Total number of tasks to generate
            sources: Specific sources to use (or all enabled)
            difficulty_distribution: Target distribution of difficulties

        Returns:
            List of validated, deduplicated tasks
        """
        if sources is None:
            sources = list(self._generators.keys())

        # Calculate tasks per source
        active_generators = {
            s: g for s, g in self._generators.items() if s in sources
        }

        if not active_generators:
            raise ValueError("No generators available for specified sources")

        tasks_per_source = n_tasks // len(active_generators) + 1
        all_tasks: list[Task] = []

        for source, generator in active_generators.items():
            source_tasks = list(generator.generate(
                tasks_per_source,
                difficulty_weights=difficulty_distribution,
            ))

            # Validate and filter
            valid_tasks = []
            for task in source_tasks:
                validation = self.validator.validate(task)
                if validation["valid"] and task.content_hash not in self._seen_hashes:
                    self._seen_hashes.add(task.content_hash)
                    valid_tasks.append(task)

            all_tasks.extend(valid_tasks)

        # Shuffle and trim to requested count
        self.rng.shuffle(all_tasks)
        final_tasks = all_tasks[:n_tasks]

        # Add to pool
        self._task_pool.extend(final_tasks)

        return final_tasks

    def generate_balanced(
        self,
        n_tasks: int,
        balance_by: str = "difficulty",
    ) -> list[Task]:
        """
        Generate tasks with balanced distribution.

        Args:
            n_tasks: Total tasks to generate
            balance_by: "difficulty", "source", or "skill"

        Returns:
            Balanced task set
        """
        if balance_by == "difficulty":
            distribution = {
                DifficultyLevel.EASY: 0.2,
                DifficultyLevel.MEDIUM: 0.35,
                DifficultyLevel.HARD: 0.30,
                DifficultyLevel.VERY_HARD: 0.15,
            }
            return self.generate(n_tasks, difficulty_distribution=distribution)

        elif balance_by == "source":
            sources = list(self._generators.keys())
            tasks_per_source = n_tasks // len(sources) + 1
            all_tasks = []

            for source in sources:
                source_tasks = self.generate(tasks_per_source, sources=[source])
                all_tasks.extend(source_tasks)

            self.rng.shuffle(all_tasks)
            return all_tasks[:n_tasks]

        else:
            # Default generation
            return self.generate(n_tasks)

    def get_task_pool_stats(self) -> dict:
        """Get statistics about the current task pool."""
        if not self._task_pool:
            return {"total": 0}

        by_source = {}
        by_difficulty = {}
        by_skill = {}

        for task in self._task_pool:
            # By source
            source = task.source.value
            by_source[source] = by_source.get(source, 0) + 1

            # By difficulty
            diff = task.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

            # By skill
            for skill in task.required_skills:
                skill_name = skill.value
                by_skill[skill_name] = by_skill.get(skill_name, 0) + 1

        return {
            "total": len(self._task_pool),
            "by_source": by_source,
            "by_difficulty": by_difficulty,
            "by_skill": by_skill,
            "unique_hashes": len(self._seen_hashes),
        }

    def sample_tasks(
        self,
        n: int,
        source: TaskSource | None = None,
        difficulty: DifficultyLevel | None = None,
    ) -> list[Task]:
        """
        Sample tasks from the pool with optional filters.
        """
        pool = self._task_pool

        if source:
            pool = [t for t in pool if t.source == source]

        if difficulty:
            pool = [t for t in pool if t.difficulty == difficulty]

        if len(pool) < n:
            # Generate more if needed
            additional = self.generate(n - len(pool), sources=[source] if source else None)
            pool = pool + additional

        return self.rng.sample(pool, min(n, len(pool)))

    def save_tasks(
        self,
        tasks: list[Task],
        filename: str | None = None,
    ) -> Path:
        """
        Save tasks to JSONL file.

        Args:
            tasks: Tasks to save
            filename: Optional filename (default: timestamped)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tasks_{timestamp}.jsonl"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for task in tasks:
                f.write(task.model_dump_json() + "\n")

        return filepath

    def load_tasks(self, filepath: str | Path) -> list[Task]:
        """
        Load tasks from JSONL file.
        """
        filepath = Path(filepath)
        tasks = []

        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    tasks.append(Task.model_validate(data))

        return tasks

    def create_evaluation_set(
        self,
        n_tasks: int = 100,
        include_robustness: bool = True,
        save: bool = True,
    ) -> dict:
        """
        Create a complete evaluation set with metadata.

        Returns:
            Dictionary containing tasks and metadata
        """
        # Generate main tasks
        tasks = self.generate_balanced(n_tasks)

        # Create evaluation set metadata
        eval_set = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
            "config": self.config.model_dump(),
            "stats": self.get_task_pool_stats(),
            "task_ids": [str(t.id) for t in tasks],
        }

        if save:
            # Save tasks
            task_path = self.save_tasks(tasks)
            eval_set["task_file"] = str(task_path)

            # Save metadata
            meta_path = self.output_dir / f"eval_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(meta_path, "w") as f:
                json.dump(eval_set, f, indent=2)
            eval_set["metadata_file"] = str(meta_path)

        eval_set["tasks"] = tasks
        return eval_set

    def get_daily_tasks(self, date: datetime | None = None) -> list[Task]:
        """
        Get deterministic task set for a specific date.

        This ensures:
        - Same tasks for same date (reproducibility)
        - Different tasks each day (freshness)
        """
        if date is None:
            date = datetime.now()

        # Create date-specific seed
        date_str = date.strftime("%Y%m%d")
        date_seed = int(hashlib.sha256(date_str.encode()).hexdigest()[:8], 16)
        combined_seed = self.seed ^ date_seed

        # Create temporary generator with date-specific seed
        temp_engine = TaskGenerationEngine(
            config=self.config,
            seed=combined_seed,
            output_dir=str(self.output_dir),
        )

        return temp_engine.generate(100)  # Daily batch
