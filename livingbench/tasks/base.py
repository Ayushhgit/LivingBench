"""
Base interface for task generators.

Task generators are responsible for producing evaluation tasks from
various sources. Each generator:
- Fetches raw data from its source
- Transforms it into Task objects
- Labels with skills and difficulty
- Ensures quality and deduplication
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from livingbench.core.types import Task, TaskSource


class TaskGeneratorBase(ABC):
    """
    Abstract base class for task generators.

    Subclasses must implement:
    - generate(): Produce tasks from the source
    - source property: Identify the task source

    Generators can be sync or async depending on the source.
    """

    @property
    @abstractmethod
    def source(self) -> TaskSource:
        """The source type for tasks from this generator."""
        ...

    @abstractmethod
    def generate(self, n_tasks: int, **kwargs) -> Iterator[Task]:
        """
        Generate tasks from this source.

        Args:
            n_tasks: Maximum number of tasks to generate
            **kwargs: Source-specific parameters

        Yields:
            Task objects ready for evaluation
        """
        ...

    async def generate_async(self, n_tasks: int, **kwargs) -> AsyncIterator[Task]:
        """
        Async version of generate for I/O-bound sources.

        Default implementation wraps sync generate.
        Override for true async sources.
        """
        for task in self.generate(n_tasks, **kwargs):
            yield task

    def validate_task(self, task: Task) -> bool:
        """
        Validate a generated task meets quality criteria.

        Override to add source-specific validation.
        """
        # Basic validation
        if not task.prompt or len(task.prompt) < 10:
            return False

        if not task.required_skills:
            return False

        # For tasks with reference answers, ensure they exist
        if task.format != "free_text" and not task.reference_answer:
            return False

        return True

    def deduplicate(self, tasks: list[Task]) -> list[Task]:
        """Remove duplicate tasks based on content hash."""
        seen_hashes: set[str] = set()
        unique_tasks: list[Task] = []

        for task in tasks:
            if task.content_hash not in seen_hashes:
                seen_hashes.add(task.content_hash)
                unique_tasks.append(task)

        return unique_tasks
