"""Tests for task generation."""

import pytest
from livingbench.core.types import Task, TaskSource, Skill, DifficultyLevel
from livingbench.tasks.generator import TaskGenerationEngine
from livingbench.tasks.sources.synthetic import (
    SyntheticReasoningGenerator,
    SyntheticMathGenerator,
)
from livingbench.tasks.labeling import SkillLabeler, DifficultyEstimator


class TestSyntheticReasoningGenerator:
    """Tests for synthetic reasoning task generation."""

    def test_generate_tasks(self):
        """Test basic task generation."""
        gen = SyntheticReasoningGenerator(seed=42)
        tasks = list(gen.generate(10))

        assert len(tasks) == 10
        assert all(isinstance(t, Task) for t in tasks)
        assert all(t.source == TaskSource.SYNTHETIC_REASONING for t in tasks)

    def test_tasks_have_required_fields(self):
        """Test that tasks have all required fields."""
        gen = SyntheticReasoningGenerator(seed=42)
        tasks = list(gen.generate(5))

        for task in tasks:
            assert task.prompt
            assert len(task.prompt) > 10
            assert task.required_skills
            assert task.difficulty

    def test_reproducibility(self):
        """Test that same seed produces same tasks."""
        gen1 = SyntheticReasoningGenerator(seed=42)
        gen2 = SyntheticReasoningGenerator(seed=42)

        tasks1 = list(gen1.generate(5))
        tasks2 = list(gen2.generate(5))

        for t1, t2 in zip(tasks1, tasks2):
            assert t1.prompt == t2.prompt
            assert t1.reference_answer == t2.reference_answer

    def test_different_seeds_different_tasks(self):
        """Test that different seeds produce different tasks."""
        gen1 = SyntheticReasoningGenerator(seed=42)
        gen2 = SyntheticReasoningGenerator(seed=123)

        tasks1 = list(gen1.generate(5))
        tasks2 = list(gen2.generate(5))

        # At least some tasks should be different
        different = sum(
            1 for t1, t2 in zip(tasks1, tasks2)
            if t1.prompt != t2.prompt
        )
        assert different > 0


class TestSyntheticMathGenerator:
    """Tests for synthetic math task generation."""

    def test_generate_tasks(self):
        """Test basic task generation."""
        gen = SyntheticMathGenerator(seed=42)
        tasks = list(gen.generate(10))

        assert len(tasks) == 10
        assert all(t.source == TaskSource.SYNTHETIC_MATH for t in tasks)

    def test_math_tasks_have_numeric_answers(self):
        """Test that math tasks have numeric reference answers."""
        gen = SyntheticMathGenerator(seed=42)
        tasks = list(gen.generate(10))

        for task in tasks:
            if task.reference_answer:
                # Should be parseable as number
                try:
                    float(task.reference_answer)
                except ValueError:
                    pytest.fail(f"Non-numeric answer: {task.reference_answer}")


class TestTaskGenerationEngine:
    """Tests for the main task generation engine."""

    def test_generate_balanced(self):
        """Test balanced task generation."""
        engine = TaskGenerationEngine(seed=42)
        tasks = engine.generate_balanced(50)

        assert len(tasks) == 50

        # Check difficulty distribution
        difficulties = [t.difficulty for t in tasks]
        # Should have variety
        assert len(set(difficulties)) > 1

    def test_deduplication(self):
        """Test that duplicate tasks are removed."""
        engine = TaskGenerationEngine(seed=42)
        tasks = engine.generate(100)

        # All content hashes should be unique
        hashes = [t.content_hash for t in tasks]
        assert len(hashes) == len(set(hashes))

    def test_pool_stats(self):
        """Test task pool statistics."""
        engine = TaskGenerationEngine(seed=42)
        engine.generate(50)

        stats = engine.get_task_pool_stats()

        assert stats["total"] == 50
        assert "by_source" in stats
        assert "by_difficulty" in stats


class TestSkillLabeler:
    """Tests for automatic skill labeling."""

    def test_detect_math_skills(self):
        """Test detection of mathematical reasoning skills."""
        labeler = SkillLabeler()

        # Create mock task with math content
        from uuid import uuid4
        task = Task(
            id=uuid4(),
            source=TaskSource.SYNTHETIC_MATH,
            prompt="Calculate 5 * 3 + 2. Compute the result.",
            required_skills=[Skill.MATHEMATICAL_REASONING],
            difficulty=DifficultyLevel.EASY,
        )

        detected = labeler.label(task)
        assert Skill.MATHEMATICAL_REASONING in detected

    def test_detect_logic_skills(self):
        """Test detection of logical reasoning skills."""
        labeler = SkillLabeler()

        from uuid import uuid4
        task = Task(
            id=uuid4(),
            source=TaskSource.SYNTHETIC_REASONING,
            prompt="If all A are B, and all B are C, therefore all A are C. Is this valid?",
            required_skills=[Skill.LOGICAL_DEDUCTION],
            difficulty=DifficultyLevel.MEDIUM,
        )

        detected = labeler.label(task)
        assert Skill.LOGICAL_DEDUCTION in detected


class TestDifficultyEstimator:
    """Tests for difficulty estimation."""

    def test_estimate_easy_task(self):
        """Test that simple tasks are rated as easy."""
        estimator = DifficultyEstimator()

        from uuid import uuid4
        task = Task(
            id=uuid4(),
            source=TaskSource.SYNTHETIC_MATH,
            prompt="What is 2 + 2?",
            required_skills=[Skill.MATHEMATICAL_REASONING],
            difficulty=DifficultyLevel.EASY,
        )

        estimated = estimator.estimate(task)
        assert estimated in [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY]

    def test_estimate_hard_task(self):
        """Test that complex tasks are rated as harder."""
        estimator = DifficultyEstimator()

        from uuid import uuid4
        task = Task(
            id=uuid4(),
            source=TaskSource.SYNTHETIC_REASONING,
            prompt="""Consider the following complex multi-step algorithm:
            First, we apply the recursive transformation to the asymptotic bounds.
            Then, we compute the derivative of the integral over the manifold.
            Finally, we solve the resulting NP-hard optimization problem.
            What is the time complexity?""",
            required_skills=[
                Skill.MATHEMATICAL_REASONING,
                Skill.CODE_UNDERSTANDING,
                Skill.MULTI_STEP_PLANNING,
                Skill.LOGICAL_DEDUCTION,
            ],
            difficulty=DifficultyLevel.VERY_HARD,
        )

        estimated = estimator.estimate(task)
        assert estimated in [DifficultyLevel.HARD, DifficultyLevel.VERY_HARD]
