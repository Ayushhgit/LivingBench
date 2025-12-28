"""Tests for evaluation pipeline and metrics."""

import pytest
from uuid import uuid4

from livingbench.core.types import (
    Task, TaskSource, TaskFormat, Skill, DifficultyLevel,
    ModelResponse, JudgeVerdict, EvaluationResult,
)
from livingbench.evaluation.metrics import (
    compute_accuracy,
    compute_accuracy_by_difficulty,
    compute_skill_scores,
    compute_calibration_error,
    compute_judge_agreement,
)
from livingbench.evaluation.skill_decomposition import SkillDecomposer
from livingbench.evaluation.capability_fingerprint import FingerprintComputer


def create_mock_result(
    is_correct: bool,
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    skills: list[Skill] | None = None,
    score: float = 0.8,
) -> EvaluationResult:
    """Create a mock evaluation result for testing."""
    skills = skills or [Skill.LOGICAL_DEDUCTION]

    task = Task(
        id=uuid4(),
        source=TaskSource.SYNTHETIC_REASONING,
        prompt="Test prompt",
        reference_answer="Test answer",
        required_skills=skills,
        difficulty=difficulty,
    )

    response = ModelResponse(
        task_id=task.id,
        model_id="test_model",
        raw_output="Test response",
    )

    verdict = JudgeVerdict(
        judge_id="test_judge",
        is_correct=is_correct,
        score=score if is_correct else score * 0.3,
        rationale="Test rationale",
        confidence=0.9 if is_correct else 0.5,
    )

    return EvaluationResult(
        task=task,
        response=response,
        verdicts=[verdict],
        is_correct=is_correct,
        agreement_ratio=1.0,
        mean_score=score if is_correct else score * 0.3,
        score_std=0.0,
    )


class TestAccuracyMetrics:
    """Tests for accuracy computation."""

    def test_compute_accuracy_all_correct(self):
        """Test accuracy when all results are correct."""
        results = [create_mock_result(is_correct=True) for _ in range(10)]
        accuracy = compute_accuracy(results)
        assert accuracy == 1.0

    def test_compute_accuracy_all_incorrect(self):
        """Test accuracy when all results are incorrect."""
        results = [create_mock_result(is_correct=False) for _ in range(10)]
        accuracy = compute_accuracy(results)
        assert accuracy == 0.0

    def test_compute_accuracy_mixed(self):
        """Test accuracy with mixed results."""
        results = [
            create_mock_result(is_correct=True),
            create_mock_result(is_correct=True),
            create_mock_result(is_correct=False),
            create_mock_result(is_correct=True),
            create_mock_result(is_correct=False),
        ]
        accuracy = compute_accuracy(results)
        assert accuracy == 0.6

    def test_compute_accuracy_weighted(self):
        """Test weighted accuracy computation."""
        results = [
            create_mock_result(is_correct=True, difficulty=DifficultyLevel.EASY),
            create_mock_result(is_correct=True, difficulty=DifficultyLevel.HARD),
            create_mock_result(is_correct=False, difficulty=DifficultyLevel.HARD),
        ]
        weighted = compute_accuracy(results, weighted=True)
        unweighted = compute_accuracy(results, weighted=False)

        # Weighted should differ from unweighted
        assert weighted != unweighted

    def test_compute_accuracy_empty(self):
        """Test accuracy with empty results."""
        accuracy = compute_accuracy([])
        assert accuracy == 0.0


class TestAccuracyByDifficulty:
    """Tests for difficulty-stratified accuracy."""

    def test_breakdown_by_difficulty(self):
        """Test accuracy breakdown by difficulty level."""
        results = [
            create_mock_result(is_correct=True, difficulty=DifficultyLevel.EASY),
            create_mock_result(is_correct=True, difficulty=DifficultyLevel.EASY),
            create_mock_result(is_correct=False, difficulty=DifficultyLevel.HARD),
            create_mock_result(is_correct=True, difficulty=DifficultyLevel.HARD),
        ]

        breakdown = compute_accuracy_by_difficulty(results)

        assert "easy" in breakdown
        assert "hard" in breakdown
        assert breakdown["easy"]["accuracy"] == 1.0
        assert breakdown["hard"]["accuracy"] == 0.5


class TestSkillScores:
    """Tests for skill-specific scoring."""

    def test_compute_skill_scores(self):
        """Test per-skill accuracy computation."""
        results = [
            create_mock_result(
                is_correct=True,
                skills=[Skill.LOGICAL_DEDUCTION]
            ),
            create_mock_result(
                is_correct=False,
                skills=[Skill.LOGICAL_DEDUCTION]
            ),
            create_mock_result(
                is_correct=True,
                skills=[Skill.MATHEMATICAL_REASONING]
            ),
        ]

        scores = compute_skill_scores(results)

        assert "logical_deduction" in scores
        assert "mathematical_reasoning" in scores
        assert scores["logical_deduction"] == 0.5
        assert scores["mathematical_reasoning"] == 1.0


class TestSkillDecomposer:
    """Tests for skill decomposition."""

    def test_decompose_basic(self):
        """Test basic skill decomposition."""
        decomposer = SkillDecomposer(min_samples_per_skill=2)

        results = [
            create_mock_result(is_correct=True, skills=[Skill.LOGICAL_DEDUCTION]),
            create_mock_result(is_correct=True, skills=[Skill.LOGICAL_DEDUCTION]),
            create_mock_result(is_correct=False, skills=[Skill.LOGICAL_DEDUCTION]),
        ]

        scores = decomposer.decompose(results)

        assert "logical_deduction" in scores
        # Should have weighted accuracy close to 0.67 (2/3)

    def test_insufficient_samples(self):
        """Test behavior with insufficient samples."""
        decomposer = SkillDecomposer(min_samples_per_skill=10)

        results = [
            create_mock_result(is_correct=True, skills=[Skill.LOGICAL_DEDUCTION]),
        ]

        scores = decomposer.decompose(results)

        # Should return None for insufficient samples
        assert scores.get("logical_deduction") is None


class TestCalibration:
    """Tests for calibration metrics."""

    def test_perfect_calibration(self):
        """Test ECE for perfectly calibrated results."""
        # Create results where confidence matches accuracy
        results = [
            create_mock_result(is_correct=True, score=0.9),
            create_mock_result(is_correct=True, score=0.9),
            create_mock_result(is_correct=False, score=0.1),
        ]

        calibration = compute_calibration_error(results)

        # ECE should be low for calibrated results
        assert calibration["ece"] < 0.3

    def test_overconfident(self):
        """Test detection of overconfidence."""
        # Create results where model is always confident but sometimes wrong
        results = []
        for i in range(10):
            # High confidence but only 50% correct
            results.append(create_mock_result(
                is_correct=(i % 2 == 0),
                score=0.9,  # Always high confidence
            ))

        calibration = compute_calibration_error(results)

        # Overconfidence should be detected
        assert calibration["overconfidence_rate"] > 0


class TestFingerprintComputer:
    """Tests for capability fingerprint computation."""

    def test_compute_fingerprint(self):
        """Test fingerprint computation."""
        computer = FingerprintComputer()

        results = [
            create_mock_result(
                is_correct=True,
                difficulty=DifficultyLevel.EASY,
                skills=[Skill.LOGICAL_DEDUCTION]
            ),
            create_mock_result(
                is_correct=True,
                difficulty=DifficultyLevel.MEDIUM,
                skills=[Skill.MATHEMATICAL_REASONING]
            ),
            create_mock_result(
                is_correct=False,
                difficulty=DifficultyLevel.HARD,
                skills=[Skill.LOGICAL_DEDUCTION]
            ),
        ] * 5  # Repeat for minimum samples

        fingerprint = computer.compute("test_model", results)

        assert fingerprint.model_id == "test_model"
        assert fingerprint.n_tasks_evaluated == 15
        assert "logical_deduction" in fingerprint.skill_scores
        assert "easy" in fingerprint.difficulty_scores

    def test_compare_fingerprints(self):
        """Test fingerprint comparison."""
        computer = FingerprintComputer()

        results1 = [create_mock_result(is_correct=True)] * 10
        results2 = [create_mock_result(is_correct=False)] * 10

        fp1 = computer.compute("model1", results1)
        fp2 = computer.compute("model2", results2)

        comparison = computer.compare_fingerprints(fp1, fp2)

        assert "cosine_similarity" in comparison
        assert "euclidean_distance" in comparison
        assert fp1.distance_to(fp2) > 0
