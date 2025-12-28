"""
Skill-Factorized Evaluation.

Instead of a single accuracy score, we decompose performance into
a multi-dimensional capability profile. This enables:

1. Fine-grained model comparison
2. Weakness identification
3. Targeted improvement tracking
4. Fair comparison across different task distributions

The key insight is that aggregate accuracy hides important
failure patterns. A model might excel at math but fail at
causal reasoning—this system makes such patterns visible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np

from livingbench.core.types import (
    Task,
    Skill,
    DifficultyLevel,
    EvaluationResult,
    CapabilityFingerprint,
)


class SkillDecomposer:
    """
    Decompose evaluation results into skill-specific scores.

    For each skill, we compute:
    - Accuracy on tasks requiring that skill
    - Weighted accuracy (by difficulty)
    - Confidence calibration per skill
    - Correlation with other skills (skill clustering)
    """

    def __init__(self, min_samples_per_skill: int = 5):
        """
        Args:
            min_samples_per_skill: Minimum tasks per skill for reliable scores
        """
        self.min_samples = min_samples_per_skill

    def decompose(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """
        Compute skill-specific scores from evaluation results.

        Returns:
            Dictionary mapping skill name to score (0-1)
        """
        # Group results by skill
        skill_results: dict[Skill, list[EvaluationResult]] = defaultdict(list)

        for result in results:
            for skill in result.task.required_skills:
                skill_results[skill].append(result)

        # Compute per-skill accuracy
        skill_scores = {}
        for skill, skill_res in skill_results.items():
            if len(skill_res) >= self.min_samples:
                # Simple accuracy
                correct = sum(1 for r in skill_res if r.is_correct)
                accuracy = correct / len(skill_res)

                # Difficulty-weighted accuracy
                weighted_correct = 0.0
                total_weight = 0.0
                for r in skill_res:
                    weight = self._difficulty_weight(r.task.difficulty)
                    weighted_correct += weight * (1.0 if r.is_correct else 0.0)
                    total_weight += weight

                weighted_accuracy = weighted_correct / total_weight if total_weight > 0 else 0.0

                # Use weighted accuracy as the skill score
                skill_scores[skill.value] = weighted_accuracy
            else:
                # Insufficient samples - mark as uncertain
                skill_scores[skill.value] = None

        return skill_scores

    def decompose_with_confidence(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, dict[str, Any]]:
        """
        Compute skill scores with confidence intervals.

        Returns detailed breakdown per skill including:
        - score: point estimate
        - ci_lower, ci_upper: 95% confidence interval
        - n_samples: number of tasks
        - by_difficulty: breakdown by difficulty level
        """
        skill_results: dict[Skill, list[EvaluationResult]] = defaultdict(list)

        for result in results:
            for skill in result.task.required_skills:
                skill_results[skill].append(result)

        detailed_scores = {}
        for skill, skill_res in skill_results.items():
            n = len(skill_res)

            if n < self.min_samples:
                detailed_scores[skill.value] = {
                    "score": None,
                    "confidence": "insufficient_data",
                    "n_samples": n,
                }
                continue

            # Compute accuracy
            correct = sum(1 for r in skill_res if r.is_correct)
            accuracy = correct / n

            # Wilson score interval for 95% CI
            z = 1.96
            denominator = 1 + z**2 / n
            center = (accuracy + z**2 / (2*n)) / denominator
            spread = z * np.sqrt((accuracy * (1-accuracy) + z**2/(4*n)) / n) / denominator

            ci_lower = max(0, center - spread)
            ci_upper = min(1, center + spread)

            # Breakdown by difficulty
            by_difficulty = {}
            for diff in DifficultyLevel:
                diff_results = [r for r in skill_res if r.task.difficulty == diff]
                if diff_results:
                    diff_acc = sum(1 for r in diff_results if r.is_correct) / len(diff_results)
                    by_difficulty[diff.value] = {
                        "accuracy": diff_acc,
                        "n": len(diff_results),
                    }

            detailed_scores[skill.value] = {
                "score": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_samples": n,
                "by_difficulty": by_difficulty,
            }

        return detailed_scores

    def compute_skill_correlations(
        self,
        results: list[EvaluationResult],
    ) -> dict[tuple[str, str], float]:
        """
        Compute correlations between skill performances.

        This reveals which skills tend to co-occur in successes/failures,
        potentially indicating shared underlying capabilities.
        """
        # Build skill x task correctness matrix
        all_skills = set()
        task_skills_correct: dict[str, dict[str, bool]] = {}

        for result in results:
            task_id = str(result.task.id)
            task_skills_correct[task_id] = {}
            for skill in result.task.required_skills:
                all_skills.add(skill.value)
                task_skills_correct[task_id][skill.value] = result.is_correct

        skills_list = sorted(all_skills)
        n_skills = len(skills_list)

        # Compute pairwise correlations
        correlations = {}
        for i, skill_a in enumerate(skills_list):
            for j, skill_b in enumerate(skills_list):
                if i >= j:
                    continue

                # Get tasks with both skills
                pairs = []
                for task_id, skill_correct in task_skills_correct.items():
                    if skill_a in skill_correct and skill_b in skill_correct:
                        pairs.append((
                            1 if skill_correct[skill_a] else 0,
                            1 if skill_correct[skill_b] else 0,
                        ))

                if len(pairs) >= self.min_samples:
                    arr = np.array(pairs)
                    if arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
                        corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
                        correlations[(skill_a, skill_b)] = corr
                    else:
                        correlations[(skill_a, skill_b)] = 0.0

        return correlations

    def identify_weakness_clusters(
        self,
        results: list[EvaluationResult],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Identify clusters of related weaknesses.

        A weakness cluster is a set of skills that:
        1. Have below-threshold accuracy
        2. Are correlated (failures tend to co-occur)

        This helps identify root capability gaps rather than
        surface-level task failures.
        """
        skill_scores = self.decompose(results)
        correlations = self.compute_skill_correlations(results)

        # Find weak skills
        weak_skills = {
            skill for skill, score in skill_scores.items()
            if score is not None and score < threshold
        }

        if not weak_skills:
            return []

        # Find correlated clusters among weak skills
        # Simple greedy clustering
        clusters = []
        remaining = set(weak_skills)

        while remaining:
            # Start new cluster with worst skill
            seed = min(remaining, key=lambda s: skill_scores.get(s, 0) or 0)
            cluster = {seed}
            remaining.remove(seed)

            # Add correlated weak skills
            for skill in list(remaining):
                for cluster_skill in cluster:
                    pair = tuple(sorted([skill, cluster_skill]))
                    corr = correlations.get(pair, 0)
                    if corr > 0.3:  # Correlation threshold
                        cluster.add(skill)
                        remaining.discard(skill)
                        break

            clusters.append({
                "skills": list(cluster),
                "avg_score": np.mean([skill_scores[s] or 0 for s in cluster]),
                "n_skills": len(cluster),
            })

        return sorted(clusters, key=lambda c: c["avg_score"])

    @staticmethod
    def _difficulty_weight(difficulty: DifficultyLevel) -> float:
        """Weight for difficulty level in weighted accuracy."""
        weights = {
            DifficultyLevel.TRIVIAL: 0.5,
            DifficultyLevel.EASY: 0.75,
            DifficultyLevel.MEDIUM: 1.0,
            DifficultyLevel.HARD: 1.5,
            DifficultyLevel.VERY_HARD: 2.0,
            DifficultyLevel.ADVERSARIAL: 2.5,
        }
        return weights.get(difficulty, 1.0)


class SkillProfile:
    """
    Rich skill profile with comparative analysis.

    Enables:
    - Comparison to baseline/reference models
    - Strength/weakness identification
    - Progress tracking over time
    """

    def __init__(
        self,
        model_id: str,
        skill_scores: dict[str, float | None],
        detailed_scores: dict[str, dict[str, Any]] | None = None,
    ):
        self.model_id = model_id
        self.skill_scores = skill_scores
        self.detailed_scores = detailed_scores or {}

    def get_strengths(self, threshold: float = 0.7) -> list[str]:
        """Get skills with above-threshold performance."""
        return [
            skill for skill, score in self.skill_scores.items()
            if score is not None and score >= threshold
        ]

    def get_weaknesses(self, threshold: float = 0.5) -> list[str]:
        """Get skills with below-threshold performance."""
        return [
            skill for skill, score in self.skill_scores.items()
            if score is not None and score < threshold
        ]

    def compare_to(self, other: SkillProfile) -> dict[str, dict[str, float]]:
        """
        Compare this profile to another.

        Returns per-skill comparison with delta and significance.
        """
        comparison = {}
        common_skills = set(self.skill_scores.keys()) & set(other.skill_scores.keys())

        for skill in common_skills:
            self_score = self.skill_scores.get(skill)
            other_score = other.skill_scores.get(skill)

            if self_score is not None and other_score is not None:
                delta = self_score - other_score
                comparison[skill] = {
                    "self": self_score,
                    "other": other_score,
                    "delta": delta,
                    "relative_change": delta / other_score if other_score > 0 else float('inf'),
                }

        return comparison

    def to_vector(self, skill_order: list[str] | None = None) -> np.ndarray:
        """Convert to numpy vector for ML analysis."""
        if skill_order is None:
            skill_order = sorted(self.skill_scores.keys())

        return np.array([
            self.skill_scores.get(s, 0) or 0 for s in skill_order
        ])

    def summarize(self) -> dict[str, Any]:
        """Get summary statistics."""
        valid_scores = [s for s in self.skill_scores.values() if s is not None]

        if not valid_scores:
            return {"status": "no_valid_scores"}

        return {
            "model_id": self.model_id,
            "n_skills_evaluated": len(valid_scores),
            "mean_score": np.mean(valid_scores),
            "std_score": np.std(valid_scores),
            "min_score": min(valid_scores),
            "max_score": max(valid_scores),
            "strengths": self.get_strengths(),
            "weaknesses": self.get_weaknesses(),
        }
