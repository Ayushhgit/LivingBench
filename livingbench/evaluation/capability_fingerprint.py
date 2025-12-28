"""
Capability Fingerprint Computation.

A capability fingerprint is a multi-dimensional representation
of a model's abilities. Unlike single-number benchmarks, it captures:

1. Skill-specific performance
2. Difficulty scaling behavior
3. Robustness metrics
4. Calibration quality
5. Tool use proficiency

This enables nuanced model comparison and failure mode detection.
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
    RobustnessResult,
)
from livingbench.evaluation.skill_decomposition import SkillDecomposer


class FingerprintComputer:
    """
    Compute comprehensive capability fingerprints from evaluation results.

    The fingerprint captures:
    - Per-skill accuracy (weighted by difficulty)
    - Difficulty scaling (how performance degrades with difficulty)
    - Robustness metrics (paraphrase, counterfactual, adversarial)
    - Calibration (ECE, overconfidence rate)
    - Tool use metrics (if applicable)
    """

    def __init__(
        self,
        min_samples_per_skill: int = 5,
        calibration_bins: int = 10,
    ):
        self.skill_decomposer = SkillDecomposer(min_samples_per_skill)
        self.calibration_bins = calibration_bins

    def compute(
        self,
        model_id: str,
        results: list[EvaluationResult],
        robustness_results: list[RobustnessResult] | None = None,
    ) -> CapabilityFingerprint:
        """
        Compute complete capability fingerprint.

        Args:
            model_id: Identifier for the evaluated model
            results: List of evaluation results
            robustness_results: Optional robustness test results

        Returns:
            CapabilityFingerprint with all metrics
        """
        # Skill scores
        skill_scores = self.skill_decomposer.decompose(results)
        # Replace None with 0 for fingerprint
        skill_scores = {k: v if v is not None else 0.0 for k, v in skill_scores.items()}

        # Difficulty breakdown
        difficulty_scores = self._compute_difficulty_scores(results)

        # Robustness metrics
        robustness = self._compute_robustness_metrics(robustness_results)

        # Calibration metrics
        calibration = self._compute_calibration(results)

        # Tool use metrics
        tool_metrics = self._compute_tool_use_metrics(results)

        # Count samples per skill
        n_per_skill: dict[str, int] = defaultdict(int)
        for result in results:
            for skill in result.task.required_skills:
                n_per_skill[skill.value] += 1

        return CapabilityFingerprint(
            model_id=model_id,
            skill_scores=skill_scores,
            difficulty_scores=difficulty_scores,
            paraphrase_consistency=robustness.get("paraphrase_consistency", 0.0),
            counterfactual_sensitivity=robustness.get("counterfactual_sensitivity", 0.0),
            adversarial_robustness=robustness.get("adversarial_robustness", 0.0),
            calibration_error=calibration.get("ece", 0.0),
            overconfidence_rate=calibration.get("overconfidence_rate", 0.0),
            abstention_rate=calibration.get("abstention_rate", 0.0),
            tool_selection_accuracy=tool_metrics.get("selection_accuracy"),
            tool_execution_success_rate=tool_metrics.get("execution_success"),
            unnecessary_tool_use_rate=tool_metrics.get("unnecessary_use_rate"),
            n_tasks_evaluated=len(results),
            n_tasks_per_skill=dict(n_per_skill),
        )

    def _compute_difficulty_scores(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """Compute accuracy breakdown by difficulty level."""
        by_difficulty: dict[DifficultyLevel, list[bool]] = defaultdict(list)

        for result in results:
            by_difficulty[result.task.difficulty].append(result.is_correct)

        scores = {}
        for diff, corrects in by_difficulty.items():
            if corrects:
                scores[diff.value] = sum(corrects) / len(corrects)

        return scores

    def _compute_robustness_metrics(
        self,
        robustness_results: list[RobustnessResult] | None,
    ) -> dict[str, float]:
        """Compute robustness metrics from robustness test results."""
        if not robustness_results:
            return {}

        # Aggregate across all robustness tests
        paraphrase_consistencies = []
        counterfactual_flips = []
        adversarial_successes = []

        for rr in robustness_results:
            paraphrase_consistencies.append(rr.paraphrase_consistency)
            counterfactual_flips.append(rr.counterfactual_flip_rate)
            adversarial_successes.append(rr.adversarial_success_rate)

        return {
            "paraphrase_consistency": np.mean(paraphrase_consistencies) if paraphrase_consistencies else 0.0,
            "counterfactual_sensitivity": np.mean(counterfactual_flips) if counterfactual_flips else 0.0,
            "adversarial_robustness": 1.0 - np.mean(adversarial_successes) if adversarial_successes else 0.0,
        }

    def _compute_calibration(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """
        Compute calibration metrics.

        ECE (Expected Calibration Error): Measures how well predicted
        confidence matches actual accuracy.

        Overconfidence: Rate of high-confidence wrong answers.

        Abstention: Rate of responses that declined to answer.
        """
        # Get confidence and correctness pairs
        confidence_correct_pairs = []

        for result in results:
            # Use mean score as confidence proxy
            confidence = result.mean_score
            correct = result.is_correct
            confidence_correct_pairs.append((confidence, correct))

        if not confidence_correct_pairs:
            return {"ece": 0.0, "overconfidence_rate": 0.0, "abstention_rate": 0.0}

        # Compute ECE
        ece = self._expected_calibration_error(confidence_correct_pairs)

        # Compute overconfidence rate
        # High confidence (>0.8) but wrong
        overconfident = sum(
            1 for conf, correct in confidence_correct_pairs
            if conf > 0.8 and not correct
        )
        overconfidence_rate = overconfident / len(confidence_correct_pairs)

        # Abstention rate (if we can detect abstention)
        # For now, use low confidence as proxy
        abstained = sum(
            1 for conf, _ in confidence_correct_pairs
            if conf < 0.3
        )
        abstention_rate = abstained / len(confidence_correct_pairs)

        return {
            "ece": ece,
            "overconfidence_rate": overconfidence_rate,
            "abstention_rate": abstention_rate,
        }

    def _expected_calibration_error(
        self,
        confidence_correct_pairs: list[tuple[float, bool]],
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum over bins of (|accuracy - confidence| * bin_weight)
        """
        n = len(confidence_correct_pairs)
        if n == 0:
            return 0.0

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_indices = np.digitize(
            [conf for conf, _ in confidence_correct_pairs],
            bin_boundaries[1:-1]
        )

        ece = 0.0
        for bin_idx in range(self.calibration_bins):
            bin_mask = bin_indices == bin_idx
            bin_pairs = [
                (conf, correct)
                for (conf, correct), in_bin in zip(confidence_correct_pairs, bin_mask)
                if in_bin
            ]

            if bin_pairs:
                bin_accuracy = sum(correct for _, correct in bin_pairs) / len(bin_pairs)
                bin_confidence = sum(conf for conf, _ in bin_pairs) / len(bin_pairs)
                bin_weight = len(bin_pairs) / n

                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def _compute_tool_use_metrics(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float | None]:
        """Compute tool use specific metrics."""
        # Filter to tool use tasks
        tool_results = [
            r for r in results
            if r.task.available_tools
        ]

        if not tool_results:
            return {
                "selection_accuracy": None,
                "execution_success": None,
                "unnecessary_use_rate": None,
            }

        # Selection accuracy: did model use the right tools?
        correct_selections = 0
        total_with_expected = 0

        for result in tool_results:
            if result.task.expected_tool_calls:
                total_with_expected += 1
                expected_tools = {tc["name"] for tc in result.task.expected_tool_calls}
                actual_tools = {tc.tool_name for tc in result.response.tool_calls}

                if expected_tools == actual_tools:
                    correct_selections += 1

        selection_accuracy = (
            correct_selections / total_with_expected
            if total_with_expected > 0 else None
        )

        # Execution success: did tool calls work?
        total_calls = 0
        successful_calls = 0

        for result in tool_results:
            for call in result.response.tool_calls:
                total_calls += 1
                if call.error is None:
                    successful_calls += 1

        execution_success = (
            successful_calls / total_calls
            if total_calls > 0 else None
        )

        # Unnecessary tool use: used tools when not needed
        no_tool_needed = [
            r for r in results
            if not r.task.expected_tool_calls  # Matches both [] and None
        ]
        unnecessary = sum(
            1 for r in no_tool_needed
            if r.response.tool_calls
        )
        unnecessary_rate = (
            unnecessary / len(no_tool_needed)
            if no_tool_needed else None
        )

        return {
            "selection_accuracy": selection_accuracy,
            "execution_success": execution_success,
            "unnecessary_use_rate": unnecessary_rate,
        }

    def compare_fingerprints(
        self,
        fp1: CapabilityFingerprint,
        fp2: CapabilityFingerprint,
    ) -> dict[str, Any]:
        """
        Compare two capability fingerprints.

        Returns detailed comparison including:
        - Per-skill deltas
        - Overall similarity
        - Significant differences
        """
        # Per-skill comparison
        all_skills = set(fp1.skill_scores.keys()) | set(fp2.skill_scores.keys())
        skill_deltas = {}

        for skill in all_skills:
            s1 = fp1.skill_scores.get(skill, 0)
            s2 = fp2.skill_scores.get(skill, 0)
            skill_deltas[skill] = {
                "model1": s1,
                "model2": s2,
                "delta": s1 - s2,
            }

        # Overall similarity (cosine of skill vectors)
        v1 = fp1.to_vector(sorted(all_skills))
        v2 = fp2.to_vector(sorted(all_skills))

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        cosine_sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        # Find significant differences (>10% absolute)
        significant_diffs = [
            skill for skill, data in skill_deltas.items()
            if abs(data["delta"]) > 0.1
        ]

        return {
            "skill_deltas": skill_deltas,
            "cosine_similarity": cosine_sim,
            "euclidean_distance": fp1.distance_to(fp2),
            "significant_differences": significant_diffs,
            "model1_strengths": [
                skill for skill, data in skill_deltas.items()
                if data["delta"] > 0.1
            ],
            "model2_strengths": [
                skill for skill, data in skill_deltas.items()
                if data["delta"] < -0.1
            ],
        }
