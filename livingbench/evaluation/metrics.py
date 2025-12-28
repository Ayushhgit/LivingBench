"""
Evaluation metrics for LivingBench.

Provides standardized metrics for:
- Accuracy computation
- Calibration analysis
- Skill-specific scoring
- Agreement metrics
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np

from livingbench.core.types import (
    EvaluationResult,
    JudgeVerdict,
    Skill,
    DifficultyLevel,
)


def compute_accuracy(
    results: list[EvaluationResult],
    weighted: bool = False,
) -> float:
    """
    Compute overall accuracy.

    Args:
        results: List of evaluation results
        weighted: Whether to weight by difficulty

    Returns:
        Accuracy score (0-1)
    """
    if not results:
        return 0.0

    if weighted:
        weights = {
            DifficultyLevel.TRIVIAL: 0.5,
            DifficultyLevel.EASY: 0.75,
            DifficultyLevel.MEDIUM: 1.0,
            DifficultyLevel.HARD: 1.5,
            DifficultyLevel.VERY_HARD: 2.0,
            DifficultyLevel.ADVERSARIAL: 2.5,
        }

        weighted_correct = sum(
            weights.get(r.task.difficulty, 1.0) * (1 if r.is_correct else 0)
            for r in results
        )
        total_weight = sum(
            weights.get(r.task.difficulty, 1.0) for r in results
        )
        return weighted_correct / total_weight if total_weight > 0 else 0.0
    else:
        correct = sum(1 for r in results if r.is_correct)
        return correct / len(results)


def compute_accuracy_by_difficulty(
    results: list[EvaluationResult],
) -> dict[str, dict[str, float]]:
    """
    Compute accuracy breakdown by difficulty level.

    Returns:
        Dictionary with per-difficulty accuracy and sample counts
    """
    by_difficulty: dict[DifficultyLevel, list[bool]] = defaultdict(list)

    for result in results:
        by_difficulty[result.task.difficulty].append(result.is_correct)

    breakdown = {}
    for diff, corrects in by_difficulty.items():
        breakdown[diff.value] = {
            "accuracy": sum(corrects) / len(corrects) if corrects else 0.0,
            "n_samples": len(corrects),
            "n_correct": sum(corrects),
        }

    return breakdown


def compute_skill_scores(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """
    Compute accuracy per skill.

    Returns:
        Dictionary mapping skill name to accuracy
    """
    by_skill: dict[Skill, list[bool]] = defaultdict(list)

    for result in results:
        for skill in result.task.required_skills:
            by_skill[skill].append(result.is_correct)

    return {
        skill.value: sum(corrects) / len(corrects) if corrects else 0.0
        for skill, corrects in by_skill.items()
    }


def compute_calibration_error(
    results: list[EvaluationResult],
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute calibration metrics.

    Returns:
        Dictionary with ECE, MCE, and overconfidence rate
    """
    if not results:
        return {"ece": 0.0, "mce": 0.0, "overconfidence_rate": 0.0}

    # Extract confidence and correctness
    pairs = [(r.mean_score, r.is_correct) for r in results]

    n = len(pairs)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize([conf for conf, _ in pairs], bin_boundaries[1:-1])

    ece = 0.0
    mce = 0.0

    for bin_idx in range(n_bins):
        bin_pairs = [
            (conf, correct) for (conf, correct), idx in zip(pairs, bin_indices)
            if idx == bin_idx
        ]

        if bin_pairs:
            bin_accuracy = sum(c for _, c in bin_pairs) / len(bin_pairs)
            bin_confidence = sum(c for c, _ in bin_pairs) / len(bin_pairs)
            bin_weight = len(bin_pairs) / n

            calibration_gap = abs(bin_accuracy - bin_confidence)
            ece += bin_weight * calibration_gap
            mce = max(mce, calibration_gap)

    # Overconfidence: high confidence, wrong answer
    overconfident = sum(
        1 for conf, correct in pairs
        if conf > 0.8 and not correct
    )
    overconfidence_rate = overconfident / n

    return {
        "ece": ece,
        "mce": mce,
        "overconfidence_rate": overconfidence_rate,
    }


def compute_judge_agreement(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """
    Compute agreement metrics across judges.

    Returns:
        Dictionary with agreement ratio, Fleiss' kappa, etc.
    """
    if not results:
        return {"agreement_ratio": 0.0, "fleiss_kappa": 0.0}

    agreement_ratios = [r.agreement_ratio for r in results]
    mean_agreement = np.mean(agreement_ratios)

    # Compute Fleiss' kappa
    # For binary classification (correct/incorrect)
    n_items = len(results)
    n_raters = len(results[0].verdicts) if results else 0

    if n_raters < 2:
        return {"agreement_ratio": mean_agreement, "fleiss_kappa": 0.0}

    # Count agreements
    p_e = 0.0  # Expected agreement by chance
    p_o = 0.0  # Observed agreement

    total_correct = 0
    total_incorrect = 0

    for result in results:
        n_correct = sum(1 for v in result.verdicts if v.is_correct)
        n_incorrect = len(result.verdicts) - n_correct

        total_correct += n_correct
        total_incorrect += n_incorrect

        # P_i for this item
        if n_raters > 1:
            p_i = (n_correct * (n_correct - 1) + n_incorrect * (n_incorrect - 1))
            p_i /= (n_raters * (n_raters - 1))
            p_o += p_i

    p_o /= n_items

    total_ratings = total_correct + total_incorrect
    if total_ratings > 0:
        p_correct = total_correct / total_ratings
        p_incorrect = total_incorrect / total_ratings
        p_e = p_correct ** 2 + p_incorrect ** 2

    # Fleiss' kappa
    if p_e < 1:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0

    return {
        "agreement_ratio": mean_agreement,
        "fleiss_kappa": kappa,
        "n_judges": n_raters,
    }


def compute_consistency_metrics(
    results: list[EvaluationResult],
    robustness_results: list[Any] | None = None,
) -> dict[str, float]:
    """
    Compute consistency metrics.

    Measures how consistent model performance is across:
    - Similar tasks
    - Paraphrases
    - Different evaluation runs
    """
    metrics = {
        "score_std": 0.0,
        "answer_consistency": 0.0,
    }

    if not results:
        return metrics

    # Score standard deviation
    scores = [r.mean_score for r in results]
    metrics["score_std"] = float(np.std(scores))

    # If robustness results available, compute paraphrase consistency
    if robustness_results:
        consistencies = []
        for rr in robustness_results:
            if hasattr(rr, "paraphrase_consistency"):
                consistencies.append(rr.paraphrase_consistency)

        if consistencies:
            metrics["paraphrase_consistency"] = float(np.mean(consistencies))

    return metrics


def compute_summary_statistics(
    results: list[EvaluationResult],
) -> dict[str, Any]:
    """
    Compute comprehensive summary statistics.

    Returns a complete statistical summary of evaluation results.
    """
    if not results:
        return {"status": "no_results"}

    # Basic accuracy
    accuracy = compute_accuracy(results)
    weighted_accuracy = compute_accuracy(results, weighted=True)

    # By difficulty
    by_difficulty = compute_accuracy_by_difficulty(results)

    # By skill
    by_skill = compute_skill_scores(results)

    # Calibration
    calibration = compute_calibration_error(results)

    # Judge agreement
    agreement = compute_judge_agreement(results)

    # Score distribution
    scores = [r.mean_score for r in results]

    return {
        "n_tasks": len(results),
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "by_difficulty": by_difficulty,
        "by_skill": by_skill,
        "calibration": calibration,
        "judge_agreement": agreement,
        "score_distribution": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
        },
    }
