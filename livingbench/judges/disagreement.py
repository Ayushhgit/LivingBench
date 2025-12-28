"""
Judge Disagreement Analysis.

Analyzes patterns of disagreement among judges:
1. Which judges tend to disagree?
2. What types of tasks cause disagreement?
3. Is disagreement a signal of task ambiguity?

Disagreement is not always bad—it can reveal:
- Ambiguous tasks that need refinement
- Judges with different evaluation criteria
- Edge cases worth investigating
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np
from itertools import combinations

from livingbench.core.types import JudgeVerdict, EvaluationResult, Task, Skill


class DisagreementAnalyzer:
    """
    Analyze disagreement patterns among judges.

    Key analyses:
    1. Pairwise agreement rates
    2. Task characteristics that predict disagreement
    3. Judge clustering by agreement patterns
    """

    def analyze_disagreement(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """
        Comprehensive disagreement analysis.

        Args:
            results: Evaluation results with multiple judge verdicts

        Returns:
            Disagreement analysis
        """
        if not results:
            return {"status": "no_results"}

        # Get all unique judges
        all_judges = set()
        for result in results:
            for verdict in result.verdicts:
                all_judges.add(verdict.judge_id)

        all_judges = list(all_judges)

        if len(all_judges) < 2:
            return {"status": "insufficient_judges"}

        return {
            "pairwise_agreement": self._pairwise_agreement(results, all_judges),
            "by_difficulty": self._disagreement_by_difficulty(results),
            "by_skill": self._disagreement_by_skill(results),
            "high_disagreement_tasks": self._find_high_disagreement_tasks(results),
            "judge_reliability": self._estimate_judge_reliability(results, all_judges),
            "disagreement_patterns": self._analyze_patterns(results),
        }

    def _pairwise_agreement(
        self,
        results: list[EvaluationResult],
        judges: list[str],
    ) -> dict[str, float]:
        """Compute pairwise agreement between all judge pairs."""
        agreements = {}

        for j1, j2 in combinations(judges, 2):
            matching = 0
            total = 0

            for result in results:
                # Find verdicts from both judges
                v1 = next((v for v in result.verdicts if v.judge_id == j1), None)
                v2 = next((v for v in result.verdicts if v.judge_id == j2), None)

                if v1 and v2:
                    total += 1
                    if v1.is_correct == v2.is_correct:
                        matching += 1

            if total > 0:
                agreements[f"{j1}_vs_{j2}"] = matching / total

        return agreements

    def _disagreement_by_difficulty(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """Compute disagreement rate by difficulty level."""
        by_difficulty = defaultdict(list)

        for result in results:
            difficulty = result.task.difficulty.value
            by_difficulty[difficulty].append(result.agreement_ratio)

        return {
            diff: 1 - np.mean(agreements)
            for diff, agreements in by_difficulty.items()
        }

    def _disagreement_by_skill(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        """Compute disagreement rate by skill."""
        by_skill = defaultdict(list)

        for result in results:
            for skill in result.task.required_skills:
                by_skill[skill.value].append(result.agreement_ratio)

        return {
            skill: 1 - np.mean(agreements)
            for skill, agreements in by_skill.items()
            if len(agreements) >= 5  # Minimum samples
        }

    def _find_high_disagreement_tasks(
        self,
        results: list[EvaluationResult],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find tasks with high judge disagreement."""
        high_disagreement = []

        for result in results:
            if result.agreement_ratio < threshold:
                # Collect judge breakdown
                verdicts_summary = [
                    {
                        "judge": v.judge_id,
                        "correct": v.is_correct,
                        "score": v.score,
                    }
                    for v in result.verdicts
                ]

                high_disagreement.append({
                    "task_id": str(result.task.id),
                    "prompt_preview": result.task.prompt[:200],
                    "agreement_ratio": result.agreement_ratio,
                    "difficulty": result.task.difficulty.value,
                    "skills": [s.value for s in result.task.required_skills],
                    "verdicts": verdicts_summary,
                })

        # Sort by agreement ratio (most disagreement first)
        return sorted(high_disagreement, key=lambda x: x["agreement_ratio"])[:20]

    def _estimate_judge_reliability(
        self,
        results: list[EvaluationResult],
        judges: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Estimate reliability of each judge.

        Uses agreement with majority as proxy for reliability.
        """
        reliability = {}

        for judge in judges:
            agreements_with_majority = []
            confidence_when_agreeing = []
            confidence_when_disagreeing = []

            for result in results:
                verdict = next(
                    (v for v in result.verdicts if v.judge_id == judge),
                    None
                )
                if verdict:
                    agrees = verdict.is_correct == result.is_correct
                    agreements_with_majority.append(agrees)

                    if agrees:
                        confidence_when_agreeing.append(verdict.confidence)
                    else:
                        confidence_when_disagreeing.append(verdict.confidence)

            if agreements_with_majority:
                reliability[judge] = {
                    "agreement_with_majority": np.mean(agreements_with_majority),
                    "mean_confidence_agreeing": np.mean(confidence_when_agreeing)
                        if confidence_when_agreeing else 0.0,
                    "mean_confidence_disagreeing": np.mean(confidence_when_disagreeing)
                        if confidence_when_disagreeing else 0.0,
                    "n_judgments": len(agreements_with_majority),
                }

        return reliability

    def _analyze_patterns(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Analyze common patterns in disagreement."""
        patterns = {
            "ambiguous_reference": 0,
            "format_mismatch": 0,
            "partial_correctness": 0,
            "confidence_disagreement": 0,
        }

        for result in results:
            if result.agreement_ratio < 0.8:
                # Check for patterns
                scores = [v.score for v in result.verdicts]
                confidences = [v.confidence for v in result.verdicts]

                # Partial correctness: high variance in scores
                if np.std(scores) > 0.2:
                    patterns["partial_correctness"] += 1

                # Confidence disagreement: judges disagree but are confident
                if np.mean(confidences) > 0.7:
                    patterns["confidence_disagreement"] += 1

                # Check detected issues
                all_issues = []
                for v in result.verdicts:
                    all_issues.extend(v.detected_issues)

                if "format_mismatch" in all_issues or "parse_error" in all_issues:
                    patterns["format_mismatch"] += 1

        n_disagreements = sum(1 for r in results if r.agreement_ratio < 0.8)
        if n_disagreements > 0:
            patterns = {k: v / n_disagreements for k, v in patterns.items()}

        return patterns


class DisagreementResolver:
    """
    Methods to resolve judge disagreement.

    Strategies:
    1. Weighted voting by reliability
    2. Appeal to additional judge
    3. Task reformulation
    """

    def __init__(self, reliability_scores: dict[str, float] | None = None):
        self.reliability_scores = reliability_scores or {}

    def resolve_by_reliability(
        self,
        verdicts: list[JudgeVerdict],
    ) -> dict[str, Any]:
        """
        Resolve disagreement using judge reliability weights.
        """
        total_weight = 0.0
        weighted_correct = 0.0

        for verdict in verdicts:
            weight = self.reliability_scores.get(verdict.judge_id, 1.0)
            total_weight += weight
            if verdict.is_correct:
                weighted_correct += weight

        if total_weight == 0:
            return {"resolved": False, "reason": "no_weights"}

        weighted_correctness = weighted_correct / total_weight
        is_correct = weighted_correctness > 0.5

        return {
            "resolved": True,
            "is_correct": is_correct,
            "confidence": abs(weighted_correctness - 0.5) * 2,
            "method": "reliability_weighted",
        }

    def identify_tiebreaker_needs(
        self,
        results: list[EvaluationResult],
    ) -> list[dict[str, Any]]:
        """
        Identify cases that need tiebreaker resolution.

        Returns tasks where judges are evenly split.
        """
        needs_tiebreaker = []

        for result in results:
            n_correct = sum(1 for v in result.verdicts if v.is_correct)
            n_total = len(result.verdicts)

            # Evenly split
            if n_correct == n_total // 2:
                needs_tiebreaker.append({
                    "task_id": str(result.task.id),
                    "current_split": f"{n_correct}/{n_total}",
                    "max_confidence_correct": max(
                        (v.confidence for v in result.verdicts if v.is_correct),
                        default=0
                    ),
                    "max_confidence_incorrect": max(
                        (v.confidence for v in result.verdicts if not v.is_correct),
                        default=0
                    ),
                })

        return needs_tiebreaker
