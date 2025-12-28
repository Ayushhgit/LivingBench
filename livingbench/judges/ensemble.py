"""
Ensemble Judge System.

Combines multiple judges to provide robust evaluation:
1. Multiple LLM judges (different models)
2. Rule-based judges for specific aspects
3. Aggregation with disagreement detection

Key insight: No single judge is reliable. Agreement among
diverse judges provides better signal than any single judge.
"""

from __future__ import annotations

from typing import Any, Callable
from datetime import datetime
import numpy as np

from livingbench.core.types import Task, ModelResponse, JudgeVerdict
from livingbench.judges.base import JudgeBase, ExactMatchJudge, RubricJudge


class EnsembleJudge:
    """
    Ensemble of multiple judges with aggregation.

    Features:
    - Majority voting for binary correctness
    - Weighted average for scores
    - Disagreement detection
    - Confidence-based weighting
    """

    def __init__(
        self,
        judges: list[JudgeBase] | None = None,
        aggregation: str = "majority",
        min_agreement: float = 0.5,
    ):
        """
        Args:
            judges: List of judges to use
            aggregation: "majority", "weighted", or "unanimous"
            min_agreement: Minimum agreement ratio for valid verdict
        """
        self.judges = judges or self._default_judges()
        self.aggregation = aggregation
        self.min_agreement = min_agreement

    def _default_judges(self) -> list[JudgeBase]:
        """Create default set of judges."""
        return [
            ExactMatchJudge(),
            RubricJudge(),
        ]

    def add_judge(self, judge: JudgeBase) -> None:
        """Add a judge to the ensemble."""
        self.judges.append(judge)

    def judge(self, task: Task, response: ModelResponse) -> list[JudgeVerdict]:
        """
        Get verdicts from all judges.

        Returns:
            List of verdicts from all judges
        """
        verdicts = []
        for judge in self.judges:
            try:
                verdict = judge.judge(task, response)
                verdicts.append(verdict)
            except Exception as e:
                # Create error verdict
                verdicts.append(JudgeVerdict(
                    judge_id=judge.judge_id,
                    is_correct=False,
                    score=0.0,
                    rationale=f"Judge error: {str(e)}",
                    confidence=0.0,
                    detected_issues=["judge_error"],
                ))

        return verdicts

    def aggregate(self, verdicts: list[JudgeVerdict]) -> dict[str, Any]:
        """
        Aggregate verdicts from multiple judges.

        Returns:
            Aggregated result with correctness, score, agreement metrics
        """
        if not verdicts:
            return {
                "is_correct": False,
                "score": 0.0,
                "agreement_ratio": 0.0,
                "aggregation_method": self.aggregation,
            }

        if self.aggregation == "majority":
            return self._majority_vote(verdicts)
        elif self.aggregation == "weighted":
            return self._weighted_aggregate(verdicts)
        elif self.aggregation == "unanimous":
            return self._unanimous_vote(verdicts)
        else:
            return self._majority_vote(verdicts)

    def _majority_vote(self, verdicts: list[JudgeVerdict]) -> dict[str, Any]:
        """Aggregate by majority vote."""
        n_correct = sum(1 for v in verdicts if v.is_correct)
        n_total = len(verdicts)

        is_correct = n_correct > n_total / 2
        agreement_ratio = max(n_correct, n_total - n_correct) / n_total

        # Average score
        scores = [v.score for v in verdicts]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        return {
            "is_correct": is_correct,
            "score": float(mean_score),
            "score_std": float(std_score),
            "agreement_ratio": agreement_ratio,
            "n_judges": n_total,
            "n_correct": n_correct,
            "aggregation_method": "majority",
        }

    def _weighted_aggregate(self, verdicts: list[JudgeVerdict]) -> dict[str, Any]:
        """Aggregate with confidence weighting."""
        total_weight = sum(v.confidence for v in verdicts)

        if total_weight == 0:
            return self._majority_vote(verdicts)

        # Weighted correctness
        weighted_correct = sum(
            v.confidence * (1 if v.is_correct else 0)
            for v in verdicts
        ) / total_weight

        is_correct = weighted_correct > 0.5

        # Weighted score
        weighted_score = sum(
            v.confidence * v.score for v in verdicts
        ) / total_weight

        return {
            "is_correct": is_correct,
            "score": float(weighted_score),
            "weighted_correctness": float(weighted_correct),
            "total_confidence": float(total_weight),
            "aggregation_method": "weighted",
        }

    def _unanimous_vote(self, verdicts: list[JudgeVerdict]) -> dict[str, Any]:
        """Require unanimous agreement."""
        all_correct = all(v.is_correct for v in verdicts)
        all_incorrect = all(not v.is_correct for v in verdicts)

        if all_correct:
            is_correct = True
            agreement_ratio = 1.0
        elif all_incorrect:
            is_correct = False
            agreement_ratio = 1.0
        else:
            # No unanimous decision
            n_correct = sum(1 for v in verdicts if v.is_correct)
            is_correct = n_correct > len(verdicts) / 2
            agreement_ratio = max(n_correct, len(verdicts) - n_correct) / len(verdicts)

        scores = [v.score for v in verdicts]

        return {
            "is_correct": is_correct,
            "score": float(np.mean(scores)),
            "unanimous": all_correct or all_incorrect,
            "agreement_ratio": agreement_ratio,
            "aggregation_method": "unanimous",
        }

    def analyze_disagreement(self, verdicts: list[JudgeVerdict]) -> dict[str, Any]:
        """
        Analyze disagreement patterns among judges.

        Returns insights about what judges disagree on.
        """
        if len(verdicts) < 2:
            return {"status": "insufficient_judges"}

        # Binary correctness disagreement
        correctness_values = [v.is_correct for v in verdicts]
        unique_values = set(correctness_values)
        has_disagreement = len(unique_values) > 1

        # Score variance
        scores = [v.score for v in verdicts]
        score_variance = np.var(scores)

        # Identify outlier judges
        mean_score = np.mean(scores)
        outliers = [
            v.judge_id for v in verdicts
            if abs(v.score - mean_score) > 2 * np.std(scores)
        ] if np.std(scores) > 0 else []

        # Collect all detected issues
        all_issues = []
        for v in verdicts:
            all_issues.extend(v.detected_issues)
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Consensus issues (detected by majority)
        consensus_issues = [
            issue for issue, count in issue_counts.items()
            if count > len(verdicts) / 2
        ]

        # Calculate agreement ratio: proportion of majority vote
        n_correct = sum(correctness_values)
        n_total = len(verdicts)
        correctness_agreement = max(n_correct, n_total - n_correct) / n_total

        return {
            "has_disagreement": has_disagreement,
            "correctness_agreement": correctness_agreement,
            "score_variance": float(score_variance),
            "outlier_judges": outliers,
            "all_issues": issue_counts,
            "consensus_issues": consensus_issues,
            "judge_breakdown": [
                {
                    "judge_id": v.judge_id,
                    "is_correct": v.is_correct,
                    "score": v.score,
                    "confidence": v.confidence,
                }
                for v in verdicts
            ],
        }


class LLMJudge(JudgeBase):
    """
    LLM-based judge for semantic evaluation.

    Uses an LLM to evaluate responses based on meaning rather
    than exact matching. Suitable for complex, open-ended tasks.
    """

    def __init__(
        self,
        model_id: str,
        inference_fn: Callable[[str], str] | None = None,
        system_prompt: str | None = None,
    ):
        """
        Args:
            model_id: Identifier for the judge model
            inference_fn: Function to call the LLM
            system_prompt: Custom system prompt for judging
        """
        self.model_id = model_id
        self.inference_fn = inference_fn
        self.system_prompt = system_prompt or self._default_system_prompt()

    @property
    def judge_id(self) -> str:
        return f"llm_{self.model_id}"

    def _default_system_prompt(self) -> str:
        return """You are an expert evaluator for AI model responses. Your task is to evaluate whether a response correctly answers the given question or task.

Evaluation criteria:
1. Correctness: Does the response contain the correct answer/solution?
2. Reasoning: Is the reasoning valid and complete?
3. Format: Does the response follow any specified format requirements?

You must respond in the following JSON format:
{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "rationale": "Brief explanation of your judgment",
    "issues": ["list", "of", "detected", "issues"]
}

Be strict but fair. Partial credit is acceptable for partially correct answers."""

    def judge(self, task: Task, response: ModelResponse) -> JudgeVerdict:
        """Use LLM to judge the response."""
        if not self.inference_fn:
            # Return placeholder if no inference function
            return JudgeVerdict(
                judge_id=self.judge_id,
                judge_model=self.model_id,
                is_correct=False,
                score=0.0,
                rationale="No inference function configured",
                confidence=0.0,
                detected_issues=["no_inference_fn"],
            )

        # Construct prompt for judge
        judge_prompt = self._construct_prompt(task, response)

        try:
            # Get judge response
            judge_response = self.inference_fn(judge_prompt)

            # Parse response
            verdict_data = self._parse_verdict(judge_response)

            return JudgeVerdict(
                judge_id=self.judge_id,
                judge_model=self.model_id,
                is_correct=verdict_data["is_correct"],
                score=verdict_data["score"],
                rationale=verdict_data["rationale"],
                confidence=verdict_data.get("confidence", 0.8),
                detected_issues=verdict_data.get("issues", []),
            )

        except Exception as e:
            return JudgeVerdict(
                judge_id=self.judge_id,
                judge_model=self.model_id,
                is_correct=False,
                score=0.0,
                rationale=f"Judgment failed: {str(e)}",
                confidence=0.0,
                detected_issues=["judgment_error"],
            )

    def _construct_prompt(self, task: Task, response: ModelResponse) -> str:
        """Construct the prompt for the judge."""
        prompt_parts = [
            self.system_prompt,
            "",
            "## Task",
            task.prompt,
        ]

        if task.reference_answer:
            prompt_parts.extend([
                "",
                "## Reference Answer",
                task.reference_answer,
            ])

        prompt_parts.extend([
            "",
            "## Model Response",
            response.raw_output,
            "",
            "## Your Evaluation (respond in JSON format)",
        ])

        return "\n".join(prompt_parts)

    def _parse_verdict(self, response: str) -> dict[str, Any]:
        """Parse verdict from judge response."""
        import json

        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = response[response.find("{"):response.rfind("}") + 1]
            data = json.loads(json_match)
            return {
                "is_correct": bool(data.get("is_correct", False)),
                "score": float(data.get("score", 0.0)),
                "rationale": str(data.get("rationale", "")),
                "issues": list(data.get("issues", [])),
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract key information
            is_correct = "correct" in response.lower() and "incorrect" not in response.lower()
            return {
                "is_correct": is_correct,
                "score": 0.7 if is_correct else 0.3,
                "rationale": response[:500],
                "issues": ["parse_error"],
            }
