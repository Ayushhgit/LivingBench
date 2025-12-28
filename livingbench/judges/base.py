"""
Base Judge Interface.

Defines the abstract interface for judges in LivingBench.
Judges evaluate model responses and produce structured verdicts.

Key principles:
1. Judges must provide rationales (no black-box scores)
2. Judges report confidence for calibration analysis
3. Judges detect specific failure modes
4. Judges can be rule-based or LLM-based
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from datetime import datetime
from typing import Any

from livingbench.core.types import Task, ModelResponse, JudgeVerdict, TaskFormat


class JudgeBase(ABC):
    """
    Abstract base class for all judges.

    Subclasses must implement the judge() method to evaluate
    a model response against a task.
    """

    @property
    @abstractmethod
    def judge_id(self) -> str:
        """Unique identifier for this judge."""
        ...

    @abstractmethod
    def judge(self, task: Task, response: ModelResponse) -> JudgeVerdict:
        """
        Evaluate a model response.

        Args:
            task: The original task
            response: The model's response

        Returns:
            JudgeVerdict with correctness, score, rationale, and detected issues
        """
        ...

    def detect_issues(self, task: Task, response: ModelResponse) -> list[str]:
        """
        Detect specific issues in the response.

        Override in subclasses for specialized detection.
        """
        issues = []

        # Check for empty response
        if not response.raw_output.strip():
            issues.append("empty_response")

        # Check for refusal
        refusal_patterns = [
            r"I cannot",
            r"I'm not able to",
            r"I don't have access",
            r"I apologize, but",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, response.raw_output, re.IGNORECASE):
                issues.append("refusal_detected")
                break

        # Check for excessive length (potential verbosity gaming)
        if len(response.raw_output) > 5000:
            issues.append("excessive_length")

        # Check for hallucination indicators
        hallucination_patterns = [
            r"As an AI",
            r"I don't have real-time",
            r"my training data",
            r"I was last updated",
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, response.raw_output, re.IGNORECASE):
                issues.append("meta_response")
                break

        return issues


class ExactMatchJudge(JudgeBase):
    """
    Simple exact match judge for tasks with deterministic answers.

    Suitable for:
    - Math problems with numeric answers
    - Multiple choice
    - Yes/No questions
    """

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    @property
    def judge_id(self) -> str:
        return "exact_match"

    def judge(self, task: Task, response: ModelResponse) -> JudgeVerdict:
        """Judge by exact match with optional normalization."""
        if task.reference_answer is None:
            return JudgeVerdict(
                judge_id=self.judge_id,
                is_correct=False,
                score=0.0,
                rationale="No reference answer provided",
                confidence=0.0,
                detected_issues=["no_reference_answer"],
            )

        reference = task.reference_answer
        predicted = response.parsed_answer or response.raw_output

        # Normalize if requested
        if self.normalize:
            reference = self._normalize(reference)
            predicted = self._normalize(predicted)

        # Case handling
        if not self.case_sensitive:
            reference = reference.lower()
            predicted = predicted.lower()

        # Check for match
        is_correct = reference in predicted or predicted == reference

        # For numeric answers, try numeric comparison
        if not is_correct and task.format == TaskFormat.NUMERIC:
            is_correct = self._numeric_match(reference, predicted)

        # Detect issues
        issues = self.detect_issues(task, response)

        return JudgeVerdict(
            judge_id=self.judge_id,
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            rationale=f"Exact match: {'match' if is_correct else 'no match'}",
            confidence=1.0,  # Rule-based, always confident
            detected_issues=issues,
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common filler
        text = re.sub(r'^(the answer is|answer:|result:)\s*', '', text, flags=re.IGNORECASE)
        # Remove punctuation at end
        text = text.rstrip('.,;:')
        return text.strip()

    def _numeric_match(self, reference: str, predicted: str) -> bool:
        """Try to match numeric values."""
        try:
            # Extract numbers
            ref_nums = re.findall(r'-?\d+\.?\d*', reference)
            pred_nums = re.findall(r'-?\d+\.?\d*', predicted)

            if ref_nums and pred_nums:
                ref_val = float(ref_nums[0])
                pred_val = float(pred_nums[0])

                # Allow small tolerance for floating point
                return abs(ref_val - pred_val) < 0.01
        except ValueError:
            pass

        return False


class ContainsJudge(JudgeBase):
    """
    Judge that checks if response contains required elements.

    Suitable for:
    - Free-form responses that should mention key concepts
    - Partial credit scoring
    """

    def __init__(self, required_elements: list[str] | None = None):
        self.required_elements = required_elements or []

    @property
    def judge_id(self) -> str:
        return "contains_check"

    def judge(self, task: Task, response: ModelResponse) -> JudgeVerdict:
        """Judge by checking for required elements."""
        text = response.raw_output.lower()

        if self.required_elements:
            elements = self.required_elements
        elif task.reference_answer:
            # Extract key terms from reference
            elements = self._extract_key_terms(task.reference_answer)
        else:
            return JudgeVerdict(
                judge_id=self.judge_id,
                is_correct=False,
                score=0.0,
                rationale="No reference or required elements",
                confidence=0.0,
            )

        # Count matches
        matches = sum(1 for elem in elements if elem.lower() in text)
        score = matches / len(elements) if elements else 0.0
        is_correct = score >= 0.8  # 80% threshold

        return JudgeVerdict(
            judge_id=self.judge_id,
            is_correct=is_correct,
            score=score,
            rationale=f"Found {matches}/{len(elements)} required elements",
            confidence=0.9,
            detected_issues=self.detect_issues(task, response),
        )

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from reference answer."""
        # Simple approach: content words
        words = re.findall(r'\b\w{4,}\b', text.lower())
        # Filter common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'would'}
        return [w for w in words if w not in stopwords][:5]


class RubricJudge(JudgeBase):
    """
    Judge responses using a scoring rubric.

    Provides multi-dimensional scoring based on predefined criteria.
    """

    def __init__(
        self,
        rubric: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Args:
            rubric: Scoring rubric with criteria and weights
                   Format: {"criterion": {"weight": float, "description": str}}
        """
        self.rubric = rubric or self._default_rubric()

    @property
    def judge_id(self) -> str:
        return "rubric"

    def _default_rubric(self) -> dict[str, dict[str, Any]]:
        """Default rubric for general evaluation."""
        return {
            "correctness": {
                "weight": 0.4,
                "description": "Is the core answer/solution correct?",
            },
            "completeness": {
                "weight": 0.2,
                "description": "Does the response address all aspects of the question?",
            },
            "reasoning": {
                "weight": 0.2,
                "description": "Is the reasoning/explanation clear and valid?",
            },
            "conciseness": {
                "weight": 0.1,
                "description": "Is the response appropriately concise?",
            },
            "format": {
                "weight": 0.1,
                "description": "Does the response follow the requested format?",
            },
        }

    def judge(self, task: Task, response: ModelResponse) -> JudgeVerdict:
        """Judge using rubric scoring."""
        scores = {}

        # Score each criterion
        for criterion, config in self.rubric.items():
            scores[criterion] = self._score_criterion(
                criterion, task, response, config
            )

        # Compute weighted total
        total_score = sum(
            scores[c] * self.rubric[c]["weight"]
            for c in scores
        )

        # Determine correctness (threshold based)
        is_correct = total_score >= 0.6

        # Build rationale
        rationale_parts = [f"{c}: {scores[c]:.2f}" for c in scores]
        rationale = f"Rubric scores: {', '.join(rationale_parts)}"

        return JudgeVerdict(
            judge_id=self.judge_id,
            is_correct=is_correct,
            score=total_score,
            rationale=rationale,
            confidence=0.8,
            skill_scores=scores,
            detected_issues=self.detect_issues(task, response),
        )

    def _score_criterion(
        self,
        criterion: str,
        task: Task,
        response: ModelResponse,
        config: dict,
    ) -> float:
        """Score a single criterion."""
        text = response.raw_output

        if criterion == "correctness":
            # Check against reference if available
            if task.reference_answer:
                ref = task.reference_answer.lower()
                if ref in text.lower():
                    return 1.0
                # Partial match
                words = set(ref.split())
                resp_words = set(text.lower().split())
                overlap = len(words & resp_words) / len(words) if words else 0
                return min(1.0, overlap * 1.5)
            return 0.5  # Unknown

        elif criterion == "completeness":
            # Check for multiple aspects addressed
            sentence_count = len(re.findall(r'[.!?]', text))
            if sentence_count >= 3:
                return 1.0
            elif sentence_count >= 2:
                return 0.7
            else:
                return 0.4

        elif criterion == "reasoning":
            # Check for reasoning indicators
            reasoning_words = ['because', 'therefore', 'since', 'thus', 'hence', 'so']
            has_reasoning = any(w in text.lower() for w in reasoning_words)
            return 1.0 if has_reasoning else 0.3

        elif criterion == "conciseness":
            # Penalize excessive length
            word_count = len(text.split())
            if word_count < 50:
                return 1.0
            elif word_count < 200:
                return 0.8
            elif word_count < 500:
                return 0.5
            else:
                return 0.2

        elif criterion == "format":
            # Check if follows expected format
            if task.format == TaskFormat.STRUCTURED_JSON:
                try:
                    json.loads(text)
                    return 1.0
                except (json.JSONDecodeError, ValueError):
                    return 0.0
            return 0.8  # Default OK for free text

        return 0.5  # Default middle score
