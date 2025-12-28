"""
Gaming and Memorization Detectors.

Detect when models are:
1. Memorizing benchmark answers (vs. reasoning)
2. Using spurious correlations
3. Over-relying on surface patterns
4. Gaming evaluation metrics

These detectors help ensure benchmark validity and model honesty.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any
import numpy as np

from livingbench.core.types import (
    Task,
    EvaluationResult,
    RobustnessResult,
    Skill,
)


class MemorizationDetector:
    """
    Detect potential memorization of benchmark answers.

    Signals of memorization:
    1. High accuracy on exact matches, low on paraphrases
    2. Verbatim reproduction of known answers
    3. Failure on novel but similar tasks
    4. Suspiciously high confidence on specific tasks
    """

    def __init__(
        self,
        consistency_threshold: float = 0.5,
        verbatim_threshold: float = 0.95,
    ):
        self.consistency_threshold = consistency_threshold
        self.verbatim_threshold = verbatim_threshold

    def detect(
        self,
        original_results: list[EvaluationResult],
        paraphrase_results: list[list[EvaluationResult]],
    ) -> dict[str, Any]:
        """
        Detect memorization patterns.

        Args:
            original_results: Results on original tasks
            paraphrase_results: Results on paraphrased versions

        Returns:
            Detection results with evidence
        """
        if not original_results or not paraphrase_results:
            return {"detected": False, "reason": "insufficient_data"}

        memorization_signals = []
        task_analyses = []

        for i, (orig_result, para_results) in enumerate(zip(original_results, paraphrase_results)):
            analysis = self._analyze_task(orig_result, para_results)
            task_analyses.append(analysis)

            if analysis["memorization_suspected"]:
                memorization_signals.append({
                    "task_id": str(orig_result.task.id),
                    "evidence": analysis["evidence"],
                })

        # Overall detection
        memorization_rate = len(memorization_signals) / len(original_results)
        detected = memorization_rate > 0.3  # More than 30% suspicious

        return {
            "detected": detected,
            "memorization_rate": memorization_rate,
            "n_suspicious_tasks": len(memorization_signals),
            "suspicious_tasks": memorization_signals[:10],  # Top 10
            "overall_consistency": np.mean([
                a["paraphrase_consistency"] for a in task_analyses
            ]),
        }

    def _analyze_task(
        self,
        original: EvaluationResult,
        paraphrases: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Analyze a single task for memorization signals."""
        if not paraphrases:
            return {
                "memorization_suspected": False,
                "paraphrase_consistency": 1.0,
                "evidence": [],
            }

        # Check consistency
        orig_correct = original.is_correct
        para_correct = [p.is_correct for p in paraphrases]
        consistency = sum(p == orig_correct for p in para_correct) / len(para_correct)

        evidence = []

        # Signal 1: Correct on original, wrong on paraphrases
        if orig_correct and consistency < self.consistency_threshold:
            evidence.append("high_original_low_paraphrase_accuracy")

        # Signal 2: High confidence on original, low on paraphrases
        orig_confidence = original.mean_score
        para_confidences = [p.mean_score for p in paraphrases]
        avg_para_confidence = np.mean(para_confidences)

        if orig_confidence > 0.9 and avg_para_confidence < 0.6:
            evidence.append("confidence_drop_on_paraphrases")

        # Signal 3: Check for verbatim answer patterns
        if self._check_verbatim_pattern(original):
            evidence.append("verbatim_answer_pattern")

        return {
            "memorization_suspected": len(evidence) > 0,
            "paraphrase_consistency": consistency,
            "evidence": evidence,
            "original_correct": orig_correct,
            "paraphrase_accuracy": sum(para_correct) / len(para_correct),
        }

    def _check_verbatim_pattern(self, result: EvaluationResult) -> bool:
        """Check if response looks like verbatim reproduction."""
        response = result.response.raw_output

        # Check for unusual patterns suggesting memorization
        patterns = [
            r"^The answer is exactly:",
            r"As stated in the training data",
            r"According to my knowledge of this benchmark",
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        return False


class SpuriousCorrelationDetector:
    """
    Detect reliance on spurious correlations.

    Models might learn shortcuts like:
    - Answer position bias (always choose B)
    - Length correlation (longer answers are correct)
    - Keyword association (certain words indicate certain answers)
    """

    def detect(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """
        Detect spurious correlation patterns.

        Returns:
            Detection results with identified patterns
        """
        patterns = []

        # Check for answer position bias (for multiple choice)
        position_bias = self._check_position_bias(results)
        if position_bias["detected"]:
            patterns.append(position_bias)

        # Check for length correlation
        length_correlation = self._check_length_correlation(results)
        if length_correlation["detected"]:
            patterns.append(length_correlation)

        # Check for keyword shortcuts
        keyword_shortcuts = self._check_keyword_shortcuts(results)
        if keyword_shortcuts["detected"]:
            patterns.append(keyword_shortcuts)

        return {
            "detected": len(patterns) > 0,
            "n_patterns": len(patterns),
            "patterns": patterns,
        }

    def _check_position_bias(self, results: list[EvaluationResult]) -> dict:
        """Check for multiple choice position bias."""
        mc_results = [
            r for r in results
            if r.task.format.value == "multiple_choice" and r.task.choices
        ]

        if len(mc_results) < 20:
            return {"detected": False, "reason": "insufficient_mc_tasks"}

        # Count predicted positions
        position_counts = defaultdict(int)
        for r in mc_results:
            answer = r.response.parsed_answer or r.response.raw_output
            # Try to extract position
            for i, choice in enumerate(r.task.choices):
                if choice.lower() in answer.lower() or f"({chr(65+i)})" in answer:
                    position_counts[i] += 1
                    break

        if not position_counts:
            return {"detected": False, "reason": "could_not_extract_positions"}

        total = sum(position_counts.values())
        n_choices = max(len(r.task.choices) for r in mc_results)
        expected_rate = 1.0 / n_choices

        # Check for significant deviation
        max_rate = max(position_counts.values()) / total
        if max_rate > expected_rate * 2:
            biased_position = max(position_counts.items(), key=lambda x: x[1])
            return {
                "detected": True,
                "type": "position_bias",
                "biased_position": biased_position[0],
                "bias_rate": max_rate,
                "expected_rate": expected_rate,
            }

        return {"detected": False}

    def _check_length_correlation(self, results: list[EvaluationResult]) -> dict:
        """Check if response length correlates with correctness."""
        lengths_correct = []
        lengths_incorrect = []

        for r in results:
            length = len(r.response.raw_output)
            if r.is_correct:
                lengths_correct.append(length)
            else:
                lengths_incorrect.append(length)

        if len(lengths_correct) < 10 or len(lengths_incorrect) < 10:
            return {"detected": False, "reason": "insufficient_samples"}

        mean_correct = np.mean(lengths_correct)
        mean_incorrect = np.mean(lengths_incorrect)

        # Check for significant difference
        ratio = mean_correct / mean_incorrect if mean_incorrect > 0 else 1.0

        if ratio > 1.5 or ratio < 0.67:
            return {
                "detected": True,
                "type": "length_correlation",
                "mean_length_correct": mean_correct,
                "mean_length_incorrect": mean_incorrect,
                "ratio": ratio,
            }

        return {"detected": False}

    def _check_keyword_shortcuts(self, results: list[EvaluationResult]) -> dict:
        """Check for keyword-based shortcuts."""
        # Collect words correlated with correctness
        word_correct = defaultdict(list)

        for r in results:
            words = set(re.findall(r'\b\w+\b', r.response.raw_output.lower()))
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_correct[word].append(r.is_correct)

        # Find words with high correlation to correctness
        shortcuts = []
        for word, corrects in word_correct.items():
            if len(corrects) >= 10:
                rate = sum(corrects) / len(corrects)
                if rate > 0.9 or rate < 0.1:
                    shortcuts.append({
                        "word": word,
                        "accuracy_when_present": rate,
                        "frequency": len(corrects),
                    })

        if shortcuts:
            return {
                "detected": True,
                "type": "keyword_shortcuts",
                "shortcuts": sorted(shortcuts, key=lambda x: abs(x["accuracy_when_present"] - 0.5), reverse=True)[:5],
            }

        return {"detected": False}


class GamingDetector:
    """
    Comprehensive gaming detection.

    Combines multiple signals to detect when a model is
    gaming the evaluation rather than genuinely solving tasks.
    """

    def __init__(self):
        self.memorization_detector = MemorizationDetector()
        self.spurious_detector = SpuriousCorrelationDetector()

    def detect(
        self,
        results: list[EvaluationResult],
        robustness_results: list[RobustnessResult] | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive gaming detection.

        Returns:
            Detection results with all signals
        """
        signals = {
            "memorization": None,
            "spurious_correlation": None,
            "verbosity_gaming": None,
            "confidence_gaming": None,
        }

        # Spurious correlation check
        signals["spurious_correlation"] = self.spurious_detector.detect(results)

        # Verbosity gaming: Is the model producing unnecessarily long responses?
        signals["verbosity_gaming"] = self._check_verbosity_gaming(results)

        # Confidence gaming: Is the model expressing false confidence?
        signals["confidence_gaming"] = self._check_confidence_gaming(results)

        # Memorization (if robustness results available)
        if robustness_results:
            # Extract paraphrase results
            paraphrase_results_grouped = self._group_paraphrase_results(robustness_results)
            signals["memorization"] = self.memorization_detector.detect(
                results, paraphrase_results_grouped
            )

        # Overall gaming score
        gaming_signals = sum(
            1 for v in signals.values()
            if v and v.get("detected", False)
        )

        return {
            "gaming_suspected": gaming_signals >= 2,
            "n_signals": gaming_signals,
            "signals": signals,
        }

    def _check_verbosity_gaming(self, results: list[EvaluationResult]) -> dict:
        """
        Check for verbosity gaming.

        Some models learn that longer responses get higher scores
        even when the extra content is filler.
        """
        # Compare response lengths for correct vs incorrect
        correct_lengths = [
            len(r.response.raw_output) for r in results if r.is_correct
        ]
        incorrect_lengths = [
            len(r.response.raw_output) for r in results if not r.is_correct
        ]

        if not correct_lengths or not incorrect_lengths:
            return {"detected": False}

        mean_correct = np.mean(correct_lengths)
        mean_incorrect = np.mean(incorrect_lengths)

        # Check for excessive length in correct answers
        if mean_correct > 1000 and mean_correct > mean_incorrect * 2:
            return {
                "detected": True,
                "type": "verbosity_gaming",
                "mean_correct_length": mean_correct,
                "mean_incorrect_length": mean_incorrect,
            }

        return {"detected": False}

    def _check_confidence_gaming(self, results: list[EvaluationResult]) -> dict:
        """
        Check for confidence gaming.

        Models might express high confidence regardless of actual certainty.
        """
        # Compute calibration
        confidences = [r.mean_score for r in results]
        correctness = [1 if r.is_correct else 0 for r in results]

        if len(confidences) < 20:
            return {"detected": False, "reason": "insufficient_samples"}

        # Check if confidence is always high
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        # Red flag: very high mean confidence with low variance
        if mean_confidence > 0.85 and std_confidence < 0.1:
            actual_accuracy = np.mean(correctness)
            if actual_accuracy < 0.7:  # High confidence but mediocre accuracy
                return {
                    "detected": True,
                    "type": "overconfidence_gaming",
                    "mean_confidence": mean_confidence,
                    "std_confidence": std_confidence,
                    "actual_accuracy": actual_accuracy,
                }

        return {"detected": False}

    def _group_paraphrase_results(
        self,
        robustness_results: list[RobustnessResult],
    ) -> list[list[EvaluationResult]]:
        """Group paraphrase results by original task."""
        grouped = []
        for rr in robustness_results:
            grouped.append(rr.paraphrase_results)
        return grouped
