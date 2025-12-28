"""
Judge Calibration Analysis.

Measures how well-calibrated judges are:
- Do confidence scores match actual accuracy?
- Are judges overconfident or underconfident?
- How does calibration vary by task type?

Well-calibrated judges are essential for reliable evaluation.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from collections import defaultdict

from livingbench.core.types import JudgeVerdict, EvaluationResult, Task


class CalibrationChecker:
    """
    Check calibration of judge confidence scores.

    A well-calibrated judge has:
    - Confidence = accuracy (on average)
    - Consistent calibration across task types
    - No systematic over/under-confidence
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def check_calibration(
        self,
        verdicts: list[JudgeVerdict],
        ground_truth: list[bool],
    ) -> dict[str, Any]:
        """
        Check calibration of a single judge.

        Args:
            verdicts: Judge verdicts
            ground_truth: Actual correctness labels

        Returns:
            Calibration metrics
        """
        if len(verdicts) != len(ground_truth):
            raise ValueError("Verdicts and ground truth must have same length")

        if not verdicts:
            return {"status": "no_data"}

        # Extract confidence and correctness
        confidences = [v.confidence for v in verdicts]
        predicted_correct = [v.is_correct for v in verdicts]

        # Compute ECE (Expected Calibration Error)
        ece = self._expected_calibration_error(confidences, ground_truth)

        # Compute MCE (Maximum Calibration Error)
        mce, mce_bin = self._max_calibration_error(confidences, ground_truth)

        # Compute reliability diagram data
        reliability = self._reliability_diagram_data(confidences, ground_truth)

        # Check for systematic bias
        mean_confidence = np.mean(confidences)
        actual_accuracy = np.mean(ground_truth)
        bias = mean_confidence - actual_accuracy

        # Overconfidence rate
        overconfident = sum(
            1 for conf, correct in zip(confidences, ground_truth)
            if conf > 0.8 and not correct
        )
        overconfidence_rate = overconfident / len(verdicts)

        return {
            "ece": float(ece),
            "mce": float(mce),
            "mce_bin": mce_bin,
            "mean_confidence": float(mean_confidence),
            "actual_accuracy": float(actual_accuracy),
            "bias": float(bias),
            "overconfidence_rate": float(overconfidence_rate),
            "reliability_diagram": reliability,
            "calibration_quality": self._classify_calibration(ece),
            "n_samples": len(verdicts),
        }

    def _expected_calibration_error(
        self,
        confidences: list[float],
        ground_truth: list[bool],
    ) -> float:
        """Compute Expected Calibration Error."""
        n = len(confidences)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

        ece = 0.0
        for bin_idx in range(self.n_bins):
            bin_mask = [i == bin_idx for i in bin_indices]
            bin_samples = [
                (conf, gt)
                for conf, gt, in_bin in zip(confidences, ground_truth, bin_mask)
                if in_bin
            ]

            if bin_samples:
                bin_accuracy = sum(gt for _, gt in bin_samples) / len(bin_samples)
                bin_confidence = sum(conf for conf, _ in bin_samples) / len(bin_samples)
                bin_weight = len(bin_samples) / n

                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def _max_calibration_error(
        self,
        confidences: list[float],
        ground_truth: list[bool],
    ) -> tuple[float, int]:
        """Compute Maximum Calibration Error and which bin."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

        max_error = 0.0
        max_bin = 0

        for bin_idx in range(self.n_bins):
            bin_mask = [i == bin_idx for i in bin_indices]
            bin_samples = [
                (conf, gt)
                for conf, gt, in_bin in zip(confidences, ground_truth, bin_mask)
                if in_bin
            ]

            if bin_samples:
                bin_accuracy = sum(gt for _, gt in bin_samples) / len(bin_samples)
                bin_confidence = sum(conf for conf, _ in bin_samples) / len(bin_samples)

                error = abs(bin_accuracy - bin_confidence)
                if error > max_error:
                    max_error = error
                    max_bin = bin_idx

        return max_error, max_bin

    def _reliability_diagram_data(
        self,
        confidences: list[float],
        ground_truth: list[bool],
    ) -> list[dict[str, float]]:
        """Generate data for reliability diagram."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

        diagram_data = []

        for bin_idx in range(self.n_bins):
            bin_mask = [i == bin_idx for i in bin_indices]
            bin_samples = [
                (conf, gt)
                for conf, gt, in_bin in zip(confidences, ground_truth, bin_mask)
                if in_bin
            ]

            if bin_samples:
                bin_accuracy = sum(gt for _, gt in bin_samples) / len(bin_samples)
                bin_confidence = sum(conf for conf, _ in bin_samples) / len(bin_samples)
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_boundaries[bin_idx] + bin_boundaries[bin_idx + 1]) / 2

            diagram_data.append({
                "bin": bin_idx,
                "lower": float(bin_boundaries[bin_idx]),
                "upper": float(bin_boundaries[bin_idx + 1]),
                "confidence": float(bin_confidence),
                "accuracy": float(bin_accuracy),
                "n_samples": len(bin_samples),
                "gap": float(bin_confidence - bin_accuracy),
            })

        return diagram_data

    def _classify_calibration(self, ece: float) -> str:
        """Classify calibration quality based on ECE."""
        if ece < 0.05:
            return "excellent"
        elif ece < 0.1:
            return "good"
        elif ece < 0.2:
            return "moderate"
        elif ece < 0.3:
            return "poor"
        else:
            return "very_poor"

    def compare_judges(
        self,
        judge_results: dict[str, list[tuple[JudgeVerdict, bool]]],
    ) -> dict[str, Any]:
        """
        Compare calibration across multiple judges.

        Args:
            judge_results: Dict mapping judge_id to list of (verdict, ground_truth)

        Returns:
            Comparative calibration analysis
        """
        judge_calibrations = {}

        for judge_id, results in judge_results.items():
            verdicts = [r[0] for r in results]
            ground_truth = [r[1] for r in results]
            judge_calibrations[judge_id] = self.check_calibration(verdicts, ground_truth)

        # Rank by ECE
        ranked = sorted(
            judge_calibrations.items(),
            key=lambda x: x[1].get("ece", float('inf'))
        )

        return {
            "per_judge": judge_calibrations,
            "ranking": [j[0] for j in ranked],
            "best_calibrated": ranked[0][0] if ranked else None,
            "worst_calibrated": ranked[-1][0] if ranked else None,
            "mean_ece": np.mean([j.get("ece", 0) for j in judge_calibrations.values()]),
        }


class CalibrationImprover:
    """
    Methods to improve judge calibration.

    Techniques:
    - Temperature scaling
    - Platt scaling
    - Isotonic regression
    """

    def __init__(self):
        self.calibration_params: dict[str, Any] = {}

    def fit_temperature_scaling(
        self,
        confidences: list[float],
        ground_truth: list[bool],
    ) -> float:
        """
        Fit temperature scaling parameter.

        Temperature scaling: calibrated_conf = sigmoid(logit(conf) / T)
        """
        # Convert to numpy and clamp to avoid numerical issues
        confs = np.array(confidences)
        confs = np.clip(confs, 1e-7, 1 - 1e-7)  # Clamp to avoid log(0) and log(inf)
        labels = np.array(ground_truth, dtype=float)

        # Grid search for best temperature
        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.linspace(0.1, 5.0, 50):
            # Apply temperature scaling with safe logit computation
            logits = np.log(confs / (1 - confs))
            scaled_logits = np.clip(logits / temp, -20, 20)  # Clamp to avoid overflow
            scaled_confs = 1 / (1 + np.exp(-scaled_logits))

            # Compute ECE
            checker = CalibrationChecker()
            ece = checker._expected_calibration_error(
                scaled_confs.tolist(), ground_truth
            )

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.calibration_params["temperature"] = best_temp
        return best_temp

    def apply_temperature_scaling(
        self,
        confidences: list[float],
    ) -> list[float]:
        """Apply fitted temperature scaling."""
        temp = self.calibration_params.get("temperature", 1.0)

        confs = np.array(confidences)
        confs = np.clip(confs, 1e-7, 1 - 1e-7)  # Clamp to avoid numerical issues
        logits = np.log(confs / (1 - confs))
        scaled_logits = np.clip(logits / temp, -20, 20)  # Clamp to avoid overflow
        scaled_confs = 1 / (1 + np.exp(-scaled_logits))

        return scaled_confs.tolist()
