"""
Performance Degradation Analysis.

Detect and analyze performance degradation:
1. Catastrophic forgetting
2. Skill regression
3. Drift detection
4. Stability analysis

Critical for:
- Continuous learning systems
- Production monitoring
- Model versioning decisions
"""

from __future__ import annotations

from typing import Any
from collections import defaultdict
import numpy as np

from livingbench.core.types import TemporalTrace, CapabilityFingerprint


class DegradationAnalyzer:
    """
    Analyze performance degradation over time.

    Detects:
    - Catastrophic forgetting
    - Gradual skill decay
    - Sudden performance drops
    - Capability drift
    """

    def __init__(
        self,
        significance_threshold: float = 0.1,
        catastrophic_threshold: float = 0.3,
    ):
        """
        Args:
            significance_threshold: Min drop to be considered significant
            catastrophic_threshold: Min drop for catastrophic forgetting
        """
        self.significance_threshold = significance_threshold
        self.catastrophic_threshold = catastrophic_threshold

    def analyze(
        self,
        trace: TemporalTrace,
        fingerprints: list[CapabilityFingerprint] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive degradation analysis.

        Args:
            trace: Temporal trace to analyze
            fingerprints: Optional list of fingerprints over time

        Returns:
            Degradation analysis results
        """
        results = {
            "overall_trend": trace.performance_trend,
            "forgetting_analysis": self._analyze_forgetting(trace),
            "stability_analysis": self._analyze_stability(trace),
        }

        if fingerprints:
            results["capability_drift"] = self._analyze_capability_drift(fingerprints)

        # Summary
        results["degradation_detected"] = self._is_degradation_detected(results)

        return results

    def _analyze_forgetting(self, trace: TemporalTrace) -> dict[str, Any]:
        """Analyze forgetting patterns."""
        retention_rate = trace.knowledge_retention_rate or 1.0
        forgetting_rate = 1.0 - retention_rate

        # Classify forgetting severity
        if forgetting_rate > self.catastrophic_threshold:
            severity = "catastrophic"
        elif forgetting_rate > self.significance_threshold:
            severity = "significant"
        elif forgetting_rate > 0.05:
            severity = "mild"
        else:
            severity = "minimal"

        # Count forgotten tasks - those that were correct before but now wrong
        # Check for various possible field names in the error dict
        n_forgotten = len([
            e for e in trace.persistent_errors
            if e.get("was_previously_correct", False)
            or e.get("previously_correct", False)
            or e.get("is_regression", False)
        ])

        return {
            "forgetting_rate": forgetting_rate,
            "retention_rate": retention_rate,
            "severity": severity,
            "n_forgotten_tasks": n_forgotten,
        }

    def _analyze_stability(self, trace: TemporalTrace) -> dict[str, Any]:
        """Analyze performance stability across sessions."""
        if len(trace.sessions) < 3:
            return {"status": "insufficient_sessions"}

        # Compute session accuracies
        accuracies = []
        for session in trace.sessions:
            n_tasks = len(session.task_history)
            n_errors = len(session.errors_made)
            if n_tasks > 0:
                acc = (n_tasks - n_errors) / n_tasks
            else:
                acc = 0.5
            accuracies.append(acc)

        # Stability metrics
        variance = np.var(accuracies)
        std = np.std(accuracies)
        range_val = max(accuracies) - min(accuracies)

        # Detect sudden drops
        sudden_drops = []
        for i in range(1, len(accuracies)):
            drop = accuracies[i-1] - accuracies[i]
            if drop > self.significance_threshold:
                sudden_drops.append({
                    "session": i,
                    "drop": drop,
                    "from_accuracy": accuracies[i-1],
                    "to_accuracy": accuracies[i],
                })

        # Stability classification
        if std < 0.05:
            stability = "highly_stable"
        elif std < 0.1:
            stability = "stable"
        elif std < 0.2:
            stability = "moderately_unstable"
        else:
            stability = "unstable"

        return {
            "variance": float(variance),
            "std": float(std),
            "range": float(range_val),
            "stability_classification": stability,
            "sudden_drops": sudden_drops,
            "n_sudden_drops": len(sudden_drops),
        }

    def _analyze_capability_drift(
        self,
        fingerprints: list[CapabilityFingerprint],
    ) -> dict[str, Any]:
        """Analyze drift in capability fingerprints over time."""
        if len(fingerprints) < 2:
            return {"status": "insufficient_fingerprints"}

        # Track per-skill changes
        skill_changes: dict[str, list[float]] = defaultdict(list)

        for fp in fingerprints:
            for skill, score in fp.skill_scores.items():
                skill_changes[skill].append(score)

        # Compute drift for each skill
        skill_drift = {}
        degraded_skills = []
        improved_skills = []

        for skill, scores in skill_changes.items():
            if len(scores) >= 2:
                # Compute trend
                first_half = np.mean(scores[:len(scores)//2])
                second_half = np.mean(scores[len(scores)//2:])
                change = second_half - first_half

                skill_drift[skill] = {
                    "initial": scores[0],
                    "final": scores[-1],
                    "change": change,
                    "trend": "improving" if change > 0.05 else
                             "degrading" if change < -0.05 else "stable",
                }

                if change < -self.significance_threshold:
                    degraded_skills.append((skill, change))
                elif change > self.significance_threshold:
                    improved_skills.append((skill, change))

        # Overall drift
        if fingerprints:
            initial_avg = np.mean(list(fingerprints[0].skill_scores.values()))
            final_avg = np.mean(list(fingerprints[-1].skill_scores.values()))
            overall_change = final_avg - initial_avg
        else:
            overall_change = 0.0

        return {
            "skill_drift": skill_drift,
            "degraded_skills": sorted(degraded_skills, key=lambda x: x[1]),
            "improved_skills": sorted(improved_skills, key=lambda x: x[1], reverse=True),
            "overall_drift": overall_change,
            "drift_classification": "positive" if overall_change > 0.05 else
                                    "negative" if overall_change < -0.05 else "neutral",
        }

    def _is_degradation_detected(self, results: dict) -> bool:
        """Determine if significant degradation was detected."""
        # Check overall trend
        if results.get("overall_trend") == "degrading":
            return True

        # Check forgetting severity
        forgetting = results.get("forgetting_analysis", {})
        if forgetting.get("severity") in ["catastrophic", "significant"]:
            return True

        # Check stability
        stability = results.get("stability_analysis", {})
        if stability.get("n_sudden_drops", 0) >= 2:
            return True

        # Check capability drift
        drift = results.get("capability_drift", {})
        if drift.get("drift_classification") == "negative":
            return True

        return False

    def generate_report(self, analysis: dict) -> str:
        """Generate human-readable degradation report."""
        lines = ["# Performance Degradation Analysis Report", ""]

        # Overall status
        if analysis.get("degradation_detected"):
            lines.append("**STATUS: DEGRADATION DETECTED**")
        else:
            lines.append("**STATUS: STABLE**")
        lines.append("")

        # Trend
        lines.append(f"## Overall Trend: {analysis.get('overall_trend', 'unknown')}")
        lines.append("")

        # Forgetting
        fg = analysis.get("forgetting_analysis", {})
        if fg:
            lines.append("## Forgetting Analysis")
            lines.append(f"- Severity: {fg.get('severity', 'unknown')}")
            lines.append(f"- Forgetting Rate: {fg.get('forgetting_rate', 0):.2%}")
            lines.append(f"- Retention Rate: {fg.get('retention_rate', 0):.2%}")
            lines.append("")

        # Stability
        st = analysis.get("stability_analysis", {})
        if st and st.get("status") != "insufficient_sessions":
            lines.append("## Stability Analysis")
            lines.append(f"- Classification: {st.get('stability_classification', 'unknown')}")
            lines.append(f"- Standard Deviation: {st.get('std', 0):.3f}")
            lines.append(f"- Sudden Drops: {st.get('n_sudden_drops', 0)}")
            lines.append("")

        # Capability drift
        cd = analysis.get("capability_drift", {})
        if cd and cd.get("status") != "insufficient_fingerprints":
            lines.append("## Capability Drift")
            lines.append(f"- Overall Drift: {cd.get('drift_classification', 'unknown')}")
            if cd.get("degraded_skills"):
                lines.append("- Degraded Skills:")
                for skill, change in cd["degraded_skills"][:5]:
                    lines.append(f"  - {skill}: {change:+.2f}")
            lines.append("")

        return "\n".join(lines)
