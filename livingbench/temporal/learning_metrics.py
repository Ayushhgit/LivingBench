"""
Learning-Over-Time Metrics.

Compute metrics that measure how model performance changes:
1. Error correction rate
2. Knowledge transfer efficiency
3. Learning curve analysis
4. Forgetting rate

These metrics are essential for evaluating:
- Few-shot learning
- In-context learning
- Fine-tuning effectiveness
"""

from __future__ import annotations

from typing import Any
from collections import defaultdict
import numpy as np

from livingbench.core.types import EvaluationResult, TemporalTrace


class LearningMetricsComputer:
    """
    Compute learning-related metrics from temporal traces.

    Focus areas:
    1. Error correction: How well does the model learn from mistakes?
    2. Skill acquisition: How do individual skills improve over time?
    3. Knowledge transfer: Does improvement in one area help others?
    4. Forgetting: Does the model lose previously correct answers?
    """

    def compute_all_metrics(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """
        Compute all learning metrics from a temporal trace.

        Returns:
            Dictionary with all computed metrics
        """
        metrics = {
            "error_correction": self._compute_error_correction_metrics(trace),
            "skill_progression": self._compute_skill_progression(trace),
            "forgetting": self._compute_forgetting_metrics(trace),
            "learning_curve": self._compute_learning_curve(trace),
        }

        # Overall learning score
        metrics["overall_learning_score"] = self._compute_overall_learning_score(metrics)

        return metrics

    def _compute_error_correction_metrics(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """Compute detailed error correction metrics."""
        return {
            "correction_rate": trace.error_correction_rate or 0.0,
            "n_persistent_errors": len(trace.persistent_errors),
            "n_learned_corrections": len(trace.learned_corrections),
            "avg_attempts_to_correct": self._avg_attempts_to_correct(trace),
        }

    def _avg_attempts_to_correct(self, trace: TemporalTrace) -> float:
        """Average number of attempts before successful correction."""
        if not trace.learned_corrections:
            return float('inf')

        attempts = [lc["corrected_at_attempt"] for lc in trace.learned_corrections]
        return np.mean(attempts)

    def _compute_skill_progression(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """Track how individual skills improve over sessions."""
        if len(trace.sessions) < 2:
            return {"status": "insufficient_sessions"}

        # Track errors per skill across sessions
        # Format: skill -> list of (session_idx, n_errors, n_total)
        skill_by_session: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

        for session_idx, session in enumerate(trace.sessions):
            # Count errors per skill in this session
            skill_errors: dict[str, int] = defaultdict(int)
            n_tasks = len(session.task_history)

            for error in session.errors_made:
                # Extract skills from error dict (supports both "skills" list and "skill" string)
                skills = error.get("skills", error.get("skill", []))
                if isinstance(skills, str):
                    skills = [skills]
                for skill in skills:
                    skill_errors[skill] += 1

            # Record data for each skill encountered
            for skill, n_errors in skill_errors.items():
                # Estimate tasks per skill proportionally
                skill_total = max(1, n_tasks // max(len(skill_errors), 1))
                skill_by_session[skill].append((session_idx, n_errors, skill_total))

        # Compute per-skill trends
        skill_trends = {}
        improvements = []
        declines = []

        for skill, session_data in skill_by_session.items():
            if len(session_data) >= 2:
                # Calculate error rates for first and last half of sessions
                mid_idx = len(session_data) // 2
                early_errors = sum(e for _, e, _ in session_data[:mid_idx])
                early_total = sum(t for _, _, t in session_data[:mid_idx])
                late_errors = sum(e for _, e, _ in session_data[mid_idx:])
                late_total = sum(t for _, _, t in session_data[mid_idx:])

                early_rate = early_errors / early_total if early_total > 0 else 0
                late_rate = late_errors / late_total if late_total > 0 else 0

                # Improvement means fewer errors (rate decreased)
                change = early_rate - late_rate  # Positive = improvement

                skill_trends[skill] = {
                    "early_error_rate": early_rate,
                    "late_error_rate": late_rate,
                    "improvement": change,
                    "trend": "improving" if change > 0.05 else "declining" if change < -0.05 else "stable",
                }

                if change > 0.05:
                    improvements.append((skill, change))
                elif change < -0.05:
                    declines.append((skill, abs(change)))

        return {
            "skill_trends": skill_trends,
            "most_improved_skill": max(improvements, key=lambda x: x[1])[0] if improvements else None,
            "most_declined_skill": max(declines, key=lambda x: x[1])[0] if declines else None,
        }

    def _compute_forgetting_metrics(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """
        Compute forgetting-related metrics.

        Forgetting = answering wrong after previously answering correctly
        """
        retention_rate = trace.knowledge_retention_rate or 1.0
        forgetting_rate = 1.0 - retention_rate

        return {
            "retention_rate": retention_rate,
            "forgetting_rate": forgetting_rate,
            "status": "high_retention" if retention_rate > 0.9 else
                      "moderate_retention" if retention_rate > 0.7 else
                      "low_retention",
        }

    def _compute_learning_curve(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """
        Compute learning curve characteristics.

        Analyzes how accuracy changes over sessions.
        """
        if len(trace.sessions) < 3:
            return {"status": "insufficient_sessions_for_curve"}

        # Compute per-session accuracy
        session_accuracies = []
        for session in trace.sessions:
            n_tasks = len(session.task_history)
            n_errors = len(session.errors_made)
            if n_tasks > 0:
                acc = (n_tasks - n_errors) / n_tasks
            else:
                acc = 0.5
            session_accuracies.append(acc)

        # Fit simple learning curve
        # Power law: accuracy = a - b * n^(-c)
        # Simplified: just compute trend

        early_acc = np.mean(session_accuracies[:len(session_accuracies)//3])
        mid_acc = np.mean(session_accuracies[len(session_accuracies)//3:2*len(session_accuracies)//3])
        late_acc = np.mean(session_accuracies[2*len(session_accuracies)//3:])

        # Learning rate (slope)
        if len(session_accuracies) >= 2:
            x = np.arange(len(session_accuracies))
            y = np.array(session_accuracies)
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        return {
            "early_accuracy": early_acc,
            "mid_accuracy": mid_acc,
            "late_accuracy": late_acc,
            "learning_slope": slope,
            "curve_type": "improving" if slope > 0.01 else
                          "degrading" if slope < -0.01 else
                          "stable",
            "session_accuracies": session_accuracies,
        }

    def _compute_overall_learning_score(
        self,
        metrics: dict[str, Any],
    ) -> float:
        """
        Compute overall learning score (0-1).

        Combines multiple metrics into single score.
        """
        components = []

        # Error correction component
        ec = metrics.get("error_correction", {})
        if "correction_rate" in ec:
            components.append(ec["correction_rate"])

        # Retention component
        fg = metrics.get("forgetting", {})
        if "retention_rate" in fg:
            components.append(fg["retention_rate"])

        # Learning curve component
        lc = metrics.get("learning_curve", {})
        if "learning_slope" in lc:
            # Normalize slope to 0-1 range
            slope = lc["learning_slope"]
            normalized_slope = min(1.0, max(0.0, 0.5 + slope * 5))
            components.append(normalized_slope)

        if not components:
            return 0.5

        return np.mean(components)


class LearningPatternAnalyzer:
    """
    Identify patterns in learning behavior.

    Detects:
    - Fast learners vs slow learners
    - Skill-specific learning patterns
    - Plateaus and breakthroughs
    """

    def analyze_patterns(
        self,
        trace: TemporalTrace,
    ) -> dict[str, Any]:
        """
        Analyze learning patterns in a temporal trace.

        Returns:
            Dictionary with identified patterns
        """
        patterns = {
            "learner_type": self._classify_learner(trace),
            "plateau_detected": self._detect_plateau(trace),
            "breakthrough_detected": self._detect_breakthrough(trace),
        }

        return patterns

    def _classify_learner(self, trace: TemporalTrace) -> str:
        """Classify the model's learning type."""
        if not trace.sessions or len(trace.sessions) < 2:
            return "unknown"

        correction_rate = trace.error_correction_rate or 0.0
        retention_rate = trace.knowledge_retention_rate or 1.0

        if correction_rate > 0.7 and retention_rate > 0.9:
            return "fast_stable_learner"
        elif correction_rate > 0.5:
            return "moderate_learner"
        elif correction_rate > 0.2:
            return "slow_learner"
        else:
            return "non_learner"

    def _detect_plateau(self, trace: TemporalTrace) -> dict[str, Any]:
        """Detect if learning has plateaued."""
        if len(trace.sessions) < 5:
            return {"detected": False, "reason": "insufficient_sessions"}

        # Compute per-session accuracy
        accuracies = []
        for session in trace.sessions:
            n_tasks = len(session.task_history)
            n_errors = len(session.errors_made)
            if n_tasks > 0:
                acc = (n_tasks - n_errors) / n_tasks
            else:
                acc = 0.5
            accuracies.append(acc)

        # Check last 3 sessions for plateau
        recent = accuracies[-3:]
        variance = np.var(recent)

        if variance < 0.01:  # Very low variance
            return {
                "detected": True,
                "plateau_level": np.mean(recent),
                "plateau_start_session": len(accuracies) - 3,
            }

        return {"detected": False}

    def _detect_breakthrough(self, trace: TemporalTrace) -> dict[str, Any]:
        """Detect sudden performance improvements."""
        if len(trace.sessions) < 3:
            return {"detected": False, "reason": "insufficient_sessions"}

        # Compute per-session accuracy
        accuracies = []
        for session in trace.sessions:
            n_tasks = len(session.task_history)
            n_errors = len(session.errors_made)
            if n_tasks > 0:
                acc = (n_tasks - n_errors) / n_tasks
            else:
                acc = 0.5
            accuracies.append(acc)

        # Look for sudden jumps
        for i in range(1, len(accuracies)):
            jump = accuracies[i] - accuracies[i-1]
            if jump > 0.2:  # 20% improvement
                return {
                    "detected": True,
                    "breakthrough_session": i,
                    "improvement": jump,
                }

        return {"detected": False}
