"""
Session Tracking for Multi-Turn Evaluation.

Tracks model performance across multiple evaluation sessions to measure:
1. Error correction: Does the model learn from mistakes?
2. Memory persistence: Does knowledge transfer between sessions?
3. Performance trends: Is the model improving or degrading?

This is critical for evaluating:
- Fine-tuned models
- Retrieval-augmented systems
- Agents with persistent memory
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
from collections import defaultdict

from livingbench.core.types import (
    Task,
    EvaluationResult,
    SessionState,
    TemporalTrace,
)


class SessionTracker:
    """
    Track model performance across multiple sessions.

    A session represents a coherent evaluation period. Between sessions,
    models might:
    - Be fine-tuned on errors
    - Update their retrieval index
    - Receive feedback

    This tracker measures the effect of such updates.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.sessions: list[SessionState] = []
        self.current_session: SessionState | None = None

        # Cross-session tracking
        self._error_history: dict[str, list[dict]] = defaultdict(list)
        self._task_history: dict[str, list[EvaluationResult]] = {}

    def start_session(self) -> SessionState:
        """
        Start a new evaluation session.

        Returns:
            The new session state
        """
        if self.current_session:
            # Finalize previous session
            self.sessions.append(self.current_session)

        self.current_session = SessionState(
            model_id=self.model_id,
        )
        return self.current_session

    def record_result(self, result: EvaluationResult) -> None:
        """
        Record an evaluation result in the current session.

        Args:
            result: The evaluation result to record
        """
        if not self.current_session:
            self.start_session()

        task_id = str(result.task.id)

        # Update session state
        self.current_session.task_history.append(result.task.id)
        self.current_session.response_history.append(result.response.id)

        # Track errors
        if not result.is_correct:
            error_info = {
                "task_id": task_id,
                "session_id": str(self.current_session.session_id),
                "timestamp": datetime.utcnow().isoformat(),
                "response": result.response.raw_output[:500],
                "expected": result.task.reference_answer,
            }
            self.current_session.errors_made.append(error_info)
            self._error_history[task_id].append(error_info)

        # Track task history
        if task_id not in self._task_history:
            self._task_history[task_id] = []
        self._task_history[task_id].append(result)

    def record_correction_attempt(
        self,
        task_id: str,
        correction_result: EvaluationResult,
    ) -> None:
        """
        Record an attempt to correct a previous error.

        Args:
            task_id: The task being corrected
            correction_result: Result of the correction attempt
        """
        if not self.current_session:
            return

        correction_info = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": correction_result.is_correct,
            "new_response": correction_result.response.raw_output[:500],
        }
        self.current_session.corrections_attempted.append(correction_info)

    def end_session(self) -> SessionState:
        """
        End the current session.

        Returns:
            The finalized session state
        """
        if self.current_session:
            session = self.current_session
            self.sessions.append(session)
            self.current_session = None
            return session
        raise ValueError("No active session to end")

    def get_error_correction_rate(self) -> dict[str, Any]:
        """
        Calculate the rate of successful error corrections.

        Returns:
            Dictionary with correction metrics
        """
        total_errors = 0
        corrected_errors = 0

        for task_id, errors in self._error_history.items():
            total_errors += len(errors)

            # Check if task was later answered correctly
            if task_id in self._task_history:
                task_results = self._task_history[task_id]
                # Find first error
                first_error_idx = next(
                    (i for i, r in enumerate(task_results) if not r.is_correct),
                    None
                )
                if first_error_idx is not None:
                    # Check if any later result was correct
                    later_correct = any(
                        r.is_correct for r in task_results[first_error_idx + 1:]
                    )
                    if later_correct:
                        corrected_errors += 1

        correction_rate = corrected_errors / total_errors if total_errors > 0 else 0.0

        return {
            "total_errors": total_errors,
            "corrected_errors": corrected_errors,
            "correction_rate": correction_rate,
        }

    def get_temporal_trace(self) -> TemporalTrace:
        """
        Generate a complete temporal trace for this model.

        Returns:
            TemporalTrace with all sessions and metrics
        """
        # Make sure current session is included
        all_sessions = list(self.sessions)
        if self.current_session:
            all_sessions.append(self.current_session)

        # Compute metrics
        correction_info = self.get_error_correction_rate()

        # Compute performance trend
        if len(all_sessions) >= 2:
            session_accuracies = self._compute_session_accuracies()
            trend = self._determine_trend(session_accuracies)
        else:
            trend = None

        # Find persistent errors (errors that were never corrected)
        persistent_errors = self._find_persistent_errors()

        # Find learned corrections
        learned = self._find_learned_corrections()

        return TemporalTrace(
            model_id=self.model_id,
            sessions=all_sessions,
            error_correction_rate=correction_info["correction_rate"],
            knowledge_retention_rate=self._compute_retention_rate(),
            performance_trend=trend,
            persistent_errors=persistent_errors,
            learned_corrections=learned,
        )

    def _compute_session_accuracies(self) -> list[float]:
        """Compute accuracy for each session."""
        accuracies = []

        for session in self.sessions:
            n_tasks = len(session.task_history)
            n_errors = len(session.errors_made)

            if n_tasks > 0:
                accuracy = (n_tasks - n_errors) / n_tasks
            else:
                accuracy = 0.0

            accuracies.append(accuracy)

        return accuracies

    def _determine_trend(self, accuracies: list[float]) -> str:
        """Determine if performance is improving, degrading, or stable."""
        if len(accuracies) < 2:
            return "stable"

        # Simple linear trend
        first_half = sum(accuracies[:len(accuracies)//2]) / (len(accuracies)//2)
        second_half = sum(accuracies[len(accuracies)//2:]) / (len(accuracies) - len(accuracies)//2)

        delta = second_half - first_half

        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "degrading"
        else:
            return "stable"

    def _find_persistent_errors(self) -> list[dict]:
        """Find errors that were never corrected."""
        persistent = []

        for task_id, errors in self._error_history.items():
            if task_id in self._task_history:
                task_results = self._task_history[task_id]
                # If last result is still wrong, it's persistent
                if not task_results[-1].is_correct:
                    persistent.append({
                        "task_id": task_id,
                        "n_attempts": len(task_results),
                        "first_error_session": errors[0]["session_id"],
                    })

        return persistent

    def _find_learned_corrections(self) -> list[dict]:
        """Find errors that were successfully corrected."""
        learned = []

        for task_id, errors in self._error_history.items():
            if task_id in self._task_history:
                task_results = self._task_history[task_id]

                # Find transition from wrong to correct
                for i in range(len(task_results) - 1):
                    if not task_results[i].is_correct and task_results[i + 1].is_correct:
                        learned.append({
                            "task_id": task_id,
                            "corrected_at_attempt": i + 2,
                            "total_attempts": len(task_results),
                        })
                        break

        return learned

    def _compute_retention_rate(self) -> float:
        """
        Compute knowledge retention rate.

        Measures: If correct in session N, still correct in session N+1?
        """
        retained = 0
        total_opportunities = 0

        for task_id, results in self._task_history.items():
            for i in range(len(results) - 1):
                if results[i].is_correct:
                    total_opportunities += 1
                    if results[i + 1].is_correct:
                        retained += 1

        if total_opportunities == 0:
            return 1.0

        return retained / total_opportunities
