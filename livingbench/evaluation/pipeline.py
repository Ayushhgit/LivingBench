"""
Main Evaluation Pipeline.

Orchestrates the complete evaluation process:
1. Load tasks
2. Run model inference
3. Collect judge verdicts
4. Aggregate results
5. Compute metrics and fingerprints

The pipeline is designed for:
- Reproducibility (deterministic where possible)
- Efficiency (parallel execution, caching)
- Flexibility (pluggable components)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from livingbench.core.types import (
    Task,
    ModelResponse,
    EvaluationResult,
    JudgeVerdict,
    CapabilityFingerprint,
    ExperimentConfig,
    ExperimentResults,
)
from livingbench.core.config import LivingBenchConfig
from livingbench.evaluation.skill_decomposition import SkillDecomposer
from livingbench.evaluation.capability_fingerprint import FingerprintComputer
from livingbench.evaluation.metrics import compute_summary_statistics


class EvaluationPipeline:
    """
    Main evaluation pipeline for LivingBench.

    Coordinates:
    - Model inference
    - Multi-judge evaluation
    - Result aggregation
    - Fingerprint computation
    """

    def __init__(
        self,
        config: LivingBenchConfig | None = None,
        output_dir: str = "outputs",
    ):
        self.config = config or LivingBenchConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.skill_decomposer = SkillDecomposer()
        self.fingerprint_computer = FingerprintComputer()

        # Model adapters (registered at runtime)
        self._model_adapters: dict[str, Callable] = {}

        # Judge instances
        self._judges: list[Callable] = []

    def register_model(
        self,
        model_id: str,
        inference_fn: Callable[[Task], ModelResponse],
    ) -> None:
        """
        Register a model for evaluation.

        Args:
            model_id: Unique identifier for the model
            inference_fn: Function that takes a Task and returns ModelResponse
        """
        self._model_adapters[model_id] = inference_fn

    def register_judge(
        self,
        judge_fn: Callable[[Task, ModelResponse], JudgeVerdict],
    ) -> None:
        """
        Register a judge for evaluation.

        Args:
            judge_fn: Function that evaluates a response and returns verdict
        """
        self._judges.append(judge_fn)

    async def evaluate_task(
        self,
        task: Task,
        model_id: str,
    ) -> EvaluationResult:
        """
        Evaluate a single task with registered model and judges.

        Args:
            task: The task to evaluate
            model_id: Which model to use

        Returns:
            Complete evaluation result with all judge verdicts
        """
        if model_id not in self._model_adapters:
            raise ValueError(f"Model '{model_id}' not registered")

        if not self._judges:
            raise ValueError("No judges registered")

        # Get model response
        inference_fn = self._model_adapters[model_id]
        response = inference_fn(task)

        # Collect verdicts from all judges
        verdicts = []
        for judge_fn in self._judges:
            verdict = judge_fn(task, response)
            verdicts.append(verdict)

        # Aggregate verdicts
        result = self._aggregate_verdicts(task, response, verdicts)
        return result

    def evaluate_task_sync(
        self,
        task: Task,
        model_id: str,
    ) -> EvaluationResult:
        """Synchronous version of evaluate_task."""
        try:
            # Try to get existing event loop (e.g., in Jupyter)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self.evaluate_task(task, model_id))

        # If there's a running loop, use nest_asyncio or run in executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.evaluate_task(task, model_id))
            return future.result()

    async def evaluate_batch(
        self,
        tasks: list[Task],
        model_id: str,
        max_concurrent: int = 10,
    ) -> list[EvaluationResult]:
        """
        Evaluate a batch of tasks.

        Args:
            tasks: List of tasks to evaluate
            model_id: Which model to use
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def eval_with_semaphore(task: Task) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_task(task, model_id)

        results = await asyncio.gather(
            *[eval_with_semaphore(task) for task in tasks]
        )
        return list(results)

    def evaluate_batch_sync(
        self,
        tasks: list[Task],
        model_id: str,
    ) -> list[EvaluationResult]:
        """Synchronous batch evaluation."""
        results = []
        for task in tasks:
            result = self.evaluate_task_sync(task, model_id)
            results.append(result)
        return results

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        tasks: list[Task],
    ) -> ExperimentResults:
        """
        Run a complete experiment.

        Args:
            experiment_config: Configuration for the experiment
            tasks: Tasks to evaluate

        Returns:
            Complete experiment results
        """
        started_at = datetime.utcnow()

        all_results: list[EvaluationResult] = []
        fingerprints: dict[str, CapabilityFingerprint] = {}

        for model_id in experiment_config.models:
            if model_id not in self._model_adapters:
                continue

            # Evaluate all tasks for this model
            model_results = self.evaluate_batch_sync(tasks, model_id)
            all_results.extend(model_results)

            # Compute fingerprint
            fingerprint = self.fingerprint_computer.compute(
                model_id=model_id,
                results=model_results,
            )
            fingerprints[model_id] = fingerprint

        completed_at = datetime.utcnow()

        # Compute summary
        summary = {
            "n_models": len(experiment_config.models),
            "n_tasks": len(tasks),
            "n_results": len(all_results),
            "overall_accuracy": sum(1 for r in all_results if r.is_correct) / len(all_results)
            if all_results else 0.0,
        }

        return ExperimentResults(
            config=experiment_config,
            started_at=started_at,
            completed_at=completed_at,
            fingerprints=fingerprints,
            evaluation_results=all_results,
            robustness_results=[],
            summary=summary,
        )

    def _aggregate_verdicts(
        self,
        task: Task,
        response: ModelResponse,
        verdicts: list[JudgeVerdict],
    ) -> EvaluationResult:
        """
        Aggregate multiple judge verdicts into a single result.

        Uses majority voting for correctness and averages for scores.
        """
        n_correct = sum(1 for v in verdicts if v.is_correct)
        n_judges = len(verdicts)

        # Majority vote for correctness
        is_correct = n_correct > n_judges / 2

        # Agreement ratio
        majority_count = max(n_correct, n_judges - n_correct)
        agreement_ratio = majority_count / n_judges

        # Score aggregation
        scores = [v.score for v in verdicts]
        mean_score = sum(scores) / len(scores)
        score_std = (
            (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            if len(scores) > 1 else 0.0
        )

        # Aggregate skill scores
        skill_scores: dict[str, list[float]] = {}
        for verdict in verdicts:
            for skill, score in verdict.skill_scores.items():
                if skill not in skill_scores:
                    skill_scores[skill] = []
                skill_scores[skill].append(score)

        aggregated_skill_scores = {
            skill: sum(scores) / len(scores)
            for skill, scores in skill_scores.items()
        }

        # Collect all detected issues
        all_issues = []
        for verdict in verdicts:
            all_issues.extend(verdict.detected_issues)
        unique_issues = list(set(all_issues))

        return EvaluationResult(
            task=task,
            response=response,
            verdicts=verdicts,
            is_correct=is_correct,
            agreement_ratio=agreement_ratio,
            mean_score=mean_score,
            score_std=score_std,
            skill_scores=aggregated_skill_scores,
            detected_issues=unique_issues,
        )

    def save_results(
        self,
        results: ExperimentResults,
        name: str | None = None,
    ) -> Path:
        """Save experiment results to disk."""
        if name is None:
            name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result_dir = self.output_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = result_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(results.config.model_dump(), f, indent=2, default=str)

        # Save fingerprints
        fp_path = result_dir / "fingerprints.json"
        fp_data = {
            model_id: fp.model_dump()
            for model_id, fp in results.fingerprints.items()
        }
        with open(fp_path, "w") as f:
            json.dump(fp_data, f, indent=2, default=str)

        # Save summary
        summary_path = result_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results.summary, f, indent=2, default=str)

        # Save detailed results
        results_path = result_dir / "results.jsonl"
        with open(results_path, "w") as f:
            for result in results.evaluation_results:
                f.write(result.model_dump_json() + "\n")

        return result_dir


class SimplePipeline:
    """
    Simplified pipeline for quick evaluation without external dependencies.

    Useful for:
    - Testing
    - Offline evaluation
    - Demo purposes
    """

    def __init__(self):
        self.skill_decomposer = SkillDecomposer()
        self.fingerprint_computer = FingerprintComputer()

    def evaluate_with_model(
        self,
        tasks: list[Task],
        model_fn: Callable[[str], str],
        judge_fn: Callable[[Task, str, str | None], tuple[bool, float, str]],
        model_id: str = "model",
        judge_id: str = "simple_judge",
    ) -> list[EvaluationResult]:
        """
        Evaluate tasks with a model function.

        Args:
            tasks: Tasks to evaluate
            model_fn: Function that takes prompt and returns response string
            judge_fn: Function that takes (task, response, reference) and returns
                     (is_correct, score, rationale)
            model_id: Identifier for the model being evaluated
            judge_id: Identifier for the judge being used

        Returns:
            List of evaluation results
        """
        results = []

        for task in tasks:
            # Get model response
            raw_output = model_fn(task.prompt)

            response = ModelResponse(
                task_id=task.id,
                model_id=model_id,
                raw_output=raw_output,
                parsed_answer=raw_output.strip(),
            )

            # Judge the response
            is_correct, score, rationale = judge_fn(
                task, raw_output, task.reference_answer
            )

            verdict = JudgeVerdict(
                judge_id=judge_id,
                is_correct=is_correct,
                score=score,
                rationale=rationale,
                confidence=0.8,
            )

            result = EvaluationResult(
                task=task,
                response=response,
                verdicts=[verdict],
                is_correct=is_correct,
                agreement_ratio=1.0,
                mean_score=score,
                score_std=0.0,
            )

            results.append(result)

        return results

    def compute_fingerprint(
        self,
        model_id: str,
        results: list[EvaluationResult],
    ) -> CapabilityFingerprint:
        """Compute capability fingerprint from results."""
        return self.fingerprint_computer.compute(model_id, results)
