"""
LivingBench Experiment Runner.

Orchestrates complete evaluation experiments:
1. Task generation or loading
2. Model inference
3. Multi-judge evaluation
4. Robustness testing
5. Metrics computation
6. Report generation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from livingbench.core.types import (
    Task,
    ModelResponse,
    EvaluationResult,
    JudgeVerdict,
    CapabilityFingerprint,
    ExperimentConfig,
    ExperimentResults,
)
from livingbench.core.config import LivingBenchConfig, load_config
from livingbench.tasks.generator import TaskGenerationEngine
from livingbench.evaluation.pipeline import EvaluationPipeline, SimplePipeline
from livingbench.evaluation.skill_decomposition import SkillDecomposer
from livingbench.evaluation.capability_fingerprint import FingerprintComputer
from livingbench.evaluation.metrics import compute_summary_statistics
from livingbench.robustness.paraphrase import ParaphraseGenerator
from livingbench.robustness.counterfactual import CounterfactualGenerator
from livingbench.robustness.adversarial import AdversarialGenerator
from livingbench.robustness.detectors import GamingDetector
from livingbench.judges.base import ExactMatchJudge, RubricJudge
from livingbench.judges.ensemble import EnsembleJudge


class ExperimentRunner:
    """
    Run complete LivingBench experiments.

    Handles:
    - Configuration loading
    - Task generation/loading
    - Model evaluation
    - Robustness testing
    - Result aggregation
    - Output generation
    """

    def __init__(
        self,
        config: LivingBenchConfig | None = None,
        output_dir: str = "outputs",
    ):
        self.config = config or LivingBenchConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.task_engine = TaskGenerationEngine(
            config=self.config.task_generation,
            seed=self.config.random_seed,
        )
        self.pipeline = SimplePipeline()
        self.fingerprint_computer = FingerprintComputer()

        # Robustness components
        self.paraphrase_gen = ParaphraseGenerator(seed=self.config.random_seed)
        self.counterfactual_gen = CounterfactualGenerator(seed=self.config.random_seed)
        self.adversarial_gen = AdversarialGenerator(seed=self.config.random_seed)
        self.gaming_detector = GamingDetector()

        # Model functions (to be registered)
        self._model_fns: dict[str, Callable[[str], str]] = {}

        # Results storage
        self.results: dict[str, Any] = {}

    def register_model(
        self,
        model_id: str,
        inference_fn: Callable[[str], str],
    ) -> None:
        """
        Register a model for evaluation.

        Args:
            model_id: Unique identifier
            inference_fn: Function that takes prompt and returns response
        """
        self._model_fns[model_id] = inference_fn

    def run(
        self,
        n_tasks: int = 100,
        models: list[str] | None = None,
        enable_robustness: bool = True,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run complete experiment.

        Args:
            n_tasks: Number of tasks to evaluate
            models: List of model IDs to evaluate (or all registered)
            enable_robustness: Whether to run robustness tests
            save_results: Whether to save results to disk

        Returns:
            Complete experiment results
        """
        start_time = datetime.now()
        print(f"Starting LivingBench experiment at {start_time.isoformat()}")

        # Get models to evaluate
        if models is None:
            models = list(self._model_fns.keys())

        if not models:
            raise ValueError(
                "No models registered. Please register at least one model using "
                "runner.register_model(model_id, inference_fn) before running experiments."
            )

        # Generate tasks
        print(f"\nGenerating {n_tasks} tasks...")
        tasks = self.task_engine.generate_balanced(n_tasks)
        print(f"Generated {len(tasks)} tasks")
        print(f"  Task pool stats: {self.task_engine.get_task_pool_stats()}")

        # Evaluate each model
        all_results: dict[str, list[EvaluationResult]] = {}
        fingerprints: dict[str, CapabilityFingerprint] = {}

        for model_id in models:
            print(f"\nEvaluating model: {model_id}")
            model_results = self._evaluate_model(model_id, tasks)
            all_results[model_id] = model_results

            # Compute fingerprint
            fingerprint = self.fingerprint_computer.compute(
                model_id=model_id,
                results=model_results,
            )
            fingerprints[model_id] = fingerprint

            # Print summary
            summary = compute_summary_statistics(model_results)
            print(f"  Accuracy: {summary['accuracy']:.2%}")
            print(f"  Weighted accuracy: {summary['weighted_accuracy']:.2%}")

        # Robustness testing
        robustness_results = {}
        if enable_robustness:
            print("\nRunning robustness tests...")
            for model_id in models:
                robustness_results[model_id] = self._run_robustness_tests(
                    model_id, tasks[:20]  # Subset for efficiency
                )

        # Gaming detection
        gaming_analysis = {}
        for model_id in models:
            gaming_analysis[model_id] = self.gaming_detector.detect(
                all_results[model_id]
            )

        # Compile results
        end_time = datetime.now()
        experiment_results = {
            "metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "n_tasks": len(tasks),
                "models": models,
                "config": self.config.model_dump(),
            },
            "fingerprints": {
                model_id: fp.model_dump()
                for model_id, fp in fingerprints.items()
            },
            "summaries": {
                model_id: compute_summary_statistics(results)
                for model_id, results in all_results.items()
            },
            "robustness": robustness_results,
            "gaming_detection": gaming_analysis,
            "comparison": self._compare_models(fingerprints) if len(models) > 1 else None,
        }

        # Save results
        if save_results:
            self._save_results(experiment_results)

        self.results = experiment_results
        return experiment_results

    def _evaluate_model(
        self,
        model_id: str,
        tasks: list[Task],
    ) -> list[EvaluationResult]:
        """Evaluate a single model on all tasks."""
        model_fn = self._model_fns.get(model_id)
        if not model_fn:
            raise ValueError(f"Model {model_id} not registered")

        # Define judge function
        def judge_fn(
            task: Task,
            response: str,
            reference: str | None,
        ) -> tuple[bool, float, str]:
            """Simple judge implementation."""
            if reference is None:
                return False, 0.5, "No reference answer"

            # Normalize for comparison
            response_norm = response.lower().strip()
            reference_norm = reference.lower().strip()

            # Exact match
            if reference_norm in response_norm:
                return True, 1.0, "Exact match found"

            # Numeric match for math tasks
            resp_nums = re.findall(r'-?\d+\.?\d*', response_norm)
            ref_nums = re.findall(r'-?\d+\.?\d*', reference_norm)

            if resp_nums and ref_nums:
                try:
                    # Compare last number from response with last from reference
                    if abs(float(resp_nums[-1]) - float(ref_nums[-1])) < 0.01:
                        return True, 1.0, "Numeric match"
                except ValueError:
                    pass

            # Partial match
            ref_words = set(reference_norm.split())
            resp_words = set(response_norm.split())
            overlap = len(ref_words & resp_words) / len(ref_words) if ref_words else 0

            if overlap > 0.8:
                return True, 0.8, f"Partial match ({overlap:.0%})"

            return False, overlap * 0.5, "No match"

        # Evaluate
        results = self.pipeline.evaluate_with_model(
            tasks=tasks,
            model_fn=model_fn,
            judge_fn=judge_fn,
            model_id=model_id,
        )

        return results

    def _run_robustness_tests(
        self,
        model_id: str,
        tasks: list[Task],
    ) -> dict[str, Any]:
        """Run robustness tests for a model."""
        model_fn = self._model_fns.get(model_id)
        if not model_fn:
            return {}

        paraphrase_results = []
        counterfactual_results = []
        adversarial_results = []

        for task in tasks:
            # Paraphrase testing
            paraphrases = self.paraphrase_gen.generate_paraphrases(task, n_paraphrases=2)
            original_response = model_fn(task.prompt)

            para_responses = []
            for para in paraphrases:
                para_response = model_fn(para.prompt)
                para_responses.append(para_response)

            # Check consistency
            consistency = sum(
                1 for pr in para_responses
                if self._responses_similar(original_response, pr)
            ) / len(para_responses) if para_responses else 1.0

            paraphrase_results.append({
                "task_id": str(task.id),
                "consistency": consistency,
            })

            # Adversarial testing
            adversarials = self.adversarial_gen.generate_adversarial(task)
            for adv_task in adversarials:
                adv_response = model_fn(adv_task.prompt)
                still_correct = self._responses_similar(original_response, adv_response)
                adversarial_results.append({
                    "task_id": str(task.id),
                    "attack_type": adv_task.variant_type,
                    "robust": still_correct,
                })

        return {
            "paraphrase_consistency": sum(r["consistency"] for r in paraphrase_results) / len(paraphrase_results) if paraphrase_results else 0,
            "adversarial_robustness": sum(1 for r in adversarial_results if r["robust"]) / len(adversarial_results) if adversarial_results else 0,
            "n_paraphrase_tests": len(paraphrase_results),
            "n_adversarial_tests": len(adversarial_results),
        }

    def _compare_models(
        self,
        fingerprints: dict[str, CapabilityFingerprint],
    ) -> dict[str, Any]:
        """Compare capability fingerprints across models."""
        if len(fingerprints) < 2:
            return {}

        model_ids = list(fingerprints.keys())
        comparisons = {}

        for i, m1 in enumerate(model_ids):
            for m2 in model_ids[i+1:]:
                fp1 = fingerprints[m1]
                fp2 = fingerprints[m2]

                comparison = self.fingerprint_computer.compare_fingerprints(fp1, fp2)
                comparisons[f"{m1}_vs_{m2}"] = comparison

        return comparisons

    def _responses_similar(self, resp1: str, resp2: str) -> bool:
        """Check if two responses are similar."""
        # Normalize
        r1 = resp1.lower().strip()
        r2 = resp2.lower().strip()

        if r1 == r2:
            return True

        # Word overlap
        words1 = set(r1.split())
        words2 = set(r2.split())

        if not words1 or not words2:
            return False

        jaccard = len(words1 & words2) / len(words1 | words2)
        return jaccard > 0.7

    def _save_results(self, results: dict) -> Path:
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / f"experiment_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(result_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save fingerprints separately for easy access
        with open(result_dir / "fingerprints.json", "w") as f:
            json.dump(results.get("fingerprints", {}), f, indent=2, default=str)

        # Save summaries
        with open(result_dir / "summaries.json", "w") as f:
            json.dump(results.get("summaries", {}), f, indent=2, default=str)

        print(f"\nResults saved to: {result_dir}")
        return result_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LivingBench evaluation experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=50,
        help="Number of tasks to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip robustness testing",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = LivingBenchConfig(
            random_seed=args.seed,
            output_dir=args.output_dir,
        )

    # Create and run experiment
    runner = ExperimentRunner(config=config, output_dir=args.output_dir)

    print("=" * 60)
    print("LivingBench Experiment Runner")
    print("=" * 60)

    results = runner.run(
        n_tasks=args.n_tasks,
        enable_robustness=not args.no_robustness,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    for model_id, summary in results.get("summaries", {}).items():
        print(f"\n{model_id}:")
        print(f"  Overall Accuracy: {summary.get('accuracy', 0):.2%}")
        print(f"  Weighted Accuracy: {summary.get('weighted_accuracy', 0):.2%}")

        by_diff = summary.get("by_difficulty", {})
        if by_diff:
            print("  By Difficulty:")
            for diff, data in by_diff.items():
                print(f"    {diff}: {data.get('accuracy', 0):.2%} ({data.get('n_samples', 0)} tasks)")


if __name__ == "__main__":
    main()
