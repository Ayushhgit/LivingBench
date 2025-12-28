"""
LivingBench Experiment with Groq Models.
Run evaluation across multiple Groq-hosted models to compare their capabilities.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

from livingbench.models.groq import GroqModel
from livingbench.core.config import LivingBenchConfig
from experiments.run_experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run LivingBench evaluation with Groq models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="Groq models to evaluate",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=50,
        help="Number of tasks to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/groq_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip robustness testing",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LivingBench Groq Experiment")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Tasks: {args.n_tasks}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)

    # Create configuration
    config = LivingBenchConfig(
        random_seed=args.seed,
        output_dir=args.output_dir,
    )

    # Create experiment runner
    runner = ExperimentRunner(config=config, output_dir=args.output_dir)

    # Register Groq models
    registered_models = []
    for model_name in args.models:
        print(f"\nInitializing model: {model_name}")
        try:
            model = GroqModel(
                model_name=model_name,
                temperature=args.temperature,
                max_tokens=1024,
            )
            runner.register_model(model_name, model.generate)
            registered_models.append(model_name)
            print(f"  Registered successfully")
        except Exception as e:
            print(f"  Failed to initialize: {e}")
            continue

    if not registered_models:
        print("\nERROR: No models were successfully registered. Exiting.")
        print("Check your GROQ_API_KEY and model names.")
        print("Available models: llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it")
        sys.exit(1)

    # Run experiment
    print("\n" + "=" * 60)
    print(f"Running evaluation with {len(registered_models)} model(s): {registered_models}")
    results = runner.run(
        n_tasks=args.n_tasks,
        models=registered_models,  # Only use successfully registered models
        enable_robustness=not args.no_robustness,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    summaries = results.get("summaries", {})
    for model_id in sorted(summaries.keys()):
        summary = summaries[model_id]
        print(f"\n{model_id}:")
        print(f"  Overall Accuracy: {summary.get('accuracy', 0):.2%}")
        print(f"  Weighted Accuracy: {summary.get('weighted_accuracy', 0):.2%}")

        by_diff = summary.get("by_difficulty", {})
        if by_diff:
            print("  By Difficulty:")
            for diff in ["easy", "medium", "hard", "very_hard"]:
                if diff in by_diff:
                    data = by_diff[diff]
                    print(f"    {diff}: {data.get('accuracy', 0):.2%}")

    # Print robustness comparison if available
    robustness = results.get("robustness", {})
    if robustness:
        print("\n" + "-" * 40)
        print("ROBUSTNESS")
        print("-" * 40)
        for model_id, rob_data in robustness.items():
            print(f"\n{model_id}:")
            print(f"  Paraphrase consistency: {rob_data.get('paraphrase_consistency', 0):.2%}")
            print(f"  Adversarial robustness: {rob_data.get('adversarial_robustness', 0):.2%}")

    print(f"\nResults saved to: {args.output_dir}")
    return results


if __name__ == "__main__":
    main()
