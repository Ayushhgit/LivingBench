"""
Visualization tools for LivingBench results.

Generates research-quality plots:
1. Capability fingerprint radar charts
2. Skill performance heatmaps
3. Difficulty scaling curves
4. Robustness comparison bars
5. Learning curves over sessions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


class ResultVisualizer:
    """
    Generate visualizations from experiment results.

    All plots are designed for research paper quality:
    - Clear labels and legends
    - Appropriate color schemes
    - Publication-ready resolution
    """

    def __init__(self, results: dict[str, Any] | None = None, output_dir: str = "outputs/plots"):
        self.results = results or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.has_matplotlib = False
            print("Warning: matplotlib not available. Visualizations will be text-based.")

    def load_results(self, results_path: str | Path) -> None:
        """Load results from JSON file."""
        with open(results_path) as f:
            self.results = json.load(f)

    def plot_capability_fingerprints(
        self,
        fingerprints: dict[str, dict] | None = None,
        save: bool = True,
    ) -> Path | str:
        """
        Create radar chart of capability fingerprints.

        Shows each model's skill profile for easy comparison.
        """
        fingerprints = fingerprints or self.results.get("fingerprints", {})

        if not fingerprints:
            return "No fingerprints to plot"

        if not self.has_matplotlib:
            return self._text_fingerprint_comparison(fingerprints)

        # Get all skills
        all_skills = set()
        for fp in fingerprints.values():
            all_skills.update(fp.get("skill_scores", {}).keys())
        skills = sorted(all_skills)

        if not skills:
            return "No skills in fingerprints"

        # Create radar chart
        fig, ax = self.plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = self.plt.cm.Set2(np.linspace(0, 1, len(fingerprints)))

        for (model_id, fp), color in zip(fingerprints.items(), colors):
            skill_scores = fp.get("skill_scores", {})
            values = [skill_scores.get(s, 0) for s in skills]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model_id, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([s.replace('_', '\n') for s in skills], size=8)
        ax.set_ylim(0, 1)
        ax.set_title("Capability Fingerprint Comparison", size=14, weight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        if save:
            path = self.output_dir / "capability_fingerprints.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            return path

        return fig

    def _text_fingerprint_comparison(self, fingerprints: dict) -> str:
        """Text-based fingerprint comparison."""
        lines = ["=" * 60, "CAPABILITY FINGERPRINT COMPARISON", "=" * 60, ""]

        # Get all skills
        all_skills = set()
        for fp in fingerprints.values():
            all_skills.update(fp.get("skill_scores", {}).keys())
        skills = sorted(all_skills)

        # Header
        header = f"{'Skill':<30}"
        for model_id in fingerprints.keys():
            header += f" {model_id[:12]:>12}"
        lines.append(header)
        lines.append("-" * (30 + 13 * len(fingerprints)))

        # Data rows
        for skill in skills:
            row = f"{skill:<30}"
            for fp in fingerprints.values():
                score = fp.get("skill_scores", {}).get(skill, 0)
                row += f" {score:>12.2%}"
            lines.append(row)

        return "\n".join(lines)

    def plot_difficulty_scaling(
        self,
        summaries: dict[str, dict] | None = None,
        save: bool = True,
    ) -> Path | str:
        """
        Plot how accuracy scales with difficulty.

        Key insight: How gracefully does performance degrade?
        """
        summaries = summaries or self.results.get("summaries", {})

        if not summaries:
            return "No summaries to plot"

        if not self.has_matplotlib:
            return self._text_difficulty_scaling(summaries)

        fig, ax = self.plt.subplots(figsize=(10, 6))

        difficulty_order = ["trivial", "easy", "medium", "hard", "very_hard", "adversarial"]
        colors = self.plt.cm.Set1(np.linspace(0, 1, len(summaries)))

        for (model_id, summary), color in zip(summaries.items(), colors):
            by_diff = summary.get("by_difficulty", {})

            x = []
            y = []
            for i, diff in enumerate(difficulty_order):
                if diff in by_diff:
                    x.append(i)
                    y.append(by_diff[diff].get("accuracy", 0))

            if x:
                ax.plot(x, y, 'o-', linewidth=2, markersize=8, label=model_id, color=color)

        ax.set_xticks(range(len(difficulty_order)))
        ax.set_xticklabels([d.replace('_', ' ').title() for d in difficulty_order])
        ax.set_xlabel("Difficulty Level", size=12)
        ax.set_ylabel("Accuracy", size=12)
        ax.set_ylim(0, 1)
        ax.set_title("Performance Scaling with Difficulty", size=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            path = self.output_dir / "difficulty_scaling.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            return path

        return fig

    def _text_difficulty_scaling(self, summaries: dict) -> str:
        """Text-based difficulty scaling."""
        lines = ["=" * 60, "DIFFICULTY SCALING", "=" * 60, ""]

        difficulty_order = ["easy", "medium", "hard", "very_hard"]

        header = f"{'Model':<20}"
        for diff in difficulty_order:
            header += f" {diff:>12}"
        lines.append(header)
        lines.append("-" * (20 + 13 * len(difficulty_order)))

        for model_id, summary in summaries.items():
            by_diff = summary.get("by_difficulty", {})
            row = f"{model_id[:20]:<20}"
            for diff in difficulty_order:
                if diff in by_diff:
                    acc = by_diff[diff].get("accuracy", 0)
                    row += f" {acc:>12.2%}"
                else:
                    row += f" {'N/A':>12}"
            lines.append(row)

        return "\n".join(lines)

    def plot_skill_heatmap(
        self,
        fingerprints: dict[str, dict] | None = None,
        save: bool = True,
    ) -> Path | str:
        """
        Create heatmap of skill scores across models.
        """
        fingerprints = fingerprints or self.results.get("fingerprints", {})

        if not fingerprints:
            return "No fingerprints for heatmap"

        if not self.has_matplotlib:
            return self._text_fingerprint_comparison(fingerprints)

        try:
            import seaborn as sns
        except ImportError:
            return self._text_fingerprint_comparison(fingerprints)

        # Build matrix
        models = list(fingerprints.keys())
        all_skills = set()
        for fp in fingerprints.values():
            all_skills.update(fp.get("skill_scores", {}).keys())
        skills = sorted(all_skills)

        matrix = np.zeros((len(models), len(skills)))
        for i, model in enumerate(models):
            for j, skill in enumerate(skills):
                matrix[i, j] = fingerprints[model].get("skill_scores", {}).get(skill, 0)

        fig, ax = self.plt.subplots(figsize=(14, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            xticklabels=[s.replace('_', '\n') for s in skills],
            yticklabels=models,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Skill Performance Heatmap", size=14, weight='bold')

        if save:
            path = self.output_dir / "skill_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            return path

        return fig

    def plot_robustness_comparison(
        self,
        robustness: dict[str, dict] | None = None,
        save: bool = True,
    ) -> Path | str:
        """
        Bar chart comparing robustness metrics.
        """
        robustness = robustness or self.results.get("robustness", {})

        if not robustness:
            return "No robustness data"

        if not self.has_matplotlib:
            return self._text_robustness_comparison(robustness)

        models = list(robustness.keys())
        metrics = ["paraphrase_consistency", "adversarial_robustness"]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = self.plt.subplots(figsize=(10, 6))

        for i, metric in enumerate(metrics):
            values = [robustness[m].get(metric, 0) for m in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Robustness Comparison", size=14, weight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if save:
            path = self.output_dir / "robustness_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            return path

        return fig

    def _text_robustness_comparison(self, robustness: dict) -> str:
        """Text-based robustness comparison."""
        lines = ["=" * 60, "ROBUSTNESS COMPARISON", "=" * 60, ""]

        header = f"{'Model':<20} {'Paraphrase':>15} {'Adversarial':>15}"
        lines.append(header)
        lines.append("-" * 50)

        for model_id, data in robustness.items():
            para = data.get("paraphrase_consistency", 0)
            adv = data.get("adversarial_robustness", 0)
            lines.append(f"{model_id:<20} {para:>15.2%} {adv:>15.2%}")

        return "\n".join(lines)

    def generate_all_plots(self) -> list[Path | str]:
        """Generate all available plots."""
        plots = []

        plots.append(self.plot_capability_fingerprints())
        plots.append(self.plot_difficulty_scaling())
        plots.append(self.plot_skill_heatmap())
        plots.append(self.plot_robustness_comparison())

        return plots
