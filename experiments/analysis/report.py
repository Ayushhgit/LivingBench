"""
Report generation for LivingBench experiments.

Generates comprehensive research-style reports:
1. Executive summary
2. Methodology overview
3. Results tables
4. Key findings
5. Failure analysis
6. Recommendations
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json


class ReportGenerator:
    """
    Generate research-style reports from experiment results.

    Reports are formatted as Markdown for easy rendering
    and can include embedded visualizations.
    """

    def __init__(self, results: dict[str, Any] | None = None):
        self.results = results or {}

    def load_results(self, results_path: str | Path) -> None:
        """Load results from JSON file."""
        with open(results_path) as f:
            self.results = json.load(f)

    def generate_full_report(
        self,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate complete experiment report.

        Returns:
            Report as Markdown string
        """
        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_methodology(),
            self._generate_results_overview(),
            self._generate_detailed_results(),
            self._generate_robustness_analysis(),
            self._generate_failure_analysis(),
            self._generate_conclusions(),
        ]

        report = "\n\n".join(sections)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

        return report

    def _generate_header(self) -> str:
        """Generate report header."""
        metadata = self.results.get("metadata", {})
        timestamp = metadata.get("end_time", datetime.now().isoformat())

        return f"""# LivingBench Experiment Report

**Generated:** {timestamp}
**Models Evaluated:** {', '.join(metadata.get('models', ['N/A']))}
**Tasks Evaluated:** {metadata.get('n_tasks', 'N/A')}
**Duration:** {metadata.get('duration_seconds', 0):.1f} seconds

---"""

    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        summaries = self.results.get("summaries", {})

        if not summaries:
            return "## Executive Summary\n\nNo results available."

        # Find best model
        best_model = max(summaries.items(), key=lambda x: x[1].get("accuracy", 0))
        best_accuracy = best_model[1].get("accuracy", 0)

        # Gaming detection summary
        gaming = self.results.get("gaming_detection", {})
        gaming_suspected = any(g.get("gaming_suspected", False) for g in gaming.values())

        lines = [
            "## Executive Summary",
            "",
            "### Key Findings",
            "",
            f"1. **Best Performing Model:** {best_model[0]} ({best_accuracy:.1%} accuracy)",
        ]

        # Add accuracy range
        accuracies = [s.get("accuracy", 0) for s in summaries.values()]
        if len(accuracies) > 1:
            lines.append(f"2. **Accuracy Range:** {min(accuracies):.1%} - {max(accuracies):.1%}")

        # Robustness finding
        robustness = self.results.get("robustness", {})
        if robustness:
            avg_robustness = sum(
                r.get("adversarial_robustness", 0) for r in robustness.values()
            ) / len(robustness)
            lines.append(f"3. **Average Adversarial Robustness:** {avg_robustness:.1%}")

        # Gaming warning
        if gaming_suspected:
            lines.append("4. **Warning:** Potential gaming behavior detected in some models")
        else:
            lines.append("4. **Gaming Detection:** No significant gaming patterns detected")

        return "\n".join(lines)

    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        metadata = self.results.get("metadata", {})
        config = metadata.get("config", {})

        return f"""## Methodology

### Evaluation Framework

LivingBench uses a multi-faceted evaluation approach:

1. **Dynamic Task Generation:** Tasks are generated from multiple sources including
   synthetic reasoning, mathematical problems, code understanding, and tool use scenarios.

2. **Skill-Factorized Evaluation:** Rather than a single accuracy score, we decompose
   performance into a capability fingerprint covering multiple cognitive skills.

3. **Robustness Testing:** Each model is tested with paraphrased, counterfactual,
   and adversarial variants to detect gaming and memorization.

4. **Multi-Judge Evaluation:** Responses are evaluated by multiple judges with
   disagreement analysis to ensure reliable verdicts.

### Task Distribution

- **Total Tasks:** {metadata.get('n_tasks', 'N/A')}
- **Task Sources:** Synthetic reasoning, synthetic math, tool use scenarios
- **Difficulty Distribution:** Balanced across easy, medium, hard, and very hard"""

    def _generate_results_overview(self) -> str:
        """Generate results overview table."""
        summaries = self.results.get("summaries", {})

        if not summaries:
            return "## Results Overview\n\nNo results available."

        lines = [
            "## Results Overview",
            "",
            "### Overall Performance",
            "",
            "| Model | Accuracy | Weighted Acc | Calibration ECE |",
            "|-------|----------|--------------|-----------------|",
        ]

        for model_id, summary in summaries.items():
            acc = summary.get("accuracy", 0)
            wacc = summary.get("weighted_accuracy", 0)
            ece = summary.get("calibration", {}).get("ece", 0)
            lines.append(f"| {model_id} | {acc:.1%} | {wacc:.1%} | {ece:.3f} |")

        return "\n".join(lines)

    def _generate_detailed_results(self) -> str:
        """Generate detailed results by skill and difficulty."""
        summaries = self.results.get("summaries", {})
        fingerprints = self.results.get("fingerprints", {})

        lines = [
            "## Detailed Results",
            "",
            "### Performance by Difficulty",
            "",
        ]

        # Difficulty table
        difficulty_order = ["easy", "medium", "hard", "very_hard"]

        header = "| Model |"
        for d in difficulty_order:
            header += f" {d.replace('_', ' ').title()} |"
        lines.append(header)

        sep = "|-------|"
        for _ in difficulty_order:
            sep += "--------|"
        lines.append(sep)

        for model_id, summary in summaries.items():
            by_diff = summary.get("by_difficulty", {})
            row = f"| {model_id} |"
            for d in difficulty_order:
                if d in by_diff:
                    acc = by_diff[d].get("accuracy", 0)
                    row += f" {acc:.1%} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")
        lines.append("### Top Skills and Weaknesses")
        lines.append("")

        for model_id, fp in fingerprints.items():
            skills = fp.get("skill_scores", {})
            if skills:
                sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_skills[:3]
                bottom_3 = sorted_skills[-3:]

                lines.append(f"**{model_id}:**")
                lines.append(f"- Strengths: {', '.join(f'{s[0]} ({s[1]:.0%})' for s in top_3)}")
                lines.append(f"- Weaknesses: {', '.join(f'{s[0]} ({s[1]:.0%})' for s in bottom_3)}")
                lines.append("")

        return "\n".join(lines)

    def _generate_robustness_analysis(self) -> str:
        """Generate robustness analysis section."""
        robustness = self.results.get("robustness", {})

        if not robustness:
            return "## Robustness Analysis\n\nRobustness testing was not performed."

        lines = [
            "## Robustness Analysis",
            "",
            "### Robustness Metrics",
            "",
            "| Model | Paraphrase Consistency | Adversarial Robustness |",
            "|-------|------------------------|------------------------|",
        ]

        for model_id, data in robustness.items():
            para = data.get("paraphrase_consistency", 0)
            adv = data.get("adversarial_robustness", 0)
            lines.append(f"| {model_id} | {para:.1%} | {adv:.1%} |")

        lines.append("")
        lines.append("### Interpretation")
        lines.append("")

        # Find models with low robustness
        low_robustness = [
            m for m, d in robustness.items()
            if d.get("paraphrase_consistency", 1) < 0.7
        ]

        if low_robustness:
            lines.append(f"- Models with low paraphrase consistency ({', '.join(low_robustness)}) "
                        "may be relying on surface-level patterns rather than genuine understanding.")
        else:
            lines.append("- All models show reasonable paraphrase consistency, suggesting "
                        "robust semantic understanding.")

        return "\n".join(lines)

    def _generate_failure_analysis(self) -> str:
        """Generate failure analysis section."""
        gaming = self.results.get("gaming_detection", {})
        summaries = self.results.get("summaries", {})

        lines = [
            "## Failure Analysis",
            "",
            "### Gaming Detection",
            "",
        ]

        for model_id, detection in gaming.items():
            if detection.get("gaming_suspected"):
                lines.append(f"**{model_id}:** Potential gaming detected")
                signals = detection.get("signals", {})
                for signal_type, signal_data in signals.items():
                    if signal_data and signal_data.get("detected"):
                        lines.append(f"  - {signal_type}: {signal_data.get('type', 'unknown')}")
            else:
                lines.append(f"**{model_id}:** No gaming patterns detected")

        lines.append("")
        lines.append("### Common Failure Modes")
        lines.append("")

        # Analyze by skill
        for model_id, summary in summaries.items():
            by_skill = summary.get("by_skill", {})
            weak_skills = [s for s, acc in by_skill.items() if acc < 0.5]
            if weak_skills:
                lines.append(f"**{model_id}:** Struggles with {', '.join(weak_skills[:3])}")

        return "\n".join(lines)

    def _generate_conclusions(self) -> str:
        """Generate conclusions section."""
        summaries = self.results.get("summaries", {})

        lines = [
            "## Conclusions and Recommendations",
            "",
            "### Key Takeaways",
            "",
        ]

        if summaries:
            # Best model
            best = max(summaries.items(), key=lambda x: x[1].get("accuracy", 0))
            lines.append(f"1. **{best[0]}** achieved the highest overall accuracy at {best[1].get('accuracy', 0):.1%}")

            # Difficulty scaling
            lines.append("2. All models show expected accuracy degradation with increasing difficulty")

            # Skill gaps
            lines.append("3. Skill-factorized analysis reveals model-specific strengths and weaknesses")

        lines.extend([
            "",
            "### Recommendations for Future Work",
            "",
            "1. Expand task coverage to include more domain-specific evaluations",
            "2. Implement temporal evaluation to track learning over sessions",
            "3. Add human evaluation calibration for judge validation",
            "4. Investigate identified failure modes in depth",
            "",
            "---",
            "",
            "*Report generated by LivingBench*",
        ])

        return "\n".join(lines)


def generate_comparison_report(
    results1: dict,
    results2: dict,
    model1_name: str = "Model A",
    model2_name: str = "Model B",
) -> str:
    """Generate a head-to-head comparison report between two models."""
    lines = [
        f"# Model Comparison: {model1_name} vs {model2_name}",
        "",
        "## Overview",
        "",
    ]

    # Get summaries
    sum1 = results1.get("summaries", {}).get(model1_name, {})
    sum2 = results2.get("summaries", {}).get(model2_name, {})

    acc1 = sum1.get("accuracy", 0)
    acc2 = sum2.get("accuracy", 0)

    lines.append(f"| Metric | {model1_name} | {model2_name} | Winner |")
    lines.append("|--------|----------|----------|--------|")

    winner = model1_name if acc1 > acc2 else model2_name if acc2 > acc1 else "Tie"
    lines.append(f"| Overall Accuracy | {acc1:.1%} | {acc2:.1%} | {winner} |")

    wacc1 = sum1.get("weighted_accuracy", 0)
    wacc2 = sum2.get("weighted_accuracy", 0)
    winner = model1_name if wacc1 > wacc2 else model2_name if wacc2 > wacc1 else "Tie"
    lines.append(f"| Weighted Accuracy | {wacc1:.1%} | {wacc2:.1%} | {winner} |")

    return "\n".join(lines)
