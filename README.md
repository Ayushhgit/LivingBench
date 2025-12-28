# LivingBench

**A Continually Updating, Self-Auditing Benchmark for LLM Reasoning, Tool Use, and Learning Over Time**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Abstract

Current LLM benchmarks suffer from three fundamental problems: **benchmark saturation** (models are optimized for fixed test sets), **single-score reductionism** (complex capabilities collapsed to one number), and **gaming susceptibility** (models exploit surface patterns rather than demonstrating genuine understanding).

**LivingBench** addresses these challenges through:

1. **Dynamic Task Generation**: Tasks are generated continuously from multiple sources (synthetic reasoning, code understanding, tool use scenarios), ensuring the benchmark never freezes.

2. **Skill-Factorized Evaluation**: Instead of single accuracy scores, we produce **capability fingerprints**—multi-dimensional profiles decomposing performance across cognitive skills.

3. **Anti-Gaming Robustness Layer**: Automatic generation of paraphrases, counterfactuals, and adversarial variants to detect memorization and spurious correlations.

4. **Learning-Over-Time Measurement**: Multi-session evaluation tracking error correction, knowledge retention, and performance trends.

5. **Calibrated Multi-Judge Evaluation**: Ensemble of judges with disagreement analysis and calibration checking—no blind trust in any single evaluator.

---

## Motivation

### The Benchmark Saturation Problem

Modern LLMs are increasingly trained or fine-tuned with benchmark performance as a direct objective. This creates a fundamental tension: benchmarks designed to measure capability become targets for optimization, degrading their validity as capability measures (Goodhart's Law in action).

### Beyond Single-Score Accuracy

A model with 80% accuracy might excel at factual recall but fail at causal reasoning. Another 80%-accuracy model might show the opposite pattern. Single scores hide these critical differences that matter for deployment decisions.

### The Gaming Problem

Models can achieve high scores through:
- Memorizing benchmark-specific patterns
- Exploiting spurious correlations (answer position bias, keyword triggers)
- Over-verbose responses that contain correct substrings by chance

LivingBench is designed to detect and penalize these failure modes.

---

## System Architecture

```
livingbench/
├── core/                    # Type system and configuration
│   ├── types.py            # Core data structures (Task, EvaluationResult, etc.)
│   ├── config.py           # Hierarchical configuration management
│   └── registry.py         # Component registration
├── tasks/                   # Task generation engine
│   ├── generator.py        # Main orchestration
│   ├── labeling.py         # Skill and difficulty estimation
│   └── sources/            # Task sources
│       ├── synthetic.py    # Synthetic reasoning/math tasks
│       ├── github.py       # GitHub issues/PRs
│       ├── arxiv.py        # ArXiv abstracts
│       └── tool_use.py     # Tool use scenarios
├── evaluation/              # Evaluation pipeline
│   ├── pipeline.py         # Main evaluation orchestration
│   ├── skill_decomposition.py  # Skill-factorized scoring
│   ├── capability_fingerprint.py  # Multi-dimensional profiles
│   └── metrics.py          # Standardized metrics
├── robustness/              # Anti-gaming layer
│   ├── paraphrase.py       # Paraphrase generation
│   ├── counterfactual.py   # Counterfactual variants
│   ├── adversarial.py      # Adversarial perturbations
│   └── detectors.py        # Gaming/memorization detection
├── temporal/                # Learning-over-time evaluation
│   ├── session_tracker.py  # Multi-session tracking
│   ├── learning_metrics.py # Learning curve analysis
│   └── degradation.py      # Performance degradation detection
├── judges/                  # LLM-as-Judge system
│   ├── base.py             # Judge interfaces
│   ├── ensemble.py         # Multi-judge aggregation
│   ├── calibration.py      # Calibration analysis
│   └── disagreement.py     # Disagreement patterns
└── utils/                   # Utilities
    ├── logging.py          # Structured logging
    └── reproducibility.py  # Deterministic execution
```

---

## Key Components

### 1. Task Generation Engine

Tasks are generated from multiple sources to ensure diversity and freshness:

| Source | Description | Skills Tested |
|--------|-------------|---------------|
| Synthetic Reasoning | Syllogisms, constraint satisfaction, propositional logic | Logical deduction, multi-step planning |
| Synthetic Math | Word problems, algebra, probability, number theory | Mathematical reasoning, reading comprehension |
| GitHub Issues | Real bug reports and feature requests | Code understanding, causal inference |
| ArXiv Abstracts | Paper comprehension and analysis | Reading comprehension, knowledge integration |
| Tool Use Scenarios | Selection, composition, error recovery | Tool selection, error recovery |

Each task is labeled with:
- **Required skills** (from a taxonomy of 20+ cognitive capabilities)
- **Difficulty level** (trivial → adversarial)
- **Content hash** (for deduplication)

### 2. Skill-Factorized Evaluation

Rather than single accuracy, we produce **capability fingerprints**:

```python
fingerprint = CapabilityFingerprint(
    model_id="gpt-4-turbo",
    skill_scores={
        "logical_deduction": 0.85,
        "mathematical_reasoning": 0.78,
        "causal_inference": 0.72,
        "tool_selection": 0.91,
        "error_recovery": 0.65,
        # ... 15+ more skills
    },
    difficulty_scores={
        "easy": 0.95,
        "medium": 0.82,
        "hard": 0.61,
        "very_hard": 0.38,
    },
    paraphrase_consistency=0.89,
    adversarial_robustness=0.76,
    calibration_error=0.08,
)
```

This enables:
- Fine-grained model comparison
- Targeted capability improvement
- Deployment-specific model selection

### 3. Anti-Gaming Robustness

For each task, we automatically generate:

| Variant Type | Purpose | Expected Behavior |
|--------------|---------|-------------------|
| **Paraphrases** | Same meaning, different wording | Answer should be consistent |
| **Counterfactuals** | Changed premises | Answer should change appropriately |
| **Adversarial** | Perturbations and distractors | Answer should remain correct |

Detected gaming patterns:
- Memorization (high accuracy on originals, low on paraphrases)
- Spurious correlation (answer position bias, keyword triggers)
- Verbosity gaming (excessive length without correctness)
- Confidence gaming (uniformly high confidence regardless of difficulty)

### 4. Multi-Judge Evaluation

No single judge is trusted. We use:

```python
ensemble = EnsembleJudge(
    judges=[
        ExactMatchJudge(),
        RubricJudge(),
        LLMJudge("gpt-4"),
        LLMJudge("claude-3"),
    ],
    aggregation="weighted",  # by calibrated confidence
)
```

Analysis includes:
- **Pairwise agreement rates** between judges
- **Calibration metrics** (ECE, overconfidence rate)
- **Disagreement patterns** (which task types cause disagreement?)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Ayushhgit/LivingBench.git
cd LivingBench

# Install with pip (Python 3.10+)
pip install -e ".[all]"

# Or with specific providers
pip install -e ".[openai,anthropic]"
```

---

## Quick Start

### Run an Experiment

```bash
# Run Groq experiment (set GROQ_API_KEY in .env first)
python scripts/run_groq_experiment.py --n-tasks 50

# View results
cat outputs/groq_experiment/results.json
```

### Programmatic Usage

```python
from livingbench import LivingBenchConfig
from livingbench.tasks import TaskGenerationEngine
from livingbench.evaluation import EvaluationPipeline, SimplePipeline
from livingbench.evaluation.capability_fingerprint import FingerprintComputer

# Generate tasks
engine = TaskGenerationEngine(seed=42)
tasks = engine.generate_balanced(n_tasks=100)

# Evaluate with your model
def my_model(prompt: str) -> str:
    # Your model inference here
    return "model response"

def my_judge(task, response, reference):
    # Your judging logic
    correct = reference.lower() in response.lower()
    return correct, 1.0 if correct else 0.0, "Matched" if correct else "No match"

pipeline = SimplePipeline()
results = pipeline.evaluate_with_model(
    tasks=tasks,
    model_fn=my_model,
    judge_fn=my_judge,
    model_id="my_model",
)

# Compute capability fingerprint
computer = FingerprintComputer()
fingerprint = computer.compute("my_model", results)

print(f"Overall accuracy: {sum(r.is_correct for r in results) / len(results):.1%}")
print(f"Skill scores: {fingerprint.skill_scores}")
```

---

## Configuration

LivingBench uses hierarchical YAML configuration:

```yaml
# configs/experiments/default.yaml
experiment_name: "baseline_comparison"
random_seed: 42

task_generation:
  synthetic_reasoning_enabled: true
  synthetic_math_enabled: true
  tool_use_enabled: true
  synthetic_tasks_count: 200

evaluation:
  n_judges: 3
  judge_models:
    - "gpt-4-turbo"
    - "claude-3-sonnet"
  compute_skill_scores: true
  compute_calibration: true

robustness:
  paraphrase_enabled: true
  n_paraphrases: 3
  adversarial_enabled: true

temporal:
  enabled: false  # Enable for multi-session evaluation
```

---

## Research Applications

### Model Development
- Identify capability gaps to target during training
- Track progress across model versions
- Compare fine-tuning strategies

### Safety Evaluation
- Detect memorization and gaming
- Measure robustness to adversarial inputs
- Identify overconfidence patterns

### Deployment Decisions
- Match model capabilities to application requirements
- Quantify reliability and consistency
- Monitor for capability degradation

---

## Limitations

1. **Synthetic Task Bias**: Generated tasks may not fully represent real-world distributions
2. **Judge Reliability**: LLM judges inherit their own biases and failure modes
3. **Skill Taxonomy**: Our skill categories are approximations of latent cognitive capabilities
4. **Temporal Evaluation**: Currently limited to simulated multi-session scenarios

---

## Future Work

- **Human Calibration**: Collect human judgments to calibrate LLM judges
- **Dynamic Difficulty**: Adaptive testing based on demonstrated capability
- **Cross-Lingual Extension**: Evaluate multilingual capabilities
- **Agentic Evaluation**: Long-horizon tool use and planning tasks
- **Benchmark-of-Benchmarks**: Meta-evaluation against other benchmark suites

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---
