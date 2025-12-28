"""
LivingBench: A Continually Updating, Self-Auditing Benchmark for LLM Evaluation

This package provides a research-grade evaluation system for large language models
that emphasizes:
- Dynamic task generation (benchmark never freezes)
- Skill-factorized evaluation (capability fingerprints)
- Anti-gaming robustness (paraphrase, counterfactual, adversarial)
- Learning-over-time measurement
- Multi-judge calibrated evaluation
"""

__version__ = "0.1.0"
__author__ = "Ayush"

from livingbench.core.types import (
    Task,
    TaskSource,
    Skill,
    DifficultyLevel,
    ModelResponse,
    EvaluationResult,
    CapabilityFingerprint,
)
from livingbench.core.config import LivingBenchConfig

__all__ = [
    "Task",
    "TaskSource",
    "Skill",
    "DifficultyLevel",
    "ModelResponse",
    "EvaluationResult",
    "CapabilityFingerprint",
    "LivingBenchConfig",
]
