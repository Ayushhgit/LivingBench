"""
Core type definitions for LivingBench.

This module defines the foundational data structures used throughout the evaluation
system. All types are immutable (frozen) Pydantic models to ensure reproducibility
and prevent accidental mutation during evaluation.

Design Principles:
- Types are semantic, not just structural
- Every field has explicit meaning and constraints
- Serialization is built-in for logging/caching
- Type hints enable static analysis across the codebase
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field


class Skill(str, Enum):
    """
    Latent cognitive skills required for task completion.

    These skills are not mutually exclusive—a single task may require
    multiple skills. The skill taxonomy is designed to be:
    - Exhaustive: covers all major LLM capabilities
    - Discriminative: different models have different skill profiles
    - Measurable: each skill can be isolated and tested
    """
    # Core reasoning
    LOGICAL_DEDUCTION = "logical_deduction"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CAUSAL_INFERENCE = "causal_inference"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"

    # Knowledge and retrieval
    FACTUAL_RECALL = "factual_recall"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    TEMPORAL_REASONING = "temporal_reasoning"

    # Language understanding
    READING_COMPREHENSION = "reading_comprehension"
    INSTRUCTION_FOLLOWING = "instruction_following"
    AMBIGUITY_RESOLUTION = "ambiguity_resolution"

    # Generation and planning
    MULTI_STEP_PLANNING = "multi_step_planning"
    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    STRUCTURED_OUTPUT = "structured_output"

    # Tool use
    TOOL_SELECTION = "tool_selection"
    TOOL_COMPOSITION = "tool_composition"
    ERROR_RECOVERY = "error_recovery"

    # Meta-cognition
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    SELF_CORRECTION = "self_correction"
    ABSTENTION = "abstention"  # knowing when not to answer


class DifficultyLevel(str, Enum):
    """
    Task difficulty calibrated against human and model baselines.

    Difficulty is not just about correctness rate—it considers:
    - Cognitive load (working memory requirements)
    - Solution path length
    - Ambiguity and underspecification
    - Required domain knowledge
    """
    TRIVIAL = "trivial"       # >95% accuracy for frontier models
    EASY = "easy"             # 80-95% accuracy
    MEDIUM = "medium"         # 50-80% accuracy
    HARD = "hard"             # 20-50% accuracy
    VERY_HARD = "very_hard"   # <20% accuracy
    ADVERSARIAL = "adversarial"  # Specifically designed to exploit weaknesses


class TaskSource(str, Enum):
    """Origin of the evaluation task."""
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"
    ARXIV_ABSTRACT = "arxiv_abstract"
    STACKOVERFLOW = "stackoverflow"
    SYNTHETIC_REASONING = "synthetic_reasoning"
    SYNTHETIC_MATH = "synthetic_math"
    TOOL_USE_SCENARIO = "tool_use_scenario"
    ADVERSARIAL_GENERATED = "adversarial_generated"
    HUMAN_CURATED = "human_curated"


class TaskFormat(str, Enum):
    """Expected response format."""
    FREE_TEXT = "free_text"
    MULTIPLE_CHOICE = "multiple_choice"
    CODE = "code"
    STRUCTURED_JSON = "structured_json"
    TOOL_CALLS = "tool_calls"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"


class ToolDefinition(BaseModel):
    """Definition of a tool available during evaluation."""
    name: str
    description: str
    parameters: dict[str, Any]
    required_params: list[str] = Field(default_factory=list)
    returns: str = "string"

    class Config:
        frozen = True


class Task(BaseModel):
    """
    A single evaluation task in LivingBench.

    Tasks are the atomic unit of evaluation. Each task is:
    - Self-contained: includes all context needed for evaluation
    - Labeled: has explicit skill and difficulty annotations
    - Versioned: tracked with unique ID and generation timestamp
    - Reproducible: deterministic given the same random seed

    The task format supports:
    - Free-form QA
    - Multiple choice
    - Code generation
    - Tool use scenarios
    - Multi-turn interactions
    """
    # Identity
    id: UUID = Field(default_factory=uuid4)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Source and provenance
    source: TaskSource
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    # Task content
    prompt: str
    context: str | None = None  # Additional context (document, code, etc.)
    format: TaskFormat = TaskFormat.FREE_TEXT

    # For multiple choice
    choices: list[str] | None = None

    # Ground truth
    reference_answer: str | None = None
    reference_explanation: str | None = None

    # For code/tool tasks
    available_tools: list[ToolDefinition] = Field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] | None = None

    # Skill and difficulty labels
    required_skills: list[Skill]
    difficulty: DifficultyLevel

    # Robustness variants (populated by robustness layer)
    is_variant: bool = False
    parent_task_id: UUID | None = None
    variant_type: str | None = None  # "paraphrase", "counterfactual", "adversarial"

    # Validation metadata
    human_validated: bool = False
    validation_notes: str | None = None

    class Config:
        frozen = True

    @computed_field
    @property
    def content_hash(self) -> str:
        """Deterministic hash of task content for deduplication."""
        content = f"{self.prompt}|{self.context}|{self.reference_answer}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: list[str] | None, info) -> list[str] | None:
        if info.data.get("format") == TaskFormat.MULTIPLE_CHOICE:
            if not v or len(v) < 2:
                raise ValueError("Multiple choice tasks require at least 2 choices")
        return v


class ToolCall(BaseModel):
    """A single tool invocation by the model."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    latency_ms: float | None = None

    class Config:
        frozen = True


class ModelResponse(BaseModel):
    """
    Complete response from an evaluated model.

    Captures not just the output but also:
    - Thinking/reasoning traces (if available)
    - Tool usage patterns
    - Token counts and latency
    - Confidence signals
    """
    # Identity
    id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    model_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Response content
    raw_output: str
    parsed_answer: str | None = None  # Extracted final answer
    reasoning_trace: str | None = None  # Chain-of-thought if available

    # Tool use
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Metadata
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None

    # Self-reported confidence (if model provides it)
    stated_confidence: float | None = None

    # For multi-turn
    turn_number: int = 1
    session_id: UUID | None = None

    class Config:
        frozen = True

    @computed_field
    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens and self.completion_tokens:
            return self.prompt_tokens + self.completion_tokens
        return None


class JudgeVerdict(BaseModel):
    """
    Verdict from a single judge on a model response.

    Each judge provides:
    - Binary correctness (for aggregation)
    - Scalar score (for fine-grained comparison)
    - Rationale (for interpretability)
    - Confidence (for calibration analysis)
    """
    judge_id: str
    judge_model: str | None = None  # For LLM judges

    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)

    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Specific failure modes detected
    detected_issues: list[str] = Field(default_factory=list)

    # For skill-specific scoring
    skill_scores: dict[str, float] = Field(default_factory=dict)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float | None = None

    class Config:
        frozen = True


class EvaluationResult(BaseModel):
    """
    Complete evaluation result for a single task-response pair.

    Aggregates verdicts from multiple judges and computes:
    - Consensus correctness
    - Agreement metrics
    - Skill-factorized scores
    """
    id: UUID = Field(default_factory=uuid4)
    task: Task
    response: ModelResponse

    # Individual judge verdicts
    verdicts: list[JudgeVerdict]

    # Aggregated results
    is_correct: bool  # Majority vote
    agreement_ratio: float  # Proportion of judges agreeing with majority
    mean_score: float
    score_std: float

    # Skill-factorized scores
    skill_scores: dict[str, float] = Field(default_factory=dict)

    # Detected issues (union across judges)
    detected_issues: list[str] = Field(default_factory=list)

    # Metadata
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class CapabilityFingerprint(BaseModel):
    """
    Multi-dimensional capability profile for a model.

    This replaces single-scalar accuracy with a rich capability vector
    that enables:
    - Fine-grained model comparison
    - Weakness identification
    - Progress tracking over time
    """
    model_id: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Core skill scores (0-1)
    skill_scores: dict[str, float]

    # Difficulty breakdown
    difficulty_scores: dict[str, float]

    # Robustness metrics
    paraphrase_consistency: float = Field(ge=0.0, le=1.0)
    counterfactual_sensitivity: float = Field(ge=0.0, le=1.0)  # Higher = more sensitive
    adversarial_robustness: float = Field(ge=0.0, le=1.0)

    # Calibration
    calibration_error: float  # ECE or similar
    overconfidence_rate: float
    abstention_rate: float

    # Tool use metrics
    tool_selection_accuracy: float | None = None
    tool_execution_success_rate: float | None = None
    unnecessary_tool_use_rate: float | None = None

    # Meta statistics
    n_tasks_evaluated: int
    n_tasks_per_skill: dict[str, int] = Field(default_factory=dict)

    class Config:
        frozen = True

    def to_vector(self, skill_order: list[str] | None = None) -> np.ndarray:
        """Convert to numpy vector for ML analysis."""
        if skill_order is None:
            skill_order = sorted(self.skill_scores.keys())
        return np.array([self.skill_scores.get(s, 0.0) for s in skill_order])

    def distance_to(self, other: CapabilityFingerprint) -> float:
        """Compute capability distance to another fingerprint."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        # Pad to same length
        max_len = max(len(v1), len(v2))
        v1 = np.pad(v1, (0, max_len - len(v1)))
        v2 = np.pad(v2, (0, max_len - len(v2)))
        return float(np.linalg.norm(v1 - v2))


class RobustnessResult(BaseModel):
    """Results from robustness testing on a task."""
    original_task_id: UUID

    # Variant results
    paraphrase_results: list[EvaluationResult] = Field(default_factory=list)
    counterfactual_results: list[EvaluationResult] = Field(default_factory=list)
    adversarial_results: list[EvaluationResult] = Field(default_factory=list)

    # Computed metrics
    paraphrase_consistency: float  # Agreement across paraphrases
    counterfactual_flip_rate: float  # Rate of answer changes
    adversarial_success_rate: float  # Attack success rate

    # Detected gaming patterns
    detected_memorization: bool = False
    memorization_evidence: str | None = None

    detected_spurious_correlation: bool = False
    spurious_correlation_evidence: str | None = None

    class Config:
        frozen = True


class SessionState(BaseModel):
    """State tracked across multi-turn evaluation sessions."""
    session_id: UUID = Field(default_factory=uuid4)
    model_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)

    # Accumulated context
    task_history: list[UUID] = Field(default_factory=list)
    response_history: list[UUID] = Field(default_factory=list)

    # Error tracking
    errors_made: list[dict[str, Any]] = Field(default_factory=list)
    corrections_attempted: list[dict[str, Any]] = Field(default_factory=list)

    # Memory updates (for models that support it)
    memory_updates: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        frozen = True


class TemporalTrace(BaseModel):
    """
    Learning-over-time evaluation trace.

    Tracks model performance across multiple sessions to measure:
    - Error correction capability
    - Knowledge retention
    - Performance degradation
    """
    model_id: str
    trace_id: UUID = Field(default_factory=uuid4)

    # Session sequence
    sessions: list[SessionState] = Field(default_factory=list)

    # Aggregated temporal metrics
    error_correction_rate: float | None = None
    knowledge_retention_rate: float | None = None
    performance_trend: Literal["improving", "stable", "degrading"] | None = None

    # Specific failure patterns
    persistent_errors: list[dict[str, Any]] = Field(default_factory=list)
    learned_corrections: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        frozen = True


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""
    experiment_id: str
    name: str
    description: str

    # Models to evaluate
    models: list[str]

    # Task configuration
    task_sources: list[TaskSource]
    n_tasks_per_source: int = 100
    difficulty_distribution: dict[str, float] | None = None

    # Evaluation configuration
    n_judges: int = 3
    judge_models: list[str] = Field(default_factory=lambda: ["gpt-4", "claude-3-opus"])

    # Robustness testing
    enable_paraphrase: bool = True
    n_paraphrases: int = 3
    enable_counterfactual: bool = True
    enable_adversarial: bool = True

    # Temporal evaluation
    enable_temporal: bool = False
    n_sessions: int = 3
    tasks_per_session: int = 10

    # Reproducibility
    random_seed: int = 42

    # Output
    output_dir: str = "outputs"
    save_intermediate: bool = True

    class Config:
        frozen = True


class ExperimentResults(BaseModel):
    """Complete results from an experiment run."""
    config: ExperimentConfig
    started_at: datetime
    completed_at: datetime

    # Per-model fingerprints
    fingerprints: dict[str, CapabilityFingerprint]

    # All evaluation results
    evaluation_results: list[EvaluationResult]

    # Robustness results
    robustness_results: list[RobustnessResult]

    # Temporal traces (if enabled)
    temporal_traces: list[TemporalTrace] = Field(default_factory=list)

    # Summary statistics
    summary: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True
