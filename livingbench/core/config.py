"""
Configuration management for LivingBench.

Provides hierarchical configuration with:
- YAML file loading
- Environment variable overrides
- Validation and defaults
- Experiment-specific configs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class TaskGenerationConfig(BaseModel):
    """Configuration for task generation."""
    # GitHub source
    github_enabled: bool = True
    github_repos: list[str] = Field(default_factory=lambda: [
        "python/cpython",
        "pytorch/pytorch",
        "huggingface/transformers",
    ])
    github_issues_per_repo: int = 50
    github_min_issue_length: int = 100

    # ArXiv source
    arxiv_enabled: bool = True
    arxiv_categories: list[str] = Field(default_factory=lambda: [
        "cs.CL", "cs.LG", "cs.AI"
    ])
    arxiv_papers_per_category: int = 50

    # Synthetic sources
    synthetic_reasoning_enabled: bool = True
    synthetic_math_enabled: bool = True
    synthetic_tasks_count: int = 200

    # Tool use scenarios
    tool_use_enabled: bool = True
    tool_scenarios_count: int = 100

    # Caching
    cache_dir: str = "data/cache"
    cache_ttl_hours: int = 24


class EvaluationConfig(BaseModel):
    """Configuration for evaluation pipeline."""
    # Judges
    n_judges: int = 3
    judge_models: list[str] = Field(default_factory=lambda: [
        "gpt-4-turbo",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    ])
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024

    # Aggregation
    require_unanimous: bool = False
    min_agreement_threshold: float = 0.5

    # Skill decomposition
    compute_skill_scores: bool = True

    # Calibration
    compute_calibration: bool = True
    calibration_bins: int = 10


class RobustnessConfig(BaseModel):
    """Configuration for robustness testing."""
    # Paraphrase
    paraphrase_enabled: bool = True
    n_paraphrases: int = 3
    paraphrase_model: str = "gpt-4-turbo"

    # Counterfactual
    counterfactual_enabled: bool = True
    n_counterfactuals: int = 2
    counterfactual_model: str = "gpt-4-turbo"

    # Adversarial
    adversarial_enabled: bool = True
    adversarial_strategies: list[str] = Field(default_factory=lambda: [
        "character_perturbation",
        "semantic_trap",
        "instruction_injection",
    ])

    # Detection thresholds
    memorization_threshold: float = 0.95
    spurious_correlation_threshold: float = 0.8


class TemporalConfig(BaseModel):
    """Configuration for temporal evaluation."""
    enabled: bool = False
    n_sessions: int = 5
    tasks_per_session: int = 20
    session_interval_hours: int = 24

    # What to track
    track_error_correction: bool = True
    track_memory_updates: bool = True
    track_performance_trend: bool = True


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str
    provider: str  # "openai", "anthropic", "local"
    model_id: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 120

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000


class LivingBenchConfig(BaseModel):
    """
    Master configuration for LivingBench.

    Hierarchical configuration that can be:
    - Loaded from YAML files
    - Overridden by environment variables
    - Extended for specific experiments
    """
    # Experiment identity
    experiment_name: str = "default"
    experiment_description: str = ""
    random_seed: int = 42

    # Component configs
    task_generation: TaskGenerationConfig = Field(default_factory=TaskGenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)

    # Models to evaluate
    models: list[ModelConfig] = Field(default_factory=list)

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    log_dir: str = "logs"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    # Execution
    max_concurrent_requests: int = 10
    save_intermediate_results: bool = True
    resume_from_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> LivingBenchConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {path}: {e}") from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading config file {path}") from e

        if data is None:
            data = {}

        return cls.model_validate(data)

    @classmethod
    def from_yaml_with_overrides(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> LivingBenchConfig:
        """Load from YAML with dictionary overrides."""
        config = cls.from_yaml(path)

        if overrides:
            # Deep merge overrides
            config_dict = config.model_dump()
            cls._deep_merge(config_dict, overrides)
            config = cls.model_validate(config_dict)

        return config

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                LivingBenchConfig._deep_merge(base[key], value)
            else:
                base[key] = value

    def apply_env_overrides(self) -> LivingBenchConfig:
        """Apply environment variable overrides."""
        updates = {}

        # Check for common overrides
        if seed := os.getenv("LIVINGBENCH_SEED"):
            updates["random_seed"] = int(seed)

        if output := os.getenv("LIVINGBENCH_OUTPUT_DIR"):
            updates["output_dir"] = output

        if log_level := os.getenv("LIVINGBENCH_LOG_LEVEL"):
            updates["log_level"] = log_level

        if updates:
            return self.model_copy(update=updates)
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        except PermissionError as e:
            raise PermissionError(f"Permission denied writing config file {path}") from e
        except OSError as e:
            raise OSError(f"Failed to write config file {path}: {e}") from e

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LivingBenchConfig:
    """
    Load configuration with sensible defaults.

    Priority (highest to lowest):
    1. Environment variable overrides
    2. Explicit overrides dict
    3. Config file values
    4. Default values
    """
    if config_path:
        config = LivingBenchConfig.from_yaml_with_overrides(config_path, overrides)
    else:
        # Use defaults with any overrides
        if overrides:
            config = LivingBenchConfig.model_validate(overrides)
        else:
            config = LivingBenchConfig()

    return config.apply_env_overrides()
