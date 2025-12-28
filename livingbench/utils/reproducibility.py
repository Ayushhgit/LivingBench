"""
Reproducibility utilities for LivingBench.

Ensures deterministic execution where possible:
- Random seed management
- Deterministic hashing
- Environment capture
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from typing import Any


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Sets seed for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)

    Args:
        seed: Random seed value
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Set PYTHONHASHSEED for deterministic hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_deterministic_hash(content: str) -> str:
    """
    Get a deterministic hash of string content.

    Uses SHA-256 for consistency across runs and platforms.

    Args:
        content: String to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(content.encode()).hexdigest()


def get_short_hash(content: str, length: int = 8) -> str:
    """
    Get a short deterministic hash.

    Args:
        content: String to hash
        length: Number of hex characters to return

    Returns:
        Short hexadecimal hash string
    """
    return get_deterministic_hash(content)[:length]


def capture_environment() -> dict[str, Any]:
    """
    Capture environment information for reproducibility.

    Returns:
        Dictionary with environment details
    """
    env = {
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Capture package versions
    try:
        import pkg_resources
        packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
            if pkg.key in [
                "numpy", "scipy", "pandas", "pydantic",
                "openai", "anthropic", "httpx",
            ]
        }
        env["packages"] = packages
    except ImportError:
        pass

    # Capture relevant environment variables
    relevant_vars = [
        "PYTHONHASHSEED",
        "LIVINGBENCH_SEED",
        "OPENAI_API_KEY",  # Just presence, not value
        "ANTHROPIC_API_KEY",
    ]

    env_vars = {}
    for var in relevant_vars:
        if var in os.environ:
            if "KEY" in var:
                env_vars[var] = "***SET***"
            else:
                env_vars[var] = os.environ[var]
    env["environment_variables"] = env_vars

    return env


class DeterministicContext:
    """
    Context manager for deterministic execution.

    Usage:
        with DeterministicContext(seed=42):
            # Deterministic operations here
            ...
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        # Save current random state
        self.original_state = random.getstate()
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state:
            random.setstate(self.original_state)
