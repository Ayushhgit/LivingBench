"""Model adapters for LivingBench evaluation."""

from livingbench.models.base import ModelAdapter

# Lazy imports for optional dependencies
def get_groq_model():
    """Get GroqModel class (requires groq package)."""
    from livingbench.models.groq import GroqModel, GroqModelWithSystem
    return GroqModel, GroqModelWithSystem

__all__ = [
    "ModelAdapter",
    "get_groq_model",
]
