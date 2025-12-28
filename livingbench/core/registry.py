"""
Component registry for LivingBench.

Provides a clean way to register and retrieve:
- Task generators
- Judges
- Model adapters
- Robustness transformers

This enables modular extension without modifying core code.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for named components.

    Usage:
        task_generators = Registry[TaskGenerator]("task_generators")

        @task_generators.register("github")
        class GitHubTaskGenerator(TaskGenerator):
            ...

        generator = task_generators.get("github")
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type[T]] = {}
        self._instances: dict[str, T] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class."""
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {self.name}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def register_class(self, name: str, cls: type[T]) -> None:
        """Directly register a class (non-decorator)."""
        if name in self._registry:
            raise ValueError(f"Component '{name}' already registered in {self.name}")
        self._registry[name] = cls

    def get_class(self, name: str) -> type[T]:
        """Get registered class by name."""
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"Component '{name}' not found in {self.name}. "
                f"Available: {available}"
            )
        return self._registry[name]

    def get(self, name: str, **kwargs: Any) -> T:
        """Get or create instance by name."""
        if name not in self._instances:
            cls = self.get_class(name)
            self._instances[name] = cls(**kwargs)
        return self._instances[name]

    def create(self, name: str, **kwargs: Any) -> T:
        """Always create new instance."""
        cls = self.get_class(name)
        return cls(**kwargs)

    def list_registered(self) -> list[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# Global registries
task_generators = Registry["TaskGeneratorBase"]("task_generators")
judges = Registry["JudgeBase"]("judges")
model_adapters = Registry["ModelAdapter"]("model_adapters")
robustness_transformers = Registry["RobustnessTransformer"]("robustness_transformers")
