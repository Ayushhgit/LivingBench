"""
Base model adapter interface.

Model adapters wrap different LLM providers (OpenAI, Anthropic, local)
with a consistent interface for evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any

from livingbench.core.types import Task, ModelResponse, ToolCall


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Subclasses implement the specific API calls for different providers.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        ...

    def generate_for_task(
        self,
        task: Task,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate a response for a specific task.

        This method handles:
        - Constructing the full prompt
        - Parsing tool calls if applicable
        - Creating the ModelResponse object

        Args:
            task: The task to respond to
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with parsed output
        """
        # Build prompt
        prompt = self._construct_prompt(task)

        # Generate
        raw_output = self.generate(prompt, **kwargs)

        # Parse tool calls if applicable
        tool_calls = []
        if task.available_tools:
            tool_calls = self._parse_tool_calls(raw_output, task)

        return ModelResponse(
            task_id=task.id,
            model_id=self.model_id,
            raw_output=raw_output,
            parsed_answer=self._extract_answer(raw_output),
            tool_calls=tool_calls,
        )

    def _construct_prompt(self, task: Task) -> str:
        """Construct the full prompt for a task."""
        parts = []

        if task.context:
            parts.append(f"Context:\n{task.context}\n")

        if task.available_tools:
            tools_desc = "\n".join(
                f"- {t.name}: {t.description}"
                for t in task.available_tools
            )
            parts.append(f"Available tools:\n{tools_desc}\n")

        parts.append(task.prompt)

        return "\n".join(parts)

    def _extract_answer(self, output: str) -> str:
        """Extract the final answer from model output."""
        # Simple extraction - can be overridden for specific formats
        lines = output.strip().split('\n')
        return lines[-1] if lines else output

    def _parse_tool_calls(self, output: str, task: Task) -> list[ToolCall]:
        """Parse tool calls from model output."""
        tool_calls = []

        # Try to find JSON tool call
        json_match = re.search(r'\{[^{}]*"tool_call"[^{}]*\}', output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "tool_call" in data:
                    tc = data["tool_call"]
                    tool_calls.append(ToolCall(
                        tool_name=tc.get("name", "unknown"),
                        arguments=tc.get("arguments", {}),
                    ))
            except json.JSONDecodeError:
                pass

        return tool_calls
