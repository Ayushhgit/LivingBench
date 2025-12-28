"""
Cost tracking for API calls - tracks tokens and calculates costs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# pricing per 1M tokens - update when prices change
MODEL_PRICES = {
    # groq
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # openai
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # anthropic
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # fallback
    "default": {"input": 1.0, "output": 2.0},
}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostRecord:
    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TokenCounter:
    """Counts tokens - uses tiktoken for openai models, approximates for others"""

    def __init__(self):
        self._tiktoken = None
        self._encoders: dict[str, Any] = {}

    def count(self, text: str, model: str = "default") -> int:
        # use tiktoken for gpt models
        if "gpt" in model.lower():
            return self._count_tiktoken(text, model)
        # otherwise just approximate (~4 chars per token)
        return len(text) // 4 + 1

    def _count_tiktoken(self, text: str, model: str) -> int:
        try:
            if self._tiktoken is None:
                import tiktoken
                self._tiktoken = tiktoken

            if model not in self._encoders:
                try:
                    self._encoders[model] = self._tiktoken.encoding_for_model(model)
                except KeyError:
                    self._encoders[model] = self._tiktoken.get_encoding("cl100k_base")

            return len(self._encoders[model].encode(text))
        except ImportError:
            return len(text) // 4 + 1


class CostTracker:
    """
    Tracks costs for API calls

    Usage:
        tracker = CostTracker()
        tracker.track("llama-70b", input_tokens=500, output_tokens=200)
        print(tracker.total_cost)
    """

    def __init__(self, prices: dict | None = None):
        self.prices = prices or MODEL_PRICES
        self.records: list[CostRecord] = []
        self.counter = TokenCounter()

    def track(
        self,
        model_id: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
    ) -> CostRecord:
        """Track a single API call"""
        # count tokens from text if not provided
        if input_tokens is None and input_text:
            input_tokens = self.counter.count(input_text, model_id)
        if output_tokens is None and output_text:
            output_tokens = self.counter.count(output_text, model_id)

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0

        # calc cost
        price = self.prices.get(model_id, self.prices["default"])
        in_cost = (input_tokens / 1_000_000) * price["input"]
        out_cost = (output_tokens / 1_000_000) * price["output"]

        rec = CostRecord(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=in_cost,
            output_cost=out_cost,
            total_cost=in_cost + out_cost,
        )
        self.records.append(rec)
        return rec

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def get_summary(self) -> dict[str, Any]:
        """Get summary grouped by model"""
        by_model: dict[str, dict] = {}

        for r in self.records:
            if r.model_id not in by_model:
                by_model[r.model_id] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0.0,
                    "calls": 0,
                }
            by_model[r.model_id]["input_tokens"] += r.input_tokens
            by_model[r.model_id]["output_tokens"] += r.output_tokens
            by_model[r.model_id]["total_cost"] += r.total_cost
            by_model[r.model_id]["calls"] += 1

        return {
            "by_model": by_model,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "n_calls": len(self.records),
        }

    # keep old name for compatibility
    def summary(self) -> dict[str, Any]:
        return self.get_summary()

    def save(self, path: str) -> None:
        data = {
            "summary": self.get_summary(),
            "records": [
                {
                    "model_id": r.model_id,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "input_cost": r.input_cost,
                    "output_cost": r.output_cost,
                    "total_cost": r.total_cost,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        s = self.get_summary()
        print("\n" + "=" * 50)
        print("COST SUMMARY")
        print("=" * 50)

        for model, data in s["by_model"].items():
            print(f"\n{model}:")
            print(f"  Calls: {data['calls']}")
            print(f"  Input: {data['input_tokens']:,} tokens")
            print(f"  Output: {data['output_tokens']:,} tokens")
            print(f"  Cost: ${data['total_cost']:.4f}")

        print("\n" + "-" * 50)
        print(f"TOTAL: ${s['total_cost']:.4f} ({s['total_tokens']:,} tokens)")
