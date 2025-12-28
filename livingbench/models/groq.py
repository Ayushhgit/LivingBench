"""
Groq API adapter - for fast llama/mixtral inference
"""

from __future__ import annotations

import os
from typing import Any

from livingbench.models.base import ModelAdapter


class GroqModel(ModelAdapter):
    """
    Groq API wrapper

    Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it

    Usage:
        model = GroqModel("llama-3.3-70b-versatile")
        resp = model.generate("What is 2+2?")
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq not installed. pip install groq")

        self._model_id = model_name
        self.temp = temperature
        self.max_tok = max_tokens
        self.extra_args = kwargs

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=key)

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, **kwargs: Any) -> str:
        args = {**self.extra_args, **kwargs}

        resp = self.client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.get("temperature", self.temp),
            max_tokens=args.get("max_tokens", self.max_tok),
            **{k: v for k, v in args.items() if k not in ("temperature", "max_tokens")},
        )
        return resp.choices[0].message.content or ""


class GroqModelWithSystem(GroqModel):
    """Same as GroqModel but with a system prompt"""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.sys_prompt = system_prompt

    def generate(self, prompt: str, **kwargs: Any) -> str:
        args = {**self.extra_args, **kwargs}

        resp = self.client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=args.get("temperature", self.temp),
            max_tokens=args.get("max_tokens", self.max_tok),
            **{k: v for k, v in args.items() if k not in ("temperature", "max_tokens")},
        )
        return resp.choices[0].message.content or ""
