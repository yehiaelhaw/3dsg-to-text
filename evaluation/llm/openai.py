"""openai.py — LLMProvider backed by the OpenAI API."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from evaluation.llm.base import GenerationResult, LLMProvider

load_dotenv()


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, options: dict) -> None:
        super().__init__(model, options)
        api_key = options.pop("api_key", os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("OpenAI API key required: set OPENAI_API_KEY or pass api_key in options")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> GenerationResult:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.options,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        return GenerationResult(
            text=response.choices[0].message.content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
        )
