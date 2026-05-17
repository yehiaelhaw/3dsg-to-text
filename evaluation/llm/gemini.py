"""gemini.py — LLMProvider backed by Google Gemini (google-genai)."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from evaluation.llm.base import GenerationResult, LLMProvider

load_dotenv()


class GeminiProvider(LLMProvider):
    def __init__(self, model: str, options: dict) -> None:
        super().__init__(model, options)
        api_key = options.pop("api_key", os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            raise ValueError("Gemini API key required: set GEMINI_API_KEY or pass api_key in options")
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> GenerationResult:
        config = types.GenerateContentConfig(**self.options) if self.options else None

        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage_metadata
        return GenerationResult(
            text=response.text,
            prompt_tokens=usage.prompt_token_count or 0,
            completion_tokens=usage.candidates_token_count or 0,
            latency_ms=round(latency_ms, 2),
        )
