"""ollama.py — LLMProvider backed by a local Ollama server."""

from __future__ import annotations

import re
import time

import ollama

from evaluation.llm.base import GenerationResult, LLMProvider

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, options: dict) -> None:
        super().__init__(model, options)
        host = options.pop("host", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def generate(self, prompt: str) -> GenerationResult:
        t0 = time.perf_counter()
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options=self.options or None,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.response
        # Reasoning models (deepseek-r1) inline a <think>...</think> trace in
        # the response. The judge must only ever see the final answer, so the
        # trace is stripped; an unclosed trace (generation cut off mid-think)
        # leaves no final answer, which is stored as such. completion_tokens
        # still counts the trace -- thinking stays in the cost accounting.
        if "<think>" in text:
            text = _THINK_RE.sub("", text)
            if "<think>" in text:
                text = text.split("<think>", 1)[0]
            text = text.lstrip()

        return GenerationResult(
            text=text,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            latency_ms=round(latency_ms, 2),
        )

    def unload(self) -> None:
        """Ask the Ollama server to evict this model and free its VRAM now.

        keep_alive=0 unloads the model after the (empty) request completes, so
        the judge can claim the full GPU instead of thrashing against a still-
        resident responder. Best-effort: if it fails, the judge load will evict
        the responder anyway (one swap instead of zero)."""
        try:
            self._client.generate(model=self.model, prompt="", keep_alive=0)
        except Exception:
            pass
