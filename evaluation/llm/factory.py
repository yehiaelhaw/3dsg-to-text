"""factory.py — Instantiate an LLMProvider by backend name."""

from __future__ import annotations

from evaluation.llm.base import LLMProvider


def create_provider(backend: str, model: str, options: dict) -> LLMProvider:
    if backend == "ollama":
        from evaluation.llm.ollama import OllamaProvider
        return OllamaProvider(model, options)
    if backend == "gemini":
        from evaluation.llm.gemini import GeminiProvider
        return GeminiProvider(model, options)
    if backend == "openai":
        from evaluation.llm.openai import OpenAIProvider
        return OpenAIProvider(model, options)
    raise ValueError(f"Unknown LLM backend: '{backend}'")
