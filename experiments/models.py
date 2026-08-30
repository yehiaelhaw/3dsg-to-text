"""Responder model profiles for the evaluation matrix."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str                       # short id; becomes results/<name>/<scene>/
    backend: str                    # "ollama" | "gemini" | ...
    model: str                      # backend-specific model id
    options: dict = field(default_factory=dict)


# Ollama endpoint (the local Ollama default port).
HOST = "http://localhost:11434"

MODEL_PROFILES: list[ModelProfile] = [
    # --- standard instruction families: greedy decoding (temperature 0) ---
    ModelProfile("qwen2.5-14b", "ollama", "qwen2.5:14b",
                 {"host": HOST, "num_ctx": 32768, "temperature": 0}),
    # Lower Qwen scale point.
    ModelProfile("qwen2.5-7b", "ollama", "qwen2.5:7b",
                 {"host": HOST, "num_ctx": 32768, "temperature": 0}),
    # 32B partially offloads on HOST; keep num_ctx matched across profiles.
    ModelProfile("qwen2.5-32b", "ollama", "qwen2.5:32b",
                 {"host": HOST, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("mistral-nemo-12b", "ollama", "mistral-nemo:12b",
                 {"host": HOST, "num_ctx": 32768, "temperature": 0}),
    # Reasoning family; DeepSeek-R1 requires nonzero temperature.
    ModelProfile("deepseek-r1-14b", "ollama", "deepseek-r1:14b",
                 {"host": HOST, "num_ctx": 32768, "temperature": 0.6}),
]
