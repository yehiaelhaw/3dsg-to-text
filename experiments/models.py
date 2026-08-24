"""Responder model profiles for the evaluation matrix."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str                       # short id; becomes results/<name>/<scene>/
    backend: str                    # "ollama" | "gemini" | ...
    model: str                      # backend-specific model id
    options: dict = field(default_factory=dict)


# Ollama endpoints via SSH forwarding. Profiles use _VOXEL because _PIXEL is shared.
_VOXEL = "http://localhost:11434"
_PIXEL = "http://localhost:11435"  # unused

MODEL_PROFILES: list[ModelProfile] = [
    # --- standard instruction families: greedy decoding (temperature 0) ---
    ModelProfile("qwen2.5-14b", "ollama", "qwen2.5:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Lower Qwen scale point for screening.
    ModelProfile("qwen2.5-7b_screening", "ollama", "qwen2.5:7b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # 32B partially offloads on _VOXEL; keep num_ctx matched across profiles.
    ModelProfile("qwen2.5-32b_screening", "ollama", "qwen2.5:32b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("llama3.1-8b", "ollama", "llama3.1:8b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Independent responder for headline robustness checks; Gemini judged.
    ModelProfile("mistral-nemo-12b", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Separate screening directory to avoid mixing judge tiers.
    ModelProfile("mistral-nemo-12b_screening", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # Re-judging profiles for judge-tier validation; not new responder runs.
    ModelProfile("mistral-nemo-12b_gemini2", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("mistral-nemo-12b_gpt41", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # Reasoning-family screening profile; DeepSeek-R1 requires nonzero temperature.
    ModelProfile("deepseek-r1-14b_screening", "ollama", "deepseek-r1:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0.6}),
]
