"""models.py -- responder model profiles for the evaluation matrix.

Each profile is one responder configuration. The matrix driver
(experiments/run_experiments.py) runs every profile against every scene, writing to
results/<profile.name>/<scene_id>/. The judge is NOT swept here -- gemma2:9b
screens during the run; re-judge with Gemini later via a score_only pass.

Temperature policy: temperature is a per-model knob set to the lowest each family
runs stably at -- it is not the variable under study, so each single draw should
be the model's representative (modal) answer. Standard instruction families use
greedy decoding (temperature 0); reasoning models that degrade at 0 use their
recommended floor (deepseek-r1 ~0.6, kept to its own branch ablation).

Edit MODEL_PROFILES to the models you actually have available. The qwen/llama
entries below are placeholders -- swap in your real model ids/hosts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str                       # short id; becomes results/<name>/<scene>/
    backend: str                    # "ollama" | "gemini" | ...
    model: str                      # backend-specific model id
    options: dict = field(default_factory=dict)


# Remote ollama endpoint currently used for the larger responders.
_NGROK = "https://macaroni-nervous-busily.ngrok-free.dev"

MODEL_PROFILES: list[ModelProfile] = [
    # --- standard instruction families: greedy decoding (temperature 0) ---
    # (placeholders -- replace with the families you intend to compare)
    ModelProfile("qwen2.5-14b", "ollama", "qwen2.5:14b",
                 {"host": _NGROK, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("llama3.1-8b", "ollama", "llama3.1:8b",
                 {"num_ctx": 32768, "temperature": 0}),

    # --- reasoning family (branch ablation): nonzero floor; r1 degrades at 0 ---
    ModelProfile("deepseek-r1-14b", "ollama", "deepseek-r1:14b",
                 {"host": _NGROK, "num_ctx": 32768, "temperature": 0.6}),
]
