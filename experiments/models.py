"""Responder model profiles for the evaluation matrix; each is one configuration
run against every scene. A `_screening` suffix means the directory is judged by
gemma2:9b only; a bare name means Gemini (one judge per directory).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str                       # short id; becomes results/<name>/<scene>/
    backend: str                    # "ollama" | "gemini" | ...
    model: str                      # backend-specific model id
    options: dict = field(default_factory=dict)


# Remote ollama endpoints -- GPU servers reached over an SSH local port-forward,
# which must be up before a run that uses them.
#   _VOXEL -> RTX 3090, 24GB VRAM
#   _PIXEL -> RTX 5070 Ti, 16GB VRAM
#
# Every profile is pinned to _VOXEL: _PIXEL's card is contended by another job.
# Check free VRAM before moving one back -- ollama does not refuse a card that is
# too full, it silently falls back to system RAM and answers ~150x slower, which
# presents as a hung run rather than an error.
_VOXEL = "http://localhost:11434"
_PIXEL = "http://localhost:11435"  # unused -- see above

MODEL_PROFILES: list[ModelProfile] = [
    # --- standard instruction families: greedy decoding (temperature 0) ---
    ModelProfile("qwen2.5-14b", "ollama", "qwen2.5:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Scale rung below the primary; with 14b/32b forms the within-family scale
    # curve (screening-tier ablation).
    ModelProfile("qwen2.5-7b_screening", "ollama", "qwen2.5:7b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # 32B's KV cache spills past 24GB (partial CPU offload, slower); num_ctx stays
    # matched to the other profiles so context_exceeded cells stay comparable.
    ModelProfile("qwen2.5-32b_screening", "ollama", "qwen2.5:32b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("llama3.1-8b", "ollama", "llama3.1:8b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Second responder for the headline-cell robustness check, lineage-independent
    # of the primary responder and both judges. Gemini-judged only -- use the
    # _screening profile below for anything screening-tier.
    ModelProfile("mistral-nemo-12b", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Screening-tier coverage-gap fill, kept in its own directory so it never
    # mixes with the Gemini-judged mistral-nemo-12b rows (one judge per directory).
    ModelProfile("mistral-nemo-12b_screening", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # Judge-tier validation, NOT new responder points: re-judges mistral-nemo-12b's
    # already-generated responses.jsonl under a different judge via --score-only
    # (_gemini2 = same-judge rerun, _gpt41 = cross-vendor); never read by
    # aggregate_results.py.
    ModelProfile("mistral-nemo-12b_gemini2", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("mistral-nemo-12b_gpt41", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # --- reasoning family (branch ablation): nonzero floor; r1 degrades at 0 ---
    ModelProfile("deepseek-r1-14b_screening", "ollama", "deepseek-r1:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0.6}),
]
