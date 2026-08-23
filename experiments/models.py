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

MODEL_PROFILES below is the committed responder set: qwen2.5-14b (primary, full
matrix) and mistral-nemo-12b (second responder, headline cells); the rest are
contingency/ablation options that run only when their trigger conditions are met.

Naming convention: a `_screening` suffix means the profile's directory is judged
by gemma2:9b and nothing else; a bare name means it is Gemini-judged
(confirmatory). The suffix makes the one-judge-per-directory invariant readable
from `ls` rather than only from the `judge` column. If a screening profile is
later promoted, the Gemini pass gets a NEW bare-named profile/directory -- never a
second judge in the `_screening` one. llama3.1-8b keeps a bare name because it is
dormant and its tier is undecided.
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
    # Scale rung below the primary: with 14b/32b it forms the within-family
    # scale curve (screening-tier ablation; confirmatory only per the
    # contingency rule).
    ModelProfile("qwen2.5-7b_screening", "ollama", "qwen2.5:7b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # 32B fits voxel only (Q4 weights ~20GB); 32k-token KV cache spills past
    # 24GB, so ollama partially offloads to CPU -- slower, but num_ctx stays
    # matched to the other profiles so context_exceeded cells stay comparable.
    ModelProfile("qwen2.5-32b_screening", "ollama", "qwen2.5:32b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("llama3.1-8b", "ollama", "llama3.1:8b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Second responder for the headline-cell robustness check: dense 12B
    # scale-matched to qwen2.5-14b, lineage-independent of the responder and both
    # judges, 128k native window so the matched num_ctx stays inside it.
    # llama3.1-8b above is the dormant third point (scale read, only if nemo
    # diverges from qwen on a headline cell). This directory is judged by Gemini
    # only -- never run it with --score-only under a different judge; use the
    # profile below for anything screening-tier.
    ModelProfile("mistral-nemo-12b", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Same tag/host as mistral-nemo-12b above but a distinct profile name and
    # directory -- the second responder's screening-tier coverage-gap fill, kept
    # separate so the Gemini-confirmatory mistral-nemo-12b directory is never
    # mixed with gemma2:9b-judged rows (one judge per directory).
    ModelProfile("mistral-nemo-12b_screening", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # --- judge-tier validation: NOT new responder points. Same backend/model/
    # options as mistral-nemo-12b above -- these exist only to re-judge that
    # profile's already-generated responses.jsonl (duplicated verbatim into each
    # directory below) under a different judge, via --score-only. Neither follows
    # the bare/`_screening` convention above, because neither is a confirmatory
    # RESULT directory: both are judge-agreement diagnostics, never read by
    # aggregate_results.py or cited as a study result.
    #   _gemini2 -- same judge (gemini-2.5-flash), second draw. Isolates
    #               same-judge non-determinism.
    #   _gpt41   -- different vendor (openai gpt-4.1), tier-matched on
    #               cost/capability. Isolates cross-vendor judge disagreement.
    ModelProfile("mistral-nemo-12b_gemini2", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("mistral-nemo-12b_gpt41", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # --- reasoning family (branch ablation): nonzero floor; r1 degrades at 0 ---
    ModelProfile("deepseek-r1-14b_screening", "ollama", "deepseek-r1:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0.6}),
]
