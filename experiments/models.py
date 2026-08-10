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

MODEL_PROFILES below is the committed responder set for the study: qwen2.5-14b
(primary, full matrix), mistral-nemo-12b (second responder, headline cells --
selection rationale in docs/METHODOLOGY.md 3.6); the remaining profiles are
contingency/ablation options (llama3.1-8b dormant third point, qwen2.5-32b
upper scale rung, deepseek-r1-14b reasoning-branch ablation) that run only
when their trigger conditions are met.

Naming convention: a `_screening` suffix means the profile's directory is
judged by gemma2:9b and nothing else; a bare name means it is Gemini-judged
(confirmatory). The suffix is the directory's judge tier made visible, so the
one-judge-per-directory invariant is readable from `ls` rather than only from
the `judge` column. Renamed 2026-08-10: qwen2.5-7b, qwen2.5-32b and
deepseek-r1-14b were always screening-only but carried bare names, which read
as if they held confirmatory numbers. If any of them is later promoted under
the METHODOLOGY 3.6 contingency rule, the Gemini pass gets a NEW bare-named
profile/directory -- never a second judge in the `_screening` one. llama3.1-8b
keeps a bare name because it is dormant and its tier is undecided; give it a
suffix at the moment it is activated at screening tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str                       # short id; becomes results/<name>/<scene>/
    backend: str                    # "ollama" | "gemini" | ...
    model: str                      # backend-specific model id
    options: dict = field(default_factory=dict)


# Remote ollama endpoints -- university GPU servers reached over an SSH local
# port-forward (run scripts/ssh_tunnel.ps1 before launching a run that uses
# these hosts). Replaces the previous Kaggle+ngrok setup.
#   _VOXEL -> voxel.nes, RTX 3090  24GB VRAM
#   _PIXEL -> pixel.nes, RTX 5070 Ti 16GB VRAM
#
# 2026-08-08: every profile is pinned to _VOXEL because a third-party job
# (Hunyuan3D gradio_app, running as root) holds ~14 of pixel's 16 GB. Ollama
# does not refuse to serve a card that full -- it loads the model into system
# RAM instead and keeps answering ~150x slower (measured 0.35 tok/s vs 54
# tok/s), which presents as a hung run rather than an error. _PIXEL is kept
# defined so profiles can be moved back one constant at a time once that job
# ends; check free VRAM first with:
#   ssh pixel.nes "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv"
_VOXEL = "http://localhost:11434"
_PIXEL = "http://localhost:11435"  # currently unusable -- see note above

MODEL_PROFILES: list[ModelProfile] = [
    # --- standard instruction families: greedy decoding (temperature 0) ---
    # (placeholders -- replace with the families you intend to compare)
    ModelProfile("qwen2.5-14b", "ollama", "qwen2.5:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Scale rung below the primary: with 14b/32b it forms the within-family
    # scale curve (screening-tier ablation; confirmatory only per the
    # METHODOLOGY 3.6 contingency rule).
    ModelProfile("qwen2.5-7b_screening", "ollama", "qwen2.5:7b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # 32B fits voxel only (Q4 weights ~20GB); 32k-token KV cache spills past
    # 24GB, so ollama partially offloads to CPU -- slower, but num_ctx stays
    # matched to the other profiles so context_exceeded cells stay comparable.
    ModelProfile("qwen2.5-32b_screening", "ollama", "qwen2.5:32b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    ModelProfile("llama3.1-8b", "ollama", "llama3.1:8b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Second responder for the headline-cell robustness check (METHODOLOGY
    # 3.6): dense 12B scale-matched to qwen2.5-14b, lineage-independent of
    # responder and both judges, 128k native window so the matched num_ctx
    # stays inside it. llama3.1-8b above is the dormant third point (scale
    # read, only if nemo diverges from qwen on a headline cell). This
    # profile's directory is judged by Gemini only (confirmatory) -- never
    # run it with --score-only under a different judge; use the profile
    # below for anything screening-tier.
    # Host moved _PIXEL -> _VOXEL 2026-08-08: a long-running third-party job
    # (Hunyuan3D gradio_app) pinned ~14 of pixel's 16 GB, leaving too little
    # VRAM to load this model -- ollama silently fell back to ~97% CPU offload
    # (0.3 of 10.3 GB resident) and generation slowed ~50-100x, which reads as
    # a hung run rather than an error. Move back to _PIXEL once that card frees.
    ModelProfile("mistral-nemo-12b", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),
    # Same tag/host as mistral-nemo-12b above but a distinct profile name/directory
    # (results/mistral-nemo-12b_screening/) -- the second responder's
    # screening-tier coverage-gap fill (scripts/run_ablation_responders.ps1),
    # kept separate so the Gemini-confirmatory mistral-nemo-12b directory is
    # never mixed with gemma2:9b-judged rows (one judge per directory).
    ModelProfile("mistral-nemo-12b_screening", "ollama", "mistral-nemo:12b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0}),

    # --- reasoning family (branch ablation): nonzero floor; r1 degrades at 0 ---
    ModelProfile("deepseek-r1-14b_screening", "ollama", "deepseek-r1:14b",
                 {"host": _VOXEL, "num_ctx": 32768, "temperature": 0.6}),
]
