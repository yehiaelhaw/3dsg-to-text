"""config.py — Runtime configuration for one evaluation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalConfig:
    # --- required ---
    dataset_path:       Path
    scene_contexts_dir: Path
    output_dir:         Path

    responder_backend:  str
    responder_model:    str

    judge_backend:      str
    judge_model:        str

    # --- optional ---
    responder_options:  dict                = field(default_factory=dict)
    judge_options:      dict                = field(default_factory=dict)

    repetitions:        int                 = 1
    representations:    Optional[list[str]] = None   # None → auto-discover
    question_ids:       Optional[list[str]] = None   # None → all questions

    # Skip (rep × question) cells the rep cannot answer (scope.in_scope). On by
    # default: it only removes structurally-meaningless cells that plots masked
    # anyway. Set False to force the full cross product.
    scope_filter:       bool                = True
    # faithfulness is the secondary metric and costs an extra judge call per
    # record. Turn off for screening passes to cut LLM calls by ~a third.
    compute_faithfulness: bool              = True

    embedding_model:    str                 = "all-MiniLM-L6-v2"
    # Resume an interrupted run: skip (question, rep, repetition) cells already
    # present and non-errored in output_dir/results.csv, and append to it.
    resume:             bool                = False

    # Two-phase execution (generate -> judge). Generation streams responses to
    # a cache so the responder and judge never need to be co-resident in VRAM:
    # the responder is unloaded before the judge loads.
    #   score_only=True   -> skip generation; judge an existing responses cache
    #                        (e.g. re-score the same answers with Gemini later).
    #   responses_path    -> where the cache lives; defaults to
    #                        output_dir/responses.jsonl. Point it at a prior
    #                        run's cache to re-judge without regenerating.
    score_only:         bool                = False
    responses_path:     Optional[Path]      = None

    def __post_init__(self):
        self.dataset_path       = Path(self.dataset_path)
        self.scene_contexts_dir = Path(self.scene_contexts_dir)
        self.output_dir         = Path(self.output_dir)
        if self.responses_path is not None:
            self.responses_path = Path(self.responses_path)
