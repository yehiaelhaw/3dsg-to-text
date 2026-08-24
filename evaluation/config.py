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
    # Run filter only; scope.py still determines answerability.
    question_types:     Optional[list[str]] = None

    # Skip out-of-scope cells before generation. Thesis runs keep this enabled.
    scope_filter:       bool                = True
    # Thesis-reportable runs require fail-closed scope validation.
    strict_scope:       bool                = False
    # Faithfulness is diagnostic only; disabling it avoids the extra judge call.
    compute_faithfulness: bool              = True

    embedding_model:    str                 = "all-MiniLM-L6-v2"
    # Resume by skipping completed, non-error cells already in results.csv.
    resume:             bool                = False

    # Judge cached responses from responses_path without generating new ones.
    score_only:         bool                = False
    responses_path:     Optional[Path]      = None

    def __post_init__(self):
        self.dataset_path       = Path(self.dataset_path)
        self.scene_contexts_dir = Path(self.scene_contexts_dir)
        self.output_dir         = Path(self.output_dir)
        if self.responses_path is not None:
            self.responses_path = Path(self.responses_path)
