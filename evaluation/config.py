"""config.py — Runtime configuration for one evaluation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalConfig:   
    dataset_path:       Path    = Path("evaluation/experiment/Brinnon_QA.jsonl")
    scene_contexts_dir: Path    = Path("scene_contexts")
    output_dir:         Path    = Path("evaluation/experiment/results")

    responder_backend:  str     = "ollama"
    responder_model:    str     = "qwen2.5:7b"
    responder_options:  dict    = field(default_factory=dict)

    judge_backend:      str     = "gemini"
    judge_model:        str     = "gemini-2.5-flash"
    judge_options:      dict    = field(default_factory=dict)

    repetitions:        int     = 1
    representations:    Optional[list[str]] = None
    question_ids:       Optional[list[str]] = None

    embedding_model:    str     = "all-MiniLM-L6-v2"

    resume:             bool    = False

    def __post_init__(self):
        self.dataset_path       = Path(self.dataset_path)
        self.scene_contexts_dir = Path(self.scene_contexts_dir)
        self.output_dir         = Path(self.output_dir)