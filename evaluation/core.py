"""core.py — Central data models for the 3DSG LLM evaluation framework."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional

# A runaway responder answer (repetition loops on dense scenes) can exceed csv's
# default 128 KB per-field cap, breaking every DictReader that reads results.csv
# (resume bookkeeping, aggregate, report, plots). Raise it once here — core is
# imported by all of them. 2**31-1 is the max on Windows (32-bit C long).
csv.field_size_limit(2**31 - 1)


# Sentinel for a responder context-window overflow (prompt clipped/truncated). Terminal: resume must not retry it, the judge must not score it.
CONTEXT_EXCEEDED = "CONTEXT_EXCEEDED"


def is_context_exceeded(error: Optional[str]) -> bool:
    return bool(error) and error.startswith(CONTEXT_EXCEEDED)


@dataclass
class KeyFact:
    fact:   str
    weight: float = 1.0


@dataclass
class Question:
    id:             str
    scene_id:       str
    text:           str
    question_type:  Optional[str] = None
    # `natural`: a user could ask this without having seen a derived view's output. `constructed`: wording mirrors that output's computed vocabulary. Only set for types exposed to a derived pole.
    question_style: Optional[str] = None
    # Fact-set key: (scene_id, pair_id) groups a natural/constructed pair so report.py can difference their register effect. NOT a provenance field -- never filter/group on which member was authored first.
    # Deliberately excluded from CSV_COLUMNS; report.py joins it from the QA files at analysis time to keep results.csv byte-identical across runs.
    pair_id:        Optional[str] = None
    key_facts:      list[KeyFact] = field(default_factory=list)


@dataclass
class Response:
    question_id:        str
    scene_id:           str
    representation:     str
    repetition:         int
    raw_answer:         str
    prompt_tokens:      int
    completion_tokens:  int
    latency_ms:         float


@dataclass
class MetricScores:
    faithfulness:              Optional[float] = None  # 0.0 – 1.0
    answer_correctness:        Optional[float] = None  # 0.0 – 1.0, core (weight > 1) facts only
    # Diagnostic only: fraction of the supporting-detail (weight <= 1) facts the answer
    # carried. Never folded into answer_correctness; None when the question has no detail
    # facts. Measures volunteered detail (verbosity x representation), not correctness.
    answer_correctness_detail: Optional[float] = None  # 0.0 – 1.0
    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "faithfulness":              self.faithfulness,
            "answer_correctness":        self.answer_correctness,
            "answer_correctness_detail": self.answer_correctness_detail,
        }


@dataclass
class EvalRecord:
    question:         Question
    response:         Response
    responder:        str
    judge:            str
    scores:           MetricScores = field(default_factory=MetricScores)
    rubric_reasoning: str          = ""
    error:            Optional[str] = None

    def to_flat_dict(self) -> dict:
        return {
            "question_id":      self.question.id,
            "scene_id":         self.question.scene_id,
            "representation":   self.response.representation,
            "repetition":       self.response.repetition,
            "responder":        self.responder,
            "judge":            self.judge,
            "question_text":    self.question.text,
            "question_type":    self.question.question_type,
            "question_style":   self.question.question_style,
            "raw_answer":       self.response.raw_answer,
            "latency_ms":       self.response.latency_ms,
            "prompt_tokens":    self.response.prompt_tokens,
            "completion_tokens":self.response.completion_tokens,
            **self.scores.to_dict(),
            "rubric_reasoning": self.rubric_reasoning,
            "error":            self.error,
        }


CSV_COLUMNS = [
    "question_id", "scene_id", "representation", "repetition",
    "responder", "judge",
    "question_type", "question_style", "question_text",
    "raw_answer", "rubric_reasoning",
    "prompt_tokens", "completion_tokens",
    "faithfulness", "answer_correctness", "answer_correctness_detail",
    "latency_ms",
    "error",
]