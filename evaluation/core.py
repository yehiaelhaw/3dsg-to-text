"""core.py — Central data models for the 3DSG LLM evaluation framework."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional

# Allow unusually long responder outputs in CSV fields; 2**31 - 1 is the Windows-safe maximum.
csv.field_size_limit(2**31 - 1)


# Terminal context-overflow sentinel: resume does not retry it and judging excludes it.
CONTEXT_EXCEEDED = "CONTEXT_EXCEEDED"


def is_context_exceeded(error: Optional[str]) -> bool:
    return bool(error) and error.startswith(CONTEXT_EXCEEDED)


class ContinuityError(ValueError):
    """Raised when resuming would mix responders or judges in one output directory."""


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
    # On affected types: natural does not rely on seeing a derived view; constructed mirrors its vocabulary.
    question_style: Optional[str] = None
    # Groups the natural/constructed versions of one fact request; not a provenance field.
    # Omitted from results.csv and rejoined from the QA files during analysis.
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
    answer_correctness:        Optional[float] = None  # 0.0–1.0, core facts only
    # Supporting-detail score; excluded from primary correctness.
    answer_correctness_detail: Optional[float] = None  # 0.0–1.0; None if no detail facts
    def to_dict(self) -> dict[str, Optional[float]]:
        return {
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
    "answer_correctness", "answer_correctness_detail",
    "latency_ms",
    "error",
]