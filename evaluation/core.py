"""core.py — Central data models for the 3DSG LLM evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Sentinel stored in EvalRecord.error / the responses cache when the responder's
# context window was too small to hold the representation -- the prompt filled or
# was truncated to num_ctx, so the rep is clipped and any answer is unreliable.
# It is a TERMINAL outcome: resume must not retry it and the judge must not score
# it, and it is reported separately from real errors in the aggregate (a context
# token-cost result, not a wrong answer). See evaluation/runner.py (the guard)
# and evaluation/results.py (the aggregate split).
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
    faithfulness:       Optional[float] = None  # 0.0 – 1.0
    answer_correctness: Optional[float] = None  # 0.0 – 1.0
    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "faithfulness":       self.faithfulness,
            "answer_correctness": self.answer_correctness,
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
    "question_type", "question_text",
    "raw_answer", "rubric_reasoning",
    "prompt_tokens", "completion_tokens",
    "faithfulness", "answer_correctness",
    "latency_ms",
    "error",
]