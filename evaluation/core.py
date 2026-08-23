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
    # `natural` where a user could have asked this without ever having seen a
    # derived view's output, `constructed` where the concept mirrors that output's
    # computed vocabulary. Only the types exposed to a derived pole carry it
    # (connectivity + the 3RScan relation family); None elsewhere. This is what the
    # vocabulary-coupling re-cut in report.py splits on -- the remedy the declared
    # confound points at (thesis 4.5 / 4.6).
    question_style: Optional[str] = None
    # The balanced natural<->constructed design. A fact-set is one information request;
    # its two members ask for the SAME answer in different registers, so register is
    # manipulated within-item rather than between-item. `(scene_id, pair_id)` is the
    # fact-set key, and it exists for exactly one purpose: letting report.py group the
    # two members so their register effects can be differenced.
    #
    # It is NOT a provenance field and must not be used as one. Which member happens
    # to carry the id that became `pair_id` -- i.e. which stem was authored first --
    # is implementation history, not an experimental variable: it must never filter,
    # group, or qualify a reported number, and all 134 scoped questions participate in
    # every table. `natural` means natural and `constructed` means constructed
    # regardless of when or why the stem was written.
    #
    # It is deliberately NOT in CSV_COLUMNS. It is a property of the authored
    # question, not of a run, so report.py joins it from the QA files on
    # (scene_id, question_id) at analysis time. That keeps every committed
    # results.csv byte-identical and avoids the mixed-fieldnames failure in
    # aggregate_results._pool_rows, which takes fieldnames from the last scene it
    # reads and would reject rows carrying keys the header lacks.
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