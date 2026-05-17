"""core.py — Central data models for the 3DSG LLM evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionType(str, Enum):
    FACTUAL     = "factual"      # "What color is the sofa?"      → extract noun/value
    YES_NO      = "yes_no"       # "Is there a window?"           → extract yes | no
    COUNT       = "count"        # "How many chairs are there?"   → extract integer
    DESCRIPTIVE = "descriptive"  # "Describe the room layout."    → score whole response
    COMPARATIVE = "comparative"  # "Which is larger, A or B?"     → extract entity name


APPLICABLE_METRICS: dict[QuestionType, set[str]] = {
    QuestionType.FACTUAL:     {"exact_match", "semantic_similarity", "faithfulness"},
    QuestionType.YES_NO:      {"exact_match", "faithfulness"},
    QuestionType.COUNT:       {"exact_match", "faithfulness"},
    QuestionType.COMPARATIVE: {"exact_match", "semantic_similarity", "faithfulness"},
    QuestionType.DESCRIPTIVE: {"semantic_similarity", "faithfulness"},
}


@dataclass
class Question:
    id:             str
    scene_id:       str
    text:           str
    ground_truth:   str
    question_type:  Optional[QuestionType] = None   # None → inferred by extractor at eval time


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
class ExtractedAnswer:
    answer_text:      str
    question_type:    QuestionType
    type_is_inferred: bool          # True if inferred, False if came from dataset


@dataclass
class MetricScores:
    exact_match:         Optional[float] = None  # 0.0 or 1.0
    semantic_similarity: Optional[float] = None  # 0.0 – 1.0
    faithfulness:        Optional[float] = None  # 0.0 – 1.0
    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "exact_match":         self.exact_match,
            "semantic_similarity": self.semantic_similarity,
            "faithfulness":        self.faithfulness,
        }


@dataclass
class EvalRecord:
    question:        Question
    response:        Response
    responder:       str
    judge:           str
    extracted:       Optional[ExtractedAnswer] = None
    scores:          MetricScores              = field(default_factory=MetricScores)
    error:           Optional[str]             = None

    def to_flat_dict(self) -> dict:
        return {
            "question_id":      self.question.id,
            "scene_id":         self.question.scene_id,
            "representation":   self.response.representation,
            "repetition":       self.response.repetition,
            "responder":        self.responder,
            "judge":            self.judge,
            "question_text":    self.question.text,
            "ground_truth":     self.question.ground_truth,
            "question_type":    self.extracted.question_type.value if self.extracted else None,
            "type_is_inferred": self.extracted.type_is_inferred if self.extracted else None,
            "raw_answer":       self.response.raw_answer,
            "extracted_answer": self.extracted.answer_text if self.extracted else None,
            "latency_ms":       self.response.latency_ms,
            "prompt_tokens":    self.response.prompt_tokens,
            "completion_tokens":self.response.completion_tokens,
            **self.scores.to_dict(),
            "error":            self.error,
        }


CSV_COLUMNS = [
    "question_id", "scene_id", "representation", "repetition",
    "responder", "judge",
    "question_text", "ground_truth",
    "question_type", "type_is_inferred",
    "raw_answer", "extracted_answer",
    "latency_ms", "prompt_tokens", "completion_tokens",
    "exact_match", "semantic_similarity", "faithfulness",
    "error",
]