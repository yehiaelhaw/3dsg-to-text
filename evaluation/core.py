"""core.py — Central data models for the 3DSG LLM evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionType(str, Enum):
    FACTUAL                    = "factual"                    # "What color is the sofa?"      → extract noun/value
    YES_NO                     = "yes_no"                     # "Is there a window?"           → extract yes | no
    COUNT                      = "count"                      # "How many chairs are there?"   → extract integer
    DESCRIPTIVE                = "descriptive"                # "Describe the room layout."    → score whole response
    COMPARATIVE                = "comparative"                # "Which is larger, A or B?"     → extract entity name
    # reasoning-heavy types
    SPATIAL_REASONING          = "spatial_reasoning"          # Euclidean distance / proximity queries
    FUNCTIONAL_COMPATIBILITY   = "functional_compatibility"   # affordance-based room suitability
    CAPACITY_RESOURCE_PLANNING = "capacity_resource_planning" # counting/summing resources across rooms
    COMPARATIVE_SUPERLATIVE    = "comparative_superlative"    # largest/smallest/most-similar across geometry
    SEMANTIC_CATEGORICAL       = "semantic_categorical"       # category membership / set difference
    ANOMALY_PATTERN_DETECTION  = "anomaly_pattern_detection"  # outliers and structural patterns
    DEPENDENCY_RELATIONSHIP_MAPPING = "dependency_relationship_mapping"  # object→room dependency traversal
    MULTI_STEP_INFERENCE       = "multi_step_inference"       # multi-condition filtering
    CONDITIONAL_CONSTRAINT     = "conditional_constraint"     # constraint satisfaction queries
    MULTI_STEP_SPATIAL_REASONING = "multi_step_spatial_reasoning"  # spatial + multi-step combined
    CONDITIONAL_REASONING      = "conditional_reasoning"      # counterfactual / hypothetical
    EXPLICIT_OPTIMIZATION      = "explicit_optimization"      # single-criterion optimisation


_EXTRACTABLE = {"exact_match", "semantic_similarity", "faithfulness", "answer_correctness"}

APPLICABLE_METRICS: dict[QuestionType, set[str]] = {
    QuestionType.FACTUAL:                       _EXTRACTABLE,
    QuestionType.YES_NO:                        {"exact_match", "faithfulness", "answer_correctness"},           # "yes"/"no" embeddings are near-identical; semantic_similarity adds noise
    QuestionType.COUNT:                         {"exact_match", "faithfulness", "answer_correctness"},           # numeric tokens have poor embedding separation
    QuestionType.COMPARATIVE:                   _EXTRACTABLE,
    QuestionType.DESCRIPTIVE:                   {"semantic_similarity", "faithfulness", "answer_correctness"},   # open-ended by design; no short answer exists to extract
    # reasoning-heavy types all have specific extractable ground truths
    QuestionType.SPATIAL_REASONING:             _EXTRACTABLE,
    QuestionType.FUNCTIONAL_COMPATIBILITY:      _EXTRACTABLE,
    QuestionType.CAPACITY_RESOURCE_PLANNING:    _EXTRACTABLE,
    QuestionType.COMPARATIVE_SUPERLATIVE:       _EXTRACTABLE,
    QuestionType.SEMANTIC_CATEGORICAL:          _EXTRACTABLE,
    QuestionType.ANOMALY_PATTERN_DETECTION:     _EXTRACTABLE,
    QuestionType.DEPENDENCY_RELATIONSHIP_MAPPING: _EXTRACTABLE,
    QuestionType.MULTI_STEP_INFERENCE:          _EXTRACTABLE,
    QuestionType.CONDITIONAL_CONSTRAINT:        _EXTRACTABLE,
    QuestionType.MULTI_STEP_SPATIAL_REASONING:  _EXTRACTABLE,
    QuestionType.CONDITIONAL_REASONING:         _EXTRACTABLE,
    QuestionType.EXPLICIT_OPTIMIZATION:         _EXTRACTABLE,
}


@dataclass
class KeyFact:
    fact:   str
    weight: float = 1.0


@dataclass
class Question:
    id:             str
    scene_id:       str
    text:           str
    ground_truth:   str
    question_type:  Optional[QuestionType] = None   # None → inferred by extractor at eval time
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
class ExtractedAnswer:
    answer_text:      str
    question_type:    QuestionType
    type_is_inferred: bool          # True if inferred, False if came from dataset


@dataclass
class MetricScores:
    exact_match:         Optional[float] = None  # 0.0 or 1.0
    semantic_similarity: Optional[float] = None  # 0.0 – 1.0
    faithfulness:        Optional[float] = None  # 0.0 – 1.0
    answer_correctness:  Optional[float] = None  # 0.0 – 1.0
    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "exact_match":         self.exact_match,
            "semantic_similarity": self.semantic_similarity,
            "faithfulness":        self.faithfulness,
            "answer_correctness":  self.answer_correctness,
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
    "exact_match", "semantic_similarity", "faithfulness", "answer_correctness",
    "error",
]