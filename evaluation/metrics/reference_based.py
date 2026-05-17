"""reference_based.py — Metrics that compare prediction against a ground-truth string."""

from __future__ import annotations

import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


def semantic_similarity(prediction: str, ground_truth: str, model_name: str) -> float:
    model = _get_model(model_name)
    embeddings = model.encode([prediction, ground_truth], convert_to_tensor=True)
    score = cos_sim(embeddings[0], embeddings[1]).item()
    return round(float(score), 4)
