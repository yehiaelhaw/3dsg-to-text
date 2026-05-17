"""dataset.py — Load questions from a JSONL file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evaluation.core import Question, QuestionType


def load(
    path: Path,
    question_ids: Optional[list[str]] = None,
) -> list[Question]:
    """Return questions from a JSONL file, optionally filtered by ID."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    filter_ids = set(question_ids) if question_ids else None
    questions: list[Question] = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc

            try:
                qid = raw["id"]
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc

            if filter_ids and qid not in filter_ids:
                continue

            qt_raw = raw.get("question_type")
            question_type = QuestionType(qt_raw) if qt_raw else None

            try:
                questions.append(Question(
                    id=qid,
                    scene_id=raw["scene_id"],
                    text=raw["text"],
                    ground_truth=raw["ground_truth"],
                    question_type=question_type,
                ))
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc

    if filter_ids:
        found = {q.id for q in questions}
        missing = filter_ids - found
        if missing:
            raise ValueError(f"question_ids not found in dataset: {sorted(missing)}")

    return questions
