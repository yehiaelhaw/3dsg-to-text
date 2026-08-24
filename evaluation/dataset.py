"""dataset.py — Load questions from a JSONL file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evaluation.core import KeyFact, Question


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
            question_type = qt_raw or None
            # Style tags are used only for the matched natural/constructed questions.
            question_style = raw.get("question_style") or None
            # Groups the two wording variants of one matched fact request.
            pair_id = raw.get("pair_id") or None

            try:
                raw_facts = raw.get("key_facts")
                if not raw_facts:
                    raise ValueError(f"{path}:{lineno}: missing or empty 'key_facts'")
                key_facts = [
                    KeyFact(fact=f["fact"], weight=float(f.get("weight", 1.0)))
                    for f in raw_facts
                ]
                # The primary answer_correctness scores the core (weight > 1) tier only;
                # a question with no core fact would leave it undefined. Fail fast here
                # (before any LLM call) rather than erroring mid-run in the judge.
                if not any(kf.weight > 1.0 for kf in key_facts):
                    raise ValueError(
                        f"{path}:{lineno}: question {qid!r} has no core (weight > 1) key "
                        f"fact; the primary answer_correctness would be undefined"
                    )
                if pair_id is not None and not question_style:
                    raise ValueError(
                        f"{path}:{lineno}: question {qid!r} is in a pair but carries no "
                        f"question_style; the register is what the pair contrasts"
                    )
                questions.append(Question(
                    id=qid,
                    scene_id=raw["scene_id"],
                    text=raw["text"],
                    question_type=question_type,
                    question_style=question_style,
                    pair_id=pair_id,
                    key_facts=key_facts,
                ))
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc

    if filter_ids:
        found = {q.id for q in questions}
        missing = filter_ids - found
        if missing:
            raise ValueError(f"question_ids not found in dataset: {sorted(missing)}")

    return questions
