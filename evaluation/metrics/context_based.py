"""context_based.py — Metrics that require the source context (scene graph text)."""

from __future__ import annotations

import re

from evaluation.llm.base import LLMProvider

_FAITHFULNESS_PROMPT = """\
You are a strict faithfulness judge for a 3D scene-graph QA system.

CONTEXT (scene graph):
{context}

QUESTION:
{question}

ANSWER:
{answer}

Task: Score how well the ANSWER is supported by the CONTEXT on a scale from 0.0 to 1.0.
- 1.0: every claim in the answer is directly supported by the context.
- 0.5: the answer is partially supported; some claims are missing or uncertain.
- 0.0: the answer contradicts the context or contains information not present in it.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""


def faithfulness(
    question: str,
    context: str,
    answer: str,
    judge: LLMProvider,
) -> float:
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context.strip(),
        question=question.strip(),
        answer=answer.strip(),
    )
    result = judge.generate(prompt)
    return _parse_score(result.text)


def _parse_score(text: str) -> float:
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return round(min(max(float(match.group(1)), 0.0), 1.0), 4)
    raise ValueError(f"Could not parse faithfulness score from judge output: {text!r}")
