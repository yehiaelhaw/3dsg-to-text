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


_CORRECTNESS_PROMPT = """\
You are a correctness judge for a 3D scene-graph QA system.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL ANSWER:
{answer}

Task: Score how correct the MODEL ANSWER is relative to the GROUND TRUTH on a scale from 0.0 to 1.0,
where 0.0 means completely wrong and 1.0 means fully correct.

First identify what the QUESTION explicitly asks for. Then check whether the MODEL ANSWER correctly provides that.
Award partial credit proportional to the fraction of explicitly-requested facts answered correctly.
The GROUND TRUTH may contain supporting evidence beyond what the question requires — do not penalize for omitting details that the question did not ask for.
Order of items does not matter unless the question explicitly asks for a ranking.
Ignore stylistic differences.
Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""


def answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    judge: LLMProvider,
) -> float:
    prompt = _CORRECTNESS_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        answer=answer.strip(),
    )
    result = judge.generate(prompt)
    return _parse_score(result.text)


def _parse_score(text: str) -> float:
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return round(min(max(float(match.group(1)), 0.0), 1.0), 4)
    raise ValueError(f"Could not parse faithfulness score from judge output: {text!r}")
