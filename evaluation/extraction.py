"""extraction.py — LLM-based answer extraction for exact-match scoring."""

from __future__ import annotations

from evaluation.core import QuestionType
from evaluation.llm.base import LLMProvider

_EXTRACTION_PROMPT = """\
Extract the direct answer from the response below.

Question type: {question_type}
Question: {question}
Response: {response}

Rules by type:
- count: reply with only the integer (e.g. "7")
- yes_no: reply with only "yes" or "no"
- factual: reply with only the key entity or value (e.g. "Room 9.0" or "Bowl id=13")
- comparative: reply with only the key entity or value (e.g. "Book id=142")

Respond with ONLY the extracted answer, nothing else."""

_SKIP_EXTRACTION = {QuestionType.DESCRIPTIVE}


def extract_answer(
    question_type: QuestionType,
    question_text: str,
    raw_answer: str,
    judge: LLMProvider,
) -> str:
    if question_type in _SKIP_EXTRACTION:
        return raw_answer
    prompt = _EXTRACTION_PROMPT.format(
        question_type=question_type.value,
        question=question_text.strip(),
        response=raw_answer.strip(),
    )
    return judge.generate(prompt).text.strip()
