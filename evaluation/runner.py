"""runner.py — Main evaluation loop: questions × representations × repetitions."""

from __future__ import annotations

import traceback
from typing import Iterator

from evaluation import dataset, scene_loader
from evaluation.config import EvalConfig
from evaluation.core import (
    APPLICABLE_METRICS,
    EvalRecord,
    ExtractedAnswer,
    MetricScores,
    QuestionType,
    Response,
)
from evaluation.llm.base import LLMProvider
from evaluation.llm.factory import create_provider
from evaluation.extraction import extract_answer
from evaluation.metrics import context_based, reference_based

_RESPONDER_PROMPT = """\
You are a 3D scene understanding assistant. Use the scene graph context below to answer the question.

SCENE GRAPH:
{context}

QUESTION:
{question}

Answer concisely and directly."""


def run(config: EvalConfig) -> list[EvalRecord]:
    return list(iter_records(config))


def iter_records(config: EvalConfig) -> Iterator[EvalRecord]:
    responder = create_provider(
        config.responder_backend, config.responder_model, dict(config.responder_options)
    )
    judge = create_provider(
        config.judge_backend, config.judge_model, dict(config.judge_options)
    )

    responder_tag = f"{config.responder_backend}/{config.responder_model}"
    judge_tag = f"{config.judge_backend}/{config.judge_model}"

    questions = dataset.load(config.dataset_path, config.question_ids)

    for question in questions:
        representations = config.representations or scene_loader.list_representations(
            config.scene_contexts_dir, question.scene_id
        )

        for representation in representations:
            try:
                context = scene_loader.load(
                    config.scene_contexts_dir, question.scene_id, representation
                )
            except (FileNotFoundError, ValueError) as exc:
                yield _make_record(question, representation, 0, responder_tag, judge_tag, error=str(exc))
                continue

            for repetition in range(1, config.repetitions + 1):
                yield _eval_one(
                    question, representation, repetition, context,
                    responder, judge, responder_tag, judge_tag, config,
                )


def _eval_one(question, representation, repetition, context, responder, judge, responder_tag, judge_tag, config) -> EvalRecord:
    try:
        prompt = _RESPONDER_PROMPT.format(
            context=context.strip(),
            question=question.text.strip(),
        )
        gen = responder.generate(prompt)

        response = Response(
            question_id=question.id,
            scene_id=question.scene_id,
            representation=representation,
            repetition=repetition,
            raw_answer=gen.text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            latency_ms=gen.latency_ms,
        )

        question_type = question.question_type or QuestionType.DESCRIPTIVE
        extracted_text = extract_answer(question_type, question.text, gen.text, judge)
        extracted = ExtractedAnswer(
            answer_text=extracted_text,
            question_type=question_type,
            type_is_inferred=question.question_type is None,
        )

        applicable = APPLICABLE_METRICS[question_type]
        scores = MetricScores()

        if "exact_match" in applicable:
            scores.exact_match = reference_based.exact_match(
                extracted.answer_text, question.ground_truth
            )
        if "semantic_similarity" in applicable:
            scores.semantic_similarity = reference_based.semantic_similarity(
                extracted.answer_text, question.ground_truth, config.embedding_model
            )
        if "faithfulness" in applicable:
            scores.faithfulness = context_based.faithfulness(
                question.text, context, extracted.answer_text, judge
            )

        return EvalRecord(
            question=question,
            response=response,
            responder=responder_tag,
            judge=judge_tag,
            extracted=extracted,
            scores=scores,
        )

    except Exception:
        return _make_record(
            question, representation, repetition, responder_tag, judge_tag,
            error=traceback.format_exc(),
        )


def _make_record(question, representation, repetition, responder_tag, judge_tag, error: str) -> EvalRecord:
    return EvalRecord(
        question=question,
        response=Response(
            question_id=question.id,
            scene_id=question.scene_id,
            representation=representation,
            repetition=repetition,
            raw_answer="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
        ),
        responder=responder_tag,
        judge=judge_tag,
        error=error,
    )
