"""runner.py — Two-phase evaluation: generate all responses, then judge them.

The two phases are split so the responder and judge never need to be co-resident
in GPU VRAM (a hard limit on small cards):

  Phase 1  generate_responses() — only the responder is loaded; every
           (question x representation x repetition) answer is streamed to a
           cache file (responses.jsonl). The responder is then unloaded *only*
           if it shares an Ollama host with the judge (else the unload frees
           nothing and just forces a cold reload — see the unload site below).
  Phase 2  score_responses()    — only the judge is loaded; it reads the cache,
           reloads each context (for faithfulness), and yields scored records.

iter_records() runs both back-to-back so callers (and run scripts) are unchanged
(`save(iter_records(config), config)`). Because the cache persists, the same
answers can be re-judged later with a different judge (e.g. Gemini for the final
numbers) via config.score_only — no regeneration, no responder VRAM.
"""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path
from typing import Iterator

from evaluation import dataset, scene_loader, scope
from evaluation.config import EvalConfig
from evaluation.core import (
    CONTEXT_EXCEEDED,
    EvalRecord,
    KeyFact,
    MetricScores,
    Question,
    Response,
    is_context_exceeded,
)
from evaluation.llm.factory import create_provider
from evaluation.metrics import context_based

_RESPONDER_PROMPT = """\
You are a 3D scene understanding assistant. Use the scene graph context below to answer the question.

SCENE GRAPH:
{context}

QUESTION:
{question}

Answer the question and explain your reasoning. Include specific values (distances, counts, room IDs) that support your answer."""

# Minimum completion budget. If the prompt leaves fewer than this many tokens of
# the responder's context window free, the representation did not fit (the input
# was filled/truncated and there is no room for a real answer) -> flag the cell
# context-exceeded instead of judging a clipped answer.
_CTX_RESERVE = 256


def iter_records(config: EvalConfig) -> Iterator[EvalRecord]:
    """Generate responses (unless score_only), then yield judged records."""
    if not config.score_only:
        generate_responses(config)
    yield from score_responses(config)


# --------------------------------------------------------------------------- #
# Phase 1: generation (responder only)
# --------------------------------------------------------------------------- #

def generate_responses(config: EvalConfig) -> Path:
    """Run the responder over every in-scope cell, streaming to responses.jsonl.

    Idempotent: cells already cached without error are skipped, so an interrupted
    generation resumes where it left off. Returns the cache path.
    """
    responder = create_provider(
        config.responder_backend, config.responder_model, dict(config.responder_options)
    )
    responder_tag = f"{config.responder_backend}/{config.responder_model}"
    # Responder context window (ollama num_ctx). Used to flag cells whose prompt
    # overflows it; None for backends that do not expose it -> guard is skipped.
    num_ctx = config.responder_options.get("num_ctx")

    questions = dataset.load(config.dataset_path, config.question_ids)

    # Fail-closed pre-pass (final runs): abort before any LLM call if a rep or
    # question type would fall through to scope's fail-open defaults.
    if config.strict_scope:
        all_reps: set[str] = set()
        for q in questions:
            all_reps.update(config.representations or scene_loader.list_representations(
                config.scene_contexts_dir, q.scene_id
            ))
        scope.validate_declared(all_reps, {q.question_type for q in questions})

    path = _responses_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _cached_keys(path)

    print(f"PHASE 1/2  generate -> {path}  (responder: {responder_tag})")
    n_new = n_skip = n_ctx = 0

    with path.open("a", encoding="utf-8") as fh:
        for question in questions:
            representations = config.representations or scene_loader.list_representations(
                config.scene_contexts_dir, question.scene_id
            )
            for representation in representations:
                # Skip cells this rep structurally cannot answer -- except the
                # no-information control (inventory), which is deliberately posed
                # the spatial questions it cannot answer so its prior-driven
                # guessing forms the Axis-A floor (see METHODOLOGY 3.1.1). Without
                # this exemption inventory is never scored on the spatial types and
                # value_of_spatial_structure.png cannot populate.
                if (config.scope_filter
                        and representation != "inventory"
                        and not scope.in_scope(representation, question.question_type)):
                    continue

                try:
                    context = scene_loader.load(
                        config.scene_contexts_dir, question.scene_id, representation
                    )
                except (FileNotFoundError, ValueError) as exc:
                    rec = _gen_dict(question, representation, 0, responder_tag,
                                    raw_answer="", error=str(exc))
                    _write(fh, rec)
                    print(f"  ERR   {question.id} | {representation} | {exc}")
                    continue

                for repetition in range(1, config.repetitions + 1):
                    if (question.id, representation, str(repetition)) in done:
                        n_skip += 1
                        continue
                    rec = _generate_one(
                        question, representation, repetition, context, responder,
                        responder_tag, num_ctx
                    )
                    _write(fh, rec)
                    if not rec["error"]:
                        n_new += 1
                        print(f"  GEN   {question.id} | {representation} | rep{repetition} | {rec['latency_ms']:.0f}ms")
                    elif is_context_exceeded(rec["error"]):
                        n_ctx += 1
                        print(f"  CTX!  {question.id} | {representation} | rep{repetition} | prompt {rec['prompt_tokens']} tok >= num_ctx {num_ctx}")
                    else:
                        print(f"  ERR   {question.id} | {representation} | rep{repetition}")

    # Free the responder's VRAM before the judge loads (Phase 2) -- but ONLY when
    # they share an Ollama host. On split hosts (responder remote, judge local)
    # there is no shared device to free, so the unload buys nothing and forces a
    # cold reload (weights + CUDA graph + empty KV cache) on the next scene's
    # first prompt -- whose load time then leaks into that cell's latency_ms.
    if _responder_judge_colocated(config):
        responder.unload()
    print(f"PHASE 1/2  done — {n_new} generated, {n_ctx} context-exceeded, {n_skip} already cached\n")
    return path


def _generate_one(question, representation, repetition, context, responder,
                  responder_tag, num_ctx=None) -> dict:
    try:
        prompt = _RESPONDER_PROMPT.format(
            context=context.strip(),
            question=question.text.strip(),
        )
        gen = responder.generate(prompt)
        # Context-window guard: if the prompt filled (or was truncated to) the
        # responder's window, the representation did not fit -- mark the cell
        # context-exceeded (terminal, unscored) rather than letting the judge
        # score a clipped answer as a wrong one. Only fires for backends that
        # report both num_ctx and prompt_tokens (ollama); others skip it.
        error = None
        if (num_ctx and gen.prompt_tokens
                and gen.prompt_tokens >= num_ctx - _CTX_RESERVE):
            error = (f"{CONTEXT_EXCEEDED}: prompt {gen.prompt_tokens} tokens "
                     f">= context window {num_ctx} (representation truncated; not scored)")
        return _gen_dict(
            question, representation, repetition, responder_tag,
            raw_answer=gen.text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            latency_ms=gen.latency_ms,
            error=error,
        )
    except Exception:
        return _gen_dict(question, representation, repetition, responder_tag,
                         raw_answer="", error=traceback.format_exc())


def _gen_dict(question, representation, repetition, responder_tag, *,
              raw_answer, prompt_tokens=0, completion_tokens=0, latency_ms=0.0,
              error=None) -> dict:
    return {
        "question_id":       question.id,
        "scene_id":          question.scene_id,
        "question_text":     question.text,
        "question_type":     question.question_type,
        "key_facts":         [{"fact": kf.fact, "weight": kf.weight} for kf in question.key_facts],
        "representation":    representation,
        "repetition":        repetition,
        "responder":         responder_tag,
        "raw_answer":        raw_answer,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms":        latency_ms,
        "error":             error,
    }


def _write(fh, rec: dict) -> None:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()


def _cached_keys(path: Path) -> set[tuple[str, str, str]]:
    """(question_id, representation, repetition) already generated without error."""
    done: set[tuple[str, str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            err = rec.get("error")
            # Skip on resume if generated cleanly OR terminally context-exceeded
            # (the rep will overflow every time -- never regenerate it).
            if not err or is_context_exceeded(err):
                done.add((rec["question_id"], rec["representation"], str(rec["repetition"])))
    return done


# --------------------------------------------------------------------------- #
# Phase 2: scoring (judge only)
# --------------------------------------------------------------------------- #

def score_responses(config: EvalConfig) -> Iterator[EvalRecord]:
    """Judge every cached response, yielding one EvalRecord each.

    De-duplicates the cache by (question, representation, repetition), keeping the
    last line written — so retried/duplicated generations score once, on the most
    recent attempt.
    """
    judge = create_provider(
        config.judge_backend, config.judge_model, dict(config.judge_options)
    )
    judge_tag = f"{config.judge_backend}/{config.judge_model}"

    path = _responses_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"No responses cache to score: {path}. Run generation first "
            f"(score_only is on but the cache is missing)."
        )

    records = _load_cache(path)
    done = _load_done_keys(config) if config.resume else set()
    ctx_cache: dict[tuple[str, str], str] = {}

    print(f"PHASE 2/2  score {len(records)} responses (judge: {judge_tag})")

    for rec in records:
        key = (rec["question_id"], rec["representation"], str(rec["repetition"]))
        if key in done:
            continue
        yield _score_one(rec, config, judge, judge_tag, ctx_cache)


def _load_cache(path: Path) -> list[dict]:
    """All cache rows, de-duplicated by key (last write wins), order preserved."""
    by_key: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["question_id"], rec["representation"], str(rec["repetition"]))
            by_key[key] = rec
    return list(by_key.values())


def _score_one(rec: dict, config, judge, judge_tag, ctx_cache) -> EvalRecord:
    question = Question(
        id=rec["question_id"],
        scene_id=rec["scene_id"],
        text=rec["question_text"],
        question_type=rec["question_type"],
        key_facts=[KeyFact(fact=f["fact"], weight=f["weight"]) for f in rec.get("key_facts", [])],
    )
    response = Response(
        question_id=rec["question_id"],
        scene_id=rec["scene_id"],
        representation=rec["representation"],
        repetition=rec["repetition"],
        raw_answer=rec["raw_answer"],
        prompt_tokens=rec["prompt_tokens"],
        completion_tokens=rec["completion_tokens"],
        latency_ms=rec["latency_ms"],
    )

    # Pass generation errors straight through — nothing to judge.
    if rec.get("error"):
        return EvalRecord(
            question=question, response=response,
            responder=rec["responder"], judge=judge_tag, error=rec["error"],
        )

    try:
        scores = MetricScores()

        if config.compute_faithfulness:
            context = _get_context(config, question.scene_id, response.representation, ctx_cache)
            scores.faithfulness = context_based.faithfulness(
                question.text, context, response.raw_answer, judge
            )

        rubric_reasoning = ""
        if question.key_facts:
            (scores.answer_correctness,
             scores.answer_correctness_detail,
             rubric_reasoning) = context_based.rubric_correctness(
                question.text, response.raw_answer, question.key_facts, judge
            )

        return EvalRecord(
            question=question, response=response,
            responder=rec["responder"], judge=judge_tag,
            scores=scores, rubric_reasoning=rubric_reasoning,
        )
    except Exception:
        return EvalRecord(
            question=question, response=response,
            responder=rec["responder"], judge=judge_tag,
            error=traceback.format_exc(),
        )


def _get_context(config, scene_id, representation, cache) -> str:
    key = (scene_id, representation)
    if key not in cache:
        cache[key] = scene_loader.load(config.scene_contexts_dir, scene_id, representation)
    return cache[key]


def _load_done_keys(config: EvalConfig) -> set[tuple[str, str, str]]:
    """(question_id, representation, repetition) already scored without error."""
    path = Path(config.output_dir) / "results.csv"
    if not path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            err = row.get("error")
            # Real errors are left out so they get re-judged on resume;
            # context-exceeded cells are terminal, so treat them as done.
            if not err or is_context_exceeded(err):
                done.add((row["question_id"], row["representation"], row["repetition"]))
    return done


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #

def _responses_path(config: EvalConfig) -> Path:
    if config.responses_path is not None:
        return Path(config.responses_path)
    return Path(config.output_dir) / "responses.jsonl"


def _responder_judge_colocated(config: EvalConfig) -> bool:
    """True only when the responder and judge run on the same Ollama host, so the
    responder must free its VRAM before the judge can load. Different hosts
    (responder remote, judge local) or different backends share no device, so
    there is nothing to free -- and unloading would only cost a cold reload."""
    if config.responder_backend != config.judge_backend:
        return False
    if config.responder_backend != "ollama":
        return False
    default = "http://localhost:11434"
    return (config.responder_options.get("host", default)
            == config.judge_options.get("host", default))
