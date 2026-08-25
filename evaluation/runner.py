"""Two-phase evaluation: generate all responses, then judge them, so the responder and judge never need to be co-resident in GPU VRAM."""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path
from typing import Iterator

from evaluation import dataset, judging, scene_loader, scope, token_count
from evaluation.axes import dataset_of
from evaluation.config import EvalConfig
from evaluation.core import (
    CONTEXT_EXCEEDED,
    ContinuityError,
    EvalRecord,
    KeyFact,
    MetricScores,
    Question,
    Response,
    is_context_exceeded,
)
from evaluation.llm.factory import create_provider

_RESPONDER_PROMPT = """\
You are a 3D scene understanding assistant. Use the scene graph context below to answer the question.

SCENE GRAPH:
{context}

QUESTION:
{question}

Answer the question and explain your reasoning. Include specific values (distances, counts, room IDs) that support your answer."""

# Tokens reserved for the completion; a prompt that leaves less is flagged
# context-exceeded rather than judged on a truncated answer.
_CTX_RESERVE = 256


def _ctx_limit(num_ctx: int) -> int:
    """Effective prompt budget. One definition, used by both guards."""
    return num_ctx - _CTX_RESERVE


def iter_records(config: EvalConfig) -> Iterator[EvalRecord]:
    """Generate responses (unless score_only), then yield judged records."""
    if not config.score_only:
        generate_responses(config)
    yield from score_responses(config)


# --------------------------------------------------------------------------- #
# Phase 1: generation (responder only)
# --------------------------------------------------------------------------- #

def _select_types(questions, types, source="dataset"):
    """Narrow `questions` to `types` (None = keep all). Fails closed on empty -- a run matching no question is always a mistake and would otherwise look like a successful zero-cell run."""
    if not types:
        return questions
    wanted = set(types)
    kept = [q for q in questions if q.question_type in wanted]
    if not kept:
        have = sorted({q.question_type for q in questions if q.question_type})
        raise ValueError(
            f"question_types={sorted(wanted)} matched no question in {source} "
            f"(available: {', '.join(have) or 'none'})"
        )
    return kept


def generate_responses(config: EvalConfig) -> Path:
    """Run the responder over every in-scope cell, streaming to responses.jsonl.

    Idempotent: cells already cached without error are skipped, so an interrupted
    generation resumes where it left off. Returns the cache path.
    """
    responder = create_provider(
        config.responder_backend, config.responder_model, dict(config.responder_options)
    )
    responder_tag = f"{config.responder_backend}/{config.responder_model}"
    # num_ctx enables the overflow guard; absent on backends that don't expose it.
    num_ctx = config.responder_options.get("num_ctx")
    # Exact pre-call token counter, built once; returns None where unsupported
    # (falls back to the approximate estimate).
    sizer = token_count.make_sizer(
        config.responder_backend, config.responder_model, config.responder_options
    )
    if num_ctx:
        why = token_count.UNSUPPORTED_REASON.get(
            (config.responder_backend, config.responder_model,
             config.responder_options.get("host", "http://localhost:11434"))
        )
        print(f"  ctx guard: {'estimated (len//4) — ' + why if why else 'exact token count'}"
              f"  (limit {_ctx_limit(num_ctx)} = num_ctx {num_ctx} - reserve {_CTX_RESERVE})")

    questions = _select_types(
        dataset.load(config.dataset_path, config.question_ids),
        config.question_types, config.dataset_path,
    )

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
    if path.exists():
        _compact_cache(path)
        _check_cache_continuity(path, responder_tag)
    done = _cached_keys(path)

    print(f"PHASE 1/2  generate -> {path}  (responder: {responder_tag})")
    n_new = n_skip = n_ctx = n_err = 0

    with path.open("a", encoding="utf-8") as fh:
        for question in questions:
            host = dataset_of(question.scene_id)
            representations = config.representations or scene_loader.list_representations(
                config.scene_contexts_dir, question.scene_id
            )
            for representation in representations:
                # inventory always runs, even out-of-scope: it's the non-spatial anchor.
                if (config.scope_filter
                        and representation != "inventory"
                        and not scope.in_scope(representation, question.question_type, host)):
                    continue

                try:
                    context = scene_loader.load(
                        config.scene_contexts_dir, question.scene_id, representation
                    )
                except (FileNotFoundError, ValueError) as exc:
                    # Record load failures on the configured repetition schedule.
                    for repetition in range(1, config.repetitions + 1):
                        if (question.id, representation, str(repetition)) in done:
                            n_skip += 1
                            continue
                        rec = _gen_dict(question, representation, repetition, responder_tag,
                                        raw_answer="", error=str(exc))
                        _write(fh, rec)
                        n_err += 1
                        print(f"  {'ERROR':<8}{question.id} | {representation} | rep{repetition} | {exc}")
                    continue

                for repetition in range(1, config.repetitions + 1):
                    if (question.id, representation, str(repetition)) in done:
                        n_skip += 1
                        continue
                    rec = _generate_one(
                        question, representation, repetition, context, responder,
                        responder_tag, num_ctx, sizer
                    )
                    _write(fh, rec)
                    if not rec["error"]:
                        n_new += 1
                        print(f"  {'RESPOND':<8}{question.id} | {representation} | rep{repetition} | {rec['latency_ms']:.0f}ms")
                    elif is_context_exceeded(rec["error"]):
                        n_ctx += 1
                        print(f"  {'CONTEXT':<8}{question.id} | {representation} | rep{repetition} | prompt {rec['prompt_tokens']} tok, budget {_ctx_limit(num_ctx)}")
                    else:
                        n_err += 1
                        print(f"  {'ERROR':<8}{question.id} | {representation} | rep{repetition}")

    # Unload only when responder and judge share a host -- otherwise it just
    # forces a cold reload, leaking load time into the next cell's latency_ms.
    if _responder_judge_colocated(config):
        responder.unload()
    print(f"PHASE 1/2  done — {n_new} generated, {n_err} errored, "
          f"{n_ctx} context-exceeded, {n_skip} already cached\n")
    # A total-wipeout phase (every attempt errored) must not exit 0 -- indistinguishable from a clean run to any caller reading the exit code.
    if n_err and not n_new and not n_ctx:
        raise RuntimeError(
            f"generation produced no rows: all {n_err} attempted cells errored "
            f"(responder {responder_tag}). Check the model is pulled on the "
            f"configured host, then rerun -- cached cells are skipped."
        )
    return path


def _generate_one(question, representation, repetition, context, responder,
                  responder_tag, num_ctx=None, sizer=None) -> dict:
    try:
        prompt = _RESPONDER_PROMPT.format(
            context=context.strip(),
            question=question.text.strip(),
        )
        # Pre-call guard: exact token count preferred, approximate estimate otherwise.
        exact_tokens = sizer(prompt) if sizer else None
        if num_ctx:
            limit = _ctx_limit(num_ctx)
            if exact_tokens is not None:
                # Exact count is entitled to equality at the boundary (reserve is
                # exactly met); the estimate below stays conservative with `>=`.
                if exact_tokens > limit:
                    return _gen_dict(
                        question, representation, repetition, responder_tag,
                        raw_answer="",
                        prompt_tokens=exact_tokens,
                        error=(f"{CONTEXT_EXCEEDED}: prompt {exact_tokens} tokens "
                               f"(exact) > budget {limit} = context window {num_ctx} "
                               f"- {_CTX_RESERVE} reserved for the answer "
                               f"(representation cannot fit; not generated)"),
                    )
            else:
                est_tokens = len(prompt) // 4
                if est_tokens >= limit:
                    return _gen_dict(
                        question, representation, repetition, responder_tag,
                        raw_answer="",
                        prompt_tokens=est_tokens,
                        error=(f"{CONTEXT_EXCEEDED}: estimated prompt ~{est_tokens} tokens "
                               f"(approximate character-based estimate) >= context window {num_ctx} "
                               f"(representation cannot fit; not generated)"),
                    )
        gen = responder.generate(prompt)
        # Post-call backstop for a server whose actual window is smaller than num_ctx.
        error = None
        if (num_ctx and gen.prompt_tokens
                and gen.prompt_tokens > _ctx_limit(num_ctx)):
            error = (f"{CONTEXT_EXCEEDED}: prompt {gen.prompt_tokens} tokens "
                     f"> budget {_ctx_limit(num_ctx)} (representation truncated; not scored)")
        # Local/server token counts should agree; a mismatch signals drift
        # (re-pull, template change, silent truncation). Loud but non-fatal.
        elif exact_tokens is not None and gen.prompt_tokens and exact_tokens != gen.prompt_tokens:
            print(f"  {'WARN':<8}{question.id} | {representation} | token count drift: "
                  f"predicted {exact_tokens}, server reported {gen.prompt_tokens} "
                  f"(delta {gen.prompt_tokens - exact_tokens:+d})")
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
        "question_style":    question.question_style,
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


def _compact_cache(path: Path) -> None:
    """Drop cache lines a resumed generation supersedes, in place -- same two rules as results._compact_for_resume: last write wins per key, real-error lines dropped, context-exceeded sentinels kept."""
    parsed: list[tuple[str, dict]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append((line, json.loads(line)))
        except json.JSONDecodeError:
            continue
    last: dict[tuple[str, str, str], int] = {}
    for i, (_, rec) in enumerate(parsed):
        last[(rec["question_id"], rec["representation"], str(rec["repetition"]))] = i
    kept = []
    for i, (line, rec) in enumerate(parsed):
        err = rec.get("error")
        if (last[(rec["question_id"], rec["representation"], str(rec["repetition"]))] == i
                and not (err and not is_context_exceeded(err))):
            kept.append(line)
    if len(kept) != len(parsed):
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"resume: compacted {path.name} -- dropped {len(parsed) - len(kept)} superseded line(s)")


def _check_cache_continuity(path: Path, responder_tag: str) -> None:
    """Fail loudly if resuming would skip a different responder's cells as already-generated."""
    existing: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("responder"):
                existing.add(rec["responder"])
    if existing - {responder_tag}:
        raise ContinuityError(
            f"{path}: existing rows were generated by responder(s) "
            f"{sorted(existing - {responder_tag})}, not {responder_tag!r} -- "
            "refusing to resume into a cache that belongs to a different responder."
        )


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
        # .get: caches written before question_style existed carry no such key, and a
        # score_only re-judge must still read them.
        question_style=rec.get("question_style"),
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
            scores.faithfulness = judging.faithfulness(
                question.text, context, response.raw_answer, judge
            )

        rubric_reasoning = ""
        if question.key_facts:
            (scores.answer_correctness,
             scores.answer_correctness_detail,
             rubric_reasoning) = judging.rubric_correctness(
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
    """(question_id, representation, repetition) already scored without error by THIS config's judge -- scoping to the judge means a new judge (e.g. a Gemini confirmatory re-judge) rescores rather than resume-skipping rows another judge already scored."""
    path = Path(config.output_dir) / "results.csv"
    if not path.exists():
        return set()
    judge_tag = f"{config.judge_backend}/{config.judge_model}"
    done: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("judge") != judge_tag:
                continue
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
