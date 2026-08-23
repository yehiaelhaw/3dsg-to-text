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

from evaluation import dataset, scene_loader, scope, token_count
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
    """Narrow `questions` to `types` (None = keep all). Fails closed on empty.

    A run restricted to no questions is always a mistake -- a typo'd type name, or
    a host that has none of that family -- and it would otherwise look like a
    successful zero-cell run. Raising costs nothing and is checked before the
    responder is ever called.
    """
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
    # Responder context window (ollama num_ctx). Used to flag cells whose prompt
    # overflows it; None for backends that do not expose it -> guard is skipped.
    num_ctx = config.responder_options.get("num_ctx")
    # Exact pre-call token counter for this responder, built once (the tokenizer
    # build costs ~0.2s; per-prompt encoding is negligible). Returns None per
    # prompt on any backend/model it cannot count exactly, which drops
    # _generate_one back to its character heuristic.
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
    done = _cached_keys(path)

    print(f"PHASE 1/2  generate -> {path}  (responder: {responder_tag})")
    n_new = n_skip = n_ctx = n_err = 0

    with path.open("a", encoding="utf-8") as fh:
        for question in questions:
            representations = config.representations or scene_loader.list_representations(
                config.scene_contexts_dir, question.scene_id
            )
            for representation in representations:
                # Skip cells this rep structurally cannot answer -- except the
                # no-information control (inventory), which is deliberately posed
                # the spatial questions it cannot answer so its prior-driven
                # guessing forms the spatial-encoding floor (see METHODOLOGY 3.1.1).
                # Without
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
                    n_err += 1
                    print(f"  ERR   {question.id} | {representation} | {exc}")
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
                        print(f"  GEN   {question.id} | {representation} | rep{repetition} | {rec['latency_ms']:.0f}ms")
                    elif is_context_exceeded(rec["error"]):
                        n_ctx += 1
                        print(f"  CTX!  {question.id} | {representation} | rep{repetition} | prompt {rec['prompt_tokens']} tok, budget {_ctx_limit(num_ctx)}")
                    else:
                        n_err += 1
                        print(f"  ERR   {question.id} | {representation} | rep{repetition}")

    # Free the responder's VRAM before the judge loads (Phase 2) -- but ONLY when
    # they share an Ollama host. On split hosts (responder remote, judge local)
    # there is no shared device to free, so the unload buys nothing and forces a
    # cold reload (weights + CUDA graph + empty KV cache) on the next scene's
    # first prompt -- whose load time then leaks into that cell's latency_ms.
    if _responder_judge_colocated(config):
        responder.unload()
    print(f"PHASE 1/2  done — {n_new} generated, {n_err} errored, "
          f"{n_ctx} context-exceeded, {n_skip} already cached\n")
    # A phase where EVERY attempt errored must not exit 0. It did until
    # 2026-08-13, so the 2026-08-12 topology_metric run reported "[ok]" and
    # "FILL COMPLETE" for three responders whose model was not on the host and
    # which produced 234 errors and zero rows -- indistinguishable from a clean
    # run to any caller reading the exit code. Errors alongside successes stay
    # non-fatal (resume re-attempts them); a total wipeout is a setup fault.
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
        # Pre-call guard. Two tiers, and the first is preferred wherever it is
        # available:
        #
        #   exact  -- evaluation.token_count tokenizes the fully rendered prompt
        #             (chat template, model default system prompt, BOS) with the
        #             tokenizer read out of the GGUF blob the server is actually
        #             serving. Verified equal to prompt_eval_count on 13,320
        #             recorded cells, max delta 0.
        #   len//4 -- the fallback for backends/models token_count does not
        #             support. A true lower bound (measured 0.60-0.91 of the real
        #             count, never above it) but loose: a prompt had to be ~36%
        #             past the window before it tripped.
        #
        # Doing this before the call matters because the post-hoc guard below
        # trusts the backend's reported count, and an overflowing prompt is
        # precisely when this server misreports it -- observed 2026-07: the dense
        # 3RScan `json` prompts (the rep is now `json_pretty`; measured 49,267 and
        # 58,794 tokens on qwen2.5, not the ~45k estimated at the time) silently
        # clipped to a reported 16,386 tokens (below the 32,768 threshold) and
        # scored as real answers. Flagging here is terminal and spends no generation.
        exact_tokens = sizer(prompt) if sizer else None
        if num_ctx:
            limit = _ctx_limit(num_ctx)
            if exact_tokens is not None:
                # Strict `>`, where the heuristic below uses `>=`. A prompt of
                # exactly `limit` leaves exactly _CTX_RESERVE tokens for the
                # answer, which is the reserve's definition of enough -- and an
                # exact count is entitled to that boundary. The heuristic is a
                # lower bound, so it must stay conservative.
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
                est_min_tokens = len(prompt) // 4
                if est_min_tokens >= limit:
                    return _gen_dict(
                        question, representation, repetition, responder_tag,
                        raw_answer="",
                        prompt_tokens=est_min_tokens,
                        error=(f"{CONTEXT_EXCEEDED}: prompt is at least ~{est_min_tokens} tokens "
                               f"(character lower bound) >= context window {num_ctx} "
                               f"(representation cannot fit; not generated)"),
                    )
        gen = responder.generate(prompt)
        # Post-call backstop, unchanged in effect: if the prompt filled (or was
        # truncated to) the responder's window, the representation did not fit --
        # mark the cell context-exceeded (terminal, unscored) rather than letting
        # the judge score a clipped answer as a wrong one. With the exact guard in
        # front of it this should now be unreachable; it stays because it is the
        # only thing that catches a server whose *effective* window is smaller
        # than the num_ctx it accepted.
        error = None
        if (num_ctx and gen.prompt_tokens
                and gen.prompt_tokens >= _ctx_limit(num_ctx)):
            error = (f"{CONTEXT_EXCEEDED}: prompt {gen.prompt_tokens} tokens "
                     f">= context window {num_ctx} (representation truncated; not scored)")
        # Consistency check: the local count and the server's count must agree.
        # A mismatch means the two have drifted apart -- an ollama upgrade that
        # changed a chat template, a re-pull under the same tag, or the server
        # silently truncating. Loud but non-fatal: the recorded prompt_tokens
        # stays the server's own number either way.
        elif exact_tokens is not None and gen.prompt_tokens and exact_tokens != gen.prompt_tokens:
            print(f"  WARN  {question.id} | {representation} | token count drift: "
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
    """Drop cache lines a resumed generation supersedes, in place.

    Generation appends, and _cached_keys does not mark real-error lines done --
    so every resume re-appends retried cells and the stale lines pile up
    (scoring already reads last-write-wins via _load_cache, but the file grows
    unbounded and its keys stop mirroring results.csv, the same defect
    results._compact_for_resume fixes on the CSV side). Same two rules,
    preserving line order: last write wins per key, real-error lines dropped
    (they are about to be re-attempted), context-exceeded sentinels kept.
    """
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
    """(question_id, representation, repetition) already scored without error
    by THIS config's judge.

    Scoping the key to the judge is what makes a score_only pass with a new
    judge (the Gemini confirmatory re-judge over screening-scored rows)
    actually re-judge instead of resume-skipping every cell, while an
    interrupted pass under the same judge still resumes past its own work.
    Rows from another judge -- including its context-exceeded sentinels, which
    _score_one re-emits without a judge call -- are rescored so the directory
    ends up under a single judge tag.
    """
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
