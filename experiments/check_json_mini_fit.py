"""check_json_mini_fit.py -- does the minified ceiling fit the responder window?

Phase 0 of the ceiling-serializer split, and the gate on everything after it. The
`json` representation overflows the 32k window on the two dense 3RScan scenes and
scores CONTEXT_EXCEEDED. Roughly 45% of the file is `indent=2` whitespace, so the
question is whether minifying it -- losing no information at all -- brings the
worst scene back inside the window.

Costs ONE generation, not a scene sweep: the prompt is the representation plus a
question stem, and stems vary by only tens of tokens, so 47 questions would buy 47x
the cost for the same number. `num_predict=1` keeps the completion at one token;
prompt evaluation happens regardless, which is the part being measured.

Writes NOTHING to experiments/results/. Reads results.csv only to look up the
control's recorded token count.

Two measurements, in this order:

  CONTROL  the LARGEST prompt in the corpus with a known-good recorded count,
           re-run on that exact question. Two things must hold for the target
           number to mean anything: the harness must measure what the runner
           measures, and the server must actually be serving the full window.
           Taking the largest recorded prompt tests both at once -- a small
           control tests only the first.

           This is not ceremony. METHODOLOGY 2.4 records a 2026-07 incident where
           a ~45k-token prompt came back reported as 16,386 tokens and the clipped
           answer was scored as real. The same 16,386 reappeared on 2026-08-09
           when this script was first run: ollama had split its 32,768-token
           context across two parallel slots because another run was generating
           at the time, so this request got 16,384. The first version of this
           control used the SMALLEST recorded prompt (11,692) and passed anyway,
           because 11,692 fits inside a halved slot. Hence `max`, not `min`.

  TARGET   json_mini on 3rscan_7f30f36c (the largest scene) against its LONGEST
           question, i.e. the worst case in the corpus.

Usage (needs scripts/ssh_tunnel.ps1 -Server voxel):
    python -m experiments.check_json_mini_fit
    python -m experiments.check_json_mini_fit --model qwen2.5-14b
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import evaluation.core  # noqa: F401  -- import raises csv.field_size_limit for results.csv
from evaluation.llm.factory import create_provider
from evaluation.runner import _CTX_RESERVE, _RESPONDER_PROMPT
from experiments.models import MODEL_PROFILES

CONTROL_REP = "json"          # control scene is chosen by size, not pinned -- see below
TARGET_SCENE, TARGET_REP = "3rscan_7f30f36c", "json_mini"

QA_ROOT = Path("experiments/scripts")
CONTEXTS = Path("scene_contexts")
RESULTS = Path("experiments/results")


def _questions(scene_id: str) -> dict[str, str]:
    path = QA_ROOT / scene_id / "keyfact-qa.jsonl"
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[str(q["id"])] = q["text"]
    return out


def _recorded(model: str, rep: str) -> dict[tuple[str, str], int]:
    """(scene_id, question_id) -> prompt_tokens, over every cell that generated.

    Error rows are excluded deliberately: a CONTEXT_EXCEEDED row carries the
    pre-call character estimate in prompt_tokens (runner.py:180-186), not a
    tokenizer count, so treating one as ground truth would be circular.
    """
    out: dict[tuple[str, str], int] = {}
    for path in sorted((RESULTS / model).glob("*/results.csv")):
        if path.parent.name.startswith("_"):
            continue
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["representation"] == rep and not r["error"] and r["prompt_tokens"]:
                    out[(r["scene_id"], r["question_id"])] = int(r["prompt_tokens"])
    return out


def _generation_in_flight() -> str | None:
    """Another run_experiments process, or None.

    A concurrent request makes ollama split its context across parallel slots, so
    this one would silently get a fraction of num_ctx and its prompt would be
    truncated -- which is exactly how the 2026-08-09 false negative happened.
    Best-effort: if the probe itself fails, say so rather than blocking.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" |"
             " Where-Object { $_.CommandLine -like '*run_experiments*' } |"
             " Measure-Object).Count"],
            capture_output=True, text=True, timeout=30,
        )
        n = int((out.stdout or "0").strip() or 0)
        return f"{n} run_experiments process(es) running" if n else None
    except Exception as e:                                    # probe unavailable
        return f"could not check ({type(e).__name__}); verify manually"


def _context(scene_id: str, rep: str) -> str:
    matches = list((CONTEXTS / scene_id).glob(f"{rep}.*"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {rep}.* in {CONTEXTS/scene_id}, got {matches}")
    return matches[0].read_text(encoding="utf-8")


def _tiktoken_estimate(text: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def _measure(provider, context: str, question: str) -> int:
    """Reported prompt_eval_count for the prompt the runner would have built."""
    prompt = _RESPONDER_PROMPT.format(context=context, question=question)
    return provider.generate(prompt).prompt_tokens


def _diagnose(provider, context: str, question: str, full: int, limit: int) -> None:
    """Is the reported count real, or a truncation plateau?

    Feed growing prefixes of the same text. Token count is very nearly linear in
    characters for one document, so a real count keeps climbing; a server-side cap
    shows up as consecutive prefixes reporting the SAME number. The pretty form of
    the same scene is measured last as a reference: it is known to be far over the
    window, so whatever it reports is what truncation looks like here.
    """
    print("\nDIAGNOSE  growing prefixes of the same text (truncated mid-token on "
          "purpose; we are counting, not parsing)\n")
    print(f"          {'prefix':>7}  {'chars':>9}  {'reported':>9}  {'chars/tok':>9}")
    seen: list[int] = []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        part = context[:int(len(context) * frac)]
        n = full if frac == 1.0 else _measure(provider, part, question)
        seen.append(n)
        print(f"          {frac:>6.0%}  {len(part):>9,}  {n:>9,}  {len(part)/n:>9.2f}")

    # Two ways a cap shows up, and the count must be checked for BOTH. A plateau
    # (equal consecutive values) is the obvious one. A DROP is the other, and it is
    # the one this server actually produces: past num_ctx the reported figure falls
    # back to what was really evaluated after truncation, so the series rises and
    # then collapses. An earlier version tested only the plateau and duly declared a
    # 27,014 -> 16,386 collapse "REAL".
    honest = seen[:-1] if seen[-1] < seen[-2] else seen
    capped = len(honest) < len(seen) or seen[-1] == seen[-2]

    print()
    if not capped:
        print(f"  REAL. The count rises with every prefix, so {seen[-1]:,} is a genuine\n"
              f"  measurement and this prompt fits inside {limit:,}.")
        return

    # Extrapolate from the last honest segment: near-linear within one document.
    step = honest[-1] - honest[-2]
    frac_each = 1.0 / (len(seen) - 1)
    est = honest[-1] + step * ((1.0 - frac_each * (len(honest) - 1)) / frac_each)
    verdict = "OVER" if est >= limit else "under"
    print(f"  CAPPED, NOT MEASURED. The series rises to {honest[-1]:,} and then reports\n"
          f"  {seen[-1]:,} for MORE text -- so {seen[-1]:,} is what this server returns once a\n"
          f"  prompt passes num_ctx, not a token count. That also identifies the 16,386\n"
          f"  in METHODOLOGY 2.4: an overflow under-report, not a misconfiguration.\n\n"
          f"  Extrapolating the last honest slope (+{step:,} per {frac_each:.0%} of text):\n"
          f"      full prompt ~= {est:,.0f} tokens   vs limit {limit:,}   -> {verdict}\n")
    if est >= limit:
        print(f"  json_mini does NOT fit on this scene. Stop: do not migrate results,\n"
              f"  report or thesis. The token saving is real and separately quantifiable,\n"
              f"  but it did not solve the coverage problem that motivated the change.")
    else:
        print(f"  Inconclusive: the extrapolation clears the limit but the direct\n"
              f"  measurement was capped. Re-measure with a larger num_ctx before acting.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen2.5-14b", help="ModelProfile name")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="skip the in-flight-generation check (the result will "
                         "almost certainly be a truncated false negative)")
    ap.add_argument("--diagnose", action="store_true",
                    help="feed growing prefixes of the target and report how the "
                         "count scales -- distinguishes a genuinely cheap prompt "
                         "from one that is being silently truncated")
    args = ap.parse_args()

    busy = _generation_in_flight()
    if busy and not args.allow_concurrent:
        raise SystemExit(
            f"ABORT: {busy}.\n"
            "  A concurrent request makes ollama divide num_ctx across parallel\n"
            "  slots, so this measurement would be truncated and read as a fit\n"
            "  failure that is really a contention artefact. Wait for the run to\n"
            "  finish (pausing is not enough -- a paused process keeps its slot),\n"
            "  or pass --allow-concurrent if you know the server is idle."
        )

    profile = next((p for p in MODEL_PROFILES if p.name == args.model), None)
    if profile is None:
        raise SystemExit(f"unknown model {args.model!r}; "
                         f"known: {', '.join(p.name for p in MODEL_PROFILES)}")

    num_ctx = profile.options.get("num_ctx")
    if not num_ctx:
        raise SystemExit(f"{profile.name} declares no num_ctx; nothing to check against")
    limit = num_ctx - _CTX_RESERVE

    # Same construction as runner.generate_responses, plus a 1-token completion cap.
    options = dict(profile.options)
    options["num_predict"] = 1
    provider = create_provider(profile.backend, profile.model, options)

    print(f"responder {profile.backend}/{profile.model}   "
          f"num_ctx {num_ctx:,} - reserve {_CTX_RESERVE} = limit {limit:,}\n")

    # --- CONTROL: the largest known-good prompt, so it tests the window too ---
    recorded = _recorded(args.model, CONTROL_REP)
    if not recorded:
        raise SystemExit(f"no recorded {CONTROL_REP} rows under {args.model}; "
                         f"cannot validate the harness")
    (scene, qid), expected = max(recorded.items(), key=lambda kv: kv[1])
    got = _measure(provider, _context(scene, CONTROL_REP), _questions(scene)[qid])
    drift = abs(got - expected) / expected

    print(f"CONTROL  {scene} / {CONTROL_REP} / {qid} (largest known-good prompt)")
    print(f"         recorded {expected:>7,}   remeasured {got:>7,}   drift {drift:.2%}")
    if drift > 0.01:
        print(f"\n  STOP. The harness did not reproduce a count it has produced before.\n"
              f"  If the remeasured value is a round power of two (16,384 / 8,192 ...),\n"
              f"  the server is serving a fraction of num_ctx -- most likely another\n"
              f"  request holds a parallel slot. Otherwise check that the tunnel points\n"
              f"  at the same server and the model tag is unchanged.")
        return
    print(f"         reproduces at {expected:,} tokens -> the full window is being "
          f"served, and the target number below is trustworthy\n")

    # --- TARGET -----------------------------------------------------------
    context = _context(TARGET_SCENE, TARGET_REP)
    questions = _questions(TARGET_SCENE)
    qid = max(questions, key=lambda k: len(questions[k]))
    got = _measure(provider, context, questions[qid])
    est = _tiktoken_estimate(context)

    print(f"TARGET   {TARGET_SCENE} / {TARGET_REP} / {qid} (longest stem, worst case)")
    print(f"         reported   {got:>7,}   limit {limit:>7,}   "
          f"{'margin ' + format(limit - got, ',') if got < limit else 'OVER by ' + format(got - limit, ',')}")
    if est:
        print(f"         tiktoken   {est:>7,}  (o200k, rep only)   "
              f"ratio reported/tiktoken {got/est:.3f}")

    if args.diagnose:
        _diagnose(provider, context, questions[qid], got, limit)
        return

    print()
    if got >= limit:
        print("  DOES NOT FIT. Minification did not solve the coverage problem. Stop "
              "here: do not migrate results, report or thesis. The remaining question "
              "-- whether the ~45% token saving alone justifies adopting json_mini -- "
              "is a separate decision about the ceiling's cost, not its coverage.")
    elif est and got / est < 0.9:
        print("  SUSPICIOUS. The reported count is far below the tokenizer estimate, "
              "which is the signature of the 2026-07 under-reporting incident rather "
              "than of a prompt that fits. Do not proceed on this number.")
    else:
        print("  FITS. Proceed with the full json -> json_pretty migration and make "
              "json_mini the canonical ceiling (phases 1-8 of the plan).")


if __name__ == "__main__":
    main()
