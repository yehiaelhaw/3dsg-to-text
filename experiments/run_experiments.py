"""run_experiments.py -- run the responder x scene evaluation matrix.

For every (ModelProfile in models.MODEL_PROFILES) x (Scene in scenes.SCENES) build
one EvalConfig and run it, writing to results/<profile.name>/<scene_id>/. The
default judge is the fixed gemma2:9b screening judge; pass --score-only with
--judge-backend/--judge-model to re-judge an existing responses.jsonl cache with
a different judge instead (e.g. Gemini for final numbers) -- no regeneration, no
responder VRAM. Runs are resumable (resume=True), so a crash -- or a
--models/--scenes subset -- can be re-run without redoing finished cells.

Usage:
  python -m experiments.run_experiments                          # full matrix
  python -m experiments.run_experiments --models qwen2.5-14b     # one model, all scenes
  python -m experiments.run_experiments --scenes Brinnon,procthor_train1
  python -m experiments.run_experiments --list                   # print matrix, run nothing
  python -m experiments.run_experiments --models qwen2.5-14b --scenes Brinnon \\
      --score-only --judge-backend gemini --judge-model gemini-2.5-flash --faithfulness
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
from itertools import product

from experiments.models import MODEL_PROFILES
from experiments.scenes import SCENES

# Fixed screening judge (not swept). Gemini re-judge is a separate score_only pass.
JUDGE_BACKEND = "ollama"
JUDGE_MODEL = "gemma2:9b"
JUDGE_OPTIONS = {"temperature": 0.0}

OUTPUT_ROOT = "experiments/results"


def _select(items, names, kind, key):
    """Filter `items` to the comma-separated `names`, or return all if None."""
    if not names:
        return items
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    by = {key(i): i for i in items}
    missing = [n for n in wanted if n not in by]
    if missing:
        raise SystemExit(f"unknown {kind}: {', '.join(missing)} "
                         f"(have: {', '.join(by)})")
    return [by[n] for n in wanted]


def build_config(profile, scene, *, judge_backend=None, judge_model=None,
                  score_only=False, compute_faithfulness=False, representations=None):
    """One EvalConfig for a (model x scene) cell. Imports lazily so --list is cheap.

    judge_backend/judge_model default to the fixed gemma2:9b screening judge;
    pass overrides (e.g. from --score-only --judge-backend gemini) to re-judge
    an existing cache with a different judge instead.

    representations overrides the scene's rep set (from --reps) for subset runs --
    e.g. a second responder on headline cells only, or refilling one purged rep.
    scenes.py stays the source of truth for what a FULL run covers.
    """
    from evaluation.config import EvalConfig
    return EvalConfig(
        dataset_path=scene.dataset_path,
        scene_contexts_dir="scene_contexts",
        output_dir=f"{OUTPUT_ROOT}/{profile.name}/{scene.scene_id}",
        responder_backend=profile.backend,
        responder_model=profile.model,
        responder_options=profile.options,
        judge_backend=judge_backend or JUDGE_BACKEND,
        judge_model=judge_model or JUDGE_MODEL,
        # temperature=0.0 is a reasonable determinism default for any judge backend.
        judge_options=JUDGE_OPTIONS,
        representations=representations or scene.representations,
        # One draw per cell: the (rep x type) group mean already averages over many
        # independent questions, so a repeat is pseudo-replication, not new signal.
        repetitions=1,
        # Screening pass: gemma2's 8k window can't hold the large contexts for the
        # faithfulness check. answer_correctness needs only the answer. (Gemini later.)
        compute_faithfulness=compute_faithfulness,
        resume=True,
        score_only=score_only,
        # Fail-closed scope validation (METHODOLOGY S3.1 / S5.7): abort before the
        # first LLM call if any representation part or question type lacks a
        # scope.py declaration. Free when declarations are complete; catches the
        # silent-typo mis-scope class on every run, not just "final" ones.
        strict_scope=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run the responder x scene evaluation matrix.")
    ap.add_argument("--models", help="comma-separated ModelProfile names (default: all)")
    ap.add_argument("--scenes", help="comma-separated scene_ids (default: all)")
    ap.add_argument("--reps", help="comma-separated representation names: override the "
                                    "rep set of every selected scene (subset runs, e.g. "
                                    "a second responder on headline cells only; resume "
                                    "still skips finished cells)")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--generate-only", action="store_true",
                     help="run responder generation only, skip judging "
                          "(score later with score_only against the cached responses.jsonl)")
    ap.add_argument("--score-only", action="store_true",
                     help="skip generation; judge the existing cached responses.jsonl "
                          "instead (requires --judge-backend/--judge-model, e.g. to "
                          "re-judge with Gemini for final numbers)")
    ap.add_argument("--judge-backend", help="override the judge backend "
                                             f"(default: {JUDGE_BACKEND} screening judge)")
    ap.add_argument("--judge-model", help=f"override the judge model (default: {JUDGE_MODEL})")
    ap.add_argument("--faithfulness", action="store_true",
                     help="also compute the faithfulness metric (off by default -- "
                          "gemma2:9b's context window can't hold it during screening)")
    args = ap.parse_args()

    if args.generate_only and args.score_only:
        raise SystemExit("--generate-only and --score-only are mutually exclusive")
    if args.score_only and not (args.judge_backend and args.judge_model):
        raise SystemExit("--score-only requires --judge-backend and --judge-model "
                          "(re-judging with the same screening judge is a no-op)")

    models = _select(MODEL_PROFILES, args.models, "model", lambda m: m.name)
    scenes = _select(SCENES, args.scenes, "scene", lambda s: s.scene_id)
    reps = ([r.strip() for r in args.reps.split(",") if r.strip()]
            if args.reps else None)
    combos = list(product(models, scenes))

    print(f"matrix: {len(models)} model(s) x {len(scenes)} scene(s) = {len(combos)} run(s)")
    for prof, scene in combos:
        print(f"  {prof.name:18s} x {scene.scene_id:18s} "
              f"-> {OUTPUT_ROOT}/{prof.name}/{scene.scene_id}")
    if args.list:
        return

    if args.generate_only:
        from evaluation.runner import generate_responses
        for i, (prof, scene) in enumerate(combos, 1):
            print(f"\n=== [{i}/{len(combos)}] {prof.name} x {scene.scene_id} (generate only) ===")
            generate_responses(build_config(prof, scene, representations=reps))
        return

    from evaluation.results import save
    from evaluation.runner import iter_records
    from experiments.aggregate_results import aggregate_model
    from pathlib import Path
    for i, (prof, scene) in enumerate(combos, 1):
        tag = f"{prof.name} x {scene.scene_id}"
        if args.score_only:
            tag += f" (score only, judge={args.judge_backend}/{args.judge_model})"
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")
        cfg = build_config(
            prof, scene,
            judge_backend=args.judge_backend, judge_model=args.judge_model,
            score_only=args.score_only, compute_faithfulness=args.faithfulness,
            representations=reps,
        )
        save(iter_records(cfg), cfg)

    print("\n=== cross-scene aggregate ===")
    for prof in models:
        aggregate_model(Path(f"{OUTPUT_ROOT}/{prof.name}"))


if __name__ == "__main__":
    main()
