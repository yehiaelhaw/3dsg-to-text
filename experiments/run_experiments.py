"""Run the responder × scene evaluation matrix."""
from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
from itertools import product

from experiments.models import MODEL_PROFILES
from experiments.scenes import SCENES

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
                  score_only=False, compute_faithfulness=False, representations=None,
                  question_types=None):
    """Build one EvalConfig for a model × scene cell."""
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
        # Deterministic judge decoding.
        judge_options=JUDGE_OPTIONS,
        representations=representations or scene.representations,
        question_types=question_types,
        # One responder draw per cell.
        repetitions=1,
        # Faithfulness is disabled during Gemma screening due to context limits.
        compute_faithfulness=compute_faithfulness,
        resume=True,
        score_only=score_only,
        # Require explicit scope declarations before running.
        strict_scope=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run the responder x scene evaluation matrix.")
    ap.add_argument("--models", help="comma-separated ModelProfile names (default: all)")
    ap.add_argument("--scenes", help="comma-separated scene_ids (default: all)")
    ap.add_argument("--representations", help="comma-separated representation names "
                                    "(default: each scene's full set)")
    ap.add_argument("--question-types", help="comma-separated question types to run "
                                     "(default: every in-scope type)")
    ap.add_argument("--plan", action="store_true", help="print the planned matrix and exit")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true",
                     help="generate responses only; judge later with --score-only")
    mode.add_argument("--score-only", action="store_true",
                     help="re-judge cached responses.jsonl instead of generating "
                          "(requires --judge-backend/--judge-model)")
    ap.add_argument("--judge-backend", help="override the judge backend "
                                             f"(default: {JUDGE_BACKEND} screening judge)")
    ap.add_argument("--judge-model", help=f"override the judge model (default: {JUDGE_MODEL})")
    ap.add_argument("--faithfulness", action="store_true",
                     help="also compute the faithfulness metric (off by default)")
    ap.add_argument("--no-report", action="store_true",
                     help="skip plots/report.md (CSVs still written; rebuild "
                          "later with aggregate_results.py)")
    args = ap.parse_args()

    if args.score_only and not (args.judge_backend and args.judge_model):
        raise SystemExit("--score-only requires --judge-backend and --judge-model "
                          "(re-judging with the same screening judge is a no-op)")

    models = _select(MODEL_PROFILES, args.models, "model", lambda m: m.name)
    scenes = _select(SCENES, args.scenes, "scene", lambda s: s.scene_id)
    reps = ([r.strip() for r in args.representations.split(",") if r.strip()]
            if args.representations else None)
    qtypes = ([t.strip() for t in args.question_types.split(",") if t.strip()]
              if args.question_types else None)
    combos = list(product(models, scenes))

    print(f"matrix: {len(models)} model(s) x {len(scenes)} scene(s) = {len(combos)} run(s)")
    for prof, scene in combos:
        print(f"  {prof.name:18s} x {scene.scene_id:18s} "
              f"-> {OUTPUT_ROOT}/{prof.name}/{scene.scene_id}")
    if args.plan:
        return

    if args.generate_only:
        from evaluation.runner import generate_responses
        for i, (prof, scene) in enumerate(combos, 1):
            print(f"\n=== [{i}/{len(combos)}] {prof.name} x {scene.scene_id} (generate only) ===")
            generate_responses(build_config(prof, scene, representations=reps,
                                            question_types=qtypes))
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
            representations=reps, question_types=qtypes,
        )
        save(iter_records(cfg), cfg, write_report=not args.no_report)

    if args.no_report:
        print("\n--no-report: skipping cross-scene aggregate "
              "(rerun aggregate_results.py later to build it)")
        return

    print("\n=== cross-scene aggregate ===")
    for prof in models:
        aggregate_model(Path(f"{OUTPUT_ROOT}/{prof.name}"))


if __name__ == "__main__":
    main()
