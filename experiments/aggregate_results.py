"""aggregate_results.py -- pool per-scene results into cross-scene plots.

Each scene's results.csv is written under experiments/results/<model>/<scene>/ by
the runner. This concatenates the scenes of a *dataset* into a combined results.csv
(question_id namespaced by scene so identities stay unique across scenes),
recomputes aggregate.csv, and runs the same scope-aware charts from
evaluation/plots.py -- so the cross-scene views are identical in form to the
per-scene ones, just pooled over more questions.

Pooling is **per dataset** because each dataset hosts a different representation
set and different question types (METHODOLOGY 1.2); pooling across datasets would
average incomparable rep sets. A combined "all" group is also written for a global
overview -- scope-masking keeps out-of-scope cells blank, but read it with the
ALL-row caveat (METHODOLOGY 5.6): compare within a question type, not across.

Output goes to experiments/results/<model>/_aggregate/<group>/ (the leading
underscore keeps it sorted apart from real scene dirs). The pooled results.csv is
kept too, so the combined cells can be inspected directly.

Usage:
  python -m experiments.aggregate_results                       # every model found
  python -m experiments.aggregate_results --models qwen2.5-14b  # one model
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import csv
from pathlib import Path

from evaluation.axes import dataset_of

OUTPUT_ROOT = Path("experiments/results")
AGG_DIRNAME = "_aggregate"


def _scene_dirs(model_dir: Path) -> list[Path]:
    """Scene subdirs of a model that actually carry a results.csv (skip the
    _aggregate output dir and any incomplete scene)."""
    return sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name != AGG_DIRNAME and (d / "results.csv").exists()
    )


def _pool_rows(scene_dirs: list[Path]) -> tuple[list[str], list[dict]]:
    """Concatenate scene results.csv rows, namespacing question_id by scene so
    identities stay unique (the per-question paired deltas key on question_id).
    Returns (fieldnames, rows)."""
    fieldnames: list[str] | None = None
    rows: list[dict] = []
    for d in scene_dirs:
        with (d / "results.csv").open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or fieldnames
            for r in reader:
                r["question_id"] = f"{r['scene_id']}:{r['question_id']}"
                rows.append(r)
    return (fieldnames or []), rows


def _write_group(group_dir: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write the pooled results.csv + aggregate.csv and render the charts."""
    from evaluation import plots
    from evaluation.results import _write_aggregate_from_csv

    group_dir.mkdir(parents=True, exist_ok=True)
    detail_path = group_dir / "results.csv"
    aggregate_path = group_dir / "aggregate.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_aggregate_from_csv(detail_path, aggregate_path)
    plots.plot_aggregate(aggregate_path)
    plots.plot_per_question(detail_path)

    # Numeric report (coverage/rank-eligibility, small-n register, axis cards). For
    # the cross-dataset 'all' group the axis cards self-restrict to each axis's host
    # dataset (axes.dataset_of), so no axis ladder is pooled across datasets.
    from evaluation.report import write_report
    write_report(detail_path, aggregate_path)


def aggregate_model(model_dir: Path) -> None:
    scene_dirs = _scene_dirs(model_dir)
    if not scene_dirs:
        print(f"  (no scene results under {model_dir})")
        return

    # group scenes by dataset, plus an 'all' group spanning every scene
    groups: dict[str, list[Path]] = {"all": list(scene_dirs)}
    for d in scene_dirs:
        groups.setdefault(dataset_of(d.name), []).append(d)

    for group, dirs in sorted(groups.items()):
        scenes = ", ".join(d.name for d in dirs)
        print(f"\n[{model_dir.name}] {group}: {len(dirs)} scene(s) -> {scenes}")
        fieldnames, rows = _pool_rows(dirs)
        if not rows:
            print("  (no rows)")
            continue
        _write_group(model_dir / AGG_DIRNAME / group, fieldnames, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool per-scene results into cross-scene plots.")
    ap.add_argument("--models", help="comma-separated model dir names (default: all under results/)")
    args = ap.parse_args()

    if not OUTPUT_ROOT.exists():
        raise SystemExit(f"no results root: {OUTPUT_ROOT}")

    model_dirs = sorted(d for d in OUTPUT_ROOT.iterdir() if d.is_dir())
    if args.models:
        wanted = {n.strip() for n in args.models.split(",") if n.strip()}
        have = {d.name for d in model_dirs}
        missing = wanted - have
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(sorted(missing))} "
                             f"(have: {', '.join(sorted(have))})")
        model_dirs = [d for d in model_dirs if d.name in wanted]

    for model_dir in model_dirs:
        aggregate_model(model_dir)


if __name__ == "__main__":
    main()
