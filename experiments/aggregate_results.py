"""Pool primary-scene results.csv files into per-dataset (and cross-dataset)
aggregate reports; stress/sensitivity scenes are pooled separately via --include-nonprimary.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import csv
from pathlib import Path

from evaluation.axes import dataset_of
from experiments.scenes import PRIMARY, UnregisteredScene, role_of

OUTPUT_ROOT = Path("experiments/results")
AGG_DIRNAME = "_aggregate"
NONPRIMARY_PREFIX = "nonprimary_"

# Require all 3 primary scenes unless --allow-partial is used.
SCENES_PER_DATASET = 3


def _scene_dirs(model_dir: Path) -> list[Path]:
    """Return scene directories containing results.csv."""
    return sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name != AGG_DIRNAME and (d / "results.csv").exists()
    )


def _split_by_role(scene_dirs: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Partition scene directories into primary and non-primary groups."""
    primary: list[Path] = []
    excluded: list[tuple[Path, str]] = []
    unknown: list[str] = []
    for d in scene_dirs:
        try:
            role = role_of(d.name)
        except UnregisteredScene:
            unknown.append(d.name)
            continue
        if role == PRIMARY:
            primary.append(d)
        else:
            excluded.append((d, role))
    if unknown:
        raise SystemExit(
            f"unregistered scene results: {', '.join(sorted(unknown))}\n"
            "Every pooled scene must declare a role in experiments/scenes.py "
            "(primary | stress | sensitivity). Refusing to guess.")
    return primary, excluded


def _pool_rows(scene_dirs: list[Path]) -> tuple[list[str], list[dict]]:
    """Pool scene rows while namespacing question IDs by scene."""
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


def _write_group(group_dir: Path, fieldnames: list[str], rows: list[dict],
                 diagnostics: bool = False) -> None:
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
    plots.plot_per_question(detail_path, diagnostics=diagnostics)

    # Axis cards restrict themselves to their host dataset.
    from evaluation.report import write_report
    write_report(detail_path, aggregate_path)


def aggregate_model(model_dir: Path, diagnostics: bool = False,
                    nonprimary: bool = False, allow_partial: bool = False) -> None:
    scene_dirs = _scene_dirs(model_dir)
    if not scene_dirs:
        print(f"  (no scene results under {model_dir})")
        return

    # Split roles before grouping and report excluded scenes.
    primary_dirs, excluded = _split_by_role(scene_dirs)
    if excluded:
        print(f"  non-primary, excluded from every pooled group: "
              + ", ".join(f"{d.name} ({role})" for d, role in sorted(excluded)))
    if not primary_dirs:
        print(f"  (no PRIMARY scene results under {model_dir})")
        return

    # group scenes by dataset, plus an 'all' group spanning every primary scene
    groups: dict[str, list[Path]] = {"all": list(primary_dirs)}
    for d in primary_dirs:
        groups.setdefault(dataset_of(d.name), []).append(d)

    # Pooled into their own groups on request, never into `all`.
    if nonprimary and excluded:
        for d, _role in excluded:
            groups.setdefault(NONPRIMARY_PREFIX + dataset_of(d.name), []).append(d)

    partial = partial_groups(groups)
    for group, dirs in sorted(groups.items()):
        scenes = ", ".join(d.name for d in dirs)
        print(f"\n[{model_dir.name}] {group}: {len(dirs)} scene(s) -> {scenes}")
        if group in partial and not allow_partial:
            print(f"  SKIPPED: {group} is a transition state ({partial[group]}). "
                  f"Existing {AGG_DIRNAME}/{group}/ left untouched; pass "
                  f"--allow-partial to overwrite it with the partial pool.")
            continue
        fieldnames, rows = _pool_rows(dirs)
        if not rows:
            print("  (no rows)")
            continue
        _write_group(model_dir / AGG_DIRNAME / group, fieldnames, rows, diagnostics)


def partial_groups(groups: dict[str, list[Path]]) -> dict[str, str]:
    """Return under-filled primary groups, propagating partial status to `all`."""
    out: dict[str, str] = {}
    for group, dirs in groups.items():
        if group == "all" or group.startswith(NONPRIMARY_PREFIX):
            continue
        if len(dirs) < SCENES_PER_DATASET:
            out[group] = f"{len(dirs)}/{SCENES_PER_DATASET} primary scenes"
    if out and "all" in groups:
        out["all"] = "pools " + ", ".join(sorted(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool per-scene results into cross-scene plots.")
    ap.add_argument("--models", help="comma-separated model dir names, not ModelProfile names (default: all)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also draw the operational latency diagnostic (off by default; "
                         "confounded, not the cost axis)")
    ap.add_argument("--include-nonprimary", action="store_true",
                    help="also pool stress/sensitivity scenes into their own "
                         "nonprimary_<dataset> groups")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a primary group with fewer than "
                         f"{SCENES_PER_DATASET} scenes instead of skipping it")
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
        aggregate_model(model_dir, diagnostics=args.diagnostics,
                        nonprimary=args.include_nonprimary,
                        allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
