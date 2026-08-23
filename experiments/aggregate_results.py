"""aggregate_results.py -- pool per-scene results into cross-scene plots.

Each scene's results.csv is written under experiments/results/<model>/<scene>/ by
the runner. This concatenates the scenes of a *dataset* into a combined results.csv
(question_id namespaced by scene so identities stay unique across scenes),
recomputes aggregate.csv, and runs the same scope-aware charts from
evaluation/plots.py -- so the cross-scene views are identical in form to the
per-scene ones, just pooled over more questions.

Pooling is **per dataset** because each dataset hosts a different representation
set and different question types; pooling across datasets would
average incomparable rep sets. A combined "all" group is also written for a global
overview -- scope-masking keeps out-of-scope cells blank, but read it with the
ALL-row caveat: compare within a question type, not across.

Only scenes whose registry role is `primary` are pooled (scenes.PRIMARY_SCENE_IDS).
A `stress` or `sensitivity` scene keeps its own per-scene results.csv/report.md and
is reachable through --nonprimary, but it never enters a primary group -- its
results answer a different question (an operational limit, or how much the verdict
set depends on scene choice) and pooling it would put it inside axis verdicts and
headline counts. Role lives in the
registry, never in a directory name, so axes.dataset_of() still resolves an
excluded scene to its host dataset.

Output goes to experiments/results/<model>/_aggregate/<group>/ (the leading
underscore keeps it sorted apart from real scene dirs). The pooled results.csv is
kept too, so the combined cells can be inspected directly.

Usage:
  python -m experiments.aggregate_results                       # every model found
  python -m experiments.aggregate_results --models qwen2.5-14b  # one model
  python -m experiments.aggregate_results --diagnostics         # + latency diagnostic
  python -m experiments.aggregate_results --nonprimary          # + nonprimary_<ds>/
  python -m experiments.aggregate_results --allow-partial       # write short groups
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

# Every dataset hosts exactly three primary scenes, and scene is
# the unit of replication for the separation rule. A per-dataset group built from
# fewer than three is a TRANSITION STATE -- mid scene substitution, or a run that
# only reached some scenes -- and writing it would silently replace a full report
# with a partial one carrying the same filenames and no marker. Skipped by default,
# leaving the existing aggregate intact; --allow-partial writes it anyway.
SCENES_PER_DATASET = 3


def _scene_dirs(model_dir: Path) -> list[Path]:
    """Scene subdirs of a model that actually carry a results.csv (skip the
    _aggregate output dir and any incomplete scene)."""
    return sorted(
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name != AGG_DIRNAME and (d / "results.csv").exists()
    )


def _split_by_role(scene_dirs: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Partition scene dirs into (primary_dirs, [(dir, role), ...]).

    Fails closed on a directory with no registry entry. An unregistered results dir
    is exactly the silent-inclusion case this split exists to prevent: nothing in the
    pooled output names its scenes, so a stray scene would enter every mean and chart
    and read as more replication rather than as a mistake.
    """
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

    # Numeric report (coverage/rank-eligibility, small-n register, axis cards). For
    # the cross-dataset 'all' group the axis cards self-restrict to each axis's host
    # dataset (axes.dataset_of), so no axis ladder is pooled across datasets.
    from evaluation.report import write_report
    write_report(detail_path, aggregate_path)


def aggregate_model(model_dir: Path, diagnostics: bool = False,
                    nonprimary: bool = False, allow_partial: bool = False) -> None:
    scene_dirs = _scene_dirs(model_dir)
    if not scene_dirs:
        print(f"  (no scene results under {model_dir})")
        return

    # Role split BEFORE any grouping. Every exclusion is named on stdout, so a scene
    # leaving the primary pool is always visible in the run log -- the point is that
    # non-primary scenes stay out of the aggregate, not that they stay out of sight.
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

    # Non-primary scenes are pooled only on request, into their own clearly named
    # groups. They are never added to `all` and never share a directory with a
    # primary group, so nothing downstream can read them as replication.
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
    """Which primary groups are under-filled, and why (group -> reason).

    A per-dataset group is partial below SCENES_PER_DATASET. `all` is partial
    whenever any dataset group is, because it pools them -- otherwise the guard would
    protect each dataset report and still let the cross-dataset overview be rewritten
    from a short pool. The nonprimary_* groups have no three-scene expectation and are
    never judged partial.
    """
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
    ap.add_argument("--models", help="comma-separated model dir names (default: all under results/)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also draw the operational latency diagnostic. Off by default: "
                         "latency is confounded by batch composition and GPU contention, "
                         "so it is not a result and not the cost axis (prompt tokens are).")
    ap.add_argument("--nonprimary", action="store_true",
                    help="also pool the stress/sensitivity scenes into their own "
                         "_aggregate/nonprimary_<dataset>/ groups. They are never added "
                         "to a primary group either way; this only builds the pooled "
                         "view used by the scene-substitution sensitivity analysis.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a per-dataset primary group even when it holds fewer "
                         f"than {SCENES_PER_DATASET} scenes. Off by default so a "
                         "transition state (mid scene substitution, or a run that only "
                         "reached some scenes) cannot silently overwrite a full report "
                         "with a partial one under the same filenames.")
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
                        nonprimary=args.nonprimary,
                        allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
