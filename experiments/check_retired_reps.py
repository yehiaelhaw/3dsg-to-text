"""check_retired_reps.py -- fail closed if a retired representation name survives
anywhere in experiments/results/.

This exists because a half-migrated results tree is **indistinguishable from a
clean one by inspection**: no filename anywhere embeds a representation name (the
only rep-derived filenames in the repo are `axis_card_<id>.png` and
`cost_quality_<host>.png`, and neither carries a rep), so a stale-file sweep can
never flag one. The name lives only inside the `representation` column of
results.csv / aggregate.csv and the `representation` key of responses.jsonl. If a
directory is missed by a migration, every downstream mean silently gains a third
representation group -- the resume compactors are same-key dedupers, not garbage
collectors, so a non-error stale row is never dropped on its own.

The retired names come from `evaluation.scope.RETIRED_REPS`, so this guard and the
runner's own strict-scope abort can never disagree about what is retired.

Scopes
------
`--scope inputs` (default) checks the hand-written inputs: every results.csv and
responses.jsonl outside an `_aggregate/` directory. This is what
`aggregate_results.main` runs at start-up -- scanning `_aggregate/` there would
abort the very run that rewrites it.

`--scope all` additionally checks the derived artifacts (per-scene aggregate.csv
and everything under `_aggregate/`). Run it *after* a full re-aggregation, as the
final gate.

Not checked, deliberately:
  * `*.bak` -- there are ~25 legitimate pre-migration backups that contain the
    retired name by definition. A permanently-red guard is an ignored guard.
  * `report.md` -- prose, regenerated wholesale from the CSVs it summarises.

Usage:
  python -m experiments.check_retired_reps                  # inputs only
  python -m experiments.check_retired_reps --scope all      # + derived artifacts
  python -m experiments.check_retired_reps --root some/dir  # non-default tree
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Imported for two reasons: RETIRED_REPS is the single source of truth for what is
# retired, and evaluation.core raises csv.field_size_limit on import. That bump is
# load-bearing here -- answers and rubric_reasoning routinely contain embedded
# newlines and run to tens of kilobytes, so a naive line scan mis-parses them and a
# default-limit DictReader raises outright.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation.core  # noqa: F401,E402  (imported for its field_size_limit bump)
from evaluation.scope import RETIRED_REPS  # noqa: E402

RESULTS_ROOT = Path("experiments/results")
AGG_DIRNAME = "_aggregate"
INPUT_NAMES = {"results.csv", "responses.jsonl"}
DERIVED_NAMES = {"aggregate.csv", "results.csv"}


def _files(root: Path, scope: str) -> list[Path]:
    """Every file to scan, in a stable order.

    Walks recursively rather than assuming `<model>/<scene>/`: the tree has held a
    surprise layout before (a Copy-Item whose destination already existed nested a
    whole model directory one level deeper, where a two-level walk could not see
    it), and a guard that trusts the layout it is meant to police is not a guard.
    """
    out: list[Path] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix == ".bak":
            continue
        in_agg = AGG_DIRNAME in f.relative_to(root).parts
        if scope == "inputs":
            if not in_agg and f.name in INPUT_NAMES:
                out.append(f)
        else:
            if f.name in INPUT_NAMES or f.name in DERIVED_NAMES:
                out.append(f)
    return out


def _hits_csv(path: Path, retired: set[str]) -> dict[str, int]:
    """{retired name: row count} in this CSV's `representation` column."""
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "representation" not in reader.fieldnames:
            return counts
        for row in reader:
            rep = (row.get("representation") or "").strip()
            # A combo ("a+b") is retired if any part is.
            for part in rep.split("+"):
                if part in retired:
                    counts[part] = counts.get(part, 0) + 1
    return counts


def _hits_jsonl(path: Path, retired: set[str]) -> dict[str, int]:
    """{retired name: record count} in this JSONL's `representation` key.

    Substring pre-filter first, then parse only the lines that could match: the
    files run to hundreds of MB and the overwhelming majority of lines cannot hit.
    The pre-filter is deliberately loose (it matches `json_mini` too) because it
    only decides whether to parse -- the decision itself is made on the parsed key.
    """
    counts: dict[str, int] = {}
    needles = [f'"{name}"' for name in retired]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not any(n in line for n in needles):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rep = (rec.get("representation") or "").strip()
            for part in rep.split("+"):
                if part in retired:
                    counts[part] = counts.get(part, 0) + 1
    return counts


def scan(root: Path, scope: str = "inputs") -> tuple[list[tuple[Path, dict[str, int]]], int]:
    """(offending files with their per-name counts, number of files scanned)."""
    retired = set(RETIRED_REPS)
    files = _files(root, scope)
    if not retired:
        return [], len(files)
    bad: list[tuple[Path, dict[str, int]]] = []
    for f in files:
        hits = _hits_jsonl(f, retired) if f.suffix == ".jsonl" else _hits_csv(f, retired)
        if hits:
            bad.append((f, hits))
    return bad, len(files)


def assert_clean(root: Path, scope: str = "inputs", quiet: bool = False) -> None:
    """Raise SystemExit(1) if any retired name survives under `root`.

    Callable from other entry points -- `aggregate_results.main` uses it per model
    directory so a clean model is never blocked by an unrelated dirty one, while the
    standalone `--scope all` run still gates the whole tree.
    """
    if not RETIRED_REPS:
        return
    bad, n_files = scan(root, scope)
    if not bad:
        if not quiet:
            print(f"ok -- no retired representation in {n_files} file(s) under {root} "
                  f"(scope: {scope}; retired: {', '.join(sorted(RETIRED_REPS))})")
        return

    total = sum(sum(h.values()) for _, h in bad)
    print(f"FAIL -- {total} row(s) still name a retired representation, "
          f"in {len(bad)} of {n_files} file(s) scanned under {root}:", file=sys.stderr)
    for f, hits in bad:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(hits.items()))
        print(f"  {f}  ({detail})", file=sys.stderr)
    print("\nWhy this is retired:", file=sys.stderr)
    for name in sorted({k for _, h in bad for k in h}):
        print(f"  {name}: {RETIRED_REPS[name]}", file=sys.stderr)
    print("\nMigrate with: python -m experiments.scripts.rename_json_representation",
          file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fail if a retired representation name survives in the results tree.")
    ap.add_argument("--scope", choices=("inputs", "all"), default="inputs",
                    help="inputs: results.csv/responses.jsonl outside _aggregate (default). "
                         "all: also per-scene aggregate.csv and everything under _aggregate/.")
    ap.add_argument("--root", default=str(RESULTS_ROOT), help="results tree to scan")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no results root: {root}")
    if not RETIRED_REPS:
        print("no retired representations declared; nothing to check")
        return
    assert_clean(root, args.scope)


if __name__ == "__main__":
    main()
