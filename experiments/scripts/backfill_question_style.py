"""backfill_question_style.py -- add the `question_style` column to results.csv
files written before the tag was plumbed through the pipeline.

The tag (`natural` / `constructed`) has always lived in the authored QA files; it
was simply dropped at load time, so every committed results.csv predates the
column. The vocabulary-coupling re-cut (thesis 4.5 / 4.6 -- the remedy the declared
confound on the derived poles points at) reads it, and re-running the study to
populate one column would cost a full responder + judge pass for data we already
have. This joins it back in instead:

    (scene_id, question_id)  in  experiments/results/<model>/<scene>/results.csv
    (scene_id, id)           in  experiments/scripts/<scene>/keyfact-qa.jsonl

Only the question types exposed to a derived pole are tagged (connectivity plus the
3RScan relation family); everything else joins to an empty string, which is the
same thing a fresh run writes for an untagged question.

Two invariants, because results.csv is the expensive artifact in this repo:
  1. Idempotent -- a file that already has the column is left untouched.
  2. All-or-nothing per model dir. aggregate_results._pool_rows takes `fieldnames`
     from the last scene it reads, so a model whose scenes disagree on their columns
     raises in DictWriter. Either every scene of a model gets the column or none do.

The _aggregate/ dirs are not touched: re-running aggregate_results.py re-pools them
from the backfilled per-scene files.

Usage:
  python -m experiments.scripts.backfill_question_style --dry-run   # report only
  python -m experiments.scripts.backfill_question_style             # write
  python -m experiments.scripts.backfill_question_style --models qwen2.5-14b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Import for the column list so this can never drift from the pipeline's own order.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.core import CSV_COLUMNS  # noqa: E402

RESULTS_ROOT = Path("experiments/results")
QA_ROOT = Path("experiments/scripts")
QA_FILENAME = "keyfact-qa.jsonl"
COLUMN = "question_style"
AGG_DIRNAME = "_aggregate"


def _load_styles() -> dict[tuple[str, str], str]:
    """(scene_id, question_id) -> style, over every authored QA file."""
    styles: dict[tuple[str, str], str] = {}
    for qa_path in sorted(QA_ROOT.glob(f"*/{QA_FILENAME}")):
        with qa_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                styles[(q["scene_id"], str(q["id"]))] = q.get("question_style") or ""
    return styles


def _scene_files(model_dir: Path) -> list[Path]:
    """results.csv of every real scene dir (the _aggregate output is regenerated,
    not backfilled)."""
    return sorted(
        d / "results.csv" for d in model_dir.iterdir()
        if d.is_dir() and d.name != AGG_DIRNAME and (d / "results.csv").exists()
    )


def _fieldnames_with_column(existing: list[str]) -> list[str]:
    """Insert the column where CSV_COLUMNS puts it (after question_type), so a
    backfilled file is byte-comparable with one a fresh run would write."""
    if COLUMN in existing:
        return existing
    out = list(existing)
    anchor = CSV_COLUMNS.index(COLUMN) - 1
    after = CSV_COLUMNS[anchor] if anchor >= 0 else None
    out.insert(out.index(after) + 1 if after in out else len(out), COLUMN)
    return out


def backfill_file(path: Path, styles: dict[tuple[str, str], str],
                  dry_run: bool) -> tuple[int, int, bool]:
    """Returns (rows, unmatched, changed)."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if COLUMN in fieldnames:
        return len(rows), 0, False
    if not rows:
        return 0, 0, False

    unmatched = 0
    for r in rows:
        key = (r["scene_id"], r["question_id"])
        if key in styles:
            r[COLUMN] = styles[key]
        else:
            r[COLUMN] = ""
            unmatched += 1

    if not dry_run:
        out_fields = _fieldnames_with_column(fieldnames)
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(rows)
        # Re-read before replacing: the row count must survive the round-trip, or
        # we would be trading an expensive artifact for a truncated one.
        with tmp.open(newline="", encoding="utf-8") as fh:
            written = sum(1 for _ in csv.DictReader(fh))
        if written != len(rows):
            tmp.unlink()
            raise SystemExit(f"{path}: wrote {written} rows, expected {len(rows)} -- aborted")
        os.replace(tmp, path)

    return len(rows), unmatched, True


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill question_style into results.csv files.")
    ap.add_argument("--models", help="comma-separated model dir names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    if not RESULTS_ROOT.exists():
        raise SystemExit(f"no results root: {RESULTS_ROOT}")

    styles = _load_styles()
    tagged = sum(1 for v in styles.values() if v)
    print(f"QA tags loaded: {len(styles)} questions, {tagged} carrying a style tag\n")
    if not styles:
        raise SystemExit(f"no QA files under {QA_ROOT}/*/{QA_FILENAME}")

    model_dirs = sorted(d for d in RESULTS_ROOT.iterdir() if d.is_dir())
    if args.models:
        wanted = {n.strip() for n in args.models.split(",") if n.strip()}
        have = {d.name for d in model_dirs}
        if wanted - have:
            raise SystemExit(f"unknown model(s): {', '.join(sorted(wanted - have))} "
                             f"(have: {', '.join(sorted(have))})")
        model_dirs = [d for d in model_dirs if d.name in wanted]

    total_rows = total_unmatched = total_changed = 0
    for model_dir in model_dirs:
        files = _scene_files(model_dir)
        if not files:
            continue
        print(f"[{model_dir.name}]")
        for path in files:
            rows, unmatched, changed = backfill_file(path, styles, args.dry_run)
            state = ("would add" if args.dry_run else "added") if changed else "already present"
            flag = f"  !! {unmatched} unmatched" if unmatched else ""
            print(f"  {path.parent.name:22} {rows:5} rows  {state}{flag}")
            total_rows += rows
            total_unmatched += unmatched
            total_changed += int(changed)
        print()

    verb = "would backfill" if args.dry_run else "backfilled"
    print(f"{verb} {total_changed} file(s), {total_rows} rows, {total_unmatched} unmatched")
    if total_unmatched:
        raise SystemExit("unmatched rows found -- a results row has no authored question; "
                         "investigate before writing")


if __name__ == "__main__":
    main()
