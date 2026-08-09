"""rename_json_representation.py -- migrate stored `json` rows to `json_pretty`.

Every row recorded under the representation name `json` was produced from the
pretty-printed serialization (`json.dumps(indent=2)`), so it *is* a `json_pretty`
row. The name changed; the data did not. This rewrites the name and nothing else,
so no cell is ever re-run: the expensive artifact in this repo is results.csv.

Why a field-level rewrite and not a text substitution
-----------------------------------------------------
`raw_answer`, `rubric_reasoning` and `question_text` routinely contain the word
"json" in prose ("the json shows...", "according to the JSON..."). A sed would
corrupt them invisibly -- there is no schema check downstream that would catch it.
So the `representation` field is parsed and reassigned, and every other byte is
carried through untouched.

The faithfulness proof
----------------------
Before writing anything, each file is round-tripped **unchanged** and the result
compared byte-for-byte against the original. That proves this script's reader and
writer reproduce the file exactly, for this file, with its own quoting, escaping
and line terminators -- so the only difference in the real output is the field we
meant to change. A file that fails the identity round-trip is reported and skipped
rather than reformatted (`responses.jsonl` falls back to a single exact-substring
replacement, which is byte-preserving by construction).

Ordering constraints (both directions matter)
---------------------------------------------
Strictly AFTER the code rename: the correct next action is regeneration, and
regenerating against a stale CEILING produces a complete, confidently wrong set of
artifacts. Strictly BEFORE any new run: the resume cache keys on
(question_id, representation, repetition) with no alias, so running first would
leave `json` and `json_pretty` coexisting as two rep groups in every mean -- the
compactors are same-key dedupers, not garbage collectors, and a non-error row is
never dropped.

Derived artifacts are regenerated, never rewritten. `_aggregate/` and the per-scene
`aggregate.csv`/plots/`report.md` are pure functions of results.csv: the six
directories getting a `json_mini` fill have theirs rewritten by `results.save` on
that run, and `--render` covers the three rename-only directories.

Usage:
  python -m experiments.scripts.rename_json_representation --dry-run
  python -m experiments.scripts.rename_json_representation
  python -m experiments.scripts.rename_json_representation --models qwen2.5-14b
  python -m experiments.scripts.rename_json_representation --render   # rename-only dirs
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluation.core  # noqa: F401,E402  (imported for its csv.field_size_limit bump)

RESULTS_ROOT = Path("experiments/results")
AGG_DIRNAME = "_aggregate"
OLD_NAME = "json"
NEW_NAME = "json_pretty"
BACKUP_SUFFIX = ".json-rename.bak"

# Directories with no ModelProfile, so they get no json_mini run and nothing else
# will rewrite their derived artifacts. `--render` regenerates these three.
RENAME_ONLY = ("deepseek-r1-14b_screening", "qwen2.5-32b_screening", "qwen2.5-7b_screening")


@dataclass
class FileResult:
    path: Path
    rows: int = 0
    renamed: int = 0
    already: int = 0          # rows already carrying NEW_NAME (idempotency)
    changed: bool = False
    skipped: str = ""         # non-empty => why it was left alone
    notes: list[str] = field(default_factory=list)


def _rewrite_csv(raw: bytes, new_value: str | None) -> tuple[bytes, int, int, int]:
    """Re-emit a results/aggregate CSV.

    `new_value=None` re-emits unchanged (the identity round-trip used as the
    faithfulness proof). Otherwise rows whose representation is OLD_NAME get
    `new_value`. Returns (bytes, n_rows, n_renamed, n_already).
    """
    text = raw.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames or []
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    n = renamed = already = 0
    for row in reader:
        n += 1
        rep = row.get("representation")
        if rep == OLD_NAME:
            renamed += 1
            if new_value is not None:
                row["representation"] = new_value
        elif rep == NEW_NAME:
            already += 1
        writer.writerow(row)
    return buf.getvalue().encode("utf-8"), n, renamed, already


def _rewrite_jsonl(raw: bytes, new_value: str | None) -> tuple[bytes, int, int, int, int]:
    """Re-emit a responses.jsonl, carrying every non-matching line byte-verbatim.

    Only lines whose parsed `representation` is OLD_NAME are re-serialized, and only
    after that line's own round-trip is proven byte-identical; otherwise a single
    exact-substring replacement is used, which cannot perturb anything else.
    Returns (bytes, n_lines, n_renamed, n_already, n_fallback).
    """
    out = bytearray()
    n = renamed = already = fallback = 0
    for line in raw.splitlines(keepends=True):
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            out += line
            continue
        n += 1
        try:
            rec = json.loads(stripped.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            out += line                      # unparseable: carry through untouched
            continue
        rep = rec.get("representation")
        if rep == NEW_NAME:
            already += 1
        if rep != OLD_NAME:
            out += line                      # byte-verbatim
            continue
        renamed += 1
        if new_value is None:
            out += line
            continue
        eol = line[len(stripped):]
        reserialized = json.dumps(rec, ensure_ascii=False).encode("utf-8")
        if reserialized == stripped:
            rec["representation"] = new_value
            out += json.dumps(rec, ensure_ascii=False).encode("utf-8") + eol
        else:
            # Round-trip is not byte-faithful for this line; do the minimal edit.
            needle = f'"representation": "{OLD_NAME}"'.encode("utf-8")
            if stripped.count(needle) != 1:
                raise ValueError(
                    f"cannot safely rewrite a line: round-trip differs and the "
                    f"representation field is not uniquely locatable ({stripped[:120]!r})")
            fallback += 1
            out += stripped.replace(
                needle, f'"representation": "{new_value}"'.encode("utf-8")) + eol
    return bytes(out), n, renamed, already, fallback


def _migrate_file(path: Path, dry_run: bool) -> FileResult:
    res = FileResult(path=path)
    raw = path.read_bytes()
    is_jsonl = path.suffix == ".jsonl"

    # 1. identity round-trip -- proves reader+writer reproduce THIS file exactly
    if is_jsonl:
        ident, n, renamed, already, _ = _rewrite_jsonl(raw, None)
    else:
        ident, n, renamed, already = _rewrite_csv(raw, None)
    res.rows, res.renamed, res.already = n, renamed, already
    if ident != raw and not is_jsonl:
        res.skipped = ("identity round-trip is not byte-faithful -- refusing to "
                       "reformat; inspect this file by hand")
        return res

    if renamed == 0:
        res.skipped = "no rows to migrate (idempotent no-op)"
        return res

    # 2. the real rewrite
    if is_jsonl:
        new, n2, renamed2, _, fb = _rewrite_jsonl(raw, NEW_NAME)
        if fb:
            res.notes.append(f"{fb} line(s) migrated by exact-substring fallback")
    else:
        new, n2, renamed2, _ = _rewrite_csv(raw, NEW_NAME)
    assert (n2, renamed2) == (n, renamed), "second pass disagreed with the first"

    # 3. invariants that must hold on the NEW bytes, checked before anything moves
    if is_jsonl:
        _, vn, vr, vnew, _ = _rewrite_jsonl(new, None)
    else:
        _, vn, vr, vnew = _rewrite_csv(new, None)
    problems = []
    if vn != n:
        problems.append(f"row count changed: {n} -> {vn}")
    if vr != 0:
        problems.append(f"{vr} row(s) still name {OLD_NAME!r}")
    if vnew != already + renamed:
        problems.append(f"{NEW_NAME} count is {vnew}, expected {already + renamed}")
    if not is_jsonl and new.split(b"\r\n", 1)[0] != raw.split(b"\r\n", 1)[0]:
        problems.append("header is not byte-identical")
    if problems:
        res.skipped = "; ".join(problems)
        return res

    res.changed = True
    if dry_run:
        return res

    # 4. backup (only if absent -- a second run must not overwrite a good backup
    #    with an already-migrated copy), then atomic replace via a verified tmp file
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_bytes(raw)
        res.notes.append(f"backup -> {backup.name}")
    else:
        res.notes.append(f"backup exists, kept: {backup.name}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(new)
    if tmp.read_bytes() != new:
        tmp.unlink(missing_ok=True)
        res.changed = False
        res.skipped = "tmp file did not read back identical -- nothing replaced"
        return res
    os.replace(tmp, path)
    return res


def _render(model_dir: Path) -> None:
    """Regenerate every derived artifact of one model directory from its results.csv.

    The same four calls aggregate_results._write_group makes, applied per scene, then
    the cross-scene groups. Only needed for directories that will not be rewritten by
    a subsequent run's `results.save`.
    """
    from evaluation import plots
    from evaluation.report import write_report
    from evaluation.results import _write_aggregate_from_csv
    from experiments.aggregate_results import aggregate_model

    for scene_dir in sorted(d for d in model_dir.iterdir()
                            if d.is_dir() and d.name != AGG_DIRNAME
                            and (d / "results.csv").exists()):
        detail = scene_dir / "results.csv"
        aggregate = scene_dir / "aggregate.csv"
        print(f"  render {scene_dir.name}")
        _write_aggregate_from_csv(detail, aggregate)
        plots.plot_aggregate(aggregate)
        plots.plot_per_question(detail)
        write_report(detail, aggregate)
    aggregate_model(model_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate stored `json` rows to `json_pretty`.")
    ap.add_argument("--models", help="comma-separated model dir names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--render", action="store_true",
                    help=f"after migrating, regenerate derived artifacts for the "
                         f"rename-only directories ({', '.join(RENAME_ONLY)})")
    ap.add_argument("--root", default=str(RESULTS_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no results root: {root}")

    model_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.models:
        wanted = {n.strip() for n in args.models.split(",") if n.strip()}
        have = {d.name for d in model_dirs}
        if missing := wanted - have:
            raise SystemExit(f"unknown model(s): {', '.join(sorted(missing))} "
                             f"(have: {', '.join(sorted(have))})")
        model_dirs = [d for d in model_dirs if d.name in wanted]

    mode = "DRY RUN -- nothing will be written" if args.dry_run else "WRITING"
    print(f"{mode}: {OLD_NAME!r} -> {NEW_NAME!r} under {root}\n")

    total_renamed = total_changed = total_skipped = 0
    failures: list[FileResult] = []
    for model_dir in model_dirs:
        # Recursive, and _aggregate is excluded: those are regenerated, not rewritten.
        files = sorted(f for f in model_dir.rglob("*")
                       if f.is_file() and f.name in ("results.csv", "responses.jsonl")
                       and AGG_DIRNAME not in f.relative_to(model_dir).parts)
        if not files:
            continue
        print(f"[{model_dir.name}]")
        for f in files:
            res = _migrate_file(f, args.dry_run)
            rel = f.relative_to(model_dir)
            if res.changed:
                total_changed += 1
                total_renamed += res.renamed
                extra = ("  (" + "; ".join(res.notes) + ")") if res.notes else ""
                print(f"  {'would rename' if args.dry_run else 'renamed'} "
                      f"{res.renamed:4d}/{res.rows:5d}  {rel}{extra}")
            elif res.skipped.startswith("no rows"):
                total_skipped += 1
                print(f"  {'-':>12} {'':4s} {res.rows:5d}  {rel}  ({res.skipped})")
            else:
                failures.append(res)
                print(f"  !! SKIPPED               {rel}  ({res.skipped})")
        print()

    print(f"{'would rename' if args.dry_run else 'renamed'} {total_renamed} row(s) "
          f"across {total_changed} file(s); {total_skipped} already clean; "
          f"{len(failures)} refused")
    if failures:
        raise SystemExit(1)

    if args.render and not args.dry_run:
        print("\nregenerating derived artifacts for the rename-only directories:")
        for name in RENAME_ONLY:
            d = root / name
            if d.is_dir():
                print(f"[{name}]")
                _render(d)


if __name__ == "__main__":
    main()
