"""rename_representation.py -- migrate stored representation-name rows.

Generalized from `rename_json_representation.py` (the `json` -> `json_pretty`
migration, 2026-08-09): same field-level rewrite, same faithfulness proof, same
backup/atomic-replace discipline, parameterized by --old/--new instead of hardcoded
names. Built for the 2026-08-20 structure-presentation naming swap
(`topology` -> `topology_inventory`, then `topology_edges_only` -> `topology`), run
here as two separate invocations in that order -- see the ordering note below.

Why a field-level rewrite and not a text substitution
-----------------------------------------------------
`raw_answer`, `rubric_reasoning` and `question_text` can contain the OLD name as a
plain English word or a coincidental substring. A sed would corrupt them invisibly
-- there is no schema check downstream that would catch it. So the `representation`
field is parsed and reassigned, and every other byte is carried through untouched.

The faithfulness proof
----------------------
Before writing anything, each file is round-tripped **unchanged** and the result
compared byte-for-byte against the original. That proves this script's reader and
writer reproduce the file exactly, for this file, with its own quoting, escaping
and line terminators -- so the only difference in the real output is the field we
meant to change. A file that fails the identity round-trip is reported and skipped
rather than reformatted (`responses.jsonl` falls back to a single exact-substring
replacement, which is byte-preserving by construction).

Ordering constraint, specific to a two-name swap
-------------------------------------------------
When migrating a SWAP (A -> B and, separately, C -> A, reusing A's name), run the
A -> B pass FIRST and the C -> A pass SECOND. Reversed order collides: C -> A would
create rows named A before the A -> B pass has cleared A out, so the second pass
could not tell an old-meaning A row from a freshly-renamed one. This script does
not enforce the order itself -- verify with --dry-run before each real run, and
check the printed counts against the invariants recorded before you started.

Derived artifacts are regenerated, never rewritten. `_aggregate/` and the per-scene
`aggregate.csv`/plots/`report.md` are pure functions of results.csv: use --render
after both passes are verified complete.

Usage:
  python -m experiments.scripts.rename_representation --old topology --new topology_inventory --dry-run
  python -m experiments.scripts.rename_representation --old topology --new topology_inventory
  python -m experiments.scripts.rename_representation --old topology_edges_only --new topology --dry-run
  python -m experiments.scripts.rename_representation --old topology_edges_only --new topology
  python -m experiments.scripts.rename_representation --old X --new Y --models qwen2.5-14b
  python -m experiments.scripts.rename_representation --old X --new Y --render
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
BACKUP_SUFFIX = ".rename.bak"


@dataclass
class FileResult:
    path: Path
    rows: int = 0
    renamed: int = 0
    already: int = 0          # rows already carrying NEW_NAME (idempotency)
    changed: bool = False
    skipped: str = ""         # non-empty => why it was left alone
    notes: list[str] = field(default_factory=list)


def _rewrite_csv(raw: bytes, old: str, new: str, write: bool) -> tuple[bytes, int, int, int]:
    """Re-emit a results/aggregate CSV.

    `write=False` re-emits unchanged (the identity round-trip used as the
    faithfulness proof / post-write verification pass) but still counts rows
    already carrying `new`, regardless of whether this call writes -- `new` is
    always the migration's target name, not "whatever this call happens to write".
    `write=True` additionally rewrites rows whose representation is `old` to `new`.
    Returns (bytes, n_rows, n_renamed, n_already).
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
        if rep == old:
            renamed += 1
            if write:
                row["representation"] = new
        elif rep == new:
            already += 1
        writer.writerow(row)
    return buf.getvalue().encode("utf-8"), n, renamed, already


def _rewrite_jsonl(raw: bytes, old: str, new: str, write: bool) -> tuple[bytes, int, int, int, int]:
    """Re-emit a responses.jsonl, carrying every non-matching line byte-verbatim.

    Only lines whose parsed `representation` is `old` are re-serialized, and only
    after that line's own round-trip is proven byte-identical; otherwise a single
    exact-substring replacement is used, which cannot perturb anything else.
    `new` is always the migration's target name for the "already migrated" count,
    independent of whether this call actually writes (`write`).
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
        if rep == new:
            already += 1
        if rep != old:
            out += line                      # byte-verbatim
            continue
        renamed += 1
        if not write:
            out += line
            continue
        eol = line[len(stripped):]
        reserialized = json.dumps(rec, ensure_ascii=False).encode("utf-8")
        if reserialized == stripped:
            rec["representation"] = new
            out += json.dumps(rec, ensure_ascii=False).encode("utf-8") + eol
        else:
            # Round-trip is not byte-faithful for this line; do the minimal edit.
            needle = f'"representation": "{old}"'.encode("utf-8")
            if stripped.count(needle) != 1:
                raise ValueError(
                    f"cannot safely rewrite a line: round-trip differs and the "
                    f"representation field is not uniquely locatable ({stripped[:120]!r})")
            fallback += 1
            out += stripped.replace(
                needle, f'"representation": "{new}"'.encode("utf-8")) + eol
    return bytes(out), n, renamed, already, fallback


def _migrate_file(path: Path, old: str, new: str, dry_run: bool) -> FileResult:
    res = FileResult(path=path)
    raw = path.read_bytes()
    is_jsonl = path.suffix == ".jsonl"

    # 1. identity round-trip -- proves reader+writer reproduce THIS file exactly
    if is_jsonl:
        ident, n, renamed, already, _ = _rewrite_jsonl(raw, old, new, write=False)
    else:
        ident, n, renamed, already = _rewrite_csv(raw, old, new, write=False)
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
        new_bytes, n2, renamed2, _, fb = _rewrite_jsonl(raw, old, new, write=True)
        if fb:
            res.notes.append(f"{fb} line(s) migrated by exact-substring fallback")
    else:
        new_bytes, n2, renamed2, _ = _rewrite_csv(raw, old, new, write=True)
    assert (n2, renamed2) == (n, renamed), "second pass disagreed with the first"

    # 3. invariants that must hold on the NEW bytes, checked before anything moves
    if is_jsonl:
        _, vn, vr, vnew, _ = _rewrite_jsonl(new_bytes, old, new, write=False)
    else:
        _, vn, vr, vnew = _rewrite_csv(new_bytes, old, new, write=False)
    problems = []
    if vn != n:
        problems.append(f"row count changed: {n} -> {vn}")
    if vr != 0:
        problems.append(f"{vr} row(s) still name {old!r}")
    if vnew != already + renamed:
        problems.append(f"{new!r} count is {vnew}, expected {already + renamed}")
    if not is_jsonl and new_bytes.split(b"\r\n", 1)[0] != raw.split(b"\r\n", 1)[0]:
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
    tmp.write_bytes(new_bytes)
    if tmp.read_bytes() != new_bytes:
        tmp.unlink(missing_ok=True)
        res.changed = False
        res.skipped = "tmp file did not read back identical -- nothing replaced"
        return res
    os.replace(tmp, path)
    return res


def _render(model_dir: Path) -> None:
    """Regenerate every derived artifact of one model directory from its results.csv."""
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
    ap = argparse.ArgumentParser(description="Migrate stored representation-name rows.")
    ap.add_argument("--old", required=True, help="representation name to migrate FROM")
    ap.add_argument("--new", required=True, help="representation name to migrate TO")
    ap.add_argument("--models", help="comma-separated model dir names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--render", action="store_true",
                    help="after migrating, regenerate derived artifacts for every "
                         "touched model directory")
    ap.add_argument("--root", default=str(RESULTS_ROOT))
    args = ap.parse_args()

    old, new = args.old, args.new
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
    print(f"{mode}: {old!r} -> {new!r} under {root}\n")

    total_renamed = total_changed = total_skipped = 0
    failures: list[FileResult] = []
    touched_model_dirs: list[Path] = []
    for model_dir in model_dirs:
        # Recursive, and _aggregate is excluded: those are regenerated, not rewritten.
        files = sorted(f for f in model_dir.rglob("*")
                       if f.is_file() and f.name in ("results.csv", "responses.jsonl")
                       and AGG_DIRNAME not in f.relative_to(model_dir).parts)
        if not files:
            continue
        printed_header = False
        model_touched = False
        for f in files:
            res = _migrate_file(f, old, new, args.dry_run)
            rel = f.relative_to(model_dir)
            if res.changed:
                if not printed_header:
                    print(f"[{model_dir.name}]")
                    printed_header = True
                total_changed += 1
                total_renamed += res.renamed
                model_touched = True
                extra = ("  (" + "; ".join(res.notes) + ")") if res.notes else ""
                print(f"  {'would rename' if args.dry_run else 'renamed'} "
                      f"{res.renamed:4d}/{res.rows:5d}  {rel}{extra}")
            elif res.skipped.startswith("no rows"):
                total_skipped += 1
            else:
                if not printed_header:
                    print(f"[{model_dir.name}]")
                    printed_header = True
                failures.append(res)
                print(f"  !! SKIPPED               {rel}  ({res.skipped})")
        if printed_header:
            print()
        if model_touched:
            touched_model_dirs.append(model_dir)

    print(f"{'would rename' if args.dry_run else 'renamed'} {total_renamed} row(s) "
          f"across {total_changed} file(s); {total_skipped} already clean; "
          f"{len(failures)} refused")
    if failures:
        raise SystemExit(1)

    if args.render and not args.dry_run:
        print("\nregenerating derived artifacts for touched model directories:")
        for d in touched_model_dirs:
            print(f"[{d.name}]")
            _render(d)


if __name__ == "__main__":
    main()
