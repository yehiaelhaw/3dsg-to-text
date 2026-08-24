"""results.py — Write evaluation records to CSV files."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Iterable

from evaluation.axes import dataset_of
from evaluation.config import EvalConfig
from evaluation.core import CSV_COLUMNS, EvalRecord, is_context_exceeded

# Host-separated (dataset, representation, question_type) rows; repetitions of one
# question are averaged together before the aggregate statistics.
_AGGREGATE_COLUMNS = [
    "dataset", "representation", "question_type",
    "n", "error_count", "context_exceeded", "n_scored", "coverage",
    "faithfulness_mean", "faithfulness_std",
    "answer_correctness_mean", "answer_correctness_std", "answer_correctness_q_range",
    # Diagnostic (supporting-detail coverage, not correctness). n_detail is its OWN
    # support -- sparser than n_scored, since only questions with weight<=1 facts
    # contribute -- so it is never averaged against the primary AC.
    "answer_correctness_detail_mean", "answer_correctness_detail_std", "n_detail",
]


def _compact_for_resume(path: Path) -> None:
    """Drop rows a resumed run supersedes, in place: last write wins per (question_id,
    representation, repetition); ordinary errors are dropped for retry;
    context_exceeded sentinels are kept (terminal). Fails closed on a header mismatch.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames != CSV_COLUMNS:
        raise ValueError(
            f"{path}: header does not match the expected CSV_COLUMNS "
            f"(found {fieldnames!r}) -- refusing to compact a file whose schema "
            "looks corrupted rather than silently entrenching it; repair the "
            "header manually before resuming."
        )
    if not rows:
        return
    last: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        last[(r["question_id"], r["representation"], r["repetition"])] = r
    kept = [r for r in rows
            if last[(r["question_id"], r["representation"], r["repetition"])] is r
            and not (r["error"] and not is_context_exceeded(r["error"]))]
    if len(kept) == len(rows):
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"resume: compacted {path.name} -- dropped {len(rows) - len(kept)} superseded row(s)")


class ResultsWriter:
    """Streams EvalRecords to a per-question CSV row-by-row."""

    def __init__(self, path: Path, resume: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        # Treat a zero-byte resume file as fresh so the CSV header is written.
        append = resume and path.exists() and path.stat().st_size > 0
        if append:
            _compact_for_resume(path)
        self._fh = path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        if not append:
            self._writer.writeheader()

    def add(self, record: EvalRecord) -> None:
        self._writer.writerow(record.to_flat_dict())
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> ResultsWriter:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def save(
    records: Iterable[EvalRecord],
    config: EvalConfig,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Write all records to per-question and aggregate CSVs. Returns both paths."""
    import time
    config.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path    = config.output_dir / "results.csv"
    aggregate_path = config.output_dir / "aggregate.csv"

    t_start = time.perf_counter()
    n_ok = n_err = 0

    if verbose:
        # Responder/judge text can contain characters outside the console's
        # codepage (e.g. cp1252 on Windows) -- replace rather than crash mid-run.
        sys.stdout.reconfigure(errors="replace")

    with ResultsWriter(detail_path, resume=config.resume) as writer:
        for record in records:
            writer.add(record)

            if verbose:
                tag = f"{record.question.id} | {record.response.representation} | rep{record.response.repetition}"
                if record.error:
                    n_err += 1
                    print(f"  ERROR  {tag}")
                    print(f"         {record.error.splitlines()[-1]}")
                else:
                    n_ok += 1
                    score_str = "  ".join(
                        f"{k}={v:.2f}" for k, v in record.scores.to_dict().items() if v is not None
                    )
                    print(f"  OK     {tag} | {score_str} | {record.response.latency_ms:.0f}ms")
                    print(f"         Q:  {record.question.text}")
                    print(f"         A:  {record.response.raw_answer}")
                    if record.rubric_reasoning:
                        print("         JUDGE:")
                        for line in record.rubric_reasoning.strip().splitlines():
                            print(f"           {line}")
                    print(70 * "=")

    # Aggregate from the written CSV (not in-memory records) so a resumed run
    # summarises prior + new rows together.
    _write_aggregate_from_csv(detail_path, aggregate_path)

    from evaluation import plots
    plots.plot_aggregate(aggregate_path)
    plots.plot_per_question(detail_path)

    # The tabular half of reporting (coverage / rank-eligibility, small-n register,
    # per-axis numeric cards) -- regenerated from results.csv alongside the charts.
    from evaluation.report import write_report
    write_report(detail_path, aggregate_path)

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"\ndone in {elapsed:.1f}s — {n_ok} ok, {n_err} errors")
        print(f"detail    -> {detail_path}")
        print(f"aggregate -> {aggregate_path}")

    return detail_path, aggregate_path


def _write_aggregate_from_csv(detail_path: Path, aggregate_path: Path) -> None:
    """Recompute the aggregate CSV from the per-question CSV on disk."""
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = _compute_aggregate(rows)
    with aggregate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerows(out)


def _compute_aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        ds = dataset_of(r["scene_id"])
        rep = r["representation"]
        qt = r["question_type"] or "unknown"
        groups.setdefault((ds, rep, qt), []).append(r)

    out = []
    for (ds, rep, qt), group in sorted(groups.items()):
        out.append(_group_row(ds, rep, qt, group))

    # Operational total (n, error/context-exceeded/scored counts, coverage), NOT a
    # quality score -- pools every dataset, representation, and question type.
    # report.py never surfaces it; see its "do not read an ALL/ALL grand mean" note.
    out.append(_operational_total_row(rows))
    return out


def _questions_of(rows: list[dict], metric: str) -> dict[tuple[str, str, str], list[float]]:
    """Group valid metric values by (scene_id, question_id, representation).
    Rows with an error or a missing/blank value for the metric are ignored.
    """
    by_q: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        v = r.get(metric, "")
        if r["error"] or v in ("", None):
            continue
        key = (r["scene_id"], r["question_id"], r["representation"])
        by_q.setdefault(key, []).append(float(v))
    return by_q


def _group_row(dataset: str, representation: str, question_type: str, rows: list[dict]) -> dict:
    # Context-exceeded cells (the rep overflowed the responder window) are a
    # reportable token-cost outcome, kept separate from real errors and excluded
    # from the means -- not scored as a wrong answer.
    context_exceeded = sum(1 for r in rows if is_context_exceeded(r["error"]))
    error_count = sum(1 for r in rows if r["error"] and not is_context_exceeded(r["error"]))

    n = len(rows)
    # Scored = no error and a real primary correctness value.
    scored = [r for r in rows if not r["error"] and r["answer_correctness"] not in ("", None)]
    n_scored = len(scored)

    def per_question_means(metric: str) -> list[float]:
        # Average repetitions within question before aggregate statistics.
        return [statistics.mean(v) for v in _questions_of(rows, metric).values()]

    def mean_of(vals: list[float]) -> str:
        return f"{statistics.mean(vals):.4f}" if vals else ""

    def std_of(vals: list[float]) -> str:
        # Descriptive population spread over per-question means; not an interval estimate.
        return f"{statistics.pstdev(vals):.4f}" if vals else ""

    def q_range_of(vals: list[float]) -> str:
        # Descriptive min-max over the same per-question means.
        if len(vals) < 2:
            return ""
        return f"{min(vals):.4f}..{max(vals):.4f}"

    fth_q = per_question_means("faithfulness")
    ac_q  = per_question_means("answer_correctness")
    det_q = per_question_means("answer_correctness_detail")  # sparser: only questions with detail facts

    return {
        "dataset":                 dataset,
        "representation":          representation,
        "question_type":           question_type,
        "n":                       n,
        "error_count":             error_count,
        "context_exceeded":        context_exceeded,
        "n_scored":                n_scored,
        # Coverage is the scored fraction of raw evaluation instances.
        "coverage":                f"{n_scored / n:.4f}" if n else "",
        "faithfulness_mean":       mean_of(fth_q),
        "faithfulness_std":        std_of(fth_q),
        "answer_correctness_mean": mean_of(ac_q),
        "answer_correctness_std":  std_of(ac_q),
        "answer_correctness_q_range": q_range_of(ac_q),
        "answer_correctness_detail_mean": mean_of(det_q),
        "answer_correctness_detail_std":  std_of(det_q),
        # Number of distinct questions contributing detail scores.
        "n_detail":                       len(det_q),
    }


def _operational_total_row(rows: list[dict]) -> dict:
    """Synthetic cross-run operational counts; quality fields remain blank."""
    context_exceeded = sum(1 for r in rows if is_context_exceeded(r["error"]))
    error_count = sum(1 for r in rows if r["error"] and not is_context_exceeded(r["error"]))
    n = len(rows)
    n_scored = sum(1 for r in rows
                   if not r["error"] and r["answer_correctness"] not in ("", None))
    return {
        "dataset":                 "ALL",
        "representation":          "ALL",
        "question_type":           "operational-total",
        "n":                       n,
        "error_count":             error_count,
        "context_exceeded":        context_exceeded,
        "n_scored":                n_scored,
        "coverage":                f"{n_scored / n:.4f}" if n else "",
        "faithfulness_mean":       "",
        "faithfulness_std":        "",
        "answer_correctness_mean": "",
        "answer_correctness_std":  "",
        "answer_correctness_q_range": "",
        "answer_correctness_detail_mean": "",
        "answer_correctness_detail_std":  "",
        "n_detail":                       "",
    }
