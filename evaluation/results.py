"""results.py — Write evaluation records to CSV files."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Iterable

from evaluation.config import EvalConfig
from evaluation.core import CSV_COLUMNS, EvalRecord

_AGGREGATE_COLUMNS = [
    "representation", "question_type",
    "n", "error_count",
    "faithfulness_mean", "faithfulness_std",
    "answer_correctness_mean", "answer_correctness_std",
]


class ResultsWriter:
    """Streams EvalRecords to a per-question CSV row-by-row."""

    def __init__(self, path: Path, resume: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        append = resume and path.exists()
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
                        print(f"         JUDGE:")
                        for line in record.rubric_reasoning.strip().splitlines():
                            print(f"           {line}")
                    print(70 * "=")

    # Aggregate from the written CSV (not in-memory records) so a resumed run
    # summarises prior + new rows together.
    _write_aggregate_from_csv(detail_path, aggregate_path)

    from evaluation import plots
    plots.plot_aggregate(aggregate_path)
    plots.plot_per_question(detail_path)

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
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        rep = r["representation"]
        qt = r["question_type"] or "unknown"
        groups.setdefault((rep, qt), []).append(r)

    out = []
    for (rep, qt), group in sorted(groups.items()):
        out.append(_group_row(rep, qt, group))

    # overall row across all groups
    out.append(_group_row("ALL", "ALL", rows))
    return out


def _group_row(representation: str, question_type: str, rows: list[dict]) -> dict:
    error_count = sum(1 for r in rows if r["error"])

    def vals_of(metric: str) -> list[float]:
        out = []
        for r in rows:
            if r["error"] or r[metric] in ("", None):
                continue
            out.append(float(r[metric]))
        return out

    def mean_of(vals: list[float]) -> str:
        return f"{statistics.mean(vals):.4f}" if vals else ""

    def std_of(vals: list[float]) -> str:
        return f"{statistics.stdev(vals):.4f}" if len(vals) >= 2 else ""

    fth = vals_of("faithfulness")
    ac  = vals_of("answer_correctness")

    return {
        "representation":          representation,
        "question_type":           question_type,
        "n":                       len(rows),
        "error_count":             error_count,
        "faithfulness_mean":       mean_of(fth),
        "faithfulness_std":        std_of(fth),
        "answer_correctness_mean": mean_of(ac),
        "answer_correctness_std":  std_of(ac),
    }
