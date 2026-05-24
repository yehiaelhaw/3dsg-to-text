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
    "exact_match_mean", "semantic_similarity_mean", "faithfulness_mean", "answer_correctness_mean",
]


class ResultsWriter:
    """Streams EvalRecords to a per-question CSV row-by-row."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._records: list[EvalRecord] = []

    def add(self, record: EvalRecord) -> None:
        self._writer.writerow(record.to_flat_dict())
        self._fh.flush()
        self._records.append(record)

    def write_aggregate(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = _compute_aggregate(self._records)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_AGGREGATE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

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

    with ResultsWriter(detail_path) as writer:
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
                    print(f"         GT: {record.question.ground_truth}")
                    print(f"         A:  {record.response.raw_answer}")
                    print(70 * "=")

        writer.write_aggregate(aggregate_path)

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(f"\ndone in {elapsed:.1f}s — {n_ok} ok, {n_err} errors")
        print(f"detail    -> {detail_path}")
        print(f"aggregate -> {aggregate_path}")

    return detail_path, aggregate_path


def _compute_aggregate(records: list[EvalRecord]) -> list[dict]:
    groups: dict[tuple[str, str], list[EvalRecord]] = {}
    for r in records:
        rep = r.response.representation
        qt = r.extracted.question_type.value if r.extracted else "unknown"
        groups.setdefault((rep, qt), []).append(r)

    rows = []
    for (rep, qt), group in sorted(groups.items()):
        rows.append(_group_row(rep, qt, group))

    # overall row across all groups
    rows.append(_group_row("ALL", "ALL", records))
    return rows


def _group_row(representation: str, question_type: str, records: list[EvalRecord]) -> dict:
    error_count = sum(1 for r in records if r.error)

    def mean_of(metric: str) -> str:
        vals = [
            getattr(r.scores, metric)
            for r in records
            if getattr(r.scores, metric) is not None
        ]
        return f"{statistics.mean(vals):.4f}" if vals else ""

    return {
        "representation":          representation,
        "question_type":           question_type,
        "n":                       len(records),
        "error_count":             error_count,
        "exact_match_mean":         mean_of("exact_match"),
        "semantic_similarity_mean": mean_of("semantic_similarity"),
        "faithfulness_mean":        mean_of("faithfulness"),
        "answer_correctness_mean":  mean_of("answer_correctness"),
    }
