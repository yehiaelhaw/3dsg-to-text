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
    "exact_match_mean", "semantic_similarity_mean", "faithfulness_mean",
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


def save(records: Iterable[EvalRecord], config: EvalConfig) -> tuple[Path, Path]:
    """Write all records to per-question and aggregate CSVs. Returns both paths."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = config.output_dir / "results.csv"
    aggregate_path = config.output_dir / "aggregate.csv"

    with ResultsWriter(detail_path) as writer:
        for record in records:
            writer.add(record)
        writer.write_aggregate(aggregate_path)

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
        "exact_match_mean":        mean_of("exact_match"),
        "semantic_similarity_mean": mean_of("semantic_similarity"),
        "faithfulness_mean":       mean_of("faithfulness"),
    }
