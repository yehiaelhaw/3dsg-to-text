"""plots.py — Generate comparison charts from evaluation results."""

from __future__ import annotations

import csv
import collections
from pathlib import Path


def plot_aggregate(aggregate_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in csv.DictReader(aggregate_path.open(encoding="utf-8"))
            if r["representation"] != "ALL"]

    representations = sorted({r["representation"] for r in rows})
    question_types  = sorted({r["question_type"] for r in rows})
    labels = [qt.replace("_", " ") for qt in question_types]

    out_dir = aggregate_path.parent
    cmap = plt.cm.get_cmap("tab10", max(len(representations), 1))
    colors = [cmap(i) for i in range(len(representations))]

    def _vals(metric_mean: str, metric_std: str, rep: str) -> tuple[list[float], list[float | None]]:
        means, stds = [], []
        for qt in question_types:
            row = next((r for r in rows if r["representation"] == rep and r["question_type"] == qt), None)
            means.append(float(row[metric_mean]) if row else 0.0)
            stds.append(float(row[metric_std]) if row and row[metric_std] else None)
        return means, stds

    def _plot(metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str) -> None:
        x = np.arange(len(question_types))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, (rep, color) in enumerate(zip(representations, colors)):
            means, stds = _vals(metric_mean, metric_std, rep)
            errs = [s if s is not None else 0.0 for s in stds]
            ax.bar(x + (i - 0.5) * width, means, width, label=rep, color=color, alpha=0.85)
            ax.errorbar(x + (i - 0.5) * width, means, yerr=errs,
                        fmt="none", color="black", capsize=3, linewidth=1)

        ax.set_xlabel("Question type")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir / filename}")

    _plot("answer_correctness_mean", "answer_correctness_std",
          "Answer correctness (mean ± 1 std)", "Answer Correctness by Question Type",
          "ac_comparison.png")

    _plot("faithfulness_mean", "faithfulness_std",
          "Faithfulness (mean ± 1 std)", "Faithfulness by Question Type",
          "faithfulness_comparison.png")


def plot_per_question(results_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    out_dir = results_path.parent
    representations = sorted({r["representation"] for r in rows})
    cmap = plt.cm.get_cmap("tab10", max(len(representations), 1))
    colors = {rep: cmap(i) for i, rep in enumerate(representations)}

    # Group by (question_id, representation) → list of float values
    ac_map:      dict[tuple, list[float]] = collections.defaultdict(list)
    faith_map:   dict[tuple, list[float]] = collections.defaultdict(list)
    latency_map: dict[tuple, list[float]] = collections.defaultdict(list)

    for r in rows:
        key = (r["question_id"], r["representation"])
        if r["answer_correctness"]:
            ac_map[key].append(float(r["answer_correctness"]))
        if r["faithfulness"]:
            faith_map[key].append(float(r["faithfulness"]))
        if r["latency_ms"]:
            latency_map[key].append(float(r["latency_ms"]) / 1000)

    question_ids = sorted({r["question_id"] for r in rows}, key=lambda x: int(x))

    # --- Plot 1: AC delta (rep_a - rep_b) per question ---
    def mean_or_nan(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    if len(representations) >= 2:
        rep_a, rep_b = representations[0], representations[1]
        col_a, col_b = colors[rep_a], colors[rep_b]
        deltas, bar_colors = [], []
        for qid in question_ids:
            a = mean_or_nan(ac_map[(qid, rep_a)])
            b = mean_or_nan(ac_map[(qid, rep_b)])
            delta = a - b if not (np.isnan(a) or np.isnan(b)) else 0.0
            deltas.append(delta)
            bar_colors.append(col_a if delta >= 0 else col_b)

        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(question_ids))
        ax.bar(x, deltas, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids, rotation=45, ha="right")
        ax.set_xlabel("Question ID")
        ax.set_ylabel(f"AC delta  ({rep_a} − {rep_b})")
        ax.set_title(f"Answer Correctness Delta per Question  ({rep_a} better vs {rep_b} better)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=col_a, label=f"{rep_a} better"),
                            Patch(color=col_b, label=f"{rep_b} better")])
        fig.tight_layout()
        fig.savefig(out_dir / "ac_delta.png", dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir / 'ac_delta.png'}")

    # --- Plot 2: Latency distribution per representation (box + strip) ---
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(len(representations)))
    for pos, rep in zip(positions, representations):
        per_q_means = [mean_or_nan(latency_map[(qid, rep)]) for qid in question_ids]
        per_q_means = [v for v in per_q_means if not np.isnan(v)]
        ax.boxplot(per_q_means, positions=[pos], widths=0.4,
                   patch_artist=True,
                   boxprops=dict(facecolor=colors.get(rep, "gray"), alpha=0.5),
                   medianprops=dict(color="black", linewidth=2))
        jitter = rng.uniform(-0.08, 0.08, len(per_q_means))
        ax.scatter([pos + j for j in jitter], per_q_means,
                   color=colors.get(rep, "gray"), alpha=0.7, s=30, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(representations, rotation=15, ha="right")
    ax.set_ylabel("Mean response latency (s)")
    ax.set_title("Response Latency by Representation")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / 'latency_comparison.png'}")

    # --- Plot 3: Faithfulness vs Answer Correctness scatter ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for rep in representations:
        rep_rows = [r for r in rows if r["representation"] == rep
                    and r["faithfulness"] and r["answer_correctness"]]
        xs = np.array([float(r["faithfulness"]) for r in rep_rows])
        ys = np.array([float(r["answer_correctness"]) for r in rep_rows])
        # jitter to reveal overlapping points at identical judge scores
        xs = xs + rng.uniform(-0.018, 0.018, len(xs))
        ys = ys + rng.uniform(-0.018, 0.018, len(ys))
        ax.scatter(xs, ys, label=rep, color=colors.get(rep, "gray"), alpha=0.5, s=25)
        if len(xs) >= 2:
            m, b = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 100)
            ax.plot(xline, m * xline + b, color=colors.get(rep, "gray"), linewidth=1.5)
    ax.set_xlabel("Faithfulness")
    ax.set_ylabel("Answer correctness")
    ax.set_title("Faithfulness vs Answer Correctness  (per observation)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "faith_vs_ac.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / 'faith_vs_ac.png'}")
