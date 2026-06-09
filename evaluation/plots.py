"""plots.py — Generate comparison charts from evaluation results.

Charts are **scope-aware**: a representation is only drawn for a question type it
can actually answer. Each question type requires certain information channels
(connectivity, metric, ...); each representation carries some; a rep is in-scope
when it covers what the type needs. Plotting an out-of-scope rep (e.g. navigation,
which has no object inventory, on a containment question) would show a structural
zero as if it were a result — so those cells are masked out.

The model is fail-open: an unknown representation or question type is never
masked, so new parsers/types are shown until their scope is declared here.
"""

from __future__ import annotations

import csv
import collections
import statistics
from pathlib import Path

# --- scope model -----------------------------------------------------------
# Single source of truth lives in evaluation/scope.py (shared with the runner,
# which skips out-of-scope cells before they are ever computed).
from evaluation.scope import in_scope as _in_scope


# --- shared loading --------------------------------------------------------

def _per_question_ac(results_path: Path):
    """Return (points, types, reps).

    points[(qtype, rep)] = list of one mean answer_correctness per question
    (averaged over repetitions). This is the basis for means, error bars, dots.
    """
    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]

    # (qid, rep) -> list of ac over repetitions; remember each question's type
    by_qr: dict[tuple, list[float]] = collections.defaultdict(list)
    qtype_of: dict[str, str] = {}
    for r in rows:
        if r["answer_correctness"] == "":
            continue
        by_qr[(r["question_id"], r["representation"])].append(float(r["answer_correctness"]))
        qtype_of[r["question_id"]] = r["question_type"] or "unknown"

    points: dict[tuple, list[float]] = collections.defaultdict(list)
    for (qid, rep), vals in by_qr.items():
        points[(qtype_of[qid], rep)].append(sum(vals) / len(vals))

    types = sorted({qt for qt, _ in points})
    reps = sorted({rep for _, rep in points})
    return points, types, reps


def _colors(reps: list[str]):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("tab10", max(len(reps), 1))
    return {rep: cmap(i) for i, rep in enumerate(reps)}


# --- aggregate-level charts (AC by axis, lift over floor) -------------------

def plot_aggregate(aggregate_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    out_dir = aggregate_path.parent
    results_path = out_dir / "results.csv"
    if not results_path.exists():
        return
    points, types, reps = _per_question_ac(results_path)
    if not types:
        return
    color = _colors(reps)
    rng = np.random.default_rng(42)

    # -- Chart 1: scope-masked AC by question type (bars + std + per-question dots) --
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(types)), 6))
    used: set[str] = set()
    for gi, qt in enumerate(types):
        in_scope = [r for r in reps if _in_scope(r, qt) and points.get((qt, r))]
        k = len(in_scope)
        if k == 0:
            continue
        barw = 0.8 / k
        for j, rep in enumerate(in_scope):
            vals = points[(qt, rep)]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            xpos = gi + (j - (k - 1) / 2) * barw
            ax.bar(xpos, mean, barw * 0.9, color=color[rep], alpha=0.85,
                   label=rep if rep not in used else None)
            used.add(rep)
            ax.errorbar(xpos, mean, yerr=std, fmt="none", color="black", capsize=2, linewidth=0.8)
            jitter = rng.uniform(-barw * 0.25, barw * 0.25, len(vals))
            ax.scatter(xpos + jitter, vals, s=14, color=color[rep],
                       edgecolor="black", linewidth=0.4, zorder=3, alpha=0.9)
            ax.text(xpos, -0.06, f"n={len(vals)}", ha="center", va="top", fontsize=6, color="gray")

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_ylabel("Answer correctness")
    ax.set_title("Answer Correctness by Axis  (in-scope reps only; bar=mean, whisker=±1 std, dots=per question)")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles = [Patch(color=color[r], label=r) for r in reps if r in used]
    ax.legend(handles=handles, ncol=min(len(handles), 4), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "ac_by_axis.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / 'ac_by_axis.png'}")

    # -- Chart 2: value of spatial structure (lift over the inventory floor) --
    # Only meaningful where inventory is a genuine no-information control, i.e. a
    # question type that needs something inventory lacks (the spatial axes). For
    # content/general-reasoning types inventory is a legitimate format, not a
    # floor, so they are excluded here and compared head-to-head in ac_by_axis.
    ctrl_types = [qt for qt in types
                  if not _in_scope("inventory", qt) and points.get((qt, "inventory"))]
    if ctrl_types:
        fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(ctrl_types)), 6))
        used = set()
        for gi, qt in enumerate(ctrl_types):
            floor = statistics.mean(points[(qt, "inventory")])
            in_scope = [r for r in reps if r != "inventory" and _in_scope(r, qt) and points.get((qt, r))]
            k = len(in_scope)
            if k == 0:
                continue
            barw = 0.8 / k
            for j, rep in enumerate(in_scope):
                lift = statistics.mean(points[(qt, rep)]) - floor
                xpos = gi + (j - (k - 1) / 2) * barw
                ax.bar(xpos, lift, barw * 0.9, color=color[rep], alpha=0.85,
                       label=rep if rep not in used else None)
                used.add(rep)
        ax.set_xticks(range(len(ctrl_types)))
        ax.set_xticklabels([t.replace("_", " ") for t in ctrl_types], rotation=20, ha="right")
        ax.set_ylabel("AC minus the inventory baseline")
        ax.set_title("Value of spatial structure  (answer correctness vs the no-spatial inventory baseline; "
                     ">0 = it helped, <0 = it hurt)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        handles = [Patch(color=color[r], label=r) for r in reps if r in used]
        ax.legend(handles=handles, ncol=min(len(handles), 4), fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "value_of_spatial_structure.png", dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir / 'value_of_spatial_structure.png'}")

    # retire charts that the scope-aware set replaces / that mislead
    for stale in ("ac_comparison.png", "faithfulness_comparison.png", "ac_delta.png",
                  "ac_lift_over_floor.png"):
        p = out_dir / stale
        if p.exists():
            p.unlink()


# --- per-observation charts (guess detector, latency) ----------------------

def plot_per_question(results_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    out_dir = results_path.parent
    representations = sorted({r["representation"] for r in rows})
    colors = _colors(representations)
    rng = np.random.default_rng(42)

    # -- Faithfulness vs Answer Correctness: a guess / grounding diagnostic --
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # quadrant shading: high AC + low faithfulness = credit without grounding (lucky guess)
    ax.axhspan(0.5, 1.05, xmin=0, xmax=0.5 / 1.05, color="tab:red", alpha=0.06)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.text(0.02, 1.0, "lucky guess\n(high AC, low faithfulness)", fontsize=7, color="tab:red", va="top")
    ax.text(0.52, 0.03, "grounded but wrong", fontsize=7, color="gray", va="bottom")

    for rep in representations:
        rr = [r for r in rows if r["representation"] == rep
              and r["faithfulness"] != "" and r["answer_correctness"] != ""]
        xs = np.array([float(r["faithfulness"]) for r in rr])
        ys = np.array([float(r["answer_correctness"]) for r in rr])
        if len(xs) == 0:
            continue
        xs = xs + rng.uniform(-0.018, 0.018, len(xs))
        ys = ys + rng.uniform(-0.018, 0.018, len(ys))
        ax.scatter(xs, ys, label=rep, color=colors.get(rep, "gray"), alpha=0.6, s=28)
    ax.set_xlabel("Faithfulness (grounded in context)")
    ax.set_ylabel("Answer correctness (matches key facts)")
    ax.set_title("Faithfulness vs Answer Correctness  (per observation)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "faith_vs_ac.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / 'faith_vs_ac.png'}")

    # -- Latency per representation (operational trade-off) --
    latency_map: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["latency_ms"]:
            latency_map[(r["question_id"], r["representation"])].append(float(r["latency_ms"]) / 1000)
    question_ids = sorted({r["question_id"] for r in rows}, key=lambda x: int(x) if x.isdigit() else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pos, rep in enumerate(representations):
        per_q = [statistics.mean(latency_map[(qid, rep)])
                 for qid in question_ids if latency_map[(qid, rep)]]
        if not per_q:
            continue
        ax.boxplot(per_q, positions=[pos], widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=colors.get(rep, "gray"), alpha=0.5),
                   medianprops=dict(color="black", linewidth=2))
        jitter = rng.uniform(-0.08, 0.08, len(per_q))
        ax.scatter([pos + j for j in jitter], per_q, color=colors.get(rep, "gray"),
                   alpha=0.7, s=30, zorder=3)
    ax.set_xticks(range(len(representations)))
    ax.set_xticklabels(representations, rotation=15, ha="right")
    ax.set_ylabel("Mean response latency (s)")
    ax.set_title("Response Latency by Representation")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / 'latency_comparison.png'}")
