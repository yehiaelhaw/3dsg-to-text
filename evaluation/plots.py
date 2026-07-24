"""plots.py — Generate comparison charts from evaluation results.

Charts are **scope-aware**: a representation is only drawn for a question type it
can actually answer. Each question type requires certain information channels
(connectivity, metric, ...); each representation carries some; a rep is in-scope
when it covers what the type needs. Plotting an out-of-scope rep (e.g. navigation,
which has no object inventory, on a containment question) would show a structural
zero as if it were a result — so those cells are masked out.

The model is fail-open: an unknown representation or question type is never
masked, so new parsers/types are shown until their scope is declared here.

Charts produced
---------------
plot_aggregate (reads results.csv + aggregate.csv):
  axis_card_<id>.png           one figure per axis in evaluation.axes.AXES (named,
                               e.g. axis_card_spatial_encoding.png)
  ac_by_axis.png               AC per question type, in-scope reps, bars+std+dots
  value_of_spatial_structure.png  AC lift over the inventory floor (spatial-encoding result)
  ac_heatmap.png               rep x type mean-AC matrix, out-of-scope cells greyed
  axis_contrasts.png           paired per-question AC delta for each axis with a
                               headline pair (evaluation.axes.AXIS_PAIRS)
plot_per_question (reads results.csv):
  cost_quality.png             mean AC vs mean prompt tokens, with efficiency frontier
  faith_vs_ac.png              per-observation guess detector (only if faithfulness on)
  latency_comparison.png       response latency per representation
"""

from __future__ import annotations

import csv
import collections
import statistics
from pathlib import Path

from evaluation.core import is_context_exceeded

# --- scope + axis model ----------------------------------------------------
# scope.py: which rep can answer which type (the structural mask, shared with the
# runner, which skips out-of-scope cells before they are computed). axes.py: how
# reps group into axis ladders for reporting (ladder order, floor/ceiling roles,
# host dataset) -- so the charts and report.md tell the same axis story.
#
# AXIS_PAIRS (the same-information contrast pairs for axis_contrasts.png) is now
# derived in axes.py from the AXES registry; each delta is averaged only over
# question types where *both* poles are in scope. The spatial-encoding axis is a
# ladder (covered by axis_cards / value_of_spatial_structure), so it contributes no
# pair. A pair whose poles are absent from a scene draws no bar.
from evaluation.scope import in_scope as _in_scope
from evaluation.axes import (
    AXES, AXIS_PAIRS, CEILING, FLOOR, SMALL_N, dataset_of, rep_role,
)


# --- shared loading --------------------------------------------------------

def _per_question_ac(results_path: Path, dataset: str | None = None):
    """Return (points, types, reps).

    points[(qtype, rep)] = list of one mean answer_correctness per question
    (averaged over repetitions). This is the basis for means, error bars, dots.

    `dataset` (procthor|3rscan|gibson) restricts to one host dataset -- the axis
    cards use it so an axis ladder is never pooled across non-comparable datasets.
    """
    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    if dataset:
        rows = [r for r in rows if dataset_of(r["scene_id"]) == dataset]

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


def _per_qid_ac(results_path: Path):
    """Return (ac, qtype_of).

    ac[(qid, rep)] = mean answer_correctness over that cell's repetitions.
    qtype_of[qid]  = the question's type. Keeps question identity (which
    _per_question_ac drops) so paired per-question deltas can be computed.
    """
    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    by_qr: dict[tuple, list[float]] = collections.defaultdict(list)
    qtype_of: dict[str, str] = {}
    for r in rows:
        if r["answer_correctness"] == "":
            continue
        by_qr[(r["question_id"], r["representation"])].append(float(r["answer_correctness"]))
        qtype_of[r["question_id"]] = r["question_type"] or "unknown"
    ac = {k: sum(v) / len(v) for k, v in by_qr.items()}
    return ac, qtype_of


# Stable per-rep colours, reused across every chart so a representation keeps the
# same colour in all figures (lets a thesis reader cross-reference). Grouped by
# axis family: grey control, blue connectivity/structure, green metric/frame,
# warm relations (relation linearization), purple/black prose+json ceiling, magenta combos.
# tab10 carries only 10 hues, so the old resampling collapsed the 16-rep set into
# duplicates; a fixed map avoids that and stays stable as reps come and go.
REP_COLORS: dict[str, str] = {
    "inventory":                     "#9e9e9e",  # control / floor
    "json":                          "#1f1f1f",  # raw-coordinate ceiling
    "prose":                         "#6a3d9a",  # natural language
    # connectivity / structure (spatial-encoding connectivity, structure presentation)
    "topology":                      "#1f78b4",
    "room_tree":                     "#a6cee3",
    "graph_digest":                  "#08519c",
    # metric / frame (spatial-encoding metric, reference frame)
    "metric_relations":              "#33a02c",
    "navigation":                    "#00bcd4",
    # object relations (relation linearization)
    "relations_flat":                "#e31a1c",
    "relations_predicate":           "#ff7f00",
    "relations_subject":             "#b15928",
    "relations_digest":              "#fdbf6f",
    "relations_tree":                "#8c510a",
    # multi-view combinations
    "topology+metric_relations":     "#fec4ff",
    "graph_digest+metric_relations": "#ce1256",
}


def _colors(reps: list[str]):
    """Stable colour per rep. Unknown reps fall back to tab20 (fail-open) so a
    new parser is still drawn -- just not with a curated colour until added above."""
    import matplotlib.pyplot as plt
    extra = [r for r in reps if r not in REP_COLORS]
    fallback = {}
    if extra:
        cmap = plt.cm.get_cmap("tab20", max(len(extra), 1))
        fallback = {r: cmap(i) for i, r in enumerate(extra)}
    return {rep: REP_COLORS.get(rep, fallback.get(rep)) for rep in reps}


# --- axis cards (the headline reporting chart) -----------------------------

def _axis_card_reps(axis, qt: str, points: dict) -> list[str]:
    """Anchored, ladder-ordered reps for one axis x type with data: floor first,
    in-scope poles in rung order, ceiling last. [] when fewer than two are present.
    Mirrors report._axis_reps so the chart and the table list the same cells."""
    reps = []
    if points.get((qt, FLOOR)):
        reps.append(FLOOR)
    for p in axis.ladder:
        if p in (FLOOR, CEILING):
            continue
        if _in_scope(p, qt) and points.get((qt, p)):
            reps.append(p)
    if points.get((qt, CEILING)) and CEILING not in reps:
        reps.append(CEILING)
    return reps if len(reps) >= 2 else []


def _plot_axis_cards(results_path: Path, out_dir: Path, color: dict) -> list[str]:
    """One figure *per* design axis (axes.AXES) -> axis_card_<id>.png: a subplot per
    probe type, reps drawn as bars in ladder order with the floor (inventory) and
    ceiling (json) as a dashed/dotted band so a pole is read against them. Each rep
    is a readable x-tick label; the y-axis starts at 0. Restricted to the axis's
    host dataset (never pooled) and its probe types; combos/synthesis are excluded
    (they answer a different question -- see report.md). Bars hatch when n < SMALL_N
    (screening-only); whisker = +/-CI95; dots = per-question means. Returns the
    figure stems written, so the caller can clean up any axis that lost its data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    rng = np.random.default_rng(42)

    written: list[str] = []
    for axis in AXES:
        points, _t, _r = _per_question_ac(results_path, dataset=axis.host)
        groups = [(qt, reps) for qt in axis.probe_types
                  if (reps := _axis_card_reps(axis, qt, points))]
        if not groups:
            continue
        fig, axs = plt.subplots(1, len(groups), squeeze=False,
                                figsize=(max(4.0, 2.6 * len(groups) + 0.4 * sum(len(r) for _, r in groups)), 4.6))
        for gi, (qt, reps) in enumerate(groups):
            ax = axs[0][gi]
            floor_ac = statistics.mean(points[(qt, FLOOR)]) if points.get((qt, FLOOR)) else None
            ceil_ac = statistics.mean(points[(qt, CEILING)]) if points.get((qt, CEILING)) else None
            if floor_ac is not None:
                ax.axhline(floor_ac, color=color.get(FLOOR, "grey"), linestyle="--",
                           linewidth=1.0, alpha=0.7, zorder=1)
            if ceil_ac is not None:
                ax.axhline(ceil_ac, color="black", linestyle=":", linewidth=1.0,
                           alpha=0.7, zorder=1)
            for xi, rep in enumerate(reps):
                vals = points[(qt, rep)]
                m = statistics.mean(vals)
                ci = 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
                small = len(vals) < SMALL_N
                ax.bar(xi, m, 0.72, color=color.get(rep, "grey"), alpha=0.85,
                       hatch="//" if small else None,
                       edgecolor="black" if small else "none", linewidth=0.4, zorder=2)
                lo, hi = max(0.0, m - ci), min(1.0, m + ci)
                ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt="none",
                            color="black", capsize=2, linewidth=0.8, zorder=3)
                jit = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(xi + jit, vals, s=9, color=color.get(rep, "grey"),
                           edgecolor="black", linewidth=0.3, alpha=0.5, zorder=4)
                ax.text(xi, min(1.0, hi) + 0.015, f"n={len(vals)}", ha="center",
                        va="bottom", fontsize=6, color="gray")
            ax.set_xticks(range(len(reps)))
            ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.08)
            ax.set_title(qt.replace("_", " "), fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            if gi == 0:
                ax.set_ylabel("Answer correctness")
        fig.suptitle(f"{axis.label}   (host: {axis.host}; "
                     f"floor=dashed, json=dotted, hatch = n<{SMALL_N})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        stem = f"axis_card_{axis.id}.png"
        fig.savefig(out_dir / stem, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(stem)
        print(f"plot -> {out_dir / stem}")
    return written


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

    # -- Chart 0: axis cards (one figure per axis, ladder order, host dataset) --
    # The headline reporting view: each axis told as its own ladder. ac_by_axis
    # below stays as the per-type head-to-head (it also covers the content/general
    # types no axis ladder probes).
    written_cards = _plot_axis_cards(results_path, out_dir, color)

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
            # AC is bounded [0,1]; clip the +/-1 std whisker so it never shoots
            # past the frame (the bimodal 0/1 spread makes std large).
            lo, hi = max(0.0, mean - std), min(1.0, mean + std)
            ax.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        color="black", capsize=2, linewidth=0.8)
            jitter = rng.uniform(-barw * 0.25, barw * 0.25, len(vals))
            # low alpha so coincident points darken (honest density) without blobs
            ax.scatter(xpos + jitter, vals, s=10, color=color[rep],
                       edgecolor="black", linewidth=0.3, zorder=3, alpha=0.55)
            ax.text(xpos, -0.04, f"n={len(vals)}", ha="center", va="top",
                    fontsize=6, color="gray", rotation=90)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_ylabel("Answer correctness")
    ax.set_title("Answer Correctness by Axis  (in-scope reps only; bar=mean, whisker=±1 std, dots=per question)")
    ax.set_ylim(-0.18, 1.05)  # extra bottom room for the rotated n= labels
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Legend outside the axes -- 16 reps would otherwise sit on top of the bars.
    handles = [Patch(color=color[r], label=r) for r in reps if r in used]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.savefig(out_dir / "ac_by_axis.png", dpi=150, bbox_inches="tight")
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
        ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
                  bbox_to_anchor=(1.005, 0.5), frameon=False)
        fig.savefig(out_dir / "value_of_spatial_structure.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"plot -> {out_dir / 'value_of_spatial_structure.png'}")

    # -- Chart 3: scope-masked AC heatmap (rep x type overview) --
    # The whole matrix at a glance. In-scope cells are coloured by mean AC;
    # out-of-scope cells are greyed/hatched (a structural gap, not a zero);
    # an in-scope cell with no data stays white. Rows ordered best-first by the
    # rep's overall in-scope mean.
    counts = {(qt, rep): len(vals) for (qt, rep), vals in points.items()}
    rep_overall: dict[str, float] = {}
    for rep in reps:
        vv = [v for (qt, r), vals in points.items()
              if r == rep and _in_scope(rep, qt) for v in vals]
        if vv:
            rep_overall[rep] = statistics.mean(vv)
    # Poles first (best-first), then combos + the synthesis candidate as a separate
    # block below a divider -- they answer a different question than the axis poles,
    # so they should not read as just more rows in the same ranking.
    def _block(rep: str) -> int:
        return 1 if rep_role(rep) in ("combo", "candidate") else 0
    ordered_reps = sorted(rep_overall, key=lambda r: (_block(r), -rep_overall[r]))
    if ordered_reps:
        M = np.full((len(ordered_reps), len(types)), np.nan)
        for i, rep in enumerate(ordered_reps):
            for j, qt in enumerate(types):
                if _in_scope(rep, qt) and points.get((qt, rep)):
                    M[i, j] = statistics.mean(points[(qt, rep)])
        fig, ax = plt.subplots(
            figsize=(max(8, 1.3 * len(types) + 3), max(4, 0.55 * len(ordered_reps) + 2))
        )
        cmap = plt.cm.get_cmap("RdYlGn").copy()
        cmap.set_bad("white")
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, rep in enumerate(ordered_reps):
            for j, qt in enumerate(types):
                if not _in_scope(rep, qt):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="lightgrey", edgecolor="white",
                                               hatch="//", zorder=2))
                    ax.text(j, i, "-", ha="center", va="center", color="gray", fontsize=8)
                elif not np.isnan(M[i, j]):
                    n = counts.get((qt, rep), 0)
                    ax.text(j, i, f"{M[i, j]:.2f}\nn={n}", ha="center", va="center",
                            color="black", fontsize=7)
                    if 0 < n < SMALL_N:  # screening-only cell: red outline
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                   edgecolor="red", linewidth=1.4, zorder=4))
        # divider between the pole block and the combo/candidate block
        split = next((i for i, r in enumerate(ordered_reps) if _block(r) == 1), None)
        if split:
            ax.axhline(split - 0.5, color="black", linewidth=1.4)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
        ax.set_yticks(range(len(ordered_reps)))
        ax.set_yticklabels(ordered_reps)
        ax.set_title("Answer Correctness heatmap  (mean per rep x type; grey = out of scope, "
                     f"red outline = n<{SMALL_N}; combos/synthesis below the line)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean AC")
        fig.tight_layout()
        fig.savefig(out_dir / "ac_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir / 'ac_heatmap.png'}")

    # -- Chart 4: axis-contrast paired deltas (headline-pair axes: format,
    #    reference frame, structure presentation, relation linearization) --
    # For each axis pair, the per-question AC delta (second pole minus first),
    # over questions where *both* poles are in scope. Bar = mean, whisker = ±1
    # population std, dots = per question. >0 means the second pole scored higher.
    qid_ac, qtype_of = _per_qid_ac(results_path)
    qids = sorted({qid for (qid, _) in qid_ac})
    bars: list[tuple[str, list[float]]] = []
    for label, a, b in AXIS_PAIRS:
        deltas = []
        for qid in qids:
            qt = qtype_of[qid]
            if not (_in_scope(a, qt) and _in_scope(b, qt)):
                continue
            if (qid, a) in qid_ac and (qid, b) in qid_ac:
                deltas.append(qid_ac[(qid, b)] - qid_ac[(qid, a)])
        if deltas:
            bars.append((f"{label}: {a} -> {b}", deltas))
    if bars:
        fig, ax = plt.subplots(figsize=(10.5, max(3, 0.9 * len(bars) + 1.5)))
        for i, (lab, deltas) in enumerate(bars):
            m = statistics.mean(deltas)
            sd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            ax.barh(i, m, color="tab:green" if m >= 0 else "tab:red", alpha=0.8)
            ax.errorbar(m, i, xerr=sd, fmt="none", color="black", capsize=3, linewidth=0.8)
            jitter = rng.uniform(-0.12, 0.12, len(deltas))
            ax.scatter(deltas, [i + j for j in jitter], s=16, color="black",
                       alpha=0.5, zorder=3)
            ax.text(0.99, i, f"n={len(deltas)}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6, color="gray")
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([lab for lab, _ in bars])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("AC delta  (second pole minus first; >0 = second pole better)")
        ax.set_title("Axis contrasts  (paired per-question AC delta, both-poles-in-scope cells)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "axis_contrasts.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"plot -> {out_dir / 'axis_contrasts.png'}")

    # retire charts that the scope-aware set replaces / that mislead, plus the old
    # single combined axis card and any per-axis card whose axis lost its data.
    stale = ["ac_comparison.png", "faithfulness_comparison.png", "ac_delta.png",
             "ac_lift_over_floor.png", "axis_cards.png"]
    stale += [p.name for p in out_dir.glob("axis_card_*.png") if p.name not in written_cards]
    for name in stale:
        p = out_dir / name
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

    # -- Cost vs Quality: accuracy per prompt token (the operational trade-off) --
    # x = mean prompt tokens (context size = serialization overhead), y = mean AC,
    # one point per rep over its *in-scope* cells (the floor's out-of-scope
    # guessing is excluded so it isn't penalised for questions it isn't meant to
    # answer). The dashed line is the efficiency frontier (no rep is both cheaper
    # and more accurate than a point on it) -- this is where the combo-vs-json
    # ceiling question is read: same AC, far fewer tokens = the combo wins.
    by_rep_ac: dict[str, list[float]] = collections.defaultdict(list)
    by_rep_tok: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        rep = r["representation"]
        qt = r["question_type"] or "unknown"
        if not _in_scope(rep, qt):
            continue
        if r["answer_correctness"] != "":
            by_rep_ac[rep].append(float(r["answer_correctness"]))
        tok = r.get("prompt_tokens", "")
        if tok not in ("", "0", None):
            try:
                by_rep_tok[rep].append(float(tok))
            except ValueError:
                pass
    # Coverage per rep (1 - context_exceeded rate) from the *unfiltered* rows: a
    # rep whose AC mean rests on few survivors (json overflowing dense scenes) is
    # survivorship-biased, so its marker is shrunk and its cov% annotated -- the
    # AC alone would read as a clean point otherwise.
    exc: dict = collections.defaultdict(int)
    tot: dict = collections.defaultdict(int)
    for r in csv.DictReader(results_path.open(encoding="utf-8")):
        tot[r["representation"]] += 1
        if is_context_exceeded(r["error"]):
            exc[r["representation"]] += 1
    cov = {rep: (1 - exc[rep] / tot[rep]) if tot[rep] else 1.0 for rep in tot}

    cq_reps = [rep for rep in by_rep_ac if by_rep_tok.get(rep)]
    if cq_reps:
        xy = {rep: (statistics.mean(by_rep_tok[rep]), statistics.mean(by_rep_ac[rep]))
              for rep in cq_reps}
        # Pareto frontier: minimise tokens, maximise AC.
        frontier = [rep for rep in xy if not any(
            o != rep and xy[o][0] <= xy[rep][0] and xy[o][1] >= xy[rep][1]
            and (xy[o][0] < xy[rep][0] or xy[o][1] > xy[rep][1]) for o in xy)]
        frontier.sort(key=lambda rep: xy[rep][0])
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(frontier) > 1:
            ax.plot([xy[r][0] for r in frontier], [xy[r][1] for r in frontier],
                    color="gray", linestyle="--", linewidth=1, zorder=2,
                    label="efficiency frontier")
        for rep in cq_reps:
            x, y = xy[rep]
            c = cov.get(rep, 1.0)
            ax.scatter(x, y, color=colors.get(rep, "gray"), s=40 + 90 * c,
                       edgecolor="red" if c < 0.8 else "black",
                       linewidth=1.4 if c < 0.8 else 0.5, zorder=3)
            tag = rep if c > 0.999 else f"{rep} (cov {c * 100:.0f}%)"
            ax.annotate(tag, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Mean prompt tokens  (context size -> serialization overhead)")
        ax.set_ylabel("Mean answer correctness  (in-scope, surviving cells)")
        ax.set_title("Cost vs Quality", fontsize=12)
        # keep the explanation off the title (it overran the frame); footnote in the
        # empty lower-left so it clears the right-hand (often json) markers
        ax.text(0.01, 0.01,
                "upper-left = more accuracy per token\nmarker shrinks + red edge as coverage drops",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="gray")
        ax.grid(linestyle="--", alpha=0.4)
        if len(frontier) > 1:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "cost_quality.png", dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir / 'cost_quality.png'}")
    else:
        # No token data recorded (some backends omit it) -> drop any stale chart.
        stale = out_dir / "cost_quality.png"
        if stale.exists():
            stale.unlink()

    # -- Faithfulness vs Answer Correctness: a guess / grounding diagnostic --
    # Only meaningful when faithfulness was computed (compute_faithfulness=True);
    # otherwise there is nothing on the x-axis, so skip rather than emit an empty
    # chart, and remove a stale one from a prior faithfulness-on run.
    if any(r["faithfulness"] != "" for r in rows):
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
    else:
        stale = out_dir / "faith_vs_ac.png"
        if stale.exists():
            stale.unlink()

    # -- Latency per representation (operational trade-off) --
    # Box + whiskers per rep (median, IQR, 1.5*IQR whiskers, outliers as fliers).
    # No point overlay -- with many questions the scatter buried the box, and the
    # box's own whiskers + fliers already carry the spread. Reps go in a side
    # legend (boxes left-to-right == legend top-to-bottom) since the long combined
    # names collide when written diagonally under the axis.
    from matplotlib.patches import Patch
    latency_map: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["latency_ms"]:
            latency_map[(r["question_id"], r["representation"])].append(float(r["latency_ms"]) / 1000)
    question_ids = sorted({r["question_id"] for r in rows}, key=lambda x: int(x) if x.isdigit() else 0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn: list[str] = []
    for pos, rep in enumerate(representations):
        per_q = [statistics.mean(latency_map[(qid, rep)])
                 for qid in question_ids if latency_map[(qid, rep)]]
        if not per_q:
            continue
        c = colors.get(rep, "gray")
        ax.boxplot(per_q, positions=[pos], widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=0.75, edgecolor="black", linewidth=0.6),
                   medianprops=dict(color="black", linewidth=1.4),
                   whiskerprops=dict(color="black", linewidth=0.8),
                   capprops=dict(color="black", linewidth=0.8),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=c,
                                   markeredgecolor="black", markeredgewidth=0.3, alpha=0.6))
        drawn.append(rep)
    ax.set_xticks([])
    ax.set_xlabel("Representation (see legend)")
    ax.set_ylabel("Mean response latency (s)")
    ax.set_title("Response Latency by Representation  (box = IQR, whiskers = 1.5*IQR, dots = outliers)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles = [Patch(facecolor=colors.get(r, "gray"), edgecolor="black", label=r) for r in drawn]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.savefig(out_dir / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'latency_comparison.png'}")
