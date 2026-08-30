"""Generate scope-aware, host-separated comparison charts from evaluation results."""

from __future__ import annotations

import csv
import collections
import statistics
from pathlib import Path

from evaluation.scope import in_scope as _in_scope
from evaluation.axes import (
    AXES, FULL_RECORD_ANCHOR,
    MIN_OBSERVATIONS, NON_SPATIAL_ANCHOR, NOT_EVALUATED, PRACTICAL_MARGIN,
    VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
    VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED, dataset_of, rep_role,
)
# Formal comparison logic comes from report.py.
from evaluation.report import (
    _cells, _load_pairs, _paired_rows,
    non_spatial_anchor_lifts,
)


# --- shared loading --------------------------------------------------------

def _per_question_ac(results_path: Path, dataset: str | None = None):
    """Return (points, types, reps); points[(qtype, rep)] = one mean AC per question, averaged over repetitions."""
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


# Stable representation colours, grouped loosely by representation family.
REP_COLORS: dict[str, str] = {
    "inventory":                     "#9e9e9e",  # non-spatial anchor
    "json_mini":                     "#1f1f1f",  # full-record anchor
    "json_pretty":                   "#5c5c5c",  # formatting ablation
    "prose":                         "#6a3d9a",  # natural language
    "narrative":                     "#cab2d6",
    # connectivity / structure (spatial-encoding connectivity, graph structure)
    "topology_inventory":            "#1f78b4",
    "room_tree":                     "#a6cee3",
    "graph_digest":                  "#08519c",
    # metric / framing (spatial-encoding metric, relational/navigational framing)
    "metric_relations":              "#33a02c",
    "navigation":                    "#00bcd4",
    # Darker navigation hue for its matched framing counterpart.
    "topology_metric":               "#00707f",
    # relation organization
    "relations_flat":                "#e31a1c",
    "relations_predicate":           "#ff7f00",
    "relations_subject":             "#b15928",
    "relations_digest":              "#fdbf6f",
    "relations_tree":                "#8c510a",
}


def _colors(reps: list[str]):
    """Return stable colors; unknown reps use tab20."""
    import matplotlib.pyplot as plt
    extra = [r for r in reps if r not in REP_COLORS]
    fallback = {}
    if extra:
        cmap = plt.cm.get_cmap("tab20", max(len(extra), 1))
        fallback = {r: cmap(i) for i, r in enumerate(extra)}
    return {rep: REP_COLORS.get(rep, fallback.get(rep)) for rep in reps}


# --- ordering and selection (the two places a ranking could sneak in) ------

_ROLE_BLOCK = {"non_spatial_anchor": 0, "pole": 1, "candidate": 2, "full_record_anchor": 3}


def registry_rep_order(reps) -> list[str]:
    """Order reps by registry role and ladder, never by results."""
    ladder_rank: dict[str, int] = {}
    for axis in AXES:
        for rep in axis.ladder:
            ladder_rank.setdefault(rep, len(ladder_rank))
    unnamed = len(ladder_rank)
    return sorted(reps, key=lambda r: (_ROLE_BLOCK.get(rep_role(r), 1),
                                       ladder_rank.get(r, unnamed), r))


# --- axis cards (the headline reporting chart) -----------------------------

def _axis_card_reps(axis, qt: str, points: dict) -> list[str]:
    """Return available reps in anchor/ladder order; [] if fewer than two."""
    reps = []
    if points.get((qt, NON_SPATIAL_ANCHOR)):
        reps.append(NON_SPATIAL_ANCHOR)
    for p in axis.ladder:
        if p in (NON_SPATIAL_ANCHOR, FULL_RECORD_ANCHOR):
            continue
        if _in_scope(p, qt, axis.host) and points.get((qt, p)):
            reps.append(p)
    if points.get((qt, FULL_RECORD_ANCHOR)) and FULL_RECORD_ANCHOR not in reps:
        reps.append(FULL_RECORD_ANCHOR)
    return reps if len(reps) >= 2 else []


def _plot_axis_cards(results_path: Path, rows: list[dict], out_dir: Path, color: dict) -> list[str]:
    """Plot one host-scoped axis card with descriptive min-max whiskers."""
    import matplotlib.pyplot as plt
    import numpy as np
    rng = np.random.default_rng(42)

    written: list[str] = []
    for axis in AXES:
        points, _t, _r = _per_question_ac(results_path, dataset=axis.host)
        cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == axis.host])
        groups = [(qt, reps) for qt in axis.probe_types
                  if (reps := _axis_card_reps(axis, qt, points))]
        if not groups:
            continue
        fig, axs = plt.subplots(1, len(groups), squeeze=False,
                                figsize=(max(4.0, 2.6 * len(groups) + 0.4 * sum(len(r) for _, r in groups)), 4.6))
        for gi, (qt, reps) in enumerate(groups):
            ax = axs[0][gi]
            ns_anchor_ac = statistics.mean(points[(qt, NON_SPATIAL_ANCHOR)]) if points.get((qt, NON_SPATIAL_ANCHOR)) else None
            fr_anchor_ac = statistics.mean(points[(qt, FULL_RECORD_ANCHOR)]) if points.get((qt, FULL_RECORD_ANCHOR)) else None
            if ns_anchor_ac is not None:
                ax.axhline(ns_anchor_ac, color=color.get(NON_SPATIAL_ANCHOR, "grey"), linestyle="--",
                           linewidth=1.0, alpha=0.7, zorder=1)
            if fr_anchor_ac is not None:
                ax.axhline(fr_anchor_ac, color=REP_COLORS.get(FULL_RECORD_ANCHOR, "black"), linestyle=":",
                           linewidth=1.0, alpha=0.7, zorder=1)
            for xi, rep in enumerate(reps):
                vals = points[(qt, rep)]
                m = statistics.mean(vals)
                ok, cov = _license(cells, rep, qt)
                ax.bar(xi, m, 0.72, color=color.get(rep, "grey"), **_mark(ok), zorder=2)
                # Descriptive per-question min-max; not an interval.
                lo, hi = (min(vals), max(vals)) if len(vals) > 1 else (m, m)
                ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt="none",
                            color="black", capsize=2, linewidth=0.8, zorder=3)
                jit = rng.uniform(-0.16, 0.16, len(vals))
                ax.scatter(xi + jit, vals, s=9, color=color.get(rep, "grey"),
                           edgecolor="black", linewidth=0.3, alpha=0.5, zorder=4)
                tag = f"n={len(vals)}" if ok else f"n={len(vals)} cov {cov:.0%} n/l"
                ax.text(xi, min(1.0, hi) + 0.015, tag, ha="center",
                        va="bottom", fontsize=6, color="gray" if ok else "red")
            ax.set_xticks(range(len(reps)))
            ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.08)
            ax.set_title(qt.replace("_", " "), fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            if gi == 0:
                ax.set_ylabel("Answer correctness")
        fig.suptitle(f"{axis.label}   (host: {axis.host}; "
                     f"{NON_SPATIAL_ANCHOR}=dashed, {FULL_RECORD_ANCHOR}=dotted, "
                     f"hatch = {VERDICT_NOT_LICENSED})", fontsize=11)
        fig.text(0.01, 0.002, NOT_LICENSED_NOTE, ha="left", va="bottom",
                 fontsize=7, color="gray")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        stem = f"axis_card_{axis.id}.png"
        fig.savefig(out_dir / stem, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(stem)
        print(f"plot -> {out_dir / stem}")
    return written


# --- aggregate-level charts (AC by axis, lift over non-spatial anchor) -------------------

def _license(cells: dict, rep: str, qt: str) -> tuple[bool, float]:
    """Return (rank_eligible, coverage); missing cells are ineligible."""
    c = cells.get((rep, qt))
    return (False, 0.0) if c is None else (c.rank_eligible, c.coverage)


def _admitted(rep: str, qt: str, host: str) -> bool:
    """Admit inventory as an anchor independently of capability scope."""
    return rep == NON_SPATIAL_ANCHOR or _in_scope(rep, qt, host)


def _mark(ok: bool) -> dict:
    """Bar styling for a rank-ineligible cell: hatched, red-edged, faded."""
    if ok:
        return dict(alpha=0.85, edgecolor="none", linewidth=0.0)
    return dict(alpha=0.4, hatch="xx", edgecolor="red", linewidth=0.9)


NOT_LICENSED_NOTE = (
    f"hatch + red edge = {VERDICT_NOT_LICENSED}; drawn, not ranked.")

LATENCY_DIAGNOSTIC_NOTE = (
    "Latency is diagnostic; prompt tokens are the cost metric.")


def _plot_ac_by_type(results_path: Path, rows: list[dict], ds: str,
                     out_dir: Path, color: dict) -> set[str]:
    """Plot host-scoped AC by question type."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    rng = np.random.default_rng(42)

    points, types, reps = _per_question_ac(results_path, dataset=ds)
    if not types:
        return set()
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])

    fig, ax = plt.subplots(figsize=(max(10, 2.2 * len(types)), 6))
    used: set[str] = set()
    for gi, qt in enumerate(types):
        drawn = [r for r in reps if r not in NOT_EVALUATED
                 and _admitted(r, qt, ds) and points.get((qt, r))]
        k = len(drawn)
        if k == 0:
            continue
        barw = 0.8 / k
        for j, rep in enumerate(drawn):
            vals = points[(qt, rep)]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            ok, cov = _license(cells, rep, qt)
            xpos = gi + (j - (k - 1) / 2) * barw
            ax.bar(xpos, mean, barw * 0.9, color=color[rep],
                   label=rep if rep not in used else None, **_mark(ok))
            used.add(rep)
            # Clip ±1 SD to the AC bounds [0, 1].
            lo, hi = max(0.0, mean - std), min(1.0, mean + std)
            ax.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        color="black", capsize=2, linewidth=0.8)
            jitter = rng.uniform(-barw * 0.25, barw * 0.25, len(vals))
            # Alpha reveals overplotting.
            ax.scatter(xpos + jitter, vals, s=10, color=color[rep],
                       edgecolor="black", linewidth=0.3, zorder=3, alpha=0.55)
            tag = f"n={len(vals)}" if ok else f"n={len(vals)} cov {cov:.0%} n/l"
            ax.text(xpos, -0.04, tag, ha="center", va="top",
                    fontsize=6, color="gray" if ok else "red", rotation=90)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_ylabel("Answer correctness")
    ax.set_title(f"Answer correctness by type — host: {ds} (mean ±1 SD; dots=questions)")
    ax.set_ylim(-0.22, 1.05)  # extra bottom room for the rotated n= labels
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.text(0.01, 0.002, NOT_LICENSED_NOTE, ha="left", va="bottom",
             fontsize=7, color="gray")
    # Legend outside the axes -- 16 reps would otherwise sit on top of the bars.
    handles = [Patch(color=color[r], label=r) for r in reps if r in used]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    name = f"ac_by_axis_{ds}.png"
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / name}")
    return {name}


def _plot_non_spatial_anchor_lift(rows: list[dict], ds: str, out_dir: Path, color: dict) -> set[str]:
    """Plot matched AC lift over inventory for one host."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    rng = np.random.default_rng(42)

    lifts = [p for p in non_spatial_anchor_lifts(rows, ds) if not _in_scope(NON_SPATIAL_ANCHOR, p.qt, ds)]
    if not lifts:
        return set()
    by_type: dict[str, dict[str, object]] = collections.defaultdict(dict)
    for p in lifts:
        by_type[p.qt][p.rep_b] = p
    ctrl_types = sorted(by_type)

    fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(ctrl_types)), 6))
    used: set[str] = set()
    for gi, qt in enumerate(ctrl_types):
        drawn = registry_rep_order(by_type[qt])
        k = len(drawn)
        if k == 0:
            continue
        barw = 0.8 / k
        for j, rep in enumerate(drawn):
            p = by_type[qt][rep]
            ok = p.verdict != VERDICT_NOT_LICENSED
            xpos = gi + (j - (k - 1) / 2) * barw
            ax.bar(xpos, p.mean, barw * 0.9, color=color.get(rep, "grey"),
                   label=rep if rep not in used else None, **_mark(ok))
            used.add(rep)
            if p.q_deltas:
                jit = rng.uniform(-barw * 0.22, barw * 0.22, len(p.q_deltas))
                ax.scatter(xpos + jit, p.q_deltas, s=8, color="black",
                           alpha=0.35, zorder=3, linewidth=0)
            ax.text(xpos, -1.06, f"n={p.n_questions}" + ("" if ok else " n/l"),
                    ha="center", va="bottom", fontsize=6,
                    color="gray" if ok else "red", rotation=90)

    ax.set_xticks(range(len(ctrl_types)))
    ax.set_xticklabels([t.replace("_", " ") for t in ctrl_types], rotation=20, ha="right")
    ax.set_ylabel(f"AC lift over the {NON_SPATIAL_ANCHOR} non-spatial anchor  (matched questions)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Spatial-structure lift — host: {ds} (>0 helps, <0 hurts)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.text(0.01, 0.002,
             "bars = scene means or mean of scene values; dots = paired question differences; "
             + NOT_LICENSED_NOTE,
             ha="left", va="bottom", fontsize=7, color="gray")
    handles = [Patch(color=color.get(r, "grey"), label=r)
               for r in registry_rep_order(used)]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    name = f"value_of_spatial_structure_{ds}.png"
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / name}")
    return {name}


def _plot_heatmap(results_path: Path, rows: list[dict], ds: str,
                  out_dir: Path) -> set[str]:
    """Plot one host's rep × type AC matrix with scope and eligibility marks."""
    import matplotlib.pyplot as plt
    import numpy as np

    points, types, reps = _per_question_ac(results_path, dataset=ds)
    if not types:
        return set()
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
    counts = {(qt, rep): len(vals) for (qt, rep), vals in points.items()}
    ordered_reps = registry_rep_order(
        [r for r in reps if r not in NOT_EVALUATED
         and any(points.get((qt, r)) for qt in types)])
    if not ordered_reps:
        return set()

    M = np.full((len(ordered_reps), len(types)), np.nan)
    for i, rep in enumerate(ordered_reps):
        for j, qt in enumerate(types):
            if _admitted(rep, qt, ds) and points.get((qt, rep)):
                M[i, j] = statistics.mean(points[(qt, rep)])
    fig, ax = plt.subplots(
        figsize=(max(8, 1.3 * len(types) + 3), max(4, 0.62 * len(ordered_reps) + 2))
    )
    cmap = plt.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, rep in enumerate(ordered_reps):
        for j, qt in enumerate(types):
            if not _admitted(rep, qt, ds):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="lightgrey", edgecolor="white",
                                           hatch="//", zorder=2))
                ax.text(j, i, "-", ha="center", va="center", color="gray", fontsize=8)
            elif not np.isnan(M[i, j]):
                n = counts.get((qt, rep), 0)
                ok, cov = _license(cells, rep, qt)
                if not ok:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", hatch="xx",
                                               linewidth=1.2, alpha=0.7, zorder=3))
                label = f"{M[i, j]:.2f}\nn={n}" + ("" if ok else f"\nn/l cov {cov:.0%}")
                ax.text(j, i, label, ha="center", va="center", color="black",
                        fontsize=7, zorder=5)
                if 0 < n < MIN_OBSERVATIONS:  # under-powered cell: red outline
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", linewidth=1.4, zorder=4))
    # Separate reporting-role blocks.
    for i in range(1, len(ordered_reps)):
        if (_ROLE_BLOCK.get(rep_role(ordered_reps[i]), 1)
                != _ROLE_BLOCK.get(rep_role(ordered_reps[i - 1]), 1)):
            ax.axhline(i - 0.5, color="black", linewidth=1.4)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_yticks(range(len(ordered_reps)))
    ax.set_yticklabels(ordered_reps)
    ax.set_title(f"AC matrix — host: {ds} (grey=out of scope; red hatch={VERDICT_NOT_LICENSED})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean AC")
    fig.tight_layout()
    name = f"ac_heatmap_{ds}.png"
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / name}")
    return {name}


# --- the graded verdict figure (paired separation) --------------------------
# Verdict markers, not delta sign, encode the grade.
# Avoid red/green: delta sign is directional, not good/bad.
VERDICT_COLOR = {
    VERDICT_CONSISTENT:    "#1b6f4a",   # the only graded win
    VERDICT_DIRECTIONAL:   "#c77c17",   # consistent direction without a licensed consistent advantage, or capped
    VERDICT_MIXED:         "#9d3b8c",   # scenes disagree at the declared margin
    VERDICT_NO_SEPARATION: "#6b7280",   # no practically meaningful separation
    VERDICT_NOT_LICENSED:  "#b0b6bf",   # gates failed -- drawn, never ranked
}

# Marker shape also encodes verdict for grayscale readability.
VERDICT_MARKER = {
    VERDICT_CONSISTENT:    "o",   # filled circle
    VERDICT_DIRECTIONAL:   "^",   # hollow triangle
    VERDICT_MIXED:         "D",   # hollow diamond
    VERDICT_NO_SEPARATION: "o",   # hollow circle
    VERDICT_NOT_LICENSED:  "p",   # hollow pentagon
}
BAND = "#e8eaed"        # the +-PRACTICAL_MARGIN region
SCENE_MARK = "#4a5058"  # the individual scene deltas
GATED_BG = "#f5f6f7"    # backs the `not licensed` block in the separation forest


_VERDICT_ORDER = {VERDICT_CONSISTENT: 0, VERDICT_DIRECTIONAL: 1, VERDICT_MIXED: 2,
                  VERDICT_NO_SEPARATION: 3, VERDICT_NOT_LICENSED: 4}
# Use one right-hand annotation per row to avoid column overlap.
_COL_NOTE, _SEPARATION_RECT_RIGHT = 1.005, 0.82


def _plot_paired_separation(prs: list, out_dir: Path,
                            responder: str, judge: str) -> set[str]:
    """Plot report.Paired rows by host, keeping gated evidence visible."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not prs:
        return set()
    hosts = sorted({p.host for p in prs})

    def blocks(host):
        rows = [p for p in prs if p.host == host]
        key = lambda p: (p.axis_id, _VERDICT_ORDER.get(p.verdict, 9), -abs(p.mean))
        return (sorted([p for p in rows if p.verdict != VERDICT_NOT_LICENSED], key=key),
                sorted([p for p in rows if p.verdict == VERDICT_NOT_LICENSED], key=key))

    per_host = {h: blocks(h) for h in hosts}
    # +1 slot for the blank row that carries the block divider, where both blocks exist
    slots = {h: len(g) + len(u) + (1 if g and u else 0) for h, (g, u) in per_host.items()}

    # Scale panel height by row count while reserving fixed title/footer space.
    H = 0.34 * sum(slots.values()) + 2.6 * len(hosts)
    fig, axes = plt.subplots(
        len(hosts), 1, squeeze=False, figsize=(13.6, H),
        gridspec_kw=dict(height_ratios=[max(slots[h], 1) for h in hosts]))

    for hi, h in enumerate(hosts):
        ax = axes[hi][0]
        graded, gated = per_host[h]
        ordered = graded + gated
        ys, cursor = [], slots[h] - 1
        for i, _p in enumerate(ordered):
            if i == len(graded) and graded and gated:
                # Reserve one divider row when both blocks exist.
                cursor -= 1
            ys.append(cursor)
            cursor -= 1

        lim = max(0.55, max(abs(v) for p in ordered for v in p.scene_deltas.values()) + 0.06)
        if gated:
            # Shade the gated block.
            ax.axhspan(min(ys[len(graded):]) - 0.5, max(ys[len(graded):]) + 0.5,
                       color=GATED_BG, zorder=0)
        ax.axvspan(-PRACTICAL_MARGIN, PRACTICAL_MARGIN, color=BAND, zorder=1)
        ax.axvline(0, color="#333", linewidth=0.9, zorder=2)

        labels, prev_axis = [], None
        for i, (p, y) in enumerate(zip(ordered, ys)):
            # Restart axis labels at the gated block.
            if i == len(graded):
                prev_axis = None
            if p.axis_id != prev_axis:
                ax.axhline(y + 0.5, color="#c9ccd1", linewidth=0.8, zorder=2)
                ax.text(-lim * 0.985, y + 0.42, p.axis_id, fontsize=7.4,
                        fontweight="bold", color="#4a5058", va="top", zorder=5)
                prev_axis = p.axis_id
            labels.append(f"{p.qt}:  {p.rep_b} - {p.rep_a}")
            c = VERDICT_COLOR[p.verdict]
            gated_row = p.verdict == VERDICT_NOT_LICENSED
            for v in p.scene_deltas.values():
                ax.plot([v], [y], marker="|", markersize=8, color=SCENE_MARK,
                        alpha=0.4 if gated_row else 0.7, zorder=4, linestyle="none")
            win = p.verdict == VERDICT_CONSISTENT
            ax.plot([p.mean], [y], marker=VERDICT_MARKER[p.verdict],
                    markersize=7 if win else 5.5,
                    markerfacecolor=c if win else "white", markeredgecolor=c,
                    markeredgewidth=1.5, zorder=5, linestyle="none")
            # Show question, fact-set, and scene support.
            tag = f"n_q={p.n_questions}" + (f", fs={p.n_factsets}" if p.n_factsets else "")
            tag += f", {len(p.scene_deltas)} sc"
            note_color, weight = "#8a9099", "normal"
            if p.capped:
                tag += "   CAPPED"
                note_color, weight = VERDICT_COLOR[VERDICT_DIRECTIONAL], "bold"
            elif p.ineligible:
                tag += f"   {p.ineligible}"
            ax.text(_COL_NOTE, y, tag, transform=ax.get_yaxis_transform(),
                    fontsize=6.4, va="center", ha="left", color=note_color,
                    fontweight=weight)

        if graded and gated:
            div = ys[len(graded)] + 1.0
            ax.axhline(div, color="#6b7280", linewidth=1.3, linestyle="-", zorder=6)
            ax.text(-lim * 0.985, div - 0.08,
                    f"{VERDICT_NOT_LICENSED.upper()} - gates failed; deltas shown, "
                    "grade withheld, never ranked",
                    fontsize=7, fontweight="bold", color="#6b7280", va="top", zorder=6)

        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7.2, fontfamily="monospace")
        for tick, p in zip(ax.get_yticklabels(), ordered):
            if p.verdict == VERDICT_NOT_LICENSED:
                tick.set_color("#8a9099")
        ax.set_ylim(-0.7, slots[h] - 0.3)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"host: {h}   ({len(graded)} graded, {len(gated)} "
                     f"{VERDICT_NOT_LICENSED})", fontsize=10, fontweight="bold",
                     loc="left")
        if hi == len(hosts) - 1:
            # Rows are rep_b - rep_a; positive favors rep_b.
            ax.set_xlabel("Scene-level mean AC delta (rep_b - rep_a; >0 favors rep_b)",
                          fontsize=9)

    fig.suptitle("Declared paired comparisons under the same verdict rules",
                 fontsize=13, fontweight="bold", x=0.05, ha="left", y=1 - 0.22 / H)
    handles = [
        Line2D([], [], marker=VERDICT_MARKER[VERDICT_CONSISTENT], linestyle="none", markersize=7,
               markerfacecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               markeredgecolor=VERDICT_COLOR[VERDICT_CONSISTENT], label=VERDICT_CONSISTENT),
    ] + [
        Line2D([], [], marker=VERDICT_MARKER[v], linestyle="none", markersize=5.5,
               markerfacecolor="white", markeredgecolor=VERDICT_COLOR[v], label=v)
        for v in (VERDICT_DIRECTIONAL, VERDICT_MIXED, VERDICT_NO_SEPARATION,
                  VERDICT_NOT_LICENSED)
    ] + [
        Line2D([], [], marker="|", linestyle="none", markersize=8, color=SCENE_MARK,
               label="individual scene delta"),
        Patch(facecolor=BAND, label=f"+-{PRACTICAL_MARGIN:.2f} practical margin"),
        Patch(facecolor=GATED_BG, label=f"{VERDICT_NOT_LICENSED} block"),
    ]
    # Keep the legend figure-level across variable-height panels.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1 - 0.42 / H),
               fontsize=7.4, frameon=False, ncol=4)
    fig.text(0.05, 0.14 / H,
             "Scene ticks are the replication; no CIs. CAPPED = confound-downgraded.\n"
             f"responder: {responder}  |  judge: {judge}",
             fontsize=7, color="#8a9099", va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.70 / H, _SEPARATION_RECT_RIGHT, 1 - 1.00 / H))
    fig.savefig(out_dir / "paired_separation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'paired_separation.png'}")
    return {"paired_separation.png"}


def plot_aggregate(aggregate_path: Path) -> None:
    out_dir = aggregate_path.parent
    # Delete the permanently retired post-hoc artifact before early returns.
    stale_ceiling = out_dir / "ceiling_premium.png"
    if stale_ceiling.exists():
        stale_ceiling.unlink()

    results_path = out_dir / "results.csv"
    if not results_path.exists():
        return
    # Error rows remain in coverage; AC means exclude them.
    with results_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    points, types, reps = _per_question_ac(results_path)
    if not types:
        return
    color = _colors(reps)

    # -- Chart 0: axis cards --
    written_cards = _plot_axis_cards(results_path, rows, out_dir, color)

    # -- Charts 1-3: host-scoped AC, anchor lift, and heatmap --
    written: set[str] = set()
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        written |= _plot_ac_by_type(results_path, rows, ds, out_dir, color)
        written |= _plot_non_spatial_anchor_lift(rows, ds, out_dir, color)
        written |= _plot_heatmap(results_path, rows, ds, out_dir)

    # -- Chart 4: paired verdicts from report.py --
    responder = rows[0].get("responder") or "?"
    judge = rows[0].get("judge") or "?"
    verdict_figs = _plot_paired_separation(_paired_rows(rows, pairs=_load_pairs()),
                                           out_dir, responder, judge)

    # Remove obsolete or stale aggregate charts.
    stale = ["ac_comparison.png", "faithfulness_comparison.png", "ac_delta.png",
             "ac_lift_over_floor.png", "axis_cards.png",
             # Remove host-pooled predecessors.
             "ac_by_axis.png", "value_of_spatial_structure.png", "ac_heatmap.png",
             # Remove retired pooled-delta chart.
             "axis_contrasts.png"]
    stale += [p.name for p in out_dir.glob("axis_card_*.png") if p.name not in written_cards]
    # Remove per-host outputs whose data disappeared.
    for pattern in ("ac_by_axis_*.png", "value_of_spatial_structure_*.png",
                    "ac_heatmap_*.png"):
        stale += [p.name for p in out_dir.glob(pattern) if p.name not in written]
    # Remove stale verdict output when nothing is drawable.
    stale += [n for n in ("paired_separation.png",) if n not in verdict_figs]
    for name in stale:
        p = out_dir / name
        if p.exists():
            p.unlink()


# --- per-observation charts (grounding diagnostic, latency) ----------------

def _matched_cost_quality_point(ac_by_qid: dict[str, list[float]],
                                tok_by_qid: dict[str, list[float]]
                                ) -> tuple[float, float, int] | None:
    """Return (tokens, AC, n) over AC∩token qids, averaging repetitions per qid."""
    matched = sorted(set(ac_by_qid) & set(tok_by_qid))
    if not matched:
        return None
    ac_mean = statistics.mean(statistics.mean(ac_by_qid[q]) for q in matched)
    tok_mean = statistics.mean(statistics.mean(tok_by_qid[q]) for q in matched)
    return tok_mean, ac_mean, len(matched)


def plot_per_question(results_path: Path, diagnostics: bool = False) -> None:
    """Plot per-observation diagnostics; latency is optional."""
    import matplotlib.pyplot as plt

    all_rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    rows = [r for r in all_rows if not r["error"]]
    out_dir = results_path.parent
    representations = sorted({r["representation"] for r in rows})
    colors = _colors(representations)

    # Cost/AC panels are host- and type-scoped; coverage uses report.py cells.
    for stale_name in ("cost_quality.png",):
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()

    datasets = sorted({dataset_of(r["scene_id"]) for r in rows})
    written_cq: set[str] = set()
    for ds in datasets:
        ds_rows = [r for r in rows if dataset_of(r["scene_id"]) == ds]
        # Average repetitions within qid before averaging questions.
        ac_by_q: dict[str, dict[str, dict[str, list[float]]]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        tok_by_q: dict[str, dict[str, dict[str, list[float]]]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for r in ds_rows:
            rep = r["representation"]
            qt = r["question_type"] or "unknown"
            if rep in NOT_EVALUATED or not _in_scope(rep, qt, ds):
                continue
            qid = r["question_id"]
            if r["answer_correctness"] != "":
                ac_by_q[qt][rep][qid].append(float(r["answer_correctness"]))
            tok = r.get("prompt_tokens", "")
            if tok not in ("", "0", None):
                try:
                    tok_by_q[qt][rep][qid].append(float(tok))
                except ValueError:
                    pass

        cells = _cells([r for r in all_rows if dataset_of(r["scene_id"]) == ds])

        # Build each point from that rep/type's AC∩token qids.
        points_by_qt: dict[str, dict[str, tuple[float, float, int]]] = {}
        for qt in set(ac_by_q) | set(tok_by_q):
            pts = {}
            for rep in set(ac_by_q[qt]) | set(tok_by_q[qt]):
                pt = _matched_cost_quality_point(ac_by_q[qt].get(rep, {}),
                                                 tok_by_q[qt].get(rep, {}))
                if pt is not None:
                    pts[rep] = pt
            if pts:
                points_by_qt[qt] = pts

        qts = sorted(points_by_qt)
        if not qts:
            # No matched AC/token data.
            continue

        ncols = min(3, len(qts))
        nrows = (len(qts) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                                 squeeze=False)
        for idx, qt in enumerate(qts):
            ax = axes[idx // ncols][idx % ncols]
            reps = sorted(points_by_qt[qt])
            lic = {rep: _license(cells, rep, qt) for rep in reps}
            for rep in reps:
                tok_mean, ac_mean, n_q = points_by_qt[qt][rep]
                ok, cov = lic[rep]
                ax.scatter(tok_mean, ac_mean, color=colors.get(rep, "gray"), s=40 + 90 * cov,
                           edgecolor="red" if not ok else "black",
                           linewidth=1.4 if not ok else 0.5, zorder=3)
                tag = (f"{rep} (q_n={n_q})" if ok else
                       f"{rep} (q_n={n_q}; cov {cov * 100:.0f}%; {VERDICT_NOT_LICENSED})")
                ax.annotate(tag, (tok_mean, ac_mean), textcoords="offset points",
                            xytext=(6, 4), fontsize=7,
                            color="red" if not ok else "black")
            ax.set_title(qt, fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(linestyle="--", alpha=0.4)
        for idx in range(len(qts), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.supxlabel("Mean prompt tokens (matched to the plotted question set)", fontsize=9)
        fig.supylabel("Mean answer correctness (matched to the plotted question set)", fontsize=9)
        fig.suptitle(f"Cost vs Quality by question type — host: {ds} (descriptive audit)",
                    fontsize=12)
        fig.text(0.01, 0.005,
                 "Descriptive only; no ranking. Efficiency claims use paired comparisons.\n"
                 f"Red edge={VERDICT_NOT_LICENSED}; size=coverage; q_n=matched questions.",
                 ha="left", va="bottom", fontsize=7, color="gray")
        fig.tight_layout(rect=(0.01, 0.02, 1, 0.97))
        name = f"cost_quality_{ds}.png"
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        written_cq.add(name)
        print(f"plot -> {out_dir / name}")

    # Remove stale per-host cost-quality charts.
    for p in out_dir.glob("cost_quality_*.png"):
        if p.name not in written_cq:
            p.unlink()

    # Faithfulness metric was removed; clean up any chart left from older runs.
    stale = out_dir / "faith_vs_ac.png"
    if stale.exists():
        stale.unlink()

    # -- Latency diagnostic --
    for stale_name in ("latency_comparison.png",):
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()
    if not diagnostics:
        stale = out_dir / "latency_diagnostic.png"
        if stale.exists():
            stale.unlink()
        return

    # Boxplot per rep; legend avoids long x labels.
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
    ax.set_ylabel("Wall-clock response latency (s)")
    ax.set_title("Response latency (diagnostic; box=IQR, whiskers=1.5×IQR)")
    ax.text(0.0, -0.13, LATENCY_DIAGNOSTIC_NOTE, transform=ax.transAxes,
            fontsize=7, color="tab:red", va="top")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles = [Patch(facecolor=colors.get(r, "gray"), edgecolor="black", label=r) for r in drawn]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.savefig(out_dir / "latency_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'latency_diagnostic.png'}  (diagnostic; not a production figure)")
