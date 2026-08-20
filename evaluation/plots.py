"""plots.py — Generate comparison charts from evaluation results.

Charts are **scope-aware**: a representation is only drawn for a question type it
can actually answer. Each question type requires certain information channels
(connectivity, metric, ...); each representation carries some; a rep is in-scope
when it covers what the type needs. Plotting an out-of-scope rep (e.g. navigation,
which has no object inventory, on a containment question) would show a structural
zero as if it were a result — so those cells are masked out.

The model is fail-open: an unknown representation or question type is never
masked, so new parsers/types are shown until their scope is declared here.

Charts are also **host-separated**: every chart from which a comparison between two
representations is read is drawn once per host dataset, never pooled. ProcTHOR,
Gibson and 3RScan pose different questions over different scenes, and the reps are
not all present on all three (the relations_* family is 3RScan's, navigation is
not 3RScan's at all), so a pooled bar can be built from a question set the bar
beside it never saw. `_<host>` in a filename means exactly that.

The third rule is **rank-eligibility**: a cell whose coverage falls below
MIN_COVERAGE has an AC conditioned on the questions it survived, so it cannot be
compared on AC against a full-coverage cell. Those cells are drawn -- hidden data
is worse than flagged data -- but marked `not licensed`, excluded from every
ordering, and kept off the efficiency frontier. The threshold, the gates and every
paired delta come from `evaluation/report.py`, which owns the analysis contract;
this module draws it and does not restate it.

Charts produced
---------------
plot_aggregate (reads results.csv + aggregate.csv):
  axis_card_<id>.png           one figure per axis in evaluation.axes.AXES (named,
                               e.g. axis_card_spatial_encoding.png)
  ac_by_axis_<host>.png        AC per question type, in-scope reps, bars+std+dots
  value_of_spatial_structure_<host>.png  matched AC lift over the inventory floor
                               (the spatial-encoding result)
  ac_heatmap_<host>.png        rep x type mean-AC matrix, out-of-scope cells greyed
  axis_contrasts.png           paired per-question AC delta for each axis with a
                               headline pair (evaluation.axes.AXIS_PAIRS); already
                               per-axis-host by construction, so it is not split
  ceiling_premium.png          the headline claim, graded: the best OBSERVED derived
                               view against the json_mini ceiling, one row per
                               host x question type
  paired_separation.png        report.md's separation table drawn: one row per
                               comparison from report._paired_rows, faceted by host,
                               graded rows and `not licensed` rows in separate blocks

The last two carry every host in one file, like axis_contrasts.png and for the same
reason: each ROW is single-host by construction (its deltas come from one host's
scenes and are labelled with it), so nothing is pooled by putting them on one canvas.
Both are drawn straight off report.py -- `ceiling_premium` through the same pairing,
gate and verdict functions, `paired_separation` off the `Paired` objects themselves --
so a mark in either figure and a row in report.md are the same computation. Neither
lets the SIGN of a delta decide anything: the verdict chooses the marker, and a
positive mean the rule does not license is drawn hollow, exactly like a negative one.

A gated comparison keeps its scene deltas in both. Licensing applies to the VERDICT,
not to whether the difference exists: `not licensed` means the design cannot grade
this pair, so the row is drawn with its data, greyed, and excluded from every count
of wins. Hiding the deltas would be a second, undeclared filter on top of the gate.

plot_per_question (reads results.csv):
  cost_quality_<host>.png      mean AC vs mean prompt tokens, with efficiency frontier
  faith_vs_ac.png              per-observation guess detector (only if faithfulness on)

Diagnostics -- drawn only with `diagnostics=True`, deliberately outside the
production figure set:
  latency_diagnostic.png       wall-clock latency per representation. Not an
                               inferential chart and not the cost axis: prompt
                               tokens are. Latency is confounded by batch
                               composition, GPU contention and model residency at
                               call time -- a contended GPU silently falls back to
                               CPU and the same representation then reads an order
                               of magnitude slower -- none of which is a property
                               of the representation. It is deliberately NOT split
                               per host: host separation would fix the shallowest
                               of those confounds and make the chart look
                               comparable when it is not.
"""

from __future__ import annotations

import csv
import collections
import statistics
from pathlib import Path

# --- scope + axis model ----------------------------------------------------
# scope.py: which rep can answer which type (the structural mask, shared with the
# runner, which skips out-of-scope cells before they are computed). axes.py: how
# reps group into axis ladders for reporting (ladder order, floor/ceiling roles,
# host dataset) -- so the charts and report.md tell the same axis story.
#
# AXIS_PAIRS (the same-information contrast pairs for axis_contrasts.png) is now
# derived in axes.py from the AXES registry, and each entry carries the axis's host
# and probe_types as well as its two poles. A delta is averaged only over questions
# on that host, of those types, with *both* poles in scope -- the same restriction
# the axis cards already apply, so a bar and its card describe one question set. The
# spatial-encoding axis is a ladder (covered by axis_cards / value_of_spatial_-
# structure), so it contributes no pair. A pair with no such questions draws no bar.
from evaluation.scope import in_scope as _in_scope
from evaluation.axes import (
    AXES, AXIS_PAIRS, CANDIDATE, CEILING, FLOOR, MIN_COVERAGE,
    MIN_SCENES_SHOWING, PRACTICAL_MARGIN, SMALL_N,
    VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
    VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED, dataset_of, rep_role,
)
# report.py owns the analysis contract (thesis 4.6): what a cell's coverage is,
# whether it is rank-eligible, and how a difference between two representations is
# computed. Imported rather than reimplemented -- a chart and report.md are read
# side by side, and a second copy of these rules would agree today and drift later.
# `_cells` is private to report's own callers, not to the reporting layer.
#
# `ceiling_premium` and `paired_separation` below take that further and import the
# gate and the verdict themselves (`_gates`, `_verdict`, `paired_deltas`,
# `_summarise`, `_q_means`, `_paired_rows`). They state a GRADE, not a quantity, and
# a grading rule reimplemented in a drawing module is a rule that will disagree with
# report.md the first time either side is edited.
from evaluation.report import (
    _cells, _gates, _load_pairs, _paired_rows, _q_means, _summarise, _verdict,
    floor_lifts, paired_deltas,
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
    """Return (ac, qtype_of, host_of).

    ac[(qid, rep)] = mean answer_correctness over that cell's repetitions.
    qtype_of[qid]  = the question's type. Keeps question identity (which
    _per_question_ac drops) so paired per-question deltas can be computed.
    host_of[qid]   = the question's host dataset, so a caller can restrict a
    contrast to the host its axis declares. Read off `scene_id` (which the
    aggregate leaves bare -- only `question_id` is namespaced), because a qid
    alone does not carry its host and the alternative is pooling by default.
    """
    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    by_qr: dict[tuple, list[float]] = collections.defaultdict(list)
    qtype_of: dict[str, str] = {}
    host_of: dict[str, str] = {}
    for r in rows:
        if r["answer_correctness"] == "":
            continue
        by_qr[(r["question_id"], r["representation"])].append(float(r["answer_correctness"]))
        qtype_of[r["question_id"]] = r["question_type"] or "unknown"
        host_of[r["question_id"]] = dataset_of(r["scene_id"])
    ac = {k: sum(v) / len(v) for k, v in by_qr.items()}
    return ac, qtype_of, host_of


def axis_pair_questions(pair, qtype_of: dict, host_of: dict) -> list[str]:
    """The qids one axis-contrast bar may be computed from, in sorted order.

    Three filters, all required: the axis's declared host, its declared probe
    types, and the structural scope mask on BOTH poles. The first two come from
    the axis registry and say what the comparison IS; the third says what could
    be asked at all, and is kept because an axis may declare a type one pole
    cannot answer -- an empty bar is the correct outcome there, not a silent
    substitution of whatever else was in scope.

    Public (not underscored) because the leakage test drives this exact function
    rather than reimplementing the rule: a copy could agree with a broken
    original. See tests/test_axis_contrasts_scope.py.
    """
    return [qid for qid in sorted(qtype_of)
            if host_of.get(qid) == pair.host
            and qtype_of[qid] in pair.probe_types
            and _in_scope(pair.a, qtype_of[qid])
            and _in_scope(pair.b, qtype_of[qid])]


# Stable per-rep colours, reused across every chart so a representation keeps the
# same colour in all figures (lets a thesis reader cross-reference). Grouped by
# axis family: grey control, blue connectivity/structure, green metric/frame,
# warm relations (relation linearization), purple/black prose+json_mini ceiling.
# tab10 carries only 10 hues, so the old resampling collapsed the 16-rep set into
# duplicates; a fixed map avoids that and stays stable as reps come and go.
REP_COLORS: dict[str, str] = {
    "inventory":                     "#9e9e9e",  # control / floor
    "json_mini":                     "#1f1f1f",  # raw-coordinate ceiling (inherits
                                                 # the old `json` black, so every
                                                 # existing figure keeps its ceiling
                                                 # colour across the migration)
    "json_pretty":                   "#5c5c5c",  # same content, pretty-printed --
                                                 # a lighter grey of the ceiling's
                                                 # own hue, clear of inventory's
    "prose":                         "#6a3d9a",  # natural language
    # connectivity / structure (spatial-encoding connectivity, structure presentation)
    "topology_inventory":            "#1f78b4",
    "room_tree":                     "#a6cee3",
    "graph_digest":                  "#08519c",
    # metric / frame (spatial-encoding metric, reference frame)
    "metric_relations":              "#33a02c",
    "navigation":                    "#00bcd4",
    # navigation's matched counterpart (same channels, locative framing): a darker
    # tone of navigation's own hue, so the pair reads as a pair in every figure.
    "topology_metric":               "#00707f",
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


# --- ordering and selection (the two places a ranking could sneak in) ------

_ROLE_BLOCK = {"floor": 0, "pole": 1, "candidate": 2, "ceiling": 3}


def registry_rep_order(reps) -> list[str]:
    """Row order for the AC matrix: floor, poles in registry ladder order, reps no
    ladder names, the synthesis candidate, then the ceiling.

    Derived entirely from `axes.AXES` and `axes.rep_role` -- it reads no results at
    all. That is the point, and it is what makes "below-coverage cells cannot enter
    a ranking" true by construction here rather than by a filter someone has to
    remember to apply.

    The previous order was "best-first by the rep's overall in-scope mean", which
    fails three ways at once. It is the ALL/ALL grand mean the reporting rules
    forbid quoting (thesis 4.6) -- expressed as a row position instead of a printed
    number, but a reader still takes the top row as the best representation. It was
    incomparable row to row, because each rep's mean was taken over the question
    types THAT rep is in scope for: `navigation` (3 types) and `json_mini` (all of
    them) were ordered against each other on different question mixes, and a rep's
    position moved when a type was added to the run. And it pooled hosts, so a rep
    evaluated only on 3RScan was ranked against one evaluated only on ProcTHOR.

    The matrix is an overview, not a verdict. Ranking lives in report.md's paired
    separation table, where the comparison is matched, gated and graded.
    """
    ladder_rank: dict[str, int] = {}
    for axis in AXES:
        for rep in axis.ladder:
            ladder_rank.setdefault(rep, len(ladder_rank))
    unnamed = len(ladder_rank)
    return sorted(reps, key=lambda r: (_ROLE_BLOCK.get(rep_role(r), 1),
                                       ladder_rank.get(r, unnamed), r))


def efficiency_frontier(xy: dict[str, tuple[float, float]], eligible) -> list[str]:
    """The Pareto frontier of (mean tokens, mean AC): minimise x, maximise y.

    Computed over the RANK-ELIGIBLE points only. A point below MIN_COVERAGE is
    still drawn and still labelled, but it may not join the frontier, because the
    frontier is a ranking claim -- "nothing is both cheaper and at least as
    accurate" -- and a sub-coverage AC is conditioned on the questions that rep
    survived. This is the survivorship trap in its most persuasive form: the view
    that overflows the window on the dense scenes looks like the efficient choice
    precisely BECAUSE the questions that broke it are missing from its mean, and it
    is cheap for the same reason it is incomplete. An ineligible point cannot be
    dominated either -- it is not on the plane the frontier is drawn on at all.
    """
    pts = {r: p for r, p in xy.items() if r in eligible}
    front = [r for r in pts if not any(
        o != r and pts[o][0] <= pts[r][0] and pts[o][1] >= pts[r][1]
        and (pts[o][0] < pts[r][0] or pts[o][1] > pts[r][1]) for o in pts)]
    return sorted(front, key=lambda r: (pts[r][0], r))


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
    """One figure *per* registry entry (axes.AXES) -> axis_card_<id>.png: a subplot per
    probe type, reps drawn as bars in ladder order with the floor (inventory) and
    ceiling (json_mini) as a dashed/dotted band so a pole is read against them. Each rep
    is a readable x-tick label; the y-axis starts at 0. Restricted to the entry's
    host dataset (never pooled) and its probe types. Only what a ladder declares is
    drawn, so synthesis never appears unless an entry names it (everything else
    stays in report.md's own candidate table). Entries that
    are not one of the five design axes say so in the suptitle, from Axis.kind, so
    a card is never mistaken for an axis. Bars hatch when n < SMALL_N
    (screening-only); whisker = min-max range of the per-question means (a
    descriptive spread, NOT an interval -- overlap is never a tie rule, see
    thesis 4.6 and report.py's paired table); dots = per-question means. Returns the
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
                ax.axhline(ceil_ac, color=REP_COLORS.get(CEILING, "black"), linestyle=":",
                           linewidth=1.0, alpha=0.7, zorder=1)
            for xi, rep in enumerate(reps):
                vals = points[(qt, rep)]
                m = statistics.mean(vals)
                small = len(vals) < SMALL_N
                ax.bar(xi, m, 0.72, color=color.get(rep, "grey"), alpha=0.85,
                       hatch="//" if small else None,
                       edgecolor="black" if small else "none", linewidth=0.4, zorder=2)
                # Descriptive min-max spread of the per-question means, not an
                # interval estimate: two of these overlapping means nothing.
                lo, hi = (min(vals), max(vals)) if len(vals) > 1 else (m, m)
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
        kind = "" if axis.kind == "axis" else f" [{axis.kind}]"
        # Anchor names come from the registry, never a literal: the ceiling was
        # renamed once already (json -> json_mini) and a hardcoded label here would
        # have kept printing the retired name over correct data.
        fig.suptitle(f"{axis.label}{kind}   (host: {axis.host}; "
                     f"{FLOOR}=dashed, {CEILING}=dotted, hatch = n<{SMALL_N})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        stem = f"axis_card_{axis.id}.png"
        fig.savefig(out_dir / stem, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(stem)
        print(f"plot -> {out_dir / stem}")
    return written


# --- aggregate-level charts (AC by axis, lift over floor) -------------------

def _license(cells: dict, rep: str, qt: str) -> tuple[bool, float]:
    """(rank_eligible, coverage) for one cell -- fail-open where there is no cell.

    Fail-open matches the rest of the scope/reporting model: an absent cell is not
    evidence of a coverage problem, and a chart that flags what it merely has no
    rows about asserts something it does not know.
    """
    c = cells.get((rep, qt))
    return (True, 1.0) if c is None else (c.rank_eligible, c.coverage)


def _mark(ok: bool) -> dict:
    """Bar styling for a rank-ineligible cell: hatched, red-edged, faded."""
    if ok:
        return dict(alpha=0.85, edgecolor="none", linewidth=0.0)
    return dict(alpha=0.4, hatch="xx", edgecolor="red", linewidth=0.9)


NOT_LICENSED_NOTE = (
    f"hatch + red edge = coverage below {MIN_COVERAGE:.0%}, so the cell is "
    f"'{VERDICT_NOT_LICENSED}': its AC is conditioned on the questions that rep "
    f"survived. Drawn, never ranked.")

LATENCY_DIAGNOSTIC_NOTE = (
    "NOT INFERENTIAL: latency is confounded by batch composition, GPU contention "
    "and model residency at call time, so a difference between two boxes is not "
    "attributable to the representation. The cost axis is prompt tokens "
    "(cost_quality_<host>.png). Pooled across hosts on purpose — splitting would "
    "fix the shallowest confound only.")


def _plot_ac_by_type(results_path: Path, rows: list[dict], ds: str,
                     out_dir: Path, color: dict) -> set[str]:
    """AC per question type, in-scope reps side by side -- ONE host per figure.

    This was a single pooled figure. A `proximity` group then averaged ProcTHOR,
    Gibson and 3RScan questions into each bar, and the reps are not present on all
    three hosts, so two bars in one group could be built from disjoint question
    sets over different scenes and still be read as the head-to-head comparison
    this chart exists to support. Splitting by host is the only thing that makes
    the side-by-side reading true.
    """
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
        drawn = [r for r in reps if _in_scope(r, qt) and points.get((qt, r))]
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
            # AC is bounded [0,1]; clip the +/-1 std whisker so it never shoots
            # past the frame (the bimodal 0/1 spread makes std large).
            lo, hi = max(0.0, mean - std), min(1.0, mean + std)
            ax.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        color="black", capsize=2, linewidth=0.8)
            jitter = rng.uniform(-barw * 0.25, barw * 0.25, len(vals))
            # low alpha so coincident points darken (honest density) without blobs
            ax.scatter(xpos + jitter, vals, s=10, color=color[rep],
                       edgecolor="black", linewidth=0.3, zorder=3, alpha=0.55)
            tag = f"n={len(vals)}" if ok else f"n={len(vals)} cov {cov:.0%} n/l"
            ax.text(xpos, -0.04, tag, ha="center", va="top",
                    fontsize=6, color="gray" if ok else "red", rotation=90)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_ylabel("Answer correctness")
    ax.set_title(f"Answer correctness by question type - host: {ds}  (in-scope reps "
                 "only; bar=mean, whisker=±1 std, dots=per question)")
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


def _plot_floor_lift(rows: list[dict], ds: str, out_dir: Path, color: dict) -> set[str]:
    """AC lift over the `inventory` floor -- matched questions, one host per figure.

    Two things were wrong here and they compounded.

    The lift was `mean(rep) - mean(inventory)`: two cell means over two question
    sets. The floor is the one representation deliberately exempted from the scope
    filter (runner.py), so it answers every question of every type, while the reps
    it is subtracted from routinely do not -- the JSON views fail closed on the
    dense scenes and lose precisely the questions with the most content in them.
    The subtraction then charged the representation for the questions missing from
    its own mean. Every bar now comes from `report.floor_lifts`, which differences
    only the questions BOTH members answered and averages within scene, so the
    chart and report.md's `vs floor` column are the same computation.

    And it pooled hosts, which for this chart is worse than elsewhere: the floor's
    meaning is host-dependent. `inventory` on a ProcTHOR connectivity question is a
    room list against a door graph; on 3RScan it is an object list against a
    relation graph. One bar cannot average those and still be "the value of spatial
    structure".

    Drawn only where inventory is a genuine no-information control -- a type that
    needs a channel inventory lacks. On the content types inventory is a legitimate
    compact format rather than a floor, so those belong in the head-to-head chart.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    rng = np.random.default_rng(42)

    lifts = [p for p in floor_lifts(rows, ds) if not _in_scope(FLOOR, p.qt)]
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
    ax.set_ylabel(f"AC lift over the {FLOOR} floor  (matched questions)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Value of spatial structure - host: {ds}  (paired per-question AC "
                 f"difference vs the no-spatial {FLOOR} floor, averaged within scene; "
                 ">0 = it helped, <0 = it hurt)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.text(0.01, 0.002,
             "bar = mean of the SCENE values (the unit of replication); dots = the "
             "individual paired question differences.  " + NOT_LICENSED_NOTE,
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
    """The rep x type mean-AC matrix for one host.

    In-scope cells are coloured by mean AC; out-of-scope cells are greyed/hatched (a
    structural gap, not a zero); an in-scope cell with no data stays white; a cell
    below MIN_COVERAGE is red-hatched and labelled `n/l`.

    Rows are in `registry_rep_order` -- fixed by axes.py, reading no results. See
    that function for what the old best-first ordering was actually claiming.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    points, types, reps = _per_question_ac(results_path, dataset=ds)
    if not types:
        return set()
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
    counts = {(qt, rep): len(vals) for (qt, rep), vals in points.items()}
    ordered_reps = registry_rep_order(
        [r for r in reps if any(points.get((qt, r)) for qt in types)])
    if not ordered_reps:
        return set()

    M = np.full((len(ordered_reps), len(types)), np.nan)
    for i, rep in enumerate(ordered_reps):
        for j, qt in enumerate(types):
            if _in_scope(rep, qt) and points.get((qt, rep)):
                M[i, j] = statistics.mean(points[(qt, rep)])
    fig, ax = plt.subplots(
        figsize=(max(8, 1.3 * len(types) + 3), max(4, 0.62 * len(ordered_reps) + 2))
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
                ok, cov = _license(cells, rep, qt)
                if not ok:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", hatch="xx",
                                               linewidth=1.2, alpha=0.7, zorder=3))
                label = f"{M[i, j]:.2f}\nn={n}" + ("" if ok else f"\nn/l cov {cov:.0%}")
                ax.text(j, i, label, ha="center", va="center", color="black",
                        fontsize=7, zorder=5)
                if 0 < n < SMALL_N:  # screening-only cell: red outline
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="red", linewidth=1.4, zorder=4))
    # A rule at each reporting-role boundary (floor / poles / candidate / ceiling):
    # the candidate answers a different question than the axis poles, and the two
    # anchors bound them rather than competing with them.
    for i in range(1, len(ordered_reps)):
        if (_ROLE_BLOCK.get(rep_role(ordered_reps[i]), 1)
                != _ROLE_BLOCK.get(rep_role(ordered_reps[i - 1]), 1)):
            ax.axhline(i - 0.5, color="black", linewidth=1.4)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right")
    ax.set_yticks(range(len(ordered_reps)))
    ax.set_yticklabels(ordered_reps)
    ax.set_title(f"Answer correctness matrix - host: {ds}  (mean per rep x type; grey = "
                 f"out of scope, red outline = n<{SMALL_N}, red hatch = not licensed)\n"
                 "Rows in registry order (floor / poles / candidate / ceiling) - "
                 "NOT a ranking; separation is decided in report.md", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean AC")
    fig.tight_layout()
    name = f"ac_heatmap_{ds}.png"
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)
    print(f"plot -> {out_dir / name}")
    return {name}


# --- the two graded figures (ceiling premium, separation forest) -----------
# Everything above this line draws a QUANTITY and leaves the reader to judge it.
# These two draw a VERDICT, so they are held to a stricter rule: every number and
# every grade comes from report.py, and the sign of a delta is never allowed to
# stand in for the grade. A positive mean that the declared rule does not license
# is drawn exactly like a negative one -- hollow, in the verdict's own colour --
# because "it went up" and "it went up in a way this design can attest" are
# different claims and only the second is what the figure exists to make.

# Deliberately not a red/green scale. A delta's sign is not good or bad, and
# colouring it that way would restate the sign the markers already refuse to grade on.
VERDICT_COLOR = {
    VERDICT_CONSISTENT:    "#1b6f4a",   # the only graded win
    VERDICT_DIRECTIONAL:   "#c77c17",   # margin met but under-replicated, or capped
    VERDICT_MIXED:         "#9d3b8c",   # scenes disagree at the declared margin
    VERDICT_NO_SEPARATION: "#6b7280",   # null
    VERDICT_NOT_LICENSED:  "#b0b6bf",   # gates failed -- drawn, never ranked
}

# Marker SHAPE per verdict, shared by every thesis figure that draws a verdict mark
# (thesis_figures.py: F1, F2, and the vocabulary-sensitivity rep points), so colour is
# never the only cue separating two verdict classes in grayscale. `consistent
# advantage` and `no practically meaningful separation` share the circle -- fill state
# (filled/hollow), not shape, tells them apart, the same convention every entry here
# already uses for win vs. not-win. `not licensed` gets its own shape because it is a
# different concept (a gate failure, never ranked) and would otherwise read as `mixed`.
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

# Reps held out of the ceiling-premium candidate pool, each with the reason. Kept as
# a visible mapping (and printed in the figure's own footer) rather than inlined as a
# set literal, because the exclusion is a reporting CHOICE that moves the headline
# number: `synthesis` is both cheap and strong, so withholding it makes the premium
# look smaller, and a reader must be able to see that it was withheld and why.
CEILING_PREMIUM_EXCLUDED = {
    CEILING:       "is the comparator",
    "json_pretty": "carries the ceiling's own parse() output, so beating it is not a "
                   "derived-view result",
    CANDIDATE:     "reported as the candidate default in report.md's candidate section",
    FLOOR:         "the no-information control, not a derived view",
}


def ceiling_premium_rows(rows: list[dict]) -> list[dict]:
    """One row per (host, question type): the best OBSERVED derived view vs the ceiling.

    The delta is `rep - json_mini`, question-matched and averaged within scene by
    `report.paired_deltas`/`_summarise`, gated by `report._gates` and graded by
    `report._verdict` -- the same four functions the separation table calls, so a
    mark here and a `json_mini - rep` row in report.md are one computation with one
    sign flip between them (`_verdict` is symmetric under negation: the margin is on
    |mean| and reversal/disagreement are defined on the scene values' signs, so
    flipping every sign flips the direction and leaves the grade alone).

    NO CONFOUND IS PASSED, and that is not an omission: `Axis.confound_for` returns
    "" whenever the floor or the ceiling is a member, because an anchor comparison is
    not the design decision an axis isolates. Passing one here would cap a verdict
    report.md leaves uncapped.

    Two selections happen per row and both are disclosed in the figure. `k` counts
    the in-scope candidates the winner was chosen from -- the choice is POST-HOC on
    the same data, an oracle, not a representation anyone declared in advance. And
    the winner is the largest mean, taken over candidates in sorted-name order so
    `max` resolves a tie deterministically at the first name rather than by dict order.

    A row whose gate fires is returned WITH its scene deltas and marked
    `not licensed`. The gate says the design cannot grade the comparison; it does not
    say the difference is unknown, and dropping the deltas would be a second filter
    the analysis contract never declared. This is what makes the 3RScan rows legible:
    json_mini overflows one dense scene there, so its rows are gated on coverage while
    the derived views behind them sit at full coverage.
    """
    out: list[dict] = []
    for host in sorted({dataset_of(r["scene_id"]) for r in rows}):
        host_rows = [r for r in rows if dataset_of(r["scene_id"]) == host]
        cells = _cells(host_rows)
        q_mean, scene_of = _q_means(host_rows)
        qtype_of = {r["question_id"]: (r["question_type"] or "unknown") for r in host_rows}
        # Sorted so `max` below breaks ties at the alphabetically first candidate.
        pool = sorted({rep for rep, _q in q_mean} - set(CEILING_PREMIUM_EXCLUDED))

        for qt in sorted({qtype_of[q] for _rep, q in q_mean}):
            if (CEILING, qt) not in cells:
                continue                      # nothing to be a premium over
            qids = {q for _rep, q in q_mean if qtype_of[q] == qt}
            cands = []
            for rep in pool:
                if not _in_scope(rep, qt) or (rep, qt) not in cells:
                    continue
                by_scene = paired_deltas(q_mean, scene_of, qids, CEILING, rep)
                if not by_scene:
                    continue
                scene_deltas, n_q, q_deltas = _summarise(by_scene)
                ineligible = _gates(cells, qt, (CEILING, rep), n_q)
                tok_rep = cells[(rep, qt)].tokens_mean
                tok_ceil = cells[(CEILING, qt)].tokens_mean
                cands.append(dict(
                    host=host, qt=qt, rep=rep, scene_deltas=scene_deltas,
                    mean=statistics.mean(list(scene_deltas.values())),
                    n_questions=n_q, q_deltas=q_deltas, ineligible=ineligible,
                    verdict=_verdict(scene_deltas, "", ineligible),
                    tok_ratio=(tok_ceil / tok_rep) if tok_rep and tok_ceil else None))
            if not cands:
                continue
            best = dict(max(cands, key=lambda c: c["mean"]))
            best["k"] = len(cands)
            out.append(best)
    return out


def _plot_ceiling_premium(prem: list[dict], out_dir: Path,
                          responder: str, judge: str) -> set[str]:
    """ceiling_premium.png -- the headline claim with its grade attached."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not prem:
        return set()
    n = len(prem)
    # The caption, legend and footnote are fixed-size furniture; the plotting area is
    # not. Reserving them as FRACTIONS of the figure (which is what a row-count-driven
    # figsize makes them) works at 19 rows and collapses at 5, where the same fraction
    # is barely an inch and the caption lands on the axes. So the bands are reserved in
    # INCHES and converted, and every out-of-axes text is positioned the same way.
    H = 0.46 * n + 4.2
    TOP_IN, BOTTOM_IN = 1.15, 2.3          # caption band; legend + footnote band
    axes_h = H - TOP_IN - BOTTOM_IN        # inches of drawable height
    fig, ax = plt.subplots(figsize=(13.2, H))
    ax.axvspan(-PRACTICAL_MARGIN, PRACTICAL_MARGIN, color=BAND, zorder=0)
    ax.axvline(0, color="#333", linewidth=0.9, zorder=1)

    lo = min([v for r in prem for v in r["scene_deltas"].values()] + [0.0])
    hi = max([v for r in prem for v in r["scene_deltas"].values()] + [0.0])
    ax.set_xlim(min(-0.35, lo - 0.08), max(0.55, hi + 0.08))

    ylabels, host_bounds, prev_host = [], [], None
    for i, r in enumerate(prem):
        y = n - 1 - i
        if r["host"] != prev_host:
            host_bounds.append((y + 0.5, r["host"]))
            prev_host = r["host"]
        ylabels.append(r["qt"].replace("_", " "))
        gated = r["verdict"] == VERDICT_NOT_LICENSED
        c = VERDICT_COLOR[r["verdict"]]
        # The scene deltas are drawn whether or not the row is graded (see the
        # function above); a gated row's ticks are lighter, not absent.
        for v in r["scene_deltas"].values():
            ax.plot([v], [y], marker="|", markersize=9, color=SCENE_MARK,
                    alpha=0.35 if gated else 0.75, zorder=3, linestyle="none")
        win = r["verdict"] == VERDICT_CONSISTENT
        ax.plot([r["mean"]], [y], marker="o" if win else "D",
                markersize=8 if win else 6,
                markerfacecolor=c if win else "white", markeredgecolor=c,
                markeredgewidth=1.6, zorder=4, linestyle="none")

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(ylabels)), fontsize=8.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("Scene-paired AC delta:  best observed derived view  -  "
                  f"{CEILING} ceiling\n(per-question deltas averaged within scene, "
                  "then over the scene values)", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    x0 = ax.get_xlim()[0]
    for ybound, host in host_bounds:
        ax.axhline(ybound, color="#c9ccd1", linewidth=0.9)
        ax.text(x0 + 0.015, ybound - 0.42, host.upper(), fontsize=8.5,
                fontweight="bold", color="#4a5058", va="top")

    # Annotation columns, placed in axes-fraction x so they track the axes box.
    COL_REP, COL_TOK, COL_VERD = 1.04, 1.42, 1.62
    trans = ax.get_yaxis_transform()
    for x, label in ((COL_REP, "best observed derived view\n(of k in scope)"),
                     (COL_TOK, "prompt\ntokens"),
                     (COL_VERD, "verdict under the declared rule")):
        ax.text(x, n - 0.35, label, transform=trans, fontsize=7.4, fontweight="bold",
                va="bottom", ha="left", color="#4a5058")
    for i, r in enumerate(prem):
        y = n - 1 - i
        gated = r["verdict"] == VERDICT_NOT_LICENSED
        body = "#8a9099" if gated else "#2a2e33"
        ax.text(COL_REP, y, f"{r['rep']}  (of {r['k']})", transform=trans,
                fontsize=7.2, va="center", ha="left", color=body)
        ax.text(COL_TOK, y, f"{r['tok_ratio']:.1f}x fewer" if r["tok_ratio"] else "-",
                transform=trans, fontsize=7.2, va="center", ha="left", color=body)
        label = r["verdict"] + (f"  ({r['ineligible']})" if gated else "")
        ax.text(COL_VERD, y, label, transform=trans, fontsize=7.2, va="center",
                ha="left", color=VERDICT_COLOR[r["verdict"]],
                fontweight="bold" if r["verdict"] == VERDICT_CONSISTENT else "normal")

    graded = [r for r in prem if r["verdict"] != VERDICT_NOT_LICENSED]
    wins = sum(1 for r in graded if r["verdict"] == VERDICT_CONSISTENT)
    pos = sum(1 for r in graded if r["verdict"] != VERDICT_CONSISTENT and r["mean"] > 0)
    flat, gated_n = len(graded) - wins - pos, len(prem) - len(graded)
    fig.suptitle(f"Does a compact derived view beat the full-information "
                 f"{CEILING} ceiling?", fontsize=13.5, fontweight="bold",
                 x=0.035, ha="left", y=1 - 0.26 / H)
    # A directory where nothing is gradeable is a result, not an empty figure, and
    # "0 of 0 cells reach a consistent advantage" states it as though the derived
    # views had been tried and lost. They were never comparable: the ceiling is the
    # member that failed.
    if graded:
        caption = (f"{wins} of {len(graded)} graded host x question-type cells reach a "
                   f"CONSISTENT ADVANTAGE over {CEILING} under the declared rule.\n"
                   f"{pos} more have a positive mean the rule does not license as a "
                   f"win; {flat} {'is' if flat == 1 else 'are'} negative or flat; "
                   f"{gated_n} {'is' if gated_n == 1 else 'are'} drawn with scene "
                   f"deltas but {VERDICT_NOT_LICENSED} (the gate is on the row).")
    else:
        caption = (f"No cell here can be graded against {CEILING}: the ceiling itself "
                   f"fails the coverage gate on every one of the {gated_n} cells "
                   f"below.\nThe derived views are drawn with their scene deltas, and "
                   f"the gate that withheld each grade is printed on its row. This is "
                   f"a fact about the ceiling, not about the views beside it.")
    fig.text(0.035, 1 - 0.60 / H, caption,
             fontsize=9.2, color="#4a5058", va="top", ha="left")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               markeredgecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               label=f"consistent advantage (|mean| >= {PRACTICAL_MARGIN:.2f}, no "
                     f"scene reversed, >= {MIN_SCENES_SHOWING} scenes showing)"),
        Line2D([], [], marker="D", linestyle="none", markersize=6,
               markerfacecolor="white", markeredgecolor=VERDICT_COLOR[VERDICT_DIRECTIONAL],
               label="directional / mixed / no separation - NOT a win, whatever the sign"),
        Line2D([], [], marker="D", linestyle="none", markersize=6,
               markerfacecolor="white", markeredgecolor=VERDICT_COLOR[VERDICT_NOT_LICENSED],
               label=f"{VERDICT_NOT_LICENSED} - deltas shown, grade withheld"),
        Line2D([], [], marker="|", linestyle="none", markersize=9, color=SCENE_MARK,
               label="individual scene delta (the unit of replication)"),
        Patch(facecolor=BAND, label=f"+-{PRACTICAL_MARGIN:.2f} practical margin"),
    ]
    # Anchored a fixed 0.95in below the axes, clear of the two-line x label.
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.95 / axes_h), fontsize=7.6, frameon=False, ncol=2)

    held = "; ".join(f"{k} ({v})" for k, v in CEILING_PREMIUM_EXCLUDED.items())
    fig.text(0.035, 0.12 / H,
             "The winning view is selected POST-HOC per cell from the k in-scope "
             "candidates - an oracle choice on the same data, not a pre-declared "
             "representation.\n"
             "'x fewer' is a PROMPT-TOKEN ratio (input only; excludes completions; no "
             "monetary cost is computed anywhere in this study).\n"
             f"Held out of the candidate pool: {held}.\n"
             f"responder: {responder}  |  judge: {judge}",
             fontsize=7, color="#8a9099", va="bottom", ha="left")

    # Explicit margins rather than tight_layout: the annotation columns live outside
    # the axes box, where tight_layout cannot see them and fights the reserved space.
    fig.subplots_adjust(left=0.145, right=0.545,
                        top=1 - TOP_IN / H, bottom=BOTTOM_IN / H)
    fig.savefig(out_dir / "ceiling_premium.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'ceiling_premium.png'}")
    return {"ceiling_premium.png"}


_VERDICT_ORDER = {VERDICT_CONSISTENT: 0, VERDICT_DIRECTIONAL: 1, VERDICT_MIXED: 2,
                  VERDICT_NO_SEPARATION: 3, VERDICT_NOT_LICENSED: 4}
# The right-hand annotation is ONE text object per row, not two columns. Two columns
# at fixed axes-fraction offsets overprinted each other: tight_layout sizes the axes
# box around the longest y-label, so the inches a fixed fraction buys change with the
# panel, and a gap that clears at one host's label width does not at another's.
# Concatenating makes the collision impossible instead of tuning it away.
_COL_NOTE, _SEPARATION_RECT_RIGHT = 1.005, 0.82


def _plot_paired_separation(prs: list, out_dir: Path,
                            responder: str, judge: str) -> set[str]:
    """paired_separation.png -- report.md's separation table, drawn.

    One row per `report.Paired`, faceted by host, and within a host split into two
    BLOCKS: the graded comparisons, then the `not licensed` ones on a shaded ground
    below a rule. The split is the point of the figure on 3RScan, where json_mini
    overflows a dense scene: every `json_mini - X` row there is gated on coverage
    while the relations_* rows beside them are at full coverage on all three scenes.
    Sorted together they interleave, and a reader scanning marks would take a gated
    row for a graded one. Separated, the licensed axis reads as the licensed axis.

    Gated rows keep their scene deltas and their mean marker -- they are the evidence
    the gate exists to qualify, not to hide -- and they are excluded from the graded
    counts in every heading.

    Nothing is recomputed here: the Paired objects arrive fully decided, and the
    figure only chooses positions and colours.
    """
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

    # Height in inches, and the title/legend/footnote bands reserved in inches too --
    # as fractions they hold at 63 rows and collapse at 7, where the same fraction is
    # a tenth of an inch and the footnote lands on the x label.
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
                cursor -= 1                     # the blank divider slot
            ys.append(cursor)
            cursor -= 1

        lim = max(0.55, max(abs(v) for p in ordered for v in p.scene_deltas.values()) + 0.06)
        if gated:   # shade the gated block before anything is drawn on it
            ax.axhspan(min(ys[len(graded):]) - 0.5, max(ys[len(graded):]) + 0.5,
                       color=GATED_BG, zorder=0)
        ax.axvspan(-PRACTICAL_MARGIN, PRACTICAL_MARGIN, color=BAND, zorder=1)
        ax.axvline(0, color="#333", linewidth=0.9, zorder=2)

        labels, prev_axis = [], None
        for i, (p, y) in enumerate(zip(ordered, ys)):
            # Reset at the block boundary so the gated block labels its own axes
            # rather than inheriting the last label from the graded block above it.
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
            ax.plot([p.mean], [y], marker="o" if win else "D",
                    markersize=7 if win else 5.5,
                    markerfacecolor=c if win else "white", markeredgecolor=c,
                    markeredgewidth=1.5, zorder=5, linestyle="none")
            # Support first, then why the row is qualified. `n sc` is the scene count:
            # the deltas are a mean over THOSE scenes, and a gated row typically has
            # fewer of them than the graded rows above it -- which is the whole story
            # on 3RScan, so it is on the row rather than in the caption.
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
            # The row label above is written `rep_b - rep_a`, so a positive mean is
            # rep_b's -- the name printed FIRST. Saying "second-named" inverts every
            # sign in the figure for a reader who takes the axis at its word.
            ax.set_xlabel("Mean of the scene-level AC deltas   (rep_b - rep_a; > 0 = "
                          "the representation named first in the row scored higher)",
                          fontsize=9)

    fig.suptitle("Every comparison report.md makes, decided the same way",
                 fontsize=13, fontweight="bold", x=0.05, ha="left", y=1 - 0.22 / H)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               markeredgecolor=VERDICT_COLOR[VERDICT_CONSISTENT], label=VERDICT_CONSISTENT),
    ] + [
        Line2D([], [], marker="D", linestyle="none", markersize=5.5,
               markerfacecolor="white", markeredgecolor=VERDICT_COLOR[v], label=v)
        for v in (VERDICT_DIRECTIONAL, VERDICT_MIXED, VERDICT_NO_SEPARATION,
                  VERDICT_NOT_LICENSED)
    ] + [
        Line2D([], [], marker="|", linestyle="none", markersize=8, color=SCENE_MARK,
               label="individual scene delta"),
        Patch(facecolor=BAND, label=f"+-{PRACTICAL_MARGIN:.2f} practical margin"),
        Patch(facecolor=GATED_BG, label=f"{VERDICT_NOT_LICENSED} block"),
    ]
    # On the FIGURE, not on the first axes: anchored to a panel it drifts with that
    # panel's height, which varies with how many comparisons the first host has.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1 - 0.42 / H),
               fontsize=7.4, frameon=False, ncol=4)
    fig.text(0.05, 0.14 / H,
             "No confidence intervals: the scene ticks are the entire replication this "
             "design carries, and overlap is never a tie rule (thesis 4.6).\n"
             "CAPPED = a declared confound downgraded an otherwise-consistent result; "
             "it never promotes and never rescues.\n"
             f"responder: {responder}  |  judge: {judge}",
             fontsize=7, color="#8a9099", va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.70 / H, _SEPARATION_RECT_RIGHT, 1 - 1.00 / H))
    fig.savefig(out_dir / "paired_separation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'paired_separation.png'}")
    return {"paired_separation.png"}


def plot_aggregate(aggregate_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = aggregate_path.parent
    results_path = out_dir / "results.csv"
    if not results_path.exists():
        return
    # The raw rows, errors included: coverage is n_scored/n, so the rows that failed
    # are the denominator. Every mean below still comes from _per_question_ac, which
    # drops them.
    with results_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    points, types, reps = _per_question_ac(results_path)
    if not types:
        return
    color = _colors(reps)
    rng = np.random.default_rng(42)

    # -- Chart 0: axis cards (one figure per axis, ladder order, host dataset) --
    # The headline reporting view: each axis told as its own ladder. The per-type
    # head-to-head below covers the content/general types no axis ladder probes.
    written_cards = _plot_axis_cards(results_path, out_dir, color)

    # -- Charts 1-3: per-host head-to-head, floor lift, and the AC matrix --
    # One figure per host each. In a single-host directory (a per-scene report, or
    # _aggregate/procthor) that is one figure; in _aggregate/all it is three, and
    # they are three separate readings rather than one averaged one.
    written: set[str] = set()
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        written |= _plot_ac_by_type(results_path, rows, ds, out_dir, color)
        written |= _plot_floor_lift(rows, ds, out_dir, color)
        written |= _plot_heatmap(results_path, rows, ds, out_dir)

    # -- Chart 4: axis-contrast paired deltas (every AXES entry declaring a
    #    headline_pair: format, reference frame, structure presentation, relation
    #    linearization, and the three ProcTHOR framing exhibits) --
    # For each axis pair, the per-question AC delta (second pole minus first), over
    # the questions that axis DECLARES -- its host and its probe types, both poles in
    # scope (axis_pair_questions). Bar = mean, whisker = ±1 population std, dots =
    # per question. >0 means the second pole scored higher.
    #
    # The host and type filters are the comparison's definition, not a tidy-up: a
    # Gibson-hosted pair must not average over ProcTHOR questions, and two axes that
    # share a pair and differ only in declared type are otherwise one number drawn
    # twice. Leakage is asserted against, in tests/test_axis_contrasts_scope.py.
    qid_ac, qtype_of, host_of = _per_qid_ac(results_path)
    bars: list[tuple[str, list[float]]] = []
    for pair in AXIS_PAIRS:
        deltas = [qid_ac[(qid, pair.b)] - qid_ac[(qid, pair.a)]
                  for qid in axis_pair_questions(pair, qtype_of, host_of)
                  if (qid, pair.a) in qid_ac and (qid, pair.b) in qid_ac]
        if deltas:
            bars.append((f"{pair.label} [{pair.host}]: {pair.a} -> {pair.b}", deltas))
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
        ax.set_title("Axis contrasts  (paired per-question AC delta; each axis's own "
                     "host [bracketed] and declared question types)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "axis_contrasts.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"plot -> {out_dir / 'axis_contrasts.png'}")

    # -- Charts 5-6: the two graded figures --
    # Drawn last because they depend on nothing above them and on all of report.py:
    # `ceiling_premium` grades the headline claim (a compact derived view against the
    # json_mini ceiling) and `paired_separation` draws the separation table itself.
    # Both carry the responder and judge in-figure -- a verdict is a verdict OF a
    # responder graded BY a judge, and these two figures travel out of their directory.
    responder = ", ".join(sorted({r.get("responder") or "" for r in rows} - {""})) or "?"
    judge = ", ".join(sorted({r.get("judge") or "" for r in rows} - {""})) or "?"
    graded_figs = _plot_ceiling_premium(ceiling_premium_rows(rows), out_dir,
                                        responder, judge)
    graded_figs |= _plot_paired_separation(_paired_rows(rows, pairs=_load_pairs()),
                                           out_dir, responder, judge)

    # retire charts that the scope-aware set replaces / that mislead, plus the old
    # single combined axis card and any per-axis card whose axis lost its data.
    stale = ["ac_comparison.png", "faithfulness_comparison.png", "ac_delta.png",
             "ac_lift_over_floor.png", "axis_cards.png",
             # The host-pooled predecessors of the three per-host charts above
             # (2026-08-14). Deleted rather than left beside them: they are the same
             # chart under a shorter name, and the shorter name is the one a stale
             # reference would still resolve to.
             "ac_by_axis.png", "value_of_spatial_structure.png", "ac_heatmap.png"]
    stale += [p.name for p in out_dir.glob("axis_card_*.png") if p.name not in written_cards]
    # ...and any per-host figure whose host lost its data since the last run.
    for pattern in ("ac_by_axis_*.png", "value_of_spatial_structure_*.png",
                    "ac_heatmap_*.png"):
        stale += [p.name for p in out_dir.glob(pattern) if p.name not in written]
    # ...and either graded figure that this run had nothing to draw. A stale verdict
    # figure is worse than a missing one: it survives under the correct filename and
    # is read as current.
    stale += [n for n in ("ceiling_premium.png", "paired_separation.png")
              if n not in graded_figs]
    for name in stale:
        p = out_dir / name
        if p.exists():
            p.unlink()


# --- per-observation charts (guess detector, latency) ----------------------

def plot_per_question(results_path: Path, diagnostics: bool = False) -> None:
    """Charts read per observation rather than per cell.

    `diagnostics` adds the operational latency box plot. Off by default: latency is
    an operational diagnostic, not a result, and a figure that ships with every run
    is a figure that ends up quoted. See the module docstring.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in csv.DictReader(results_path.open(encoding="utf-8")) if not r["error"]]
    out_dir = results_path.parent
    representations = sorted({r["representation"] for r in rows})
    colors = _colors(representations)
    rng = np.random.default_rng(42)

    # -- Cost vs Quality: accuracy per prompt token (the operational trade-off) --
    # One PANEL PER QUESTION TYPE, and one figure per host dataset. The y-axis must
    # never pool AC across question types: that is the `ALL` row the reporting rules
    # forbid quoting (thesis 4.6), and it averages across exactly the boundaries the
    # axis design exists to hold apart -- a single pooled point per rep silently
    # ranks a rep that is strong on connectivity against one strong on set_logic.
    # The x-axis IS legitimately pooled: the representation is the same serialized
    # file whatever question is asked of it, so its token cost does not vary by type.
    # Within a panel: the dashed line is the efficiency frontier (no rep is both
    # cheaper and more accurate than a point on it) -- where the derived-view-vs-
    # ceiling question is read: same AC, far fewer tokens = the derived view wins.
    #
    # Coverage comes from report.py's own cell statistics, per (rep, question type)
    # and per host -- the same number report.md gates on, so a point flagged here and
    # a row flagged there are the same claim. It used to be computed locally as one
    # rate per rep over ALL that host's rows, which is a different denominator: a rep
    # that overflows only on the object-relation types read as ~90% covered
    # everywhere, diluting exactly the panel where it fails. Below MIN_COVERAGE the
    # marker shrinks, gains a red edge, is labelled `not licensed`, and is excluded
    # from the efficiency frontier -- the AC alone would read as a clean point.
    all_rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    for stale_name in ("cost_quality.png",):   # pooled-AC predecessor of this chart
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()

    datasets = sorted({dataset_of(r["scene_id"]) for r in rows})
    written_cq: set[str] = set()
    for ds in datasets:
        ds_rows = [r for r in rows if dataset_of(r["scene_id"]) == ds]
        # tokens: pooled per rep within the host (same file for every question)
        by_rep_tok: dict[str, list[float]] = collections.defaultdict(list)
        # AC: kept separate per question type -- never pooled
        by_qt_rep_ac: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for r in ds_rows:
            rep = r["representation"]
            qt = r["question_type"] or "unknown"
            if not _in_scope(rep, qt):
                continue
            if r["answer_correctness"] != "":
                by_qt_rep_ac[qt][rep].append(float(r["answer_correctness"]))
            tok = r.get("prompt_tokens", "")
            if tok not in ("", "0", None):
                try:
                    by_rep_tok[rep].append(float(tok))
                except ValueError:
                    pass

        cells = _cells([r for r in all_rows if dataset_of(r["scene_id"]) == ds])

        qts = sorted(qt for qt in by_qt_rep_ac
                     if any(by_rep_tok.get(rep) for rep in by_qt_rep_ac[qt]))
        if not qts:
            continue   # no token data recorded (some backends omit it)

        ncols = min(3, len(qts))
        nrows = (len(qts) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                                 squeeze=False)
        for idx, qt in enumerate(qts):
            ax = axes[idx // ncols][idx % ncols]
            reps = [rep for rep in by_qt_rep_ac[qt] if by_rep_tok.get(rep)]
            xy = {rep: (statistics.mean(by_rep_tok[rep]),
                        statistics.mean(by_qt_rep_ac[qt][rep])) for rep in reps}
            lic = {rep: _license(cells, rep, qt) for rep in reps}
            frontier = efficiency_frontier(xy, {r for r in reps if lic[r][0]})
            if len(frontier) > 1:
                ax.plot([xy[r][0] for r in frontier], [xy[r][1] for r in frontier],
                        color="gray", linestyle="--", linewidth=1, zorder=2,
                        label="efficiency frontier (licensed points only)")
            for rep in reps:
                x, y = xy[rep]
                ok, c = lic[rep]
                ax.scatter(x, y, color=colors.get(rep, "gray"), s=40 + 90 * c,
                           edgecolor="red" if not ok else "black",
                           linewidth=1.4 if not ok else 0.5, zorder=3)
                tag = (rep if c > 0.999
                       else f"{rep} (cov {c * 100:.0f}%"
                            + (f"; {VERDICT_NOT_LICENSED})" if not ok else ")"))
                ax.annotate(tag, (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=7,
                            color="red" if not ok else "black")
            ax.set_title(qt, fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(linestyle="--", alpha=0.4)
            if len(frontier) > 1:
                ax.legend(fontsize=7)
        for idx in range(len(qts), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.supxlabel("Mean prompt tokens  (context size -> serialization overhead; "
                      "same file for every question type)", fontsize=9)
        fig.supylabel("Mean answer correctness  (in-scope, surviving cells)", fontsize=9)
        fig.suptitle(f"Cost vs Quality by question type - host: {ds}", fontsize=12)
        fig.text(0.01, 0.005,
                 "upper-left = more accuracy per token  |  marker shrinks + red edge as "
                 f"coverage drops; below {MIN_COVERAGE:.0%} the point is "
                 f"{VERDICT_NOT_LICENSED} and cannot join the frontier  |  AC is never "
                 "pooled across question types or hosts",
                 ha="left", va="bottom", fontsize=7, color="gray")
        fig.tight_layout(rect=(0.01, 0.02, 1, 0.97))
        name = f"cost_quality_{ds}.png"
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        written_cq.add(name)
        print(f"plot -> {out_dir / name}")

    # drop per-host charts whose host lost its data since the last run
    for p in out_dir.glob("cost_quality_*.png"):
        if p.name not in written_cq:
            p.unlink()

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

    # -- Latency per representation (OPERATIONAL DIAGNOSTIC, not a result) --
    # Off unless asked for, and renamed away from `latency_comparison`: the
    # comparison it appeared to license is not supported. Two representations'
    # boxes differ by batch composition, GPU contention and whether the model was
    # resident at call time as much as by anything the representation does -- a
    # contended GPU falls back to CPU silently, so a box can move an order of
    # magnitude with the serialization held constant. Splitting per host would fix
    # only the shallowest of those confounds while making the chart look repaired,
    # so it stays pooled and stays labelled. Prompt tokens are the cost axis
    # (cost_quality_<host>.png); this is here to spot a run that went wrong.
    #
    # The previous production filename is swept whether or not diagnostics are on:
    # it exists in every results directory written before this rule.
    for stale_name in ("latency_comparison.png",):
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()
    if not diagnostics:
        stale = out_dir / "latency_diagnostic.png"
        if stale.exists():
            stale.unlink()
        return

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
    ax.set_ylabel("Wall-clock response latency (s)")
    ax.set_title("Response latency — OPERATIONAL DIAGNOSTIC, not a comparison\n"
                 "(box = IQR, whiskers = 1.5*IQR, dots = outliers)")
    ax.text(0.0, -0.13, LATENCY_DIAGNOSTIC_NOTE, transform=ax.transAxes,
            fontsize=7, color="tab:red", va="top")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles = [Patch(facecolor=colors.get(r, "gray"), edgecolor="black", label=r) for r in drawn]
    ax.legend(handles=handles, ncol=1, fontsize=8, loc="center left",
              bbox_to_anchor=(1.005, 0.5), frameon=False)
    fig.savefig(out_dir / "latency_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out_dir / 'latency_diagnostic.png'}  (diagnostic; not a production figure)")
