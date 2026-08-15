"""thesis_figures.py — page-sized renders of the graded figures, for the thesis body.

This module exists because a figure that is *correct* and a figure that is *readable
at 418pt* are two different artifacts, and the production set in `plots.py` is the
first kind. `ceiling_premium.png` is 13.2in wide and `paired_separation.png` is
13.6 x 29.6in; dropped into a `\\textwidth` float they scale by 0.44 and 0.43, which
turns their 8.5pt and 7.2pt labels into 3.7pt and 3.1pt. Nothing here shrinks those
files. They stay exactly as they are, as the audit artifacts a reader checks the
thesis against, and this module draws SEPARATE figures at final page dimensions with
native type.

Three renders and three LaTeX tables, all written from the same rows the production
figures use — `plots.ceiling_premium_rows` and `report._paired_rows` — so a mark
here, a mark there and a row in report.md are one computation:

  fig_ceiling_premium   F1. 19 host x question-type cells, best observed derived view
                        against the json_mini ceiling. The production figure's
                        right-hand annotation columns (best view / of k, token ratio,
                        verdict) are NOT drawn: they are a table rendered as pixels,
                        and they are what makes that figure 13in wide. They move to
                        tab_ceiling_premium.tex as real LaTeX text.
  fig_paired_separation F2. The 19 WITHIN-representation contrasts. Not "every
                        comparison": every (rung, CEILING) anchor belongs to F1 by the
                        same rule `Axis.confound_for` already applies -- an anchor
                        comparison is not the design decision an axis isolates -- so
                        drawing them again here would be F1 twice. Split into the
                        declared axis contrasts and the exhibits, because thesis 6.1.2
                        says an exhibit is never read as an axis contrast and a shared
                        panel would invite exactly that.
  fig_cost_quality      The 14 GRADEABLE F1 cells as a scatter: token reduction
                        against the paired AC delta. One point per host x question
                        type; nothing is pooled across question types, and the five
                        3RScan cells the ceiling's own coverage gate withheld are not
                        plotted, because a cell with no licensed delta has no y.

  tab_ceiling_premium   F1's detail columns.
  tab_comparison_catalogue  ALL 63 rows of the separation table, landscape. The
                        exhaustive catalogue F2 no longer carries.
  tab_cost_quality      The scatter's 14 points as numbers. Appendix, not chapter 7:
                        beside the figure it is the same 14 cells twice on one page.

Fonts are set so nothing renders below 7pt at final size, and no figure carries a
title, caption or footnote: those are `\\caption` in LaTeX, where they reflow, get a
number, and are searchable. Everything a caption must say is returned by
`caption_facts()` rather than typed twice.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from evaluation.axes import (
    AXES, CEILING, MIN_COVERAGE, MIN_SCENES_SHOWING, PRACTICAL_MARGIN,
    VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
    VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED, dataset_of,
)
from evaluation.report import _load_pairs, _paired_rows
from evaluation.plots import (
    BAND, CEILING_PREMIUM_EXCLUDED, SCENE_MARK, VERDICT_COLOR, ceiling_premium_rows,
)

# --- page geometry ----------------------------------------------------------
# content/setup.tex: \areaset[1.5cm]{418pt}{658pt}. A LaTeX pt is 1/72.27in, so the
# text block is 5.784 x 9.104in and a \textwidth figure is drawn at 1:1 -- which is
# the entire point of this module. Landscape (pdflscape) swaps them.
PT = 1 / 72.27
TEXTWIDTH_IN = 418 * PT
TEXTHEIGHT_IN = 658 * PT

# The thesis's own host order (tab:meth-datasets, and the order chapter 6 reads them
# in), not alphabetical. Alphabetical puts 3RScan -- the host where every ceiling row
# is gated -- at the top of F1, so the figure opens on its five withheld grades.
HOST_ORDER = ["procthor", "gibson", "3rscan"]
HOST_LABEL = {"procthor": "ProcTHOR", "gibson": "Gibson", "3rscan": "3RScan"}

_AXIS = {a.id: a for a in AXES}
_AXIS_POS = {a.id: i for i, a in enumerate(AXES)}

# Short verdict codes for the LaTeX tables. `no practically meaningful separation` is
# 36 characters and there is no column in a 418pt text block that holds it; the key is
# printed in each table's own notes rather than left to the reader.
VERDICT_SHORT = {
    VERDICT_CONSISTENT:    "CA",
    VERDICT_DIRECTIONAL:   "dir",
    VERDICT_MIXED:         "mixed",
    # `no sep.`, not `null`: the chapter says "no practically meaningful separation"
    # everywhere and never "null", and the two are not synonyms -- a null result is
    # read as evidence of no effect, which is precisely the reading the decision rule
    # of thesis 4.6 refuses to license from three scenes.
    VERDICT_NO_SEPARATION: "no sep.",
    VERDICT_NOT_LICENSED:  "n.l.",
}

MINUS = "−"          # U+2212, present in DejaVu Sans/Sans Mono


def _rc():
    """Font sizes fixed so the smallest glyph in any output is 7pt AT FINAL SIZE.

    Set once, here, rather than per-figure: the guarantee is a property of the whole
    output set, and a per-figure default is a per-figure regression waiting to happen.
    `pdf.fonttype = 42` embeds TrueType rather than rasterising, so the labels stay
    selectable text in the compiled thesis.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0.0,
    })


def load_rows(results_path: Path) -> list[dict]:
    with results_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- row selection ----------------------------------------------------------

def f1_rows(rows: list[dict]) -> list[dict]:
    """The ceiling-anchored cells, in thesis host order."""
    prem = ceiling_premium_rows(rows)
    return sorted(prem, key=lambda r: (HOST_ORDER.index(r["host"]), r["qt"]))


def f2_rows(rows: list[dict]) -> list:
    """The within-representation contrasts: `_paired_rows` minus the ceiling anchors.

    The filter is the whole selection rule, and it is one line on purpose. Any list
    curated by which comparisons the prose happens to cite is a list a reader cannot
    check and an edit to chapter 6 can silently falsify; this one they can restate
    ("every contrast the registry generates that is not against the ceiling") and
    verify against the appendix catalogue.
    """
    prs = _paired_rows(rows, pairs=_load_pairs())
    keep = [p for p in prs if CEILING not in (p.rep_a, p.rep_b)]
    return sorted(keep, key=_f2_key)


def _f2_key(p):
    ax = _AXIS[p.axis_id]
    qi = ax.probe_types.index(p.qt) if p.qt in ax.probe_types else 99
    la = ax.ladder.index(p.rep_a) if p.rep_a in ax.ladder else 99
    lb = ax.ladder.index(p.rep_b) if p.rep_b in ax.ladder else 99
    return (_AXIS_POS[p.axis_id], qi, la, lb)


def catalogue_rows(rows: list[dict]) -> list:
    """Every row of the separation table, for the appendix. Nothing is filtered."""
    prs = _paired_rows(rows, pairs=_load_pairs())
    return sorted(prs, key=lambda p: (HOST_ORDER.index(p.host), _f2_key(p)))


def _strip(rep: str, axis_id: str) -> str:
    """`relations_digest` -> `digest` inside the relation-linearization group only.

    Five reps sharing an 11-character prefix cost ~0.65in of a 5.78in figure to say
    the same word five times. The prefix is restored in the group header, so the
    figure never drops a name it does not also print.
    """
    if axis_id == "relation_linearization" and rep.startswith("relations_"):
        return rep[len("relations_"):]
    return rep


# --- F1: the ceiling premium ------------------------------------------------

_F1_LEFT, _F1_RIGHT = 0.215, 0.995

# Verdict order for a legend, so both figures key them in the same sequence.
_KEY_ORDER = (VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
              VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED)


def _verdict_key(present, win_sz: float, other_sz: float, tick_sz: float) -> list:
    """The legend, keyed to the verdicts THIS figure actually draws.

    In the production figures a verdict COLUMN spelled each row's grade out in words
    beside the marker; those columns are now LaTeX tables, which leaves marker colour
    as the only in-figure cue and makes an unkeyed colour a distinction the reader can
    see but not decode. Keying a verdict that does not appear is the opposite failure:
    `not licensed` (#b0b6bf) and `no practically meaningful separation` (#6b7280) are
    two greys, and listing the absent one sends a reader hunting for a mark that is
    not in the figure.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    seen = set(present)
    out = []
    for v in _KEY_ORDER:
        if v not in seen:
            continue
        win = v == VERDICT_CONSISTENT
        out.append(Line2D([], [], marker="o" if win else "D", linestyle="none",
                          markersize=win_sz if win else other_sz,
                          markerfacecolor=VERDICT_COLOR[v] if win else "white",
                          markeredgecolor=VERDICT_COLOR[v],
                          label=v + (" (grade withheld)"
                                     if v == VERDICT_NOT_LICENSED else "")))
    out.append(Line2D([], [], marker="|", linestyle="none", markersize=tick_sz,
                      color=SCENE_MARK, label="individual scene delta"))
    out.append(Patch(facecolor=BAND,
                     label=f"±{PRACTICAL_MARGIN:.2f} practical margin"))
    return out


def render_f1(prem: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n = len(prem)
    ROW = 0.175                       # inches per row
    HDR = 0.14                        # extra for each host rule
    LEGEND_IN, XLABEL_IN = 0.82, 0.42
    body = ROW * n + HDR * len(HOST_ORDER)
    H = body + LEGEND_IN + XLABEL_IN + 0.10
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, H))

    ax.axvspan(-PRACTICAL_MARGIN, PRACTICAL_MARGIN, color=BAND, zorder=0)
    ax.axvline(0, color="#333", linewidth=0.8, zorder=1)
    lo = min([v for r in prem for v in r["scene_deltas"].values()] + [0.0])
    hi = max([v for r in prem for v in r["scene_deltas"].values()] + [0.0])
    ax.set_xlim(min(-0.30, lo - 0.06), max(0.55, hi + 0.06))

    labels, prev = [], None
    for i, r in enumerate(prem):
        y = n - 1 - i
        if r["host"] != prev:
            ax.axhline(y + 0.5, color="#c9ccd1", linewidth=0.7, zorder=2)
            ax.text(0.012, y + 0.42, HOST_LABEL[r["host"]],
                    transform=ax.get_yaxis_transform(), fontsize=7.5,
                    fontweight="bold", color="#4a5058", va="top", zorder=6)
            prev = r["host"]
        labels.append(r["qt"].replace("_", " "))
        gated = r["verdict"] == VERDICT_NOT_LICENSED
        c = VERDICT_COLOR[r["verdict"]]
        for v in r["scene_deltas"].values():
            ax.plot([v], [y], marker="|", markersize=6.5, color=SCENE_MARK,
                    alpha=0.35 if gated else 0.75, zorder=3, linestyle="none")
        win = r["verdict"] == VERDICT_CONSISTENT
        ax.plot([r["mean"]], [y], marker="o" if win else "D",
                markersize=5.5 if win else 4.2,
                markerfacecolor=c if win else "white", markeredgecolor=c,
                markeredgewidth=1.1, zorder=4, linestyle="none")

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)), fontsize=7.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(f"Scene-paired AC delta:  best observed derived view "
                  f"{MINUS} {CEILING} ceiling", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.legend(handles=_verdict_key([r["verdict"] for r in prem], 5.5, 4.2, 6.5),
              loc="upper left", ncol=2, fontsize=7, frameon=False,
              handletextpad=0.5, columnspacing=1.2,
              bbox_to_anchor=((0.020 - _F1_LEFT) / (_F1_RIGHT - _F1_LEFT),
                              -(XLABEL_IN + 0.10) / body))
    fig.subplots_adjust(left=_F1_LEFT, right=_F1_RIGHT,
                        top=1 - 0.04 / H, bottom=(LEGEND_IN + XLABEL_IN) / H)
    _save(fig, out)


# --- F2: the within-representation contrasts --------------------------------

# F2's axes box. `_GUTTER` is where a panel title or group header starts, in
# axes-fraction x -- negative, i.e. left of the box, running across the y-label
# gutter. Derived from the margins rather than tuned beside them, because the two
# drifting apart is exactly how the longest header ran off the right of the page.
_F2_LEFT, _F2_RIGHT = 0.400, 0.900
_GUTTER = (0.020 - _F2_LEFT) / (_F2_RIGHT - _F2_LEFT)


def render_f2(prs: list, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    axis_rows = [p for p in prs if _AXIS[p.axis_id].kind != "exhibit"]
    exhibit_rows = [p for p in prs if _AXIS[p.axis_id].kind == "exhibit"]
    # (axis_id, [(question type, [rows])]). Two levels rather than one, because the
    # alternative is a `question type: rep_b - rep_a` label on every row, and
    # `relation aggregate: relations_predicate - relations_subject` is 54 characters:
    # at 7pt it needs 3.2 of the figure's 5.78in for the labels alone. Hoisting the
    # repeated parts into headers is what buys the row label enough width to be set at
    # a readable size instead of shrunk to fit.
    groups: list[tuple[str, list[tuple[str, list]]]] = []
    for p in axis_rows:
        if not groups or groups[-1][0] != p.axis_id:
            groups.append((p.axis_id, []))
        sub = groups[-1][1]
        if not sub or sub[-1][0] != p.qt:
            sub.append((p.qt, []))
        sub[-1][1].append(p)
    # A one-question-type axis folds its type into the axis header; only an axis read
    # on several types (relation linearization, on three) spends a line per type.
    n_sub = sum(len(s) for _, s in groups if len(s) > 1)

    ROW, HDR = 0.165, 0.155
    top_slots = len(axis_rows) * ROW + (len(groups) + n_sub) * HDR
    bot_slots = len(exhibit_rows) * ROW
    TITLE_IN, LEGEND_IN, XLABEL_IN = 0.20, 0.56, 0.42
    H = top_slots + bot_slots + 2 * TITLE_IN + LEGEND_IN + XLABEL_IN + 0.26
    fig, axes = plt.subplots(
        2, 1, figsize=(TEXTWIDTH_IN, H),
        gridspec_kw=dict(height_ratios=[top_slots + TITLE_IN,
                                        bot_slots + TITLE_IN]))

    lim = max(0.45, max(abs(v) for p in prs for v in p.scene_deltas.values()) + 0.05)

    def panel(ax, rows_, title, grouped):
        slots = len(rows_) + (len(groups) + n_sub if grouped else 0)
        ax.axvspan(-PRACTICAL_MARGIN, PRACTICAL_MARGIN, color=BAND, zorder=1)
        ax.axvline(0, color="#333", linewidth=0.8, zorder=2)
        ys, cursor, labels = [], slots - 1, []
        if not grouped:
            for p in rows_:
                ys.append(cursor)
                cursor -= 1
                labels.append(p.qt.replace("_", " "))
        for axis_id, subs in (groups if grouped else []):
            ax_def = _AXIS[axis_id]
            ax.axhline(cursor + 0.5, color="#c9ccd1", linewidth=0.7, zorder=2)
            head = f"{ax_def.label.upper()}   ({HOST_LABEL[ax_def.host]}"
            if len(subs) == 1:
                head += f" · {subs[0][0].replace('_', ' ')}"
            # Headers are set in axes FRACTION, starting left of the axes box, so they
            # run across the y-label gutter as well. A header slot carries no y-label,
            # so there is nothing there to collide with -- and anchored at the axes
            # edge instead, the longest of them ran off the right of the page.
            ax.text(_GUTTER, cursor + 0.40, head + ")",
                    transform=ax.get_yaxis_transform(), fontsize=7,
                    fontweight="bold", color="#4a5058", va="top", zorder=6)
            cursor -= 1
            for qt, members in subs:
                if len(subs) > 1:
                    ax.text(_GUTTER + 0.04, cursor + 0.38, qt.replace("_", " "),
                            transform=ax.get_yaxis_transform(), fontsize=6.8,
                            style="italic", color="#6b7280", va="top", zorder=6)
                    cursor -= 1
                for p in members:
                    ys.append(cursor)
                    cursor -= 1
                    labels.append(f"{_strip(p.rep_b, p.axis_id)} {MINUS} "
                                  f"{_strip(p.rep_a, p.axis_id)}")
        for p, y in zip(rows_, ys):
            c = VERDICT_COLOR[p.verdict]
            for v in p.scene_deltas.values():
                ax.plot([v], [y], marker="|", markersize=6, color=SCENE_MARK,
                        alpha=0.7, zorder=4, linestyle="none")
            win = p.verdict == VERDICT_CONSISTENT
            ax.plot([p.mean], [y], marker="o" if win else "D",
                    markersize=5.2 if win else 4.0,
                    markerfacecolor=c if win else "white", markeredgecolor=c,
                    markeredgewidth=1.1, zorder=5, linestyle="none")
            # The support columns (n_q, fact-sets, scenes) go to the appendix table;
            # CAPPED stays, because it is the only in-figure mark whose absence would
            # let a directional row be read as an ordinary under-powered one rather
            # than as a consistent result a declared confound downgraded.
            if p.capped:
                ax.text(1.008, y, "CAPPED", transform=ax.get_yaxis_transform(),
                        fontsize=6.2, va="center", ha="left", fontweight="bold",
                        color=VERDICT_COLOR[VERDICT_DIRECTIONAL])
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
        ax.set_ylim(-0.7, slots - 0.3)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=4,
                     x=_GUTTER)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        return ys

    panel(axes[0], axis_rows,
          "Declared axis contrasts   (relations_* prefix dropped throughout)", True)
    panel(axes[1], exhibit_rows,
          f"Exhibits and controls (ProcTHOR):  navigation {MINUS} topology_metric\n"
          f"not axis contrasts, and never read as one (thesis 6.1.2)", False)
    axes[1].set_xlabel("Mean of the scene-level AC deltas\n"
                       "> 0 = the representation named first in the row scored higher",
                       fontsize=8)

    handles = _verdict_key([p.verdict for p in prs], 5.2, 4.0, 6.0)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.004),
               fontsize=7, frameon=False, ncol=2, handletextpad=0.5,
               columnspacing=1.2)
    fig.subplots_adjust(left=_F2_LEFT, right=_F2_RIGHT, hspace=0.34,
                        top=1 - TITLE_IN / H, bottom=(LEGEND_IN + XLABEL_IN) / H)
    _save(fig, out)


# --- the cost/quality scatter -----------------------------------------------

HOST_MARK = {"procthor": "o", "gibson": "^", "3rscan": "s"}

# Cosmetic label placement only, keyed by (host, question type). Every point is drawn
# from the data; these move the TEXT off a neighbour it would otherwise overprint.
# Six of the fourteen cells sit inside x in [4.07, 4.14] -- a 2% spread on a log axis
# -- so a uniform offset is not an option and an automatic declutter would move labels
# on data that has not changed. Offsets are in points, (dx, dy).
CQ_NUDGE = {
    # the x ~ 4.1 cluster: six cells inside a 2% spread of each other
    ("gibson", "containment"):   (-7, 0),
    ("procthor", "aggregation"): (7, -1),
    ("gibson", "planning"):      (7, 1),
    ("gibson", "proximity"):     (7, 0),
    ("procthor", "planning"):    (7, 0),
    ("procthor", "set_logic"):   (7, 0),
    # x ~ 8: connectivity and proximity differ by 0.003 in AC and 0.6x in tokens
    ("procthor", "connectivity"): (-7, 0),
    ("procthor", "proximity"):    (7, 0),
    ("procthor", "containment"):  (7, 0),
    # x ~ 14: set logic and aggregation, likewise
    ("gibson", "set_logic"):     (-7, 0),
    ("gibson", "aggregation"):   (7, 0),
}


def render_cost_quality(prem: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pts = [r for r in prem if r["verdict"] != VERDICT_NOT_LICENSED and r["tok_ratio"]]
    H = 3.95
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, H))

    ax.axhspan(0.0, PRACTICAL_MARGIN, color=BAND, zorder=0)
    ax.axhline(0.0, color="#333", linewidth=0.8, zorder=2)
    ax.axhline(PRACTICAL_MARGIN, color="#8a9099", linewidth=0.7,
               linestyle="--", zorder=2)

    for r in pts:
        win = r["verdict"] == VERDICT_CONSISTENT
        c = VERDICT_COLOR[r["verdict"]]
        ax.plot([r["tok_ratio"]], [r["mean"]], marker=HOST_MARK[r["host"]],
                markersize=6.0 if win else 5.0,
                markerfacecolor=c if win else "white", markeredgecolor=c,
                markeredgewidth=1.2, linestyle="none", zorder=5)
        dx, dy = CQ_NUDGE.get((r["host"], r["qt"]), (6, 0))
        ax.annotate(r["qt"].replace("_", " "), (r["tok_ratio"], r["mean"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right", va="center",
                    fontsize=7, color="#3a3f45", zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(3.2, 34)
    ax.set_xticks([4, 5, 6, 8, 10, 15, 20, 25])
    ax.set_xticklabels([f"{v}×" for v in (4, 5, 6, 8, 10, 15, 20, 25)])
    ax.minorticks_off()
    ax.set_ylim(-0.10, 0.95)
    ax.set_xlabel("Prompt-token reduction vs the json_mini ceiling  "
                  "(log scale; input tokens only)", fontsize=8)
    ax.set_ylabel(f"Scene-paired AC delta  (derived view {MINUS} ceiling)", fontsize=8)
    ax.grid(True, which="major", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], marker=HOST_MARK[h], linestyle="none", markersize=5.5,
               markerfacecolor="#4a5058", markeredgecolor="#4a5058",
               label=HOST_LABEL[h]) for h in ("procthor", "gibson")
    ] + [
        Line2D([], [], marker="o", linestyle="none", markersize=6.0,
               markerfacecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               markeredgecolor=VERDICT_COLOR[VERDICT_CONSISTENT],
               label="consistent advantage (filled)"),
        Line2D([], [], marker="o", linestyle="none", markersize=5.0,
               markerfacecolor="white", markeredgecolor=VERDICT_COLOR[VERDICT_MIXED],
               label="graded, not a win (hollow)"),
        Patch(facecolor=BAND, label=f"0 to +{PRACTICAL_MARGIN:.2f}: below the margin"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False,
              ncol=2, handletextpad=0.5, columnspacing=1.2)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.985, bottom=0.135)
    _save(fig, out)


def _save(fig, out: Path) -> None:
    """PDF for the thesis, PNG beside it for eyeballing without a LaTeX run.

    `bbox_inches` is deliberately NOT 'tight': tight cropping changes the output width
    away from 418pt, which silently reintroduces the scale factor this whole module
    exists to remove.
    """
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"figure -> {out.with_suffix('.pdf')}  "
          f"({fig.get_figwidth():.3f} x {fig.get_figheight():.3f} in)")


# --- LaTeX tables -----------------------------------------------------------

def _esc(s: str) -> str:
    """LaTeX-escape free text coming out of the analysis layer.

    The gate strings are the reason this exists: `_gates` composes them from rep names
    and percentages, so `json_mini coverage 67%` carries both an underscore and a
    percent sign -- one of which is a subscript error and the other of which silently
    comments out the rest of the row.
    """
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"),
                 ("&", "\\&"), ("#", "\\#"), ("$", "\\$")):
        s = s.replace(a, b)
    return s


def _tt(s: str) -> str:
    return "\\texttt{" + _esc(s) + "}"


def token_range(ratios: list[float]) -> tuple[str, str]:
    """(ratio phrase, share phrase) for the oracle winners, from the SAME unrounded
    ratios, in the one convention the table and chapter 7 both use.

    The share is deliberately approximate and integer. Carried to a decimal it is
    3.96--24.54%, which invites `4.0--24.5` in one place and `4.0--24.6` in another
    from the same numbers -- a discrepancy that looks like a disagreement about the
    data and is only a disagreement about rounding. There is no question in this study
    that a tenth of a percent of a token budget answers, so the share is stated as
    `about 4--25%` and the precise figure is carried by the ratio beside it.
    """
    lo, hi = min(ratios), max(ratios)
    return (f"${lo:.2f}\\times$--${hi:.2f}\\times$",
            f"about {round(100 / hi)}--{round(100 / lo)}\\,\\% of the ceiling's "
            f"prompt tokens")


def table_f1(prem: list[dict], out: Path, label: str) -> None:
    """F1's right-hand annotation columns, as text.

    The Host column is a spanning subheader rather than a repeated cell: three values
    over 19 rows, and the 1.5cm it would cost is 10% of a 418pt text block.
    """
    L = [
        "% GENERATED by evaluation/thesis_figures.py -- do not edit by hand.",
        "\\begin{table}[tbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\caption[Ceiling premium: the best observed derived view per cell]{%",
        "    Detail for \\cref{fig:ceiling-premium}. For each host $\\times$ "
        "question-type cell in which the \\texttt{json\\_mini} ceiling is itself "
        "scored, the derived view with the largest scene-paired advantage over it. "
        "The winner is selected \\emph{post hoc} from the $k$ in-scope candidates "
        "listed for that cell --- an oracle choice made on the same data, not a "
        "representation declared in advance. $\\bar{d}$ is the mean of the "
        "scene-level AC deltas (view $-$ ceiling). Verdicts: CA $=$ consistent "
        "advantage, dir $=$ directional, no sep.\\ $=$ no practically meaningful "
        "separation, n.l.\\ $=$ not licensed. Token ratios are prompt (input) "
        "tokens only.}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{llrrl}",
        "    \\toprule",
        "    Question type & Best observed view (of $k$) & Tokens & $\\bar{d}$ & "
        "Verdict \\\\",
        "    & & saved & & \\\\",
        "    \\midrule",
    ]
    prev = None
    for r in prem:
        if r["host"] != prev:
            if prev is not None:
                L.append("    \\addlinespace[2pt]")
            L.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{HOST_LABEL[r['host']]}}}}}"
                     " \\\\")
            prev = r["host"]
        verd = VERDICT_SHORT[r["verdict"]]
        if r["verdict"] == VERDICT_NOT_LICENSED:
            verd += f" ({_esc(r['ineligible'].split(';')[0].strip())})"
        L.append(
            f"    \\quad {r['qt'].replace('_', ' ')} & "
            f"{_tt(r['rep'])} ({r['k']}) & "
            f"{r['tok_ratio']:.1f}$\\times$ & "
            f"${r['mean']:+.3f}$ & {verd} \\\\")
    L += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    _write(out, L)


def table_cost_quality(prem: list[dict], out: Path, label: str) -> None:
    """The 14 plotted points, as numbers. A scatter a reader cannot re-read off the
    page is a claim without a source, and these are the study's headline cells."""
    pts = [r for r in prem if r["verdict"] != VERDICT_NOT_LICENSED and r["tok_ratio"]]
    xs = [r["tok_ratio"] for r in pts]
    ys = [r["mean"] for r in pts]
    L = [
        "% GENERATED by evaluation/thesis_figures.py -- do not edit by hand.",
        "\\begin{table}[tbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{5pt}",
        "  \\caption[Cost and quality of the oracle-selected view per cell]{%",
        "    The 14 points of \\cref{fig:cost-quality}, as numbers. One row per "
        "gradeable host $\\times$ question-type cell; the view is the "
        "oracle-selected winner of \\cref{tab:ceiling-premium}. Reduction is the "
        "ratio of the ceiling's mean prompt tokens to the view's. The five 3RScan "
        "cells are absent because the ceiling itself fails the coverage gate there, "
        "so no delta against it is licensed.}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lllrrl}",
        "    \\toprule",
        "    Host & Question type & View & Reduction & $\\bar{d}$ & Verdict \\\\",
        "    \\midrule",
    ]
    for r in sorted(pts, key=lambda r: -r["tok_ratio"]):
        L.append(
            f"    {HOST_LABEL[r['host']]} & {r['qt'].replace('_', ' ')} & "
            f"{_tt(r['rep'])} & {r['tok_ratio']:.2f}$\\times$ & "
            f"${r['mean']:+.3f}$ & {VERDICT_SHORT[r['verdict']]} \\\\")
    ratio_phrase, share_phrase = token_range(xs)
    L += [
        "    \\midrule",
        f"    \\multicolumn{{6}}{{l}}{{\\footnotesize Range: {ratio_phrase} "
        f"({share_phrase}); $\\bar{{d}}$ from ${min(ys):+.3f}$ to "
        f"${max(ys):+.3f}$.}} \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    _write(out, L)


# The catalogue's column header, written once and reused for the first page and the
# continuation pages. Two hand-kept copies of a twelve-column header is two places for
# a column to be renamed in only one of them.
_HEAD = ("{prefix}\\toprule Host & Axis & Question type & A & B & $\\bar{{d}}$ & "
         "range & $n_q$ & fs & sc & Verdict & Flag \\\\ \\midrule}}")


def table_catalogue(prs: list, out: Path, label: str) -> None:
    """All 63 comparisons, landscape, page-breaking.

    `supertabular` rather than `longtable`: longtable is not loaded by
    content/packages.tex and adding a package to make a table fit is a change to
    every float in the document.
    """
    L = [
        "% GENERATED by evaluation/thesis_figures.py -- do not edit by hand.",
        "\\begin{landscape}",
        # `plain` for the span of the table: lscape leaves the running head in the
        # portrait margin, where it is set sideways across the right-hand columns of a
        # table this wide. Restored after \\end{landscape}.
        "\\pagestyle{plain}",
        # Twelve columns of which two carry free text: at \footnotesize with every
        # column set `l`, the natural width is ~850pt against the 658pt lscape gives
        # back, and the overflow lands on the Flag column -- i.e. the gate reasons, the
        # one thing chapter 6 sends a reader here to look up, run off the paper. The
        # two text columns wrap instead of setting on one line.
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\tablecaption{{Complete catalogue of the {len(prs)} paired comparisons "
        "behind \\cref{fig:ceiling-premium,fig:paired-separation}. "
        "Every contrast the axis registry generates on the primary responder "
        "(\\texttt{qwen2.5:14b}, Gemini judge), decided by the rule of "
        "\\cref{sec:meth-analysis}. $\\bar{d}$ is the mean of the scene-level AC "
        "deltas (B $-$ A); \\emph{range} is their minimum and maximum, the "
        "replication the verdict is actually read from. $n_q$ counts paired "
        "questions, \\emph{fs} the distinct information requests behind them "
        "(--- where no fact-set map joins the two sides, which is not a count of "
        "zero), \\emph{sc} the scenes. Verdicts: CA "
        "$=$ consistent advantage, dir $=$ directional, no sep.\\ $=$ no practically "
        "meaningful separation, n.l.\\ $=$ not licensed. CAPPED marks a "
        "comparison a declared confound downgraded from a consistent "
        "advantage.\\label{" + label + "}}",
        # `\\label` lives INSIDE \tablecaption: supertabular emits the caption itself,
        # so a label after \end{supertabular} would step no counter and resolve to
        # whatever float precedes the table.
        _HEAD.format(prefix="\\tablefirsthead{"),
        _HEAD.format(prefix="\\tablehead{\\multicolumn{12}{l}{\\footnotesize\\itshape "
                            "(continued from the previous page)} \\\\ "),
        "\\tabletail{\\midrule \\multicolumn{12}{r}{\\footnotesize\\itshape "
        "(continued on the next page)} \\\\}",
        "\\tablelasttail{\\bottomrule}",
        # Axis and Flag wrap (`p`, ragged so a two-word cell is not stretched); the
        # rest are single-line by construction -- a rep name, a verdict code or a
        # number -- and are left where they are so the columns still align on sight.
        "\\begin{supertabular}{l"
        ">{\\raggedright\\arraybackslash}p{64pt}"      # Axis
        ">{\\raggedright\\arraybackslash}p{46pt}"      # Question type
        "llrrrrrl"                                    # A, B, d, range, nq, fs, sc, verdict
        ">{\\raggedright\\arraybackslash}p{48pt}}",    # Flag
    ]
    for p in prs:
        lo, hi = p.spread
        if p.capped:
            flag = "CAPPED"
        elif p.ineligible:
            flag = _esc(p.ineligible)
        else:
            flag = "---"
        L.append(
            f"{HOST_LABEL[p.host]} & {_esc(_AXIS[p.axis_id].label)} & "
            f"{p.qt.replace('_', ' ')} & {_tt(p.rep_a)} & {_tt(p.rep_b)} & "
            f"${p.mean:+.3f}$ & $[{lo:+.3f},\\,{hi:+.3f}]$ & "
            # `---`, not `0`: n_factsets is 0 when no fact-set map joins the two
            # sides at all, which is a missing mapping and not a comparison drawn over
            # zero information requests. A numeral in that column is a count.
            f"{p.n_questions} & {p.n_factsets or '---'} & {len(p.scene_deltas)} & "
            f"{VERDICT_SHORT[p.verdict]} & {flag} \\\\")
    L += ["\\end{supertabular}", "\\end{landscape}", "\\pagestyle{headings}", ""]
    _write(out, L)


def _write(out: Path, lines: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"table  -> {out}  ({len(lines)} lines)")


# --- the numbers a caption has to state -------------------------------------

def caption_facts(prem: list[dict], f2: list, cat: list) -> str:
    graded = [r for r in prem if r["verdict"] != VERDICT_NOT_LICENSED]
    wins = [r for r in graded if r["verdict"] == VERDICT_CONSISTENT]
    ratios = [r["tok_ratio"] for r in graded if r["tok_ratio"]]
    over = [r for r in graded if r["mean"] >= PRACTICAL_MARGIN]
    ax_n = len([p for p in f2 if _AXIS[p.axis_id].kind != "exhibit"])
    out = [
        "F1  cells                : %d (%d graded, %d not licensed)"
        % (len(prem), len(graded), len(prem) - len(graded)),
        "F1  consistent advantage : %d of %d graded" % (len(wins), len(graded)),
        "F1  clear the margin     : %d of %d graded  (NOT the same set as the wins)"
        % (len(over), len(graded)),
        # Printed in the convention the .tex uses, so a caption is transcribed from
        # here rather than re-rounded by hand into a third variant.
        "F1  token reduction      : %.2fx to %.2fx  (about %d%%--%d%% of the ceiling)"
        % (min(ratios), max(ratios), round(100 / max(ratios)),
           round(100 / min(ratios))),
        "F1  delta range          : %+.3f to %+.3f"
        % (min(r["mean"] for r in graded), max(r["mean"] for r in graded)),
        "F2  contrasts            : %d (%d axis, %d exhibit)"
        % (len(f2), ax_n, len(f2) - ax_n),
        "CAT rows                 : %d (%d graded, %d not licensed)"
        % (len(cat), sum(1 for p in cat if p.verdict != VERDICT_NOT_LICENSED),
           sum(1 for p in cat if p.verdict == VERDICT_NOT_LICENSED)),
    ]
    return "\n".join(out)


DEFAULT_RESULTS = Path("experiments/results/qwen2.5-14b/_aggregate/all/results.csv")
DEFAULT_FIGDIR = Path("thesis/content/figures")
DEFAULT_TABDIR = Path("thesis/content/tables")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--figures", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--tables", type=Path, default=DEFAULT_TABDIR)
    ap.add_argument("--facts-only", action="store_true",
                    help="print the caption numbers and write nothing")
    a = ap.parse_args(argv)

    rows = load_rows(a.results)
    prem, f2, cat = f1_rows(rows), f2_rows(rows), catalogue_rows(rows)
    print(caption_facts(prem, f2, cat))
    if a.facts_only:
        return 0

    _rc()
    render_f1(prem, a.figures / "fig_ceiling_premium")
    render_f2(f2, a.figures / "fig_paired_separation")
    render_cost_quality(prem, a.figures / "fig_cost_quality")
    table_f1(prem, a.tables / "tab_ceiling_premium.tex", "tab:ceiling-premium")
    table_cost_quality(prem, a.tables / "tab_cost_quality.tex", "tab:cost-quality")
    table_catalogue(cat, a.tables / "tab_comparison_catalogue.tex",
                    "tab:comparison-catalogue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
