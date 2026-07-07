"""report.py — auto-generated numeric report (the tabular half of reporting).

The charts (`plots.py`) are the visual layer; this is the layer for the facts a
chart cannot legibly carry: coverage / rank-eligibility (the context_exceeded
survivorship trap), the small-n register, and per-cell n + CI. It is regenerated
from `results.csv` on every run/aggregation alongside the PNGs, so it never goes
stale, and it is the seed for publication tables at thesis-writing time.

Design rules enforced here (see METHODOLOGY 3.5):
- No `ALL/ALL` grand mean is ever surfaced; every number is within one question
  type, and axis cards are within one host dataset (no cross-dataset pooling).
- Axis cards are floor/ceiling-anchored and ladder-ordered (evaluation/axes.py).
- A cell with coverage < MIN_COVERAGE is flagged and marked not rank-eligible
  (its AC is conditioned on the surviving subset). A cell with n < SMALL_N is
  flagged screening-only.
- Combos and the `synthesis` candidate get their own tables, never mixed into a
  pole card.

These are descriptive *display* statistics over the same per-cell AC values and
the same grouping as `results._group_row` -- no new analysis. CI95 = 1.96*s/sqrt(k)
over the k per-question means (clustered by question -- repetitions are stochastic
re-draws, not independent items, so pooling question x repetition would overstate
precision); a coarse precision cue, wide and unreliable at the small k this study
runs at.
"""

from __future__ import annotations

import collections
import csv
import datetime
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from evaluation.axes import (
    AXES, CANDIDATE, CEILING, FLOOR, MIN_COVERAGE, SMALL_N,
    dataset_of, rep_role, tier_of,
)
from evaluation.core import is_context_exceeded
from evaluation.scope import in_scope


@dataclass
class Cell:
    """One (representation x question_type) group's display statistics."""
    n: int
    error_count: int
    context_exceeded: int
    n_scored: int
    ac_mean: float
    ac_ci95: float
    coverage: float
    tokens_mean: float | None
    # Diagnostic: supporting-detail coverage over its OWN support (cells with weight<=1
    # facts only), NOT correctness and NOT comparable cell-for-cell against ac_mean.
    detail_mean: float | None
    n_detail: int

    @property
    def rank_eligible(self) -> bool:
        return self.n_scored > 0 and self.coverage >= MIN_COVERAGE


def _cell(rows: list[dict]) -> Cell:
    n = len(rows)
    context_exceeded = sum(1 for r in rows if is_context_exceeded(r["error"]))
    error_count = sum(1 for r in rows if r["error"] and not is_context_exceeded(r["error"]))
    ac = [float(r["answer_correctness"]) for r in rows
          if not r["error"] and r["answer_correctness"] not in ("", None)]
    # CI clustered by question (repetitions of one question are not independent
    # evidence): per-question means first, then 1.96*s/sqrt(k) over the k means --
    # matching results._group_row and the per-question points in plots.py.
    ac_by_q: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if not r["error"] and r["answer_correctness"] not in ("", None):
            ac_by_q[r["question_id"]].append(float(r["answer_correctness"]))
    q_means = [statistics.mean(v) for v in ac_by_q.values()]
    det = [float(r["answer_correctness_detail"]) for r in rows
           if not r["error"] and r.get("answer_correctness_detail") not in ("", None)]
    toks: list[float] = []
    for r in rows:
        if r["error"]:
            continue
        t = r.get("prompt_tokens")
        if t in ("", "0", None):
            continue
        try:
            toks.append(float(t))
        except ValueError:
            pass
    n_scored = n - error_count - context_exceeded
    ac_mean = statistics.mean(ac) if ac else 0.0
    ac_ci95 = 1.96 * statistics.stdev(q_means) / math.sqrt(len(q_means)) if len(q_means) > 1 else 0.0
    coverage = n_scored / n if n else 0.0
    return Cell(n, error_count, context_exceeded, n_scored, ac_mean, ac_ci95,
                coverage, statistics.mean(toks) if toks else None,
                statistics.mean(det) if det else None, len(det))


def _cells(rows: list[dict]) -> dict[tuple[str, str], Cell]:
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["representation"], r["question_type"] or "unknown")].append(r)
    return {k: _cell(v) for k, v in groups.items()}


# --- markdown rendering ----------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _ac_str(c: Cell) -> str:
    """AC with a coverage flag: a sub-threshold cell prints `0.70* (cov 33%)` so it
    is never silently ranked against a full-coverage one."""
    if c.n_scored == 0:
        return "n/a"
    s = f"{c.ac_mean:.2f}"
    if c.coverage < MIN_COVERAGE:
        s += f"* (cov {c.coverage * 100:.0f}%)"
    return s


def _ci_str(c: Cell) -> str:
    return f"+/-{c.ac_ci95:.2f}" if c.n_scored > 1 else ""


def _axis_reps(axis, qt: str, cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Anchored, ladder-ordered rep list for one axis x type: floor first, the
    in-scope poles in rung order, ceiling last -- keeping only those with data.
    Returns [] when there is no real ladder (fewer than two reps present)."""
    reps: list[str] = []
    if (FLOOR, qt) in cells:
        reps.append(FLOOR)
    for p in axis.ladder:
        if p in (FLOOR, CEILING):
            continue
        if in_scope(p, qt) and (p, qt) in cells:
            reps.append(p)
    if (CEILING, qt) in cells and CEILING not in reps:
        reps.append(CEILING)
    return reps if len(reps) >= 2 else []


def _axis_card(axis, rows: list[dict]) -> list[str]:
    """One axis card: a per-probe-type anchored table, host dataset only."""
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == axis.host])
    body: list[str] = []
    for qt in axis.probe_types:
        reps = _axis_reps(axis, qt, cells)
        if not reps:
            continue
        floor = cells.get((FLOOR, qt))
        floor_ac = floor.ac_mean if floor and floor.n_scored else None
        trows = []
        for rep in reps:
            c = cells[(rep, qt)]
            dfloor = ("" if floor_ac is None or c.n_scored == 0
                      else f"{c.ac_mean - floor_ac:+.2f}")
            trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                          _ac_str(c), _ci_str(c), dfloor])
        body.append(f"**{qt}** (host: {axis.host})")
        body += _table(["rep", "role", "n", "cov%", "AC", "CI95", "vs floor"], trows)
        body.append("")
    if not body:
        return []
    return [f"## Axis {axis.id} - {axis.label}", f"_{axis.note}_", ""] + body


def _planning_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """The planning / real-world-utility probe, reported on its own (not an axis
    pole). A planning question states a *goal*; the model must infer the objects it
    needs (the affordance step) rather than being handed them -- that inference is
    the extra reasoning vs a containment/set_logic question on the same objects.
    inventory is the floor, json the ceiling; in-scope reps carry the inventory
    channel."""
    reps = sorted({rep for (rep, qt) in cells if qt == "planning"})
    if not reps:
        return []
    floor = cells.get((FLOOR, "planning"))
    floor_ac = floor.ac_mean if floor and floor.n_scored else None

    def order(rep: str):  # floor first, poles by AC desc, ceiling last
        rank = {"floor": 0, "ceiling": 2}.get(rep_role(rep), 1)
        return (rank, -cells[(rep, "planning")].ac_mean)

    trows = []
    for rep in sorted(reps, key=order):
        c = cells[(rep, "planning")]
        dfloor = ("" if floor_ac is None or c.n_scored == 0
                  else f"{c.ac_mean - floor_ac:+.2f}")
        trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                      _ac_str(c), _ci_str(c), dfloor])
    return (["## Planning / real-world utility (separate probe)",
             "_Goal-framed questions: the model must infer the objects a goal needs, not be "
             "handed them. Reported on its own -- not an axis pole. inventory = floor, json = ceiling._", ""]
            + _table(["rep", "role", "n", "cov%", "AC", "CI95", "vs floor"], trows) + [""])


OBJECT_RELATION_TYPES = {"object_relation", "relation_structure", "relation_aggregate"}


def _coverage_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Cells where coverage matters: anything that overflowed the window, plus the
    whole object-relation family (`object_relation`/`relation_structure`/
    `relation_aggregate` -- where json/relations_flat overflow on dense scenes).
    This is where the survivorship trap is read."""
    trows = []
    for (rep, qt), c in sorted(cells.items()):
        if c.context_exceeded > 0 or qt in OBJECT_RELATION_TYPES:
            trows.append([rep, qt, str(c.n), str(c.context_exceeded),
                          f"{c.coverage * 100:.0f}", _ac_str(c),
                          "yes" if c.rank_eligible else "no"])
    if not trows:
        return []
    return (["## Coverage & rank-eligibility",
             "_Cells that overflowed the window or carry object relations. "
             "rank_eligible = coverage >= 80%; a `no` must not be ranked on AC._", ""]
            + _table(["rep", "type", "n", "exceeded", "cov%", "AC", "rank_eligible"], trows)
            + [""])


def _small_n_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    trows = [[rep, qt, str(c.n), _ac_str(c)]
             for (rep, qt), c in sorted(cells.items()) if 0 < c.n < SMALL_N]
    if not trows:
        return []
    return (["## Small-n register (n < %d -- screening-only)" % SMALL_N,
             "_These cells are too small to rank on; treat as directional only._", ""]
            + _table(["rep", "type", "n", "AC"], trows) + [""])


def _combo_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Multi-view combos vs their best single component vs the json ceiling -- the
    'do two orthogonal views reach the ceiling' question, kept out of the pole cards."""
    combos = sorted({rep for (rep, _) in cells if "+" in rep})
    if not combos:
        return []
    trows = []
    for combo in combos:
        parts = combo.split("+")
        qts = sorted({qt for (rep, qt) in cells if rep == combo})
        for qt in qts:
            c = cells[(combo, qt)]
            comp = [(p, cells[(p, qt)].ac_mean) for p in parts
                    if (p, qt) in cells and cells[(p, qt)].n_scored]
            best = max(comp, key=lambda x: x[1]) if comp else None
            best_s = f"{best[0]} {best[1]:.2f}" if best else "n/a"
            jc = cells.get((CEILING, qt))
            json_s = _ac_str(jc) if jc else "n/a"
            dbest = (f"{c.ac_mean - best[1]:+.2f}"
                     if best and c.n_scored else "")
            trows.append([combo, qt, str(c.n), _ac_str(c), best_s, json_s, dbest])
    return (["## Multi-view combinations",
             "_Each combo vs its best single component and the json ceiling._", ""]
            + _table(["combo", "type", "n", "AC", "best single", "json AC", "vs best"], trows)
            + [""])


def _candidate_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """The synthesis candidate-default vs the json ceiling, with token cost -- the
    'match the ceiling at a fraction of the tokens' question."""
    qts = sorted({qt for (rep, qt) in cells if rep == CANDIDATE})
    if not qts:
        return []
    trows = []
    for qt in qts:
        s = cells[(CANDIDATE, qt)]
        j = cells.get((CEILING, qt))
        st = f"{s.tokens_mean:.0f}" if s.tokens_mean else "-"
        jt = f"{j.tokens_mean:.0f}" if j and j.tokens_mean else "-"
        trows.append([qt, _ac_str(s), _ac_str(j) if j else "n/a", st, jt])
    return (["## Candidate default (synthesis vs json)",
             "_Can one compact view match the json ceiling at a fraction of the tokens?_", ""]
            + _table(["type", "synthesis AC", "json AC", "synth tok", "json tok"], trows)
            + [""])


def _detail_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Supporting-detail coverage -- a DIAGNOSTIC, not a quality score.

    Reports the weight<=1 (detail) tier separately: the fraction of unasked supporting
    facts the answer volunteered. It measures verbosity x representation, not
    correctness, has its own (sparser) support n_detail, and is NOT comparable
    cell-for-cell against AC. Read it only alongside AC (e.g. high AC + low detail =
    answering correctly but thinly). Never rank on it."""
    trows = [[rep, qt, str(c.n_detail), f"{c.detail_mean:.2f}"]
             for (rep, qt), c in sorted(cells.items())
             if c.n_detail > 0 and c.detail_mean is not None]
    if not trows:
        return []
    return (["## Supporting-detail coverage (diagnostic -- NOT correctness)",
             "_The weight<=1 detail tier, scored on its own. This is volunteered "
             "supporting detail (verbosity x representation), not a quality score; "
             "n_detail is its own support and it is NOT comparable to AC. Never rank on it._", ""]
            + _table(["rep", "type", "n_detail", "detail_AC"], trows) + [""])


def write_report(results_path: Path, aggregate_path: Path | None = None) -> Path | None:
    """Render report.md next to results.csv. Returns the path, or None if no rows."""
    results_path = Path(results_path)
    if not results_path.exists():
        return None
    with results_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    cells = _cells(rows)
    responder = rows[0].get("responder", "?")
    judge = rows[0].get("judge", "?")
    tier = tier_of(judge)
    datasets = sorted({dataset_of(r["scene_id"]) for r in rows})
    today = datetime.date.today().isoformat()

    lines = [
        f"# Evaluation report - {responder}",
        "",
        f"- judge: `{judge}`  |  tier: **{tier}**"
        + ("  (ranking only, not reported effect sizes)" if tier == "screening" else ""),
        f"- datasets: {', '.join(datasets)}  |  generated: {today}",
        "",
        "> Do NOT read an ALL/ALL grand mean. Every number below is within one "
        "question type, and axis cards are within one host dataset. Compare within "
        "a card, never across types or datasets.",
        "",
    ]
    for axis in AXES:
        lines += _axis_card(axis, rows)
    lines += _planning_section(cells)
    lines += _coverage_section(cells)
    lines += _small_n_section(cells)
    lines += _combo_section(cells)
    lines += _candidate_section(cells)
    lines += _detail_section(cells)

    out_path = results_path.parent / "report.md"
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"report -> {out_path}")
    return out_path
