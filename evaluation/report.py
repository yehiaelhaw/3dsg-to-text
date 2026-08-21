"""report.py — auto-generated numeric report (the tabular half of reporting).

The charts (`plots.py`) are the visual layer; this is the layer for the facts a
chart cannot legibly carry: coverage / rank-eligibility (the context_exceeded
survivorship trap), the small-n register, per-cell n, and the paired scene-level
separation table that decides every axis verdict. It is regenerated
from `results.csv` on every run/aggregation alongside the PNGs, so it never goes
stale, and it is the seed for publication tables at thesis-writing time.

Design rules enforced here (thesis 4.6 `sec:meth-analysis`; METHODOLOGY 3.5/3.7):
- No `ALL/ALL` grand mean is ever surfaced; every number is within one question
  type, and axis cards are within one host dataset (no cross-dataset pooling).
- Axis cards are floor/ceiling-anchored and ladder-ordered (evaluation/axes.py).
- A cell with coverage < MIN_COVERAGE is flagged and marked not rank-eligible
  (its AC is conditioned on the surviving subset). A cell with n < SMALL_N is
  flagged screening-only. Where such a cell would enter a comparison or an
  ordering, it is printed as `not licensed` with the reason rather than as a
  number, and never sorted to the top of a block.
- Every difference between two representations -- the separation table and the
  `vs floor` columns alike -- is computed over the questions BOTH members
  answered (`paired_deltas`). Two cell means are never subtracted: the floor is
  exempt from the scope filter and answers everything, so that subtraction would
  difference one rep's surviving questions against the other's whole set.
- Combos and the `synthesis` candidate get their own tables, never mixed into a
  pole card.

The per-cell tables are descriptive *display* statistics over the same per-cell AC
values and the same grouping as `results._group_row` -- no new analysis. The
spread printed beside each mean is the min-max RANGE over the k per-question
means: descriptive only, never an interval estimate. No confidence interval is
computed anywhere and interval overlap is never used as a decision rule (see
thesis 4.6) -- at three scenes and a handful of questions per cell an
interval would imply a precision this design cannot support.

Separation between two representations is decided in `_paired_section` instead,
which is the actual analysis: paired per-question differences, averaged within
each host scene, judged against a pre-declared practical margin with a
consistency requirement across scenes (evaluation/axes.py).

Two further sections read the question's authored register. `_recut_section`
splits every capped comparison into its `natural` and `constructed` halves;
`_matched_section` goes one step further and differences those two halves within
each fact-set, which is what the balanced corpus was authored to make possible.
"""

from __future__ import annotations

import collections
import csv
import datetime
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from evaluation.axes import (
    AXES, AXIS_BY_ID, CANDIDATE, CAPPED_VERDICT, CEILING, FLOOR, MIN_COVERAGE,
    MIN_SCENES_SHOWING, PRACTICAL_MARGIN, SMALL_N,
    VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
    VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED,
    dataset_of, rep_role, tier_of,
)
from evaluation.core import is_context_exceeded
from evaluation.scope import in_scope

# --- the authored fact-set map ---------------------------------------------
# `pair_id` is a property of the authored question, not of a run, so it is not a
# results.csv column (see the comment on core.Question). It is joined here from the
# QA files at analysis time -- the same (scene_id, question_id) join
# experiments/scripts/backfill_question_style.py performs -- which keeps every
# committed results.csv byte-identical.
#
# It is used for exactly ONE thing: grouping the two members of a fact-set so their
# register effects can be differenced. It never filters, never groups a table, and
# is never compared against the question's own id. Whether a stem was authored
# before or during the matching exercise is history, not an experimental variable:
# `natural` means natural and `constructed` means constructed regardless, and all
# 134 scoped questions participate in every table.
QA_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
QA_FILENAME = "keyfact-qa.jsonl"


def _load_pairs(qa_root: Path | None = None) -> dict[tuple[str, str], str]:
    """(scene_id, question_id) -> pair_id, over every authored QA file.

    Returns {} when the QA files are not reachable (a report rendered outside the
    repo) -- the matched section then simply does not render, which is the honest
    outcome rather than an import-time failure in a display layer.

    `qa_root` is resolved at call time, not bound as a default, so a caller (the
    smoke test) can point the module constant at a synthetic corpus.
    """
    root = Path(qa_root if qa_root is not None else QA_ROOT)
    pairs: dict[tuple[str, str], str] = {}
    if not root.is_dir():
        return pairs
    for qa_path in sorted(root.glob(f"*/{QA_FILENAME}")):
        with qa_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                if q.get("pair_id"):
                    pairs[(q["scene_id"], str(q["id"]))] = str(q["pair_id"])
    return pairs


def _row_key(r: dict) -> tuple[str, str]:
    """A row's (scene_id, authored question id).

    aggregate_results._pool_rows namespaces `question_id` as `<scene_id>:<id>` so
    identities stay unique across pooled scenes, while leaving `scene_id` bare. The
    _aggregate/ report is the one anyone actually reads, so a join that forgot this
    would match nothing there and silently drop the matched section rather than fail.
    """
    scene, qid = r["scene_id"], r["question_id"]
    prefix = f"{scene}:"
    return scene, (qid[len(prefix):] if qid.startswith(prefix) else qid)


@dataclass
class Cell:
    """One (representation x question_type) group's display statistics."""
    n: int
    error_count: int
    context_exceeded: int
    n_scored: int
    ac_mean: float
    # Descriptive min-max spread over the k per-question means. NOT an interval
    # estimate and never a decision rule -- separation is decided in
    # `_paired_section`. (lo, hi); (0.0, 0.0) when there is nothing to spread.
    ac_range: tuple[float, float]
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
    # Per-question means first (repetitions of one question are not independent
    # evidence), then their min-max range -- matching the per-question points
    # plots.py draws. A descriptive spread, not a precision estimate.
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
    ac_range = (min(q_means), max(q_means)) if q_means else (0.0, 0.0)
    coverage = n_scored / n if n else 0.0
    return Cell(n, error_count, context_exceeded, n_scored, ac_mean, ac_range,
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


def _range_str(c: Cell) -> str:
    """Descriptive min-max spread of the per-question means. Deliberately NOT an
    interval: two of these overlapping means nothing, and overlap is never a tie
    rule (thesis 4.6). Separation is read from the paired table."""
    lo, hi = c.ac_range
    return f"{lo:.2f}-{hi:.2f}" if c.n_scored > 1 else ""


def _tok_str(p) -> str:
    """Mean prompt tokens of the two members, in the comparison's own order
    (`rep_b` vs `rep_a`), with b's share of a. This is what the cost rule of
    thesis 4.6 is read from: a pair that does not separate on correctness while
    differing sharply here is a result, not the absence of one."""
    a, b = p.tokens_a, p.tokens_b
    if not a or not b:
        return "-"
    return f"{b:.0f} vs {a:.0f} ({b / a:.2f}x)"


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
    """One card: a per-probe-type anchored table, host dataset only.

    The heading names what the entry is whenever it is not one of the five design
    axes (Axis.kind) -- a companion carding the same axis on a second host, or an
    exhibit that is not an axis contrast at all. Without that, the card count here
    reads as an axis count and contradicts the design chapter.
    """
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == axis.host])
    # Question-matched, from the one pairing primitive -- NOT this cell's mean minus
    # the floor cell's mean, which would difference two different question sets
    # wherever a rep answered fewer questions than the floor did.
    lifts = {(p.qt, p.rep_b): p for p in floor_lifts(rows, axis.host)}
    body: list[str] = []
    for qt in axis.probe_types:
        reps = _axis_reps(axis, qt, cells)
        if not reps:
            continue
        floor = cells.get((FLOOR, qt))
        has_floor = bool(floor and floor.n_scored)
        trows = []
        for rep in reps:
            c = cells[(rep, qt)]
            dfloor = ("+0.00" if rep == FLOOR and has_floor
                      else _lift_str(lifts.get((qt, rep))))
            trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                          _ac_str(c), _range_str(c), dfloor])
        body.append(f"**{qt}** (host: {axis.host})")
        body += _table(["rep", "role", "n", "cov%", "AC", "q-range", "vs floor"], trows)
        body.append("")
    if not body:
        return []
    kind = "" if axis.kind == "axis" else f" ({axis.kind})"
    return [f"## {axis.label}{kind}", f"_{axis.note}_", ""] + body


def _planning_section(rows: list[dict]) -> list[str]:
    """The planning / real-world-utility probe, reported on its own (not an axis
    pole). A planning question states a *goal*; the model must infer the objects it
    needs (the affordance step) rather than being handed them -- that inference is
    the extra reasoning vs a containment/set_logic question on the same objects.
    inventory is the floor, json_mini the ceiling; in-scope reps carry the
    inventory channel.

    Split per host dataset (never pooled): planning's floor/ceiling meaning differs
    by dataset (e.g. 3RScan planning needs the raw relation channel, not just
    inventory -- QA_DESIGN 5), so a pooled `all` row would silently average across
    incompatible scopes.

    No filtering is needed here any more: the multi-view combos this table once had
    to exclude by name were purged from the results tree on 2026-08-12, so every rep
    reaching this point is part of the design."""

    def order(cells, rep):
        """Floor first, then the poles, then the ceiling -- and within the poles the
        rank-eligible cells by AC descending, with the sub-threshold ones beneath
        them rather than interleaved.

        Row order is a ranking whether or not the word appears: a reader takes the
        top pole for the best one. A cell below MIN_COVERAGE must therefore not be
        able to claim that position on an AC its own coverage gate says cannot be
        compared against a full-coverage cell.
        """
        c = cells[(rep, "planning")]
        block = {"floor": 0, "ceiling": 2}.get(rep_role(rep), 1)
        return (block, 0 if c.rank_eligible else 1, -c.ac_mean)

    out: list[str] = []
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
        reps = sorted({rep for (rep, qt) in cells if qt == "planning"})
        if not reps:
            continue
        lifts = {(p.qt, p.rep_b): p for p in floor_lifts(rows, ds)}
        floor = cells.get((FLOOR, "planning"))
        has_floor = bool(floor and floor.n_scored)
        trows = []
        for rep in sorted(reps, key=lambda r: order(cells, r)):
            c = cells[(rep, "planning")]
            dfloor = ("+0.00" if rep == FLOOR and has_floor
                      else _lift_str(lifts.get(("planning", rep))))
            trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                          _ac_str(c), _range_str(c), dfloor])
        out += [f"## Planning / real-world utility (separate probe, host: {ds})",
                "_Goal-framed questions: the model must infer the objects a goal needs, not be "
                "handed them. Reported on its own -- not an axis pole, and never pooled across "
                f"datasets. {FLOOR} = floor, {CEILING} = ceiling._", ""]
        out += _table(["rep", "role", "n", "cov%", "AC", "q-range", "vs floor"], trows) + [""]
    return out


# --- the separation analysis (thesis 4.6) -----------------------------
# This is the only place a comparison between two representations is decided.
# Everything above is display.

@dataclass
class Paired:
    """One axis pair x question type, compared the way the design licenses.

    `scene_deltas` maps scene_id -> the mean per-question AC difference
    (rep_b - rep_a) over the questions BOTH members answered in that scene. The
    scene is the unit of replication: questions are nested within scenes, so the
    headline figure is the unweighted mean of the scene values, not a mean over
    pooled questions (which would let a question-rich scene outvote the others).
    """
    axis_id: str
    host: str
    qt: str
    rep_a: str                      # earlier rung / baseline
    rep_b: str                      # later rung, or the ceiling
    scene_deltas: dict[str, float]
    n_questions: int                # paired questions summed across scenes
    verdict: str
    confound: str
    capped: bool                    # the confound actually downgraded the grade
    ineligible: str                 # why the gates failed, or ""
    # Mean prompt tokens of each member, carried so the cost rule (thesis 4.6) is
    # decided from the same row as the verdict: a pair that fails to separate on
    # correctness while differing sharply in tokens is a result, and the cheaper
    # member is reported as the more efficient alternative -- never as the winner.
    tokens_a: float | None = None
    tokens_b: float | None = None
    # How many distinct information requests the paired questions come from. On the
    # scoped types every fact-set contributes BOTH its natural and its constructed
    # member, so `n_questions` counts two correlated observations per request and
    # overstates the support; this is the honest denominator. 0 where no fact-set
    # map joins (every unpaired question type). Display only -- no verdict reads it.
    n_factsets: int = 0
    # The individual paired differences behind `scene_deltas`, scene-major. Display
    # only: the verdict reads the SCENE values, never these. Carried so a chart can
    # draw the per-question points of a comparison without recomputing them -- a
    # recomputation is where a second, subtly different matching rule gets in.
    q_deltas: tuple[float, ...] = ()

    @property
    def mean(self) -> float:
        v = list(self.scene_deltas.values())
        return statistics.mean(v) if v else 0.0

    @property
    def spread(self) -> tuple[float, float]:
        v = list(self.scene_deltas.values())
        return (min(v), max(v)) if v else (0.0, 0.0)


def _verdict(scene_deltas: dict[str, float], confound: str, ineligible: str) -> str:
    """The verdict scale of thesis 4.6, Table 4.5.

    A scene whose delta is exactly 0.0 does not contradict a direction, but
    neither does it supply one -- which is why consistency needs both `no scene
    reversed` and `at least MIN_SCENES_SHOWING showing`. The margin-met but
    under-replicated case (e.g. +0.30 / 0.0 / 0.0) is graded down to directional:
    the direction is not contradicted but two scenes do not attest it, and the
    conservative grade is the one the design can carry.

    MIXED is decided on the scene values, not on their mean. Testing the mean
    alone made cancellation -- the *strongest* form of scene disagreement -- read
    as a null: +0.143 / -0.222 / -0.214 averages to -0.098, lands under the
    margin, and printed as `no practically meaningful separation` even though one
    scene clears the margin one way and two clear it the other. That is the
    opposite of what this verdict exists to say, so disagreement is now tested as
    `some scene clears +margin AND some scene clears -margin`, reusing the same
    declared margin rather than introducing a second threshold. A single scene
    moving while the others sit flat (+0.000 / +0.053 / -0.250) is still not a
    reversal and still grades as no separation.
    """
    if ineligible or not scene_deltas:
        return VERDICT_NOT_LICENSED

    vals = list(scene_deltas.values())
    mean = statistics.mean(vals)
    margin_met = abs(mean) >= PRACTICAL_MARGIN
    direction = (mean > 0) - (mean < 0)          # +1 / -1 / 0
    reversed_ = any((v > 0) - (v < 0) == -direction for v in vals) if direction else False
    showing = sum(1 for v in vals if (v > 0) - (v < 0) == direction) if direction else 0
    # Scenes disagree at the declared margin, whatever their mean does.
    disagree = max(vals) >= PRACTICAL_MARGIN and min(vals) <= -PRACTICAL_MARGIN

    if disagree:
        return VERDICT_MIXED
    if direction == 0:
        return VERDICT_NO_SEPARATION
    if reversed_:
        return VERDICT_MIXED if margin_met else VERDICT_NO_SEPARATION
    if margin_met and showing >= MIN_SCENES_SHOWING:
        # A declared confound caps an otherwise-consistent result. It never
        # promotes and never rescues -- only downgrades.
        return CAPPED_VERDICT if confound else VERDICT_CONSISTENT
    return VERDICT_DIRECTIONAL


def _paired_pairs(axis, qt: str, cells: dict[tuple[str, str], Cell]) -> list[tuple[str, str]]:
    """Which contrasts to compute for one axis x question type.

    The ladder baseline against each later rung (the axis's own story), the
    declared headline pair, and each rung against the json_mini ceiling (the
    anchor comparison). Deduplicated, always ordered (earlier, later) so the reported
    delta's sign is unambiguous.
    """
    rungs = [p for p in axis.ladder
             if p not in (FLOOR, CEILING) and in_scope(p, qt) and (p, qt) in cells]
    pairs: list[tuple[str, str]] = []
    if rungs:
        base = rungs[0]
        pairs += [(base, r) for r in rungs[1:]]
    if axis.headline_pair:
        a, b = axis.headline_pair
        if (a, qt) in cells and (b, qt) in cells and in_scope(a, qt) and in_scope(b, qt):
            pairs.append((a, b))
    if (CEILING, qt) in cells:
        pairs += [(r, CEILING) for r in rungs]
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        if p not in seen and p[0] != p[1]:
            seen.add(p)
            out.append(p)
    return out


def _q_means(host_rows: list[dict]) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """((rep, question_id) -> mean AC over that question's scored rows,
    question_id -> scene_id).

    Errors and context-exceeded rows carry no AC, so they simply fail to appear --
    an overflowing rep contributes no delta rather than a zero, and its coverage
    gate catches it at the call site.
    """
    ac: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    scene_of: dict[str, str] = {}
    for r in host_rows:
        if r["error"] or r["answer_correctness"] in ("", None):
            continue
        ac[(r["representation"], r["question_id"])].append(float(r["answer_correctness"]))
        scene_of[r["question_id"]] = r["scene_id"]
    return {k: statistics.mean(v) for k, v in ac.items()}, scene_of


def paired_deltas(q_mean: dict[tuple[str, str], float], scene_of: dict[str, str],
                  qids, rep_a: str, rep_b: str) -> dict[str, list[tuple[float, str]]]:
    """scene_id -> [(AC(rep_b) - AC(rep_a), question_id)] over the questions BOTH
    members answered, questions in sorted order.

    THE pairing primitive. Every difference this repo reports comes through here --
    the separation table, the `vs floor` columns, and the lift chart in plots.py --
    so that no two of them can disagree about which questions a delta was taken
    over.

    A question either member did not answer is DROPPED, never zero-filled. That is
    the whole point: a difference of two cell MEANS is a difference between one
    rep's surviving questions and the other's, and where a rep overflows the context
    window on the hard half of a type those are not the same set. The subtraction
    then silently reads `easy questions only, minus everything` as an effect of the
    representation. Dropping is also not free -- it can leave too few pairs to grade
    -- which is what the SMALL_N gate at each call site is for.

    Sorted rather than set-ordered so the summation order (and therefore the last
    bit of every mean) is reproducible: `qids` is a set of strings, and CPython
    randomizes string hashing per process, so the unsorted form varied run to run.
    Every figure is printed to 3 decimals, so this fixes an invisible instability
    rather than changing any reported number.
    """
    by_scene: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for q in sorted(qids):
        if (rep_a, q) in q_mean and (rep_b, q) in q_mean:
            by_scene[scene_of[q]].append((q_mean[(rep_b, q)] - q_mean[(rep_a, q)], q))
    return by_scene


def _summarise(by_scene: dict[str, list[tuple[float, str]]]):
    """(scene_deltas, n_questions, q_deltas) from `paired_deltas` output."""
    scene_deltas = {s: statistics.mean([d for d, _ in items])
                    for s, items in sorted(by_scene.items())}
    n_q = sum(len(v) for v in by_scene.values())
    q_deltas = tuple(d for _, items in sorted(by_scene.items()) for d, _ in items)
    return scene_deltas, n_q, q_deltas


def _gates(cells: dict[tuple[str, str], Cell], qt: str, reps: tuple[str, ...],
           n_q: int, unit: str = "paired questions") -> str:
    """Why this comparison is not rank-eligible, or "".

    Applied to the COMPARISON rather than to either cell alone: a pair is only as
    licensed as its weaker member, and the paired question count -- not either
    cell's n -- is its real support. Shared by every caller so `not licensed` means
    exactly one thing across the whole report.
    """
    reasons = [f"{rep} coverage {cells[(rep, qt)].coverage * 100:.0f}%"
               for rep in reps
               if (rep, qt) in cells and not cells[(rep, qt)].rank_eligible]
    if n_q < SMALL_N:
        reasons.append(f"only {n_q} {unit}")
    return "; ".join(reasons)


def _paired_rows(rows: list[dict], style: str | None = None, axes=None,
                 lift_confound: bool = False,
                 pairs: dict[tuple[str, str], str] | None = None) -> list[Paired]:
    """Every licensed axis contrast, compared per scene and graded.

    `style` restricts to one question-style subset (`natural` / `constructed`) --
    the vocabulary-coupling re-cut. The restriction is applied before `_cells`, so
    the eligibility gates are evaluated on the subset actually being compared
    rather than inherited from the pooled cell. `axes` narrows which axes are
    walked (the re-cut only reads the ones carrying a declared confound).

    `lift_confound` drops the cap. It exists for exactly one caller: the `natural`
    half of the re-cut, where a vocabulary confound is resolved *by construction*
    -- a natural question is one a user could have asked without ever having seen
    the derived view's output, so its wording cannot be mirroring that output.
    Capping there would suppress the very finding the re-cut was built to produce.
    Never lift a `content` confound this way; no question-style split addresses it.

    `pairs` is the authored fact-set map. It is used only to report how many
    distinct information requests a comparison rests on: with a balanced corpus the
    pooled table sees BOTH members of every fact-set, so `n_q` counts correlated
    observations twice. It changes no delta and no verdict.
    """
    out: list[Paired] = []
    for axis in (AXES if axes is None else axes):
        host_rows = [r for r in rows if dataset_of(r["scene_id"]) == axis.host]
        if style is not None:
            host_rows = [r for r in host_rows if (r.get("question_style") or "") == style]
        if not host_rows:
            continue
        cells = _cells(host_rows)
        q_mean, scene_of = _q_means(host_rows)
        # question_id (as it appears in this csv) -> (scene, pair_id), for the count.
        fs_of: dict[str, tuple[str, str]] = {}
        for r in host_rows:
            key = _row_key(r)
            if pairs and key in pairs:
                fs_of[r["question_id"]] = (key[0], pairs[key])

        for qt in axis.probe_types:
            qids = {q for (rep, q) in q_mean
                    if any(r["question_id"] == q and (r["question_type"] or "unknown") == qt
                           for r in host_rows)}
            for rep_a, rep_b in _paired_pairs(axis, qt, cells):
                by_scene = paired_deltas(q_mean, scene_of, qids, rep_a, rep_b)
                if not by_scene:
                    continue
                factsets = {fs_of[q] for items in by_scene.values()
                            for _, q in items if q in fs_of}
                scene_deltas, n_q, q_deltas = _summarise(by_scene)
                ineligible = _gates(cells, qt, (rep_a, rep_b), n_q)
                confound = "" if lift_confound else axis.confound_for(rep_a, rep_b)
                verdict = _verdict(scene_deltas, confound, ineligible)
                # The cap fired only where it changed the grade: the same pair
                # judged without its confound would have been consistent.
                capped = bool(confound) and verdict == CAPPED_VERDICT and (
                    _verdict(scene_deltas, "", ineligible) == VERDICT_CONSISTENT)
                ca, cb = cells.get((rep_a, qt)), cells.get((rep_b, qt))
                out.append(Paired(axis.id, axis.host, qt, rep_a, rep_b, scene_deltas,
                                  n_q, verdict, confound, capped, ineligible,
                                  ca.tokens_mean if ca else None,
                                  cb.tokens_mean if cb else None,
                                  len(factsets), q_deltas))
    return out


# --- the floor comparison ---------------------------------------------------
# `_paired_pairs` deliberately never pairs a rung against the floor: a floor
# comparison is not the design decision an axis isolates, so it does not belong in
# the separation table. But it is still a difference between two representations,
# and it is the headline of the spatial-encoding axis ("what does adding structure
# buy over a no-information control?"), so it gets the same treatment here rather
# than a second, weaker one of its own.
FLOOR_LIFT_ID = "floor"


def floor_lifts(rows: list[dict], host: str) -> list[Paired]:
    """Every representation against the `inventory` floor on one host, question-matched.

    Same pairing primitive, same eligibility gates and same verdict scale as
    `_paired_rows`. Both the `vs floor` columns below and
    `plots.value_of_spatial_structure` read this one function, so a lift cannot
    carry two different numbers in two artifacts describing the same run.

    What this replaces is a subtraction of two cell means. That is wrong here more
    often than anywhere else in the report, because the floor is the one rep that is
    deliberately exempted from the scope filter (runner.py) and therefore answers
    every question of every type, while the reps it is subtracted from routinely do
    not -- the JSON views overflow the window on dense scenes and lose exactly the
    questions with the most content in them. `mean(survivors) - mean(everything)`
    then reads the missing questions as an effect of the representation.

    Every type the floor has data on is returned, not only the control types where
    inventory is out of scope: the axis cards print `vs floor` on content types too,
    where inventory is a legitimate compact format rather than a no-information
    control. Restricting to the control types is the CHART's job, and it says so in
    its title.
    """
    host_rows = [r for r in rows if dataset_of(r["scene_id"]) == host]
    if not host_rows:
        return []
    cells = _cells(host_rows)
    q_mean, scene_of = _q_means(host_rows)
    qids_by_type: dict[str, set[str]] = collections.defaultdict(set)
    for r in host_rows:
        qids_by_type[r["question_type"] or "unknown"].add(r["question_id"])

    out: list[Paired] = []
    for qt in sorted(qids_by_type):
        if (FLOOR, qt) not in cells:
            continue
        reps = sorted({rep for (rep, t) in cells
                       if t == qt and rep != FLOOR and in_scope(rep, qt)})
        for rep in reps:
            by_scene = paired_deltas(q_mean, scene_of, qids_by_type[qt], FLOOR, rep)
            if not by_scene:
                continue
            scene_deltas, n_q, q_deltas = _summarise(by_scene)
            ineligible = _gates(cells, qt, (FLOOR, rep), n_q)
            ca, cb = cells.get((FLOOR, qt)), cells.get((rep, qt))
            out.append(Paired(FLOOR_LIFT_ID, host, qt, FLOOR, rep, scene_deltas, n_q,
                              _verdict(scene_deltas, "", ineligible), "", False,
                              ineligible,
                              ca.tokens_mean if ca else None,
                              cb.tokens_mean if cb else None,
                              0, q_deltas))
    return out


def _lift_str(p: Paired | None) -> str:
    """The `vs floor` cell: a question-MATCHED delta, or an explicit `not licensed`.

    A sub-threshold cell prints the reason instead of a number rather than
    alongside it. A number a reader must remember not to use is a number that gets
    used -- and unlike the AC column (where `0.70* (cov 38%)` still describes the
    surviving answers honestly) a lift is a comparison, and this one is not
    licensed to be made.
    """
    if p is None:
        return ""
    if p.verdict == VERDICT_NOT_LICENSED:
        return f"{VERDICT_NOT_LICENSED} ({p.ineligible})"
    return f"{p.mean:+.2f}"


def _paired_section(rows: list[dict], pairs=None) -> list[str]:
    """The separation table -- the analysis every axis verdict is read from."""
    prs = _paired_rows(rows, pairs=pairs)
    if not prs:
        return []
    order = {VERDICT_CONSISTENT: 0, CAPPED_VERDICT: 1, VERDICT_MIXED: 2,
             VERDICT_NO_SEPARATION: 3, VERDICT_NOT_LICENSED: 4}
    prs.sort(key=lambda p: (order.get(p.verdict, 9), -abs(p.mean)))
    trows = []
    for p in prs:
        lo, hi = p.spread
        if p.ineligible:
            note = p.ineligible
        elif p.capped:
            note = f"CAPPED from '{VERDICT_CONSISTENT}' -- {p.confound}"
        elif p.confound:
            note = f"declared confound (did not change the grade): {p.confound}"
        else:
            note = ""
        trows.append([f"{p.axis_id}", p.qt, f"`{p.rep_b}` - `{p.rep_a}`",
                      " / ".join(f"{v:+.3f}" for v in p.scene_deltas.values()),
                      f"{p.mean:+.3f}", f"{lo:+.3f}..{hi:+.3f}", str(len(p.scene_deltas)),
                      str(p.n_questions), str(p.n_factsets) if p.n_factsets else "",
                      _tok_str(p), p.verdict, note])
    return (["## Paired separation (the analysis -- thesis 4.6)",
             f"_Per-question AC differences, averaged within each host scene; the mean is "
             f"over the SCENE values (the unit of replication), not over pooled questions. "
             f"A consistent advantage needs |mean| >= {PRACTICAL_MARGIN:.2f}, no scene "
             f"reversed, and >= {MIN_SCENES_SHOWING} scenes showing the direction; a "
             f"declared confound caps it at '{CAPPED_VERDICT}'. Scenes disagreeing by "
             f"{PRACTICAL_MARGIN:.2f} in BOTH directions is '{VERDICT_MIXED}' whatever the "
             f"mean does. No confidence intervals, and overlap is never a tie rule. Where a "
             f"pair shows '{VERDICT_NO_SEPARATION}', the `tokens` column decides: report the "
             f"cheaper representation as the more efficient alternative -- not as the "
             f"winner. On the scoped types every information request contributes both its "
             f"natural and its constructed stem, so read `fact-sets` -- not `n_q` -- as the "
             f"count of independent requests behind a row._", ""]
            + _table(["axis", "type", "comparison", "scene deltas", "mean", "range",
                      "scenes", "n_q", "fact-sets", "tokens (b vs a)", "verdict", "note"],
                     trows) + [""])


STYLES = ("natural", "constructed")


def _recut_rows(rows: list[dict]) -> dict[tuple[str, str, str, str], dict[str, "Paired"]]:
    """(axis_id, qt, rep_a, rep_b) -> {style: Paired}, the vocabulary re-cut's raw data.

    Factored out of `_recut_section` so a caller other than the markdown table (the
    thesis-figure renderer) can read the same natural/constructed split without
    reimplementing the style-filtering and cap-lifting logic. Returns {} exactly
    where `_recut_section` would render nothing: no style tag in `rows`, no
    vocabulary-coupled axis, or no comparison the cap actually fires on.
    """
    if not any((r.get("question_style") or "") for r in rows):
        return {}   # results.csv predates the tag; nothing to re-cut
    coupled = [a for a in AXES if a.confound_kind == "vocabulary"]
    if not coupled:
        return {}

    # The cap is lifted on `natural` only (see _paired_rows) -- there the coupling
    # is resolved by construction. On `constructed` it stands: that subset is where
    # the pole's own vocabulary is in the question, so its verdict is exactly the
    # reading the cap exists to distrust.
    by_style = {s: _paired_rows(rows, style=s, axes=coupled,
                                lift_confound=(s == "natural")) for s in STYLES}
    keyed: dict[tuple[str, str, str, str], dict[str, Paired]] = collections.defaultdict(dict)
    for s in STYLES:
        for p in by_style[s]:
            # Only the comparisons the cap actually fires on. An anchor comparison
            # against the floor or ceiling carries no vocabulary confound, so it has
            # nothing for the re-cut to resolve and would only pad the table.
            axis = AXIS_BY_ID[p.axis_id]
            if not axis.confound_for(p.rep_a, p.rep_b):
                continue
            keyed[(p.axis_id, p.qt, p.rep_a, p.rep_b)][s] = p
    return keyed


def _recut_section(rows: list[dict]) -> list[str]:
    """The vocabulary-coupling re-cut -- the remedy the declared confound points at.

    Two axes carry a derived pole whose own computed output states the concept the
    question asks for (`graph_digest`'s hub/bottleneck, `relations_digest`'s chain
    depth/clusters). The cap on those pairs says the pooled figure cannot settle
    whether the pole reasons better or merely recites; this splits the same paired
    comparison by the question's authored style tag, which is what settles it:
    `natural` = a user could have asked it without ever having seen the derived
    view, `constructed` = the concept mirrors that view's vocabulary.

    Only the comparisons the cap actually fires on are walked: axes whose confound
    is a `vocabulary` one (the two with a derived pole), and within them only the
    pairs touching that pole. No axis currently declares a `content` confound --
    that kind is for a superset no question-style split could address, and the
    format axis's original example (`prose`'s content superset of
    `topology_inventory`) was retired in favour of `narrative`, a content-matched
    pole built to remove the confound rather than qualify it. The ordinary gates
    apply unchanged to each subset: a split that lands under SMALL_N reads `not
    licensed`, which is the honest outcome for a subset too thin to rank, not a
    reason to pool it back together. Rows are ordered so a comparison's two
    styles sit adjacent -- that adjacency is the argument.

    On the scoped types the two halves are balanced by construction: every
    information request is authored in both registers, so each half holds exactly one
    member of every fact-set and `n_q` here IS the fact-set count. That balance also
    sharpens the lift above -- with the facts held constant across the halves, the
    tag encodes register and nothing else, so `a natural question cannot be mirroring
    the view` is a claim about wording alone rather than about which concept the
    author happened to pick. What this table still cannot do is difference the two
    halves question by question, because its rows compare SETS; `_matched_section`
    below does that.
    """
    keyed = _recut_rows(rows)
    if not keyed:
        return []

    trows = []
    for (axis_id, qt, rep_a, rep_b) in sorted(keyed):
        for s in STYLES:
            p = keyed[(axis_id, qt, rep_a, rep_b)].get(s)
            if p is None:
                # A style with no scored questions of this type at all -- recorded as
                # absent, not as a null. The scoped types are authored balanced (one
                # member of every fact-set per style), so this now only fires where a
                # whole subset failed to score, never by construction as it once did.
                trows.append([axis_id, qt, f"`{rep_b}` - `{rep_a}`", s,
                              "-", "-", "0", "no scored questions of this style"])
                continue
            trows.append([axis_id, qt, f"`{rep_b}` - `{rep_a}`", s,
                          " / ".join(f"{v:+.3f}" for v in p.scene_deltas.values()),
                          f"{p.mean:+.3f}", str(p.n_questions),
                          p.verdict + (f" ({p.ineligible})" if p.ineligible else "")])
    return (["## Vocabulary re-cut: natural vs constructed (thesis 4.6)",
             "_Every capped comparison, split by the question's authored style tag. A "
             "derived pole that only leads on `constructed` questions is reciting its own "
             "printed vocabulary, not reasoning better -- which is exactly what the capped "
             "verdict in the table above leaves open. Read the two rows of a comparison "
             "against each other; that adjacency is the argument. The cap is lifted on "
             "`natural` (a question a user could have asked without seeing the derived view "
             "cannot be mirroring it) and stands on `constructed`. Gates apply per subset, "
             "so a thin split reads `not licensed` rather than being pooled back. Only "
             "vocabulary-coupled axes appear here; no axis currently declares a content "
             "confound (a superset no style split could address)._", ""]
            + _table(["axis", "type", "comparison", "style", "scene deltas", "mean",
                      "n_q", "verdict"], trows) + [""])


def _matched_rows(rows: list[dict], pairs: dict[tuple[str, str], str]) -> list[dict]:
    """The matched within-fact-set comparison's raw numbers, one dict per comparison.

    Factored out of `_matched_section` so a caller other than the markdown table
    (the thesis-figure renderer) can read `d_natural`, `d_constructed`, the matched
    `mean` (= d_constructed - d_natural) and `scene_deltas` without recomputing them
    -- the whole point being that a figure and report.md's table are one
    computation, not two independently maintained ones. See `_matched_section`'s
    docstring for the method; this function differs from it only in returning data
    instead of formatted strings.
    """
    if not pairs:
        return []
    coupled = [a for a in AXES if a.confound_kind == "vocabulary"]
    if not coupled:
        return []

    out: list[dict] = []
    for axis in coupled:
        host_rows = [r for r in rows if dataset_of(r["scene_id"]) == axis.host]
        if not host_rows:
            continue
        cells = _cells(host_rows)
        q_mean, _ = _q_means(host_rows)

        # (scene, pair_id, question_type) -> {style: question_id as this csv spells it}
        members: dict[tuple[str, str, str], dict[str, str]] = collections.defaultdict(dict)
        for r in host_rows:
            key = _row_key(r)
            pid = pairs.get(key)
            style = r.get("question_style") or ""
            if pid and style in STYLES:
                members[(key[0], pid, r["question_type"] or "unknown")][style] = r["question_id"]

        for qt in axis.probe_types:
            for rep_a, rep_b in _paired_pairs(axis, qt, cells):
                if not axis.confound_for(rep_a, rep_b):
                    continue
                by_scene: dict[str, list[float]] = collections.defaultdict(list)
                nat_d: dict[str, list[float]] = collections.defaultdict(list)
                con_d: dict[str, list[float]] = collections.defaultdict(list)
                for (scene, _pid, mqt), m in members.items():
                    if mqt != qt:
                        continue
                    nat, con = m.get("natural"), m.get("constructed")
                    if not nat or not con:
                        continue
                    needed = [(rep, q) for rep in (rep_a, rep_b) for q in (nat, con)]
                    if not all(k in q_mean for k in needed):
                        continue
                    d_nat = q_mean[(rep_b, nat)] - q_mean[(rep_a, nat)]
                    d_con = q_mean[(rep_b, con)] - q_mean[(rep_a, con)]
                    by_scene[scene].append(d_con - d_nat)
                    nat_d[scene].append(d_nat)
                    con_d[scene].append(d_con)
                if not by_scene:
                    continue

                scene_deltas = {s: statistics.mean(v) for s, v in sorted(by_scene.items())}
                n_f = sum(len(v) for v in by_scene.values())
                ineligible = _gates(cells, qt, (rep_a, rep_b), n_f, "matched fact-sets")
                verdict = _verdict(scene_deltas, "", ineligible)

                def _m(d):  # mean over scene means, same unit of replication
                    return statistics.mean([statistics.mean(v) for v in d.values()])

                out.append({
                    "axis_id": axis.id, "qt": qt, "rep_a": rep_a, "rep_b": rep_b,
                    "d_natural": _m(nat_d), "d_constructed": _m(con_d),
                    "scene_deltas": scene_deltas,
                    "mean": statistics.mean(list(scene_deltas.values())),
                    "n_factsets": n_f, "verdict": verdict, "ineligible": ineligible,
                })
    return out


def _matched_section(rows: list[dict], pairs: dict[tuple[str, str], str]) -> list[str]:
    """The matched within-fact-set comparison -- what the balanced corpus buys.

    The re-cut above compares a natural SUBSET against a constructed SUBSET, so any
    difference between them mixes the register manipulation with whatever else the
    two sets of facts differ in. On the scoped types the corpus removes that: each
    information request is authored twice, once in each register, over the same key
    facts and with the same expected answer. Differencing the two within a fact-set
    cancels fact selection exactly, leaving the wording:

        d_nat(f) = AC(rep_b, natural f)     - AC(rep_a, natural f)
        d_con(f) = AC(rep_b, constructed f) - AC(rep_a, constructed f)
        m(f)     = d_con(f) - d_nat(f)

    averaged within each host scene and then over the scene values, which is the
    same unit of replication as every other table here.

    Only the comparisons the cap actually fires on are walked, exactly as in the
    re-cut: an anchor comparison against the floor or the ceiling carries no
    vocabulary confound, so it has nothing to resolve and would only pad the table.

    A fact-set missing any of its four cells is DROPPED, not zero-filled -- mirroring
    the membership test in `_paired_rows`. Zero-filling would read a context overflow
    on one member as `wording made no difference here`, which is the one conclusion
    the missing data cannot support.

    No confound is passed to `_verdict`. The coupling is the estimand in this table,
    not a threat to it, so capping would grade down the very quantity the cap exists
    to point at -- the same reasoning that lifts the cap on the natural half of the
    re-cut. The gates are otherwise untouched: SMALL_N applies to the fact-set count
    (the real number of independent requests), and each rep still has to clear its
    coverage threshold.
    """
    rows_ = _matched_rows(rows, pairs)
    if not rows_:
        return []

    trows: list[list[str]] = []
    for r in rows_:
        trows.append([
            r["axis_id"], r["qt"], f"`{r['rep_b']}` - `{r['rep_a']}`",
            f"{r['d_natural']:+.3f}", f"{r['d_constructed']:+.3f}",
            " / ".join(f"{v:+.3f}" for v in r["scene_deltas"].values()),
            f"{r['mean']:+.3f}", str(r["n_factsets"]),
            r["verdict"] + (f" ({r['ineligible']})" if r["ineligible"] else ""),
        ])

    return (["## Matched fact-sets: does the lead depend on wording? (thesis 4.6)",
             "_Each scoped information request is authored twice, once `natural` and once "
             "`constructed`, over the same key facts and with the same expected answer. "
             "This table differences the two WITHIN each fact-set, so fact selection "
             "cancels and only the register remains -- which the re-cut above cannot do, "
             "since its two halves ask about different facts. `mean` is "
             "d(constructed) - d(natural): positive means the derived pole's lead is "
             "larger when the question is worded in that pole's own vocabulary. A fact-set "
             "with any member unscored is dropped, never zero-filled. Read the verdict as "
             "a statement about the COUPLING, not about the representation: "
             f"'{VERDICT_CONSISTENT}' here means the lead is consistently bigger on "
             f"constructed stems, so the wording is doing work; "
             f"'{VERDICT_NO_SEPARATION}' means the lead does not depend on wording, which "
             "is the result that would clear the derived pole._", ""]
            + _table(["axis", "type", "comparison", "d_natural", "d_constructed",
                      "scene deltas", "mean", "fact-sets", "verdict"], trows) + [""])


def _sibling_groups(results_path: Path) -> list[tuple[str, Path]]:
    """(responder directory name, results.csv) for every other model directory
    holding the same aggregate group.

    Eligibility is four explicit conditions, none of them inferred: the caller is
    an aggregate-group report, the candidate is a different directory, it holds the
    same group, and that group has rows. The judge condition is deliberately NOT
    applied here -- a judge mismatch must be reported as a refusal rather than
    silently filtered, so `_cross_section` applies it where it can be printed.

    Returns [] for a per-scene report: one scene is not the unit of replication, so
    agreement measured there would be a single draw dressed up as a replication.
    Iteration is sorted, so adding a directory can never reorder an existing report.
    """
    if results_path.parent.parent.name != "_aggregate":
        return []
    group = results_path.parent.name
    model_dir, root = results_path.parents[2], results_path.parents[3]
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d == model_dir:
            continue
        p = d / "_aggregate" / group / "results.csv"
        if p.exists():
            out.append((d.name, p))
    return out


# A comparison's state under one responder. A delta below the practical margin has
# NO DIRECTION -- it is not a small win, it is an absence of separation -- so the
# cross-responder reading is a three-state comparison rather than a sign comparison.
# This is what keeps `REVERSED` off noise: deltas of +0.03 and -0.02 disagree in
# sign while neither separates, and that is a null in both, not a failed
# replication. Where a sub-margin delta's raw sign is shown it is called a LEAN,
# never a direction, because naming it a direction would reintroduce the very
# state this three-way split exists to remove.
AHEAD, NULL, BEHIND = 1, 0, -1

CROSS_PRESERVED   = "ordering preserved"
CROSS_REVERSED    = "REVERSED"
CROSS_LEAN_SAME   = "below margin there, leans same way"
CROSS_LEAN_OPPOSE = "below margin there, leans opposite way"
CROSS_ONLY_THERE  = "no separation here, separates there"
CROSS_NULL_BOTH   = "no separation in either"


def _dir_state(d: float) -> int:
    if d >= PRACTICAL_MARGIN:
        return AHEAD
    if d <= -PRACTICAL_MARGIN:
        return BEHIND
    return NULL


def _cross_outcome(d_here: float, d_there: float) -> str:
    """Direction-and-separation state, never sign alone.

    `REVERSED` requires BOTH sides to clear the practical margin in opposite
    directions; nothing else can produce it. Where only `here` separates, the
    sub-margin delta is reported as a lean -- informative, but not a reversal,
    because a delta that does not separate has no direction to reverse.

    Where `here` does not separate the lean is not split at all: this report is
    written from `here`'s perspective, so `here` supplies the reference ordering,
    and with no ordering to preserve there is nothing for `there` to agree with.
    """
    sh, st = _dir_state(d_here), _dir_state(d_there)
    if sh != NULL and st != NULL:
        return CROSS_PRESERVED if sh == st else CROSS_REVERSED
    if sh != NULL:
        return CROSS_LEAN_SAME if (d_there >= 0) == (d_here >= 0) else CROSS_LEAN_OPPOSE
    if st != NULL:
        return CROSS_ONLY_THERE
    return CROSS_NULL_BOTH


# A full `aggregate_results` run renders every group of every directory, and each
# report reads every sibling group -- so without a cache each sibling CSV is parsed
# and run through `_paired_rows` once per reporting directory, which is the whole
# quadratic term. Keyed by (path, mtime, size) rather than path alone: the run
# REWRITES these same files as it proceeds, so a path-only key would serve a report
# the pre-aggregation contents of a directory aggregated later in the same run.
_SIB_ROWS: dict[tuple[str, int, int], list[dict]] = {}
_SIB_PAIRED: dict[tuple[str, int, int], dict] = {}


def _sib_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)


def _read_sibling(path: Path) -> list[dict]:
    key = _sib_key(path)
    if key not in _SIB_ROWS:
        with path.open(encoding="utf-8") as fh:
            _SIB_ROWS[key] = list(csv.DictReader(fh))
    return _SIB_ROWS[key]


def _sibling_paired(path: Path, other_rows: list[dict], pairs) -> dict:
    key = _sib_key(path)
    if key not in _SIB_PAIRED:
        _SIB_PAIRED[key] = {(p.axis_id, p.qt, p.rep_a, p.rep_b): p
                            for p in _paired_rows(other_rows, pairs=pairs)}
    return _SIB_PAIRED[key]


def _cross_section(rows: list[dict], results_path: Path,
                   pairs: dict[tuple[str, str], str] | None = None) -> list[str]:
    """Does each paired comparison point the same way under a second responder?

    Three rules are enforced in code rather than left to the reader.

    *Orderings, never levels.* Each responder's delta is computed inside its own
    directory and only the direction states are compared. Two responders differ in
    general capability for reasons unrelated to representation, so reading the gap
    between their scores would be reading the responder, not the representation --
    in either direction. This holds for any pair of directories, not only the
    scale-matched confirmatory pair of thesis 3.6, since the section renders
    wherever two same-judge directories exist. No absolute correctness crosses the
    boundary.

    *One judge per comparison.* A directory under a different judge is refused and
    the refusal is printed. `feedback_one_judge_per_directory` keeps judges out of a
    single directory; the same confound reappears across directories the moment two
    are compared, and a judge change is not a responder change.

    *Matched evidence.* A comparison is read only where both responders averaged
    over the SAME scenes. Otherwise a disagreement could come from the responder or
    from which scenes entered the mean, and this section exists to isolate the
    first. Mismatches are excluded and listed rather than recomputed on the
    intersection: a recomputed delta would disagree with the same comparison's
    figure in the paired separation table above, and one comparison must not carry
    two different numbers in one document.

    Pair orientation needs no canonicalization. `_paired_pairs` derives it from the
    ladder order in `axes.py`, so for any unordered pair the emitted orientation is
    identical in every directory; and because the join key is the ORDERED tuple, a
    hypothetical flip would drop the comparison out of the shared set rather than
    invert its sign -- an under-report, never a false `REVERSED`. The invariant is
    asserted in the test suite so a future data-derived ordering fails loudly.
    """
    sibs = _sibling_groups(results_path)
    if not sibs:
        return []
    here = {(p.axis_id, p.qt, p.rep_a, p.rep_b): p for p in _paired_rows(rows, pairs=pairs)}
    if not here:
        return []
    judges_here = sorted({r.get("judge", "?") for r in rows})
    responder = rows[0].get("responder", "?")

    # Gather first, render second. Every exclusion is COLLECTED rather than dropped, so
    # the compact summary and the collapsed tables are two views of one computation:
    # nothing is shown that was not computed, and nothing computed is discarded. The
    # display hierarchy is deliberate -- a reader acts on a reversal or a coverage gap,
    # never on a preserved row, so the preserved rows are the part that collapses.
    refused: dict[str, list[str]] = {}
    read: list[dict] = []
    for name, path in sibs:
        other_rows = _read_sibling(path)
        if not other_rows:
            continue
        judges_there = sorted({r.get("judge", "?") for r in other_rows})
        if judges_there != judges_here:
            refused.setdefault(", ".join(f"`{j}`" for j in judges_there), []).append(name)
            continue

        there = _sibling_paired(path, other_rows, pairs)
        shared, mismatched = [], []
        for k in here:
            if k not in there:
                continue
            if set(here[k].scene_deltas) == set(there[k].scene_deltas):
                shared.append(k)
            else:
                # Resolved now, not at render time: the excluded row's whole point is
                # WHICH scenes differed, and `there` is not in scope further down.
                mismatched.append((k, sorted(here[k].scene_deltas),
                                   sorted(there[k].scene_deltas)))
        reps_there = {r["representation"] for r in other_rows}
        missing = sorted({r for k in here if k not in there
                          for r in (k[2], k[3]) if r not in reps_there})
        uncovered = sorted({k[0] for k in here if k not in there} - {k[0] for k in shared})

        trows, tally, reversals = [], collections.Counter(), []
        for k in shared:
            a, b = here[k], there[k]
            outcome = _cross_outcome(a.mean, b.mean)
            tally[outcome] += 1
            trows.append([k[0], k[1], f"`{k[3]}` - `{k[2]}`",
                          f"{a.mean:+.3f}", f"{b.mean:+.3f}",
                          str(len(a.scene_deltas)), outcome])
            if outcome == CROSS_REVERSED:
                reversals.append([f"`{name}`", k[0], k[1], f"`{k[3]}` - `{k[2]}`",
                                  f"{a.mean:+.3f}", f"{b.mean:+.3f}"])
        order = {CROSS_REVERSED: 0, CROSS_LEAN_OPPOSE: 1, CROSS_PRESERVED: 2,
                 CROSS_LEAN_SAME: 3, CROSS_ONLY_THERE: 4, CROSS_NULL_BOTH: 5}
        trows.sort(key=lambda t: (order.get(t[-1], 9), t[0], t[1]))
        reversals.sort(key=lambda t: (t[1], t[2]))
        read.append(dict(name=name, tally=tally, trows=trows, reversals=reversals,
                         mismatched=mismatched, uncovered=uncovered, missing=missing,
                         shared=len(shared)))

    if not read and not refused:
        return []

    out = ["## Cross-responder ordering check (thesis 3.6)", "",
           f"_Holding the evaluated items fixed, does changing the responder preserve the "
           f"ordering within each comparison? A directory is read only when it holds the same "
           f"aggregate group, under an identical judge, on matched scenes -- a judge change is "
           f"not a responder change, so agreement across judges would measure both at once. "
           f"Each responder's delta is computed inside its own directory and classified "
           f"independently as ahead, behind, or no separation at the {PRACTICAL_MARGIN:.2f} "
           f"margin; absolute correctness LEVELS are never compared. Two responders differ in "
           f"general capability for reasons that have nothing to do with representation, so a "
           f"gap between their scores reads the responder rather than the representations -- "
           f"in either direction, whichever is stronger. `{CROSS_REVERSED}` "
           f"requires clear separation in opposite directions; a sub-margin delta has no "
           f"direction at all, and its raw sign is shown only as a lean. Outcomes are reported "
           f"relative to the ordering in this report's responder, `{responder}`. These results "
           f"test pairwise ordering stability, NOT cross-responder generalization, which the "
           f"study does not claim. Every exclusion is printed rather than filtered._", ""]

    if read:
        srows = []
        for x in read:
            t = x["tally"]
            clear = (t[CROSS_PRESERVED] + t[CROSS_REVERSED]
                     + t[CROSS_LEAN_SAME] + t[CROSS_LEAN_OPPOSE])
            srows.append([f"`{x['name']}`", str(x["shared"]), str(clear),
                          str(t[CROSS_PRESERVED]), str(t[CROSS_REVERSED]),
                          str(t[CROSS_LEAN_SAME]), str(t[CROSS_LEAN_OPPOSE]),
                          str(t[CROSS_ONLY_THERE]), str(t[CROSS_NULL_BOTH])])
        # `matched` and `neither` are carried so the row accounts for itself: matched =
        # preserved + reversed + both leans + only there + neither. Without them a reader
        # cannot tell a thin comparison from a broad one that mostly agreed.
        out += _table(["responder", "matched", "clear here", "preserved", "reversed",
                       "weakened same lean", "weakened opposite lean", "only there",
                       "neither"], srows) + [""]

    for judges, names in sorted(refused.items()):
        out += [f"**Refused (judge mismatch):** "
                f"{', '.join(f'`{n}`' for n in sorted(names))} -- scored by {judges}, not this "
                f"directory's {', '.join(f'`{j}`' for j in judges_here)}. Listed rather than "
                f"dropped, so \"not comparable\" stays distinguishable from \"not found\".", ""]

    if not read:
        # Nothing was comparable. The headings below would each assert something about a
        # comparison that never happened -- "no reversals" most of all.
        return out

    revs = [r for x in read for r in x["reversals"]]
    out += ["### Reversals", ""]
    out += (_table(["responder", "axis", "type", "comparison", "d here", "d there"], revs) + [""]
            if revs else
            ["**None.** Every comparison that separates on both responders points the same "
             "way.", ""])

    gaps = [x for x in read if x["uncovered"] or not x["shared"]]
    if gaps:
        out += ["### Coverage gaps", ""]
        for x in gaps:
            if not x["shared"]:
                out += [f"- `{x['name']}`: **no comparison** present in both directories on "
                        f"matched scenes -- a coverage statement, not a result."]
            if x["uncovered"]:
                why = (" -- never run on " + ", ".join(f"`{r}`" for r in x["missing"])
                       if x["missing"] else "")
                out += [f"- `{x['name']}`: no coverage for "
                        + ", ".join(f"`{a}`" for a in x["uncovered"]) + why + "."]
        out += ["", "Nothing about these axes is preserved or reversed at this judge tier. "
                    "Stated explicitly because an absent section otherwise reads as "
                    "agreement.", ""]

    if any(x["mismatched"] for x in read):
        out += ["### Excluded -- evidence not matched", "",
                "Both directories hold these comparisons but averaged them over different "
                "scenes, so a difference would confound the responder with the scene set. "
                "Excluded rather than recomputed on the intersection: a recomputed delta "
                "would disagree with the same comparison's figure in the paired separation "
                "table above, and one comparison must not carry two numbers in one document.",
                ""]
        for x in read:
            out += [f"- `{x['name']}` / {k[0]} / {k[1]} / `{k[3]}` - `{k[2]}`: "
                    f"here {sh}, there {st}" for k, sh, st in x["mismatched"]]
        out += [""]

    detailed = [x for x in read if x["trows"]]
    if detailed:
        out += ["### Full pairwise evidence", ""]
        for x in detailed:
            out += ["<details>",
                    f"<summary>Full pairwise table vs <code>{x['name']}</code> "
                    f"({x['shared']} matched comparisons)</summary>", ""]
            out += _table(["axis", "type", "comparison", "d here", f"d {x['name']}",
                           "scenes (matched)", "outcome"], x["trows"])
            out += ["", "</details>", ""]
    return out


OBJECT_RELATION_TYPES = {"object_relation", "relation_structure", "relation_aggregate"}


def _coverage_section(rows: list[dict]) -> list[str]:
    """Cells where coverage matters: anything that overflowed the window, plus the
    whole object-relation family (`object_relation`/`relation_structure`/
    `relation_aggregate` -- where json_mini/json_pretty/relations_flat overflow on
    dense scenes).
    This is where the survivorship trap is read.

    Split per host dataset (never pooled): overflow is a host property (the JSON
    views only exceed the window on dense 3RScan scenes -- json_pretty on two of the
    three, json_mini on 7f30f36c alone), so a pooled row would average a
    host where a rep is fully scoreable with one where it fails closed and hide
    exactly the survivorship signal this table exists to surface."""
    out: list[str] = []
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
        trows = []
        for (rep, qt), c in sorted(cells.items()):
            if c.context_exceeded > 0 or qt in OBJECT_RELATION_TYPES:
                trows.append([rep, qt, str(c.n), str(c.context_exceeded),
                              f"{c.coverage * 100:.0f}", _ac_str(c),
                              "yes" if c.rank_eligible else "no"])
        if not trows:
            continue
        out += (["## Coverage & rank-eligibility (host: %s)" % ds,
                 "_Cells that overflowed the window or carry object relations. "
                 "rank_eligible = coverage >= 80%; a `no` must not be ranked on AC._", ""]
                + _table(["rep", "type", "n", "exceeded", "cov%", "AC", "rank_eligible"], trows)
                + [""])
    return out


def _small_n_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    trows = [[rep, qt, str(c.n), _ac_str(c)]
             for (rep, qt), c in sorted(cells.items()) if 0 < c.n < SMALL_N]
    if not trows:
        return []
    return (["## Small-n register (n < %d -- screening-only)" % SMALL_N,
             "_These cells are too small to rank on; treat as directional only._", ""]
            + _table(["rep", "type", "n", "AC"], trows) + [""])


# The multi-view-combination section that used to sit here was deleted on
# 2026-08-12 along with the rows it printed. It rendered an ungraded
# difference-of-means table for the two concatenated views; those were superseded
# as a route baseline by `topology_metric` (fact-matched to `navigation` rather
# than a channel superset of it) and are no longer part of the study.


def _candidate_section(rows: list[dict]) -> list[str]:
    """The synthesis candidate-default vs the json_mini ceiling, with token cost -- the
    'match the ceiling at a fraction of the tokens' question.

    Split per host dataset (never pooled): most question types here are asked on
    more than one dataset (aggregation/containment/direction/planning/proximity/
    route/set_logic all appear on 2-3 hosts) with different floor/ceiling scope
    semantics per host (same trap as `_planning_section`), and `synthesis`'s own
    composition is host-dependent (e.g. its object-relation section only fires
    where `has_object_relations` holds) -- a pooled `all` row would silently
    average across incompatible scopes and compositions."""
    out: list[str] = []
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
        qts = sorted({qt for (rep, qt) in cells if rep == CANDIDATE})
        if not qts:
            continue
        trows = []
        for qt in qts:
            s = cells[(CANDIDATE, qt)]
            j = cells.get((CEILING, qt))
            st = f"{s.tokens_mean:.0f}" if s.tokens_mean else "-"
            jt = f"{j.tokens_mean:.0f}" if j and j.tokens_mean else "-"
            trows.append([qt, _ac_str(s), _ac_str(j) if j else "n/a", st, jt])
        out += [f"## Candidate default ({CANDIDATE} vs {CEILING}, host: {ds})",
                f"_Can one compact view match the {CEILING} ceiling at a fraction of the "
                "tokens? Never pooled across datasets._", ""]
        out += _table(["type", f"{CANDIDATE} AC", f"{CEILING} AC",
                       f"{CANDIDATE} tok", f"{CEILING} tok"], trows) + [""]
    return out


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
    judges = sorted({r.get("judge", "?") for r in rows})
    datasets = sorted({dataset_of(r["scene_id"]) for r in rows})
    today = datetime.date.today().isoformat()

    if len(judges) == 1:
        judge = judges[0]
        tier = tier_of(judge)
        judge_line = (f"- judge: `{judge}`  |  tier: **{tier}**"
                      + ("  (ranking only, not reported effect sizes)" if tier == "screening" else ""))
    else:
        # Directory mixes judges (e.g. a headline subset re-judged by Gemini, then a
        # coverage-gap fill run with the default screening judge on newly-added reps).
        # A single global tier would be true for some cells and false for others --
        # naming rows[0]'s judge as if it applied everywhere risks presenting a
        # screening cell as confirmatory. Force the reader to the per-row column.
        tiers = sorted({tier_of(j) for j in judges})
        judge_line = (f"- judges: {', '.join(f'`{j}`' for j in judges)}  |  tiers: **{'/'.join(tiers)}** "
                      "(MIXED -- judge varies by row; check the `judge` column in results.csv "
                      "before citing any cell, never assume confirmatory)")

    lines = [
        f"# Evaluation report - {responder}",
        "",
        judge_line,
        f"- datasets: {', '.join(datasets)}  |  generated: {today}",
        "",
        "> Do NOT read an ALL/ALL grand mean. Every number below is within one "
        "question type, and axis cards are within one host dataset. Compare within "
        "a card, never across types or datasets.",
        "",
    ]
    pairs = _load_pairs()
    lines += _paired_section(rows, pairs)
    lines += _recut_section(rows)
    lines += _matched_section(rows, pairs)
    lines += _cross_section(rows, results_path, pairs)
    for axis in AXES:
        lines += _axis_card(axis, rows)
    lines += _planning_section(rows)
    lines += _coverage_section(rows)
    lines += _small_n_section(cells)
    lines += _candidate_section(rows)
    lines += _detail_section(cells)

    out_path = results_path.parent / "report.md"
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"report -> {out_path}")
    return out_path
