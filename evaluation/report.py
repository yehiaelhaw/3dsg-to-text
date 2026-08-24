"""Evaluation report with coverage, eligibility, and paired scene-level verdicts."""

from __future__ import annotations

import collections
import csv
import datetime
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from evaluation.axes import (
    AXES, AXIS_BY_ID, CANDIDATE, CAPPED_VERDICT, FULL_RECORD_ANCHOR, MIN_COVERAGE,
    MIN_OBSERVATIONS, MIN_SCENES_SHOWING, NON_SPATIAL_ANCHOR, PRACTICAL_MARGIN,
    VERDICT_CONSISTENT, VERDICT_DIRECTIONAL, VERDICT_MIXED,
    VERDICT_NO_SEPARATION, VERDICT_NOT_LICENSED,
    dataset_of, rep_role, tier_of,
)
from evaluation.core import is_context_exceeded
from evaluation.scope import in_scope

# Load QA pair_ids for matched fact-set analysis.
QA_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "qa"
QA_FILENAME = "keyfact-qa.jsonl"


class MixedComparisonError(ValueError):
    """Raised when comparative results mix responders or judges."""


def require_single_responder_judge(rows: list[dict]) -> None:
    """Require exactly one nonblank responder and judge."""
    responders = {(r.get("responder") or "").strip() for r in rows}
    judges = {(r.get("judge") or "").strip() for r in rows}
    if len(responders) != 1 or len(judges) != 1 or "" in responders or "" in judges:
        raise MixedComparisonError(
            "Thesis comparative outputs require one responder and one judge per results.csv."
        )


def _load_pairs(qa_root: Path | None = None) -> dict[tuple[str, str], str]:
    """Load (scene_id, qid) -> pair_id; return {} if QA files are unavailable."""
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
    """Return (scene_id, authored qid), stripping aggregate scene prefixes."""
    scene, qid = r["scene_id"], r["question_id"]
    prefix = f"{scene}:"
    return scene, (qid[len(prefix):] if qid.startswith(prefix) else qid)


@dataclass
class Cell:
    """Display statistics for one representation × question-type cell."""
    n: int
    error_count: int
    context_exceeded: int
    n_scored: int
    ac_mean: float
    # Descriptive per-question min-max; not an interval or verdict input.
    ac_range: tuple[float, float]
    coverage: float
    tokens_mean: float | None
    # Supporting-detail diagnostic; separate from primary correctness.
    detail_mean: float | None
    n_detail: int

    @property
    def rank_eligible(self) -> bool:
        return self.n >= MIN_OBSERVATIONS and self.n_scored > 0 and self.coverage >= MIN_COVERAGE


def _cell(rows: list[dict]) -> Cell:
    n = len(rows)
    context_exceeded = sum(1 for r in rows if is_context_exceeded(r["error"]))
    error_count = sum(1 for r in rows if r["error"] and not is_context_exceeded(r["error"]))
    # Scored rows require no error and a real AC value.
    scored = [r for r in rows
              if not r["error"] and r["answer_correctness"] not in ("", None)]
    n_scored = len(scored)
    # Average repetitions within question before cell summaries.
    ac_by_q: dict[str, list[float]] = collections.defaultdict(list)
    for r in scored:
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
    ac_mean = statistics.mean(q_means) if q_means else 0.0
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
    """Format AC and its failing eligibility gate."""
    if c.n_scored == 0:
        return "n/a"
    s = f"{c.ac_mean:.2f}"
    if c.n < MIN_OBSERVATIONS:
        s += f"* (n={c.n})"
    elif c.coverage < MIN_COVERAGE:
        s += f"* (cov {c.coverage * 100:.0f}%)"
    return s


def _range_str(c: Cell) -> str:
    """Format descriptive per-question min-max; not an interval."""
    lo, hi = c.ac_range
    return f"{lo:.2f}-{hi:.2f}" if c.n_scored > 1 else ""


def _tok_str(p) -> str:
    """Format matched token means and tok_n coverage."""
    if not p.n_token_pairs:
        return "-"
    a, b = p.tokens_a, p.tokens_b
    return f"{b:.0f} vs {a:.0f} ({b / a:.2f}x; tok_n={p.n_token_pairs}/{p.n_questions})"


def _axis_reps(axis, qt: str, cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Return available reps in anchor/ladder order; [] if fewer than two."""
    reps: list[str] = []
    if (NON_SPATIAL_ANCHOR, qt) in cells:
        reps.append(NON_SPATIAL_ANCHOR)
    for p in axis.ladder:
        if p in (NON_SPATIAL_ANCHOR, FULL_RECORD_ANCHOR):
            continue
        if in_scope(p, qt, axis.host) and (p, qt) in cells:
            reps.append(p)
    if (FULL_RECORD_ANCHOR, qt) in cells and FULL_RECORD_ANCHOR not in reps:
        reps.append(FULL_RECORD_ANCHOR)
    return reps if len(reps) >= 2 else []


def _axis_card(axis, rows: list[dict]) -> list[str]:
    """Render one host-scoped axis card."""
    cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == axis.host])
    # Anchor lifts are question-matched, not differences of cell means.
    lifts = {(p.qt, p.rep_b): p for p in non_spatial_anchor_lifts(rows, axis.host)}
    body: list[str] = []
    for qt in axis.probe_types:
        reps = _axis_reps(axis, qt, cells)
        if not reps:
            continue
        anchor = cells.get((NON_SPATIAL_ANCHOR, qt))
        has_anchor = bool(anchor and anchor.n_scored)
        trows = []
        for rep in reps:
            c = cells[(rep, qt)]
            d_anchor = ("+0.00" if rep == NON_SPATIAL_ANCHOR and has_anchor
                        else _lift_str(lifts.get((qt, rep))))
            trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                          _ac_str(c), _range_str(c), d_anchor])
        body.append(f"**{qt}** (host: {axis.host})")
        body += _table(["rep", "role", "n", "cov%", "AC", "q-range", "vs non-spatial anchor"], trows)
        body.append("")
    if not body:
        return []
    kind = "" if axis.kind == "axis" else f" ({axis.kind})"
    return [f"## {axis.label}{kind}", f"_{axis.note}_", ""] + body


def _planning_section(rows: list[dict]) -> list[str]:
    """Render planning separately by host; planning is not an axis pole."""

    def order(cells, rep):
        """Order planning rows by role, eligibility, then AC."""
        c = cells[(rep, "planning")]
        block = {"non_spatial_anchor": 0, "full_record_anchor": 2}.get(rep_role(rep), 1)
        return (block, 0 if c.rank_eligible else 1, -c.ac_mean)

    out: list[str] = []
    for ds in sorted({dataset_of(r["scene_id"]) for r in rows}):
        cells = _cells([r for r in rows if dataset_of(r["scene_id"]) == ds])
        reps = sorted({rep for (rep, qt) in cells if qt == "planning"})
        if not reps:
            continue
        lifts = {(p.qt, p.rep_b): p for p in non_spatial_anchor_lifts(rows, ds)}
        anchor = cells.get((NON_SPATIAL_ANCHOR, "planning"))
        has_anchor = bool(anchor and anchor.n_scored)
        trows = []
        for rep in sorted(reps, key=lambda r: order(cells, r)):
            c = cells[(rep, "planning")]
            d_anchor = ("+0.00" if rep == NON_SPATIAL_ANCHOR and has_anchor
                        else _lift_str(lifts.get(("planning", rep))))
            trows.append([rep, rep_role(rep), str(c.n), f"{c.coverage * 100:.0f}",
                          _ac_str(c), _range_str(c), d_anchor])
        license_note = (
            "Formal planning representation comparisons are licensed only on 3RScan; "
            "planning is a separate real-world-utility probe, not a spatial axis."
            if ds == "3rscan" else
            "Descriptive planning scores only; no representation-level planning "
            "comparison is licensed on this host.")
        out += [f"## Planning / real-world utility (separate probe, host: {ds})",
                f"_Goal-framed planning probe; separate by host, not an axis. {license_note}_", ""]
        out += _table(["rep", "role", "n", "cov%", "AC", "q-range", "vs non-spatial anchor"], trows) + [""]
    return out


# --- paired separation analysis ---
# Formal verdicts come from the paired analysis below.

@dataclass
class Paired:
    """One paired comparison; scene_deltas are equally weighted scene means of matched-question deltas."""
    axis_id: str
    host: str
    qt: str
    rep_a: str
    rep_b: str
    scene_deltas: dict[str, float]
    n_questions: int
    verdict: str
    confound: str
    capped: bool
    ineligible: str
    # Matched prompt-token means for the paired questions.
    tokens_a: float | None = None
    tokens_b: float | None = None
    # Correctness-paired qids with token data on both reps.
    n_token_pairs: int = 0
    # Distinct information requests; display only, never a gate.
    n_factsets: int = 0
    # Per-question deltas for plots; verdicts use scene_deltas.
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
    """Apply the declared scene-level verdict and confound-cap rules."""
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
        # Confounds only downgrade consistent results.
        return CAPPED_VERDICT if confound else VERDICT_CONSISTENT
    return VERDICT_DIRECTIONAL


def _paired_pairs(axis, qt: str, cells: dict[tuple[str, str], Cell]) -> list[tuple[str, str]]:
    """Return declared axis contrasts in earlier-to-later order."""
    rungs = [p for p in axis.ladder
             if p not in (NON_SPATIAL_ANCHOR, FULL_RECORD_ANCHOR) and in_scope(p, qt, axis.host) and (p, qt) in cells]
    pairs: list[tuple[str, str]] = []
    if rungs:
        base = rungs[0]
        pairs += [(base, r) for r in rungs[1:]]
    if axis.headline_pair:
        a, b = axis.headline_pair
        if (a, qt) in cells and (b, qt) in cells and in_scope(a, qt, axis.host) and in_scope(b, qt, axis.host):
            pairs.append((a, b))
    if (FULL_RECORD_ANCHOR, qt) in cells:
        pairs += [(r, FULL_RECORD_ANCHOR) for r in rungs]
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        if p not in seen and p[0] != p[1]:
            seen.add(p)
            out.append(p)
    return out


def _q_means(host_rows: list[dict]) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """Return per-question AC means and qid->scene; omit unscored rows."""
    ac: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    scene_of: dict[str, str] = {}
    for r in host_rows:
        if r["error"] or r["answer_correctness"] in ("", None):
            continue
        ac[(r["representation"], r["question_id"])].append(float(r["answer_correctness"]))
        scene_of[r["question_id"]] = r["scene_id"]
    return {k: statistics.mean(v) for k, v in ac.items()}, scene_of


def _q_token_means(host_rows: list[dict]) -> dict[tuple[str, str], float]:
    """Return per-question token means, averaging repetitions first."""
    toks: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in host_rows:
        if r["error"]:
            continue
        t = r.get("prompt_tokens")
        if t in ("", "0", None):
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        toks[(r["representation"], r["question_id"])].append(v)
    return {k: statistics.mean(v) for k, v in toks.items()}


def _paired_token_summary(q_tok: dict[tuple[str, str], float], qids,
                          rep_a: str, rep_b: str) -> tuple[float | None, float | None, int]:
    """Average tokens on correctness-paired qids available for both reps."""
    a_vals, b_vals = [], []
    for q in sorted(set(qids)):
        if (rep_a, q) in q_tok and (rep_b, q) in q_tok:
            a_vals.append(q_tok[(rep_a, q)])
            b_vals.append(q_tok[(rep_b, q)])
    if not a_vals:
        return None, None, 0
    return statistics.mean(a_vals), statistics.mean(b_vals), len(a_vals)


def paired_deltas(q_mean: dict[tuple[str, str], float], scene_of: dict[str, str],
                  qids, rep_a: str, rep_b: str) -> dict[str, list[tuple[float, str]]]:
    """Return scene-grouped matched AC deltas (rep_b-rep_a); drop unmatched qids."""
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


def _gates(cells: dict[tuple[str, str], Cell], qt: str, reps: tuple[str, ...]) -> str:
    """Return failed cell-level gates; paired/fact-set support never gates."""
    reasons = []
    for rep in reps:
        c = cells.get((rep, qt))
        if c is None or c.rank_eligible:
            continue
        if c.n < MIN_OBSERVATIONS:
            reasons.append(f"{rep} n={c.n} (< {MIN_OBSERVATIONS})")
        else:
            reasons.append(f"{rep} coverage {c.coverage * 100:.0f}%")
    return "; ".join(reasons)


def _paired_rows(rows: list[dict], style: str | None = None, axes=None,
                 lift_confound: bool = False,
                 pairs: dict[tuple[str, str], str] | None = None) -> list[Paired]:
    """Build paired axis rows; style filtering occurs before eligibility checks."""
    out: list[Paired] = []
    for axis in (AXES if axes is None else axes):
        host_rows = [r for r in rows if dataset_of(r["scene_id"]) == axis.host]
        if style is not None:
            host_rows = [r for r in host_rows if (r.get("question_style") or "") == style]
        if not host_rows:
            continue
        cells = _cells(host_rows)
        q_mean, scene_of = _q_means(host_rows)
        q_tok = _q_token_means(host_rows)
        # qid -> (scene, pair_id) for fact-set support.
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
                paired_qids = [q for items in by_scene.values() for _, q in items]
                tokens_a, tokens_b, n_token_pairs = _paired_token_summary(
                    q_tok, paired_qids, rep_a, rep_b)
                ineligible = _gates(cells, qt, (rep_a, rep_b))
                confound = "" if lift_confound else axis.confound_for(rep_a, rep_b)
                verdict = _verdict(scene_deltas, confound, ineligible)
                # capped is true only when the confound changed a consistent verdict.
                capped = bool(confound) and verdict == CAPPED_VERDICT and (
                    _verdict(scene_deltas, "", ineligible) == VERDICT_CONSISTENT)
                out.append(Paired(axis.id, axis.host, qt, rep_a, rep_b, scene_deltas,
                                  n_q, verdict, confound, capped, ineligible,
                                  tokens_a, tokens_b, n_token_pairs,
                                  len(factsets), q_deltas))
    return out


# --- the anchor comparison ---------------------------------------------------
# Inventory lift is a separate anchor comparison.
NON_SPATIAL_ANCHOR_LIFT_ID = "non_spatial_anchor"


def non_spatial_anchor_lifts(rows: list[dict], host: str) -> list[Paired]:
    """Build matched rep-vs-inventory comparisons for one host."""
    host_rows = [r for r in rows if dataset_of(r["scene_id"]) == host]
    if not host_rows:
        return []
    cells = _cells(host_rows)
    q_mean, scene_of = _q_means(host_rows)
    q_tok = _q_token_means(host_rows)
    qids_by_type: dict[str, set[str]] = collections.defaultdict(set)
    for r in host_rows:
        qids_by_type[r["question_type"] or "unknown"].add(r["question_id"])

    out: list[Paired] = []
    for qt in sorted(qids_by_type):
        if (NON_SPATIAL_ANCHOR, qt) not in cells:
            continue
        reps = sorted({rep for (rep, t) in cells
                       if t == qt and rep != NON_SPATIAL_ANCHOR and in_scope(rep, qt, host)})
        for rep in reps:
            by_scene = paired_deltas(q_mean, scene_of, qids_by_type[qt], NON_SPATIAL_ANCHOR, rep)
            if not by_scene:
                continue
            scene_deltas, n_q, q_deltas = _summarise(by_scene)
            paired_qids = [q for items in by_scene.values() for _, q in items]
            tokens_a, tokens_b, n_token_pairs = _paired_token_summary(
                q_tok, paired_qids, NON_SPATIAL_ANCHOR, rep)
            ineligible = _gates(cells, qt, (NON_SPATIAL_ANCHOR, rep))
            out.append(Paired(NON_SPATIAL_ANCHOR_LIFT_ID, host, qt, NON_SPATIAL_ANCHOR, rep, scene_deltas, n_q,
                              _verdict(scene_deltas, "", ineligible), "", False,
                              ineligible,
                              tokens_a, tokens_b, n_token_pairs,
                              0, q_deltas))
    return out


def _lift_str(p: Paired | None) -> str:
    """Format matched lift or its eligibility reason."""
    if p is None:
        return ""
    if p.verdict == VERDICT_NOT_LICENSED:
        return f"{VERDICT_NOT_LICENSED} ({p.ineligible})"
    return f"{p.mean:+.2f}"


def _paired_section(rows: list[dict], pairs=None) -> list[str]:
    """Render the formal paired-separation table."""
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
    return (["## Paired separation",
             f"_Scene-first matched AC deltas; verdicts use the declared margin rules. "
             f"Tokens are qid-matched (tok_n=X/Y); fact-sets count distinct requests._", ""]
            + _table(["axis", "type", "comparison", "scene deltas", "mean", "range",
                      "scenes", "n_q", "fact-sets", "tokens (b vs a)", "verdict", "note"],
                     trows) + [""])


STYLES = ("natural", "constructed")


def _recut_rows(rows: list[dict]) -> dict[tuple[str, str, str, str], dict[str, "Paired"]]:
    """Return vocabulary-confound comparisons split by question style."""
    if not any((r.get("question_style") or "") for r in rows):
        return {}   # No question-style metadata available.
    coupled = [a for a in AXES if a.confound_kind == "vocabulary"]
    if not coupled:
        return {}

    # Lift the vocabulary cap for natural wording only.
    by_style = {s: _paired_rows(rows, style=s, axes=coupled,
                                lift_confound=(s == "natural")) for s in STYLES}
    keyed: dict[tuple[str, str, str, str], dict[str, Paired]] = collections.defaultdict(dict)
    for s in STYLES:
        for p in by_style[s]:
            # Re-cut only comparisons with a vocabulary confound.
            axis = AXIS_BY_ID[p.axis_id]
            if not axis.confound_for(p.rep_a, p.rep_b):
                continue
            keyed[(p.axis_id, p.qt, p.rep_a, p.rep_b)][s] = p
    return keyed


def _recut_section(rows: list[dict]) -> list[str]:
    """Render natural/constructed re-cut; each subset keeps its own gates."""
    keyed = _recut_rows(rows)
    if not keyed:
        return []

    trows = []
    for (axis_id, qt, rep_a, rep_b) in sorted(keyed):
        for s in STYLES:
            p = keyed[(axis_id, qt, rep_a, rep_b)].get(s)
            if p is None:
                # Missing style subsets are absent, not null.
                trows.append([axis_id, qt, f"`{rep_b}` - `{rep_a}`", s,
                              "-", "-", "0", "no scored questions of this style"])
                continue
            trows.append([axis_id, qt, f"`{rep_b}` - `{rep_a}`", s,
                          " / ".join(f"{v:+.3f}" for v in p.scene_deltas.values()),
                          f"{p.mean:+.3f}", str(p.n_questions),
                          p.verdict + (f" ({p.ineligible})" if p.ineligible else "")])
    return (["## Vocabulary re-cut: natural vs constructed (thesis 4.6)",
             "_Split vocabulary-confounded comparisons by natural/constructed wording; each subset keeps its own gates._", ""]
            + _table(["axis", "type", "comparison", "style", "scene deltas", "mean",
                      "n_q", "verdict"], trows) + [""])


def _matched_rows(rows: list[dict], pairs: dict[tuple[str, str], str]) -> list[dict]:
    """Compute matched within-fact-set wording effects."""
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

        # (scene, pair_id, type) -> {style: qid}
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
                ineligible = _gates(cells, qt, (rep_a, rep_b))
                verdict = _verdict(scene_deltas, "", ineligible)

                def _m(d):
                    # Unweighted mean of scene means.
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
    """Render d_constructed-d_natural within fact-set; drop incomplete sets."""
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
             "_Within each fact-set, mean = d_constructed - d_natural; incomplete sets are dropped._", ""]
            + _table(["axis", "type", "comparison", "d_natural", "d_constructed",
                      "scene deltas", "mean", "fact-sets", "verdict"], trows) + [""])


def _sibling_groups(results_path: Path) -> list[tuple[str, Path]]:
    """Find sibling responder results; judge filtering happens downstream."""
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


# Sub-margin deltas are NULL; their raw sign is only a lean.
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
    """Classify ordering; reversal requires opposite margin-clearing deltas."""
    sh, st = _dir_state(d_here), _dir_state(d_there)
    if sh != NULL and st != NULL:
        return CROSS_PRESERVED if sh == st else CROSS_REVERSED
    if sh != NULL:
        return CROSS_LEAN_SAME if (d_there >= 0) == (d_here >= 0) else CROSS_LEAN_OPPOSE
    if st != NULL:
        return CROSS_ONLY_THERE
    return CROSS_NULL_BOTH


# Cache by path/mtime/size so rewritten results invalidate entries.
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
    """Compare pairwise ordering across responders with the same judge and scenes."""
    sibs = _sibling_groups(results_path)
    if not sibs:
        return []
    here = {(p.axis_id, p.qt, p.rep_a, p.rep_b): p for p in _paired_rows(rows, pairs=pairs)}
    if not here:
        return []
    judges_here = sorted({r.get("judge", "?") for r in rows})
    responder = rows[0].get("responder", "?")

    # Collect exclusions before rendering so summary and details use the same evidence.
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
                # Store mismatched scene sets before `other_rows` leaves scope.
                mismatched.append((k, sorted(here[k].scene_deltas),
                                   sorted(there[k].scene_deltas)))
        reps_there = {r["representation"] for r in other_rows}
        missing = sorted({r for k in here if k not in there
                          for r in (k[2], k[3]) if r not in reps_there})
        # Keep matched/neither counts so summary totals reconcile.
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

    out = ["## Cross-responder ordering check", "",
           "_Same judge/scenes only; reversal requires opposite margin-clearing deltas. "
           "Ordering-stability diagnostic only._", ""]

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
        # matched = preserved + reversed + both leans + only there + neither.
        out += _table(["responder", "matched", "clear here", "preserved", "reversed",
                       "weakened same lean", "weakened opposite lean", "only there",
                       "neither"], srows) + [""]

    for judges, names in sorted(refused.items()):
        out += [f"**Refused (judge mismatch):** "
                f"{', '.join(f'`{n}`' for n in sorted(names))} -- judge: {judges}; expected: "
                f"{', '.join(f'`{j}`' for j in judges_here)}.", ""]

    if not read:
        return out

    revs = [r for x in read for r in x["reversals"]]
    out += ["### Reversals", ""]
    out += (_table(["responder", "axis", "type", "comparison", "d here", "d there"], revs) + [""]
            if revs else
            ["**None.** All comparisons separating on both responders preserve direction.", ""])

    gaps = [x for x in read if x["uncovered"] or not x["shared"]]
    if gaps:
        out += ["### Coverage gaps", ""]
        for x in gaps:
            if not x["shared"]:
                out += [f"- `{x['name']}`: no matched comparison."]
            if x["uncovered"]:
                why = (" -- never run on " + ", ".join(f"`{r}`" for r in x["missing"])
                       if x["missing"] else "")
                out += [f"- `{x['name']}`: no coverage for "
                        + ", ".join(f"`{a}`" for a in x["uncovered"]) + why + "."]
        out += ["", "No ordering claim for uncovered axes.", ""]

    if any(x["mismatched"] for x in read):
        out += ["### Excluded -- evidence not matched", "",
                "Excluded because responders used different scene sets; no intersection recomputation.",
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
    """Render host-scoped overflow and object-relation coverage."""
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
                 "_Overflow/object-relation cells; eligible iff coverage >= 80% and n >= 6._", ""]
                + _table(["rep", "type", "n", "exceeded", "cov%", "AC", "rank_eligible"], trows)
                + [""])
    return out


def _small_n_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    trows = [[rep, qt, str(c.n), _ac_str(c)]
             for (rep, qt), c in sorted(cells.items()) if 0 < c.n < MIN_OBSERVATIONS]
    if not trows:
        return []
    return (["## Small-n register (n < %d -- descriptive only)" % MIN_OBSERVATIONS,
             "_These cells are below the reporting threshold; inspect descriptively only "
             "and assign no comparative verdict._", ""]
            + _table(["rep", "type", "n", "AC"], trows) + [""])


# --- candidate audit ---
# Fixed candidate comparison: synthesis - json_mini.
CANDIDATE_AUDIT_ID = "candidate"


def candidate_rows(rows: list[dict]) -> list[Paired]:
    """Build the fixed candidate-vs-json_mini paired audit."""
    out: list[Paired] = []
    for host in sorted({dataset_of(r["scene_id"]) for r in rows}):
        host_rows = [r for r in rows if dataset_of(r["scene_id"]) == host]
        cells = _cells(host_rows)
        cand_qts = {qt for (rep, qt) in cells if rep == CANDIDATE}
        anchor_qts = {qt for (rep, qt) in cells if rep == FULL_RECORD_ANCHOR}
        qts = sorted(cand_qts & anchor_qts)
        if not qts:
            continue
        q_mean, scene_of = _q_means(host_rows)
        q_tok = _q_token_means(host_rows)
        for qt in qts:
            qids = {q for (rep, q) in q_mean
                    if any(r["question_id"] == q and (r["question_type"] or "unknown") == qt
                           for r in host_rows)}
            by_scene = paired_deltas(q_mean, scene_of, qids, FULL_RECORD_ANCHOR, CANDIDATE)
            if not by_scene:
                continue
            scene_deltas, n_q, q_deltas = _summarise(by_scene)
            paired_qids = [q for items in by_scene.values() for _, q in items]
            tokens_a, tokens_b, n_token_pairs = _paired_token_summary(
                q_tok, paired_qids, FULL_RECORD_ANCHOR, CANDIDATE)
            ineligible = _gates(cells, qt, (FULL_RECORD_ANCHOR, CANDIDATE))
            verdict = _verdict(scene_deltas, "", ineligible)
            out.append(Paired(CANDIDATE_AUDIT_ID, host, qt, FULL_RECORD_ANCHOR, CANDIDATE,
                              scene_deltas, n_q, verdict, "", False, ineligible,
                              tokens_a, tokens_b, n_token_pairs, 0, q_deltas))
    return out


def _candidate_reading(p: Paired) -> str:
    """Translate the candidate-audit verdict and matched cost evidence."""
    if p.verdict == VERDICT_CONSISTENT:
        if p.mean > 0:
            return "consistent correctness advantage for synthesis"
        if p.mean < 0:
            return "consistent correctness advantage for json_mini"
        return "consistent correctness advantage"
    if p.verdict == VERDICT_DIRECTIONAL:
        return "directional correctness evidence only; no efficiency conclusion"
    if p.verdict == VERDICT_MIXED:
        return "scene-dependent correctness result; no efficiency conclusion"
    if p.verdict == VERDICT_NOT_LICENSED:
        return "comparative claim not licensed"
    complete = (p.n_token_pairs == p.n_questions and p.n_questions > 0
                and p.tokens_a is not None and p.tokens_b is not None)
    if not complete:
        return "no correctness separation; token-cost reading incomplete"
    if p.tokens_b < p.tokens_a:
        return "no correctness separation; synthesis is the cheaper alternative"
    if p.tokens_b > p.tokens_a:
        return "no correctness separation; json_mini is the cheaper alternative"
    return "no correctness separation; no token-cost separation"


def _candidate_section(rows: list[dict]) -> list[str]:
    """Render the host/type synthesis-vs-json_mini paired audit."""
    prs = candidate_rows(rows)
    if not prs:
        return []
    out: list[str] = []
    for host in sorted({p.host for p in prs}):
        trows = []
        for p in sorted((p for p in prs if p.host == host), key=lambda p: p.qt):
            lo, hi = p.spread
            trows.append([
                p.qt, f"`{CANDIDATE}` - `{FULL_RECORD_ANCHOR}`",
                " / ".join(f"{v:+.3f}" for v in p.scene_deltas.values()),
                f"{p.mean:+.3f}", f"{lo:+.3f}..{hi:+.3f}", str(len(p.scene_deltas)),
                str(p.n_questions), _tok_str(p), p.verdict, _candidate_reading(p),
            ])
        out += [f"## Candidate audit ({CANDIDATE} vs {FULL_RECORD_ANCHOR}, host: {host})",
                f"_Fixed candidate comparison: synthesis - json_mini. "
                f"Cost reading requires no correctness separation and complete matched-token coverage (tok_n=X/Y)._", ""]
        out += _table(["type", "comparison", "scene deltas", "mean", "range", "scenes",
                       "n_q", f"tokens ({CANDIDATE} vs {FULL_RECORD_ANCHOR})", "verdict",
                       "audit reading"], trows) + [""]
    return out


def _detail_section(cells: dict[tuple[str, str], Cell]) -> list[str]:
    """Render supporting-detail diagnostics; never treat them as correctness."""
    trows = [[rep, qt, str(c.n_detail), f"{c.detail_mean:.2f}"]
             for (rep, qt), c in sorted(cells.items())
             if c.n_detail > 0 and c.detail_mean is not None]
    if not trows:
        return []
    return (["## Supporting-detail coverage (diagnostic -- NOT correctness)",
             "_Supporting-detail diagnostic only; not correctness and not ranked._", ""]
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
    require_single_responder_judge(rows)

    cells = _cells(rows)
    responder = rows[0].get("responder", "?")
    judge = rows[0].get("judge", "?")
    tier = tier_of(judge)
    judge_line = (f"- judge: `{judge}`  |  tier: **{tier}**"
                  + ("  (candidate-selection tier only)" if tier == "screening" else ""))
    datasets = sorted({dataset_of(r["scene_id"]) for r in rows})
    today = datetime.date.today().isoformat()

    lines = [
        f"# Evaluation report - {responder}",
        "",
        judge_line,
        f"- datasets: {', '.join(datasets)}  |  generated: {today}",
        "",
        "> No ALL/ALL mean: compare only within question type and host.",
        "",
    ]
    pairs = _load_pairs()
    lines += _paired_section(rows, pairs)
    lines += _recut_section(rows)
    lines += _matched_section(rows, pairs)
    # Cross-responder check is auxiliary and omitted from the default report.
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
