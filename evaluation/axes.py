"""axes.py — axis + representation-role registry (the reporting backbone).

Single source of truth for *how the design axes map onto representations*: which
reps form each axis ladder, on which host dataset, probing which question types,
plus each rep's reporting role (floor / ceiling / candidate / pole).

This is distinct from `scope.py`: scope says which rep *can answer* which type
(the structural mask); this says how those reps are *grouped and ordered for
reporting* (the axis story). `plots.py` and `report.py` both read it so the same
axis ladder is told the same way — ladder order, floor/ceiling anchored — in every
artifact. Keeping it here (not duplicated in plots.py) means a new parser joins the
story in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# --- reporting roles -------------------------------------------------------
# floor and ceiling anchor every axis card (they bound the band a pole sits in).
# The candidate (synthesis) answers a *different* question than the axis poles, so
# it is reported in its own tables, never mixed into a card.
FLOOR = "inventory"
# The complete-information anchor of the primary ladder, minified (2026-08-09).
# CEILING bounds what a derived view can *express*, not what can score highest:
# json_pretty carries identical information and is graded against it on the
# json_formatting cards. Minifying is a cost-accounting correction -- the ceiling's
# token cost is the denominator of the headline claim, so charging it for
# pretty-print whitespace overstates every derived view's apparent saving.
CEILING = "json_mini"
CANDIDATE = "synthesis"

# Multi-view combinations (`topology+metric_relations`, `graph_digest+metric_relations`)
# were retired from the argument on 2026-08-11 and PURGED from the results tree on
# 2026-08-12: their rows are gone, and so is every reporting role, table and plot
# branch that existed to file them. There is no archived heading any more, and no
# `ARCHIVED_COMBOS` list -- nothing here is combo-aware, because nothing is left to
# be aware of. The names are declared in scope.RETIRED_REPS so a reappearance aborts
# rather than being silently reported as a pole.
#
# They went because no distinct research question required a concatenation any
# more. The one that did was the ProcTHOR route baseline -- a route question needs
# connectivity and metric_edges together, and no single view carried both, so the
# baseline had to be assembled. `topology_metric` now carries exactly that pair by
# design and is matched to `navigation` fact-for-fact, which is strictly better:
# the combo was a channel SUPERSET of navigation (it also carried object
# inventories, stated twice, and metric between unconnected rooms), so its delta
# confounded the contrast with that surplus. The combos' other question -- "can
# salient views match the json_mini ceiling?" -- is already owned by `synthesis`,
# a curated single document rather than a concatenation.


def rep_role(rep: str) -> str:
    """Reporting role of a representation (drives where it is shown)."""
    if rep == FLOOR:
        return "floor"
    if rep == CEILING:
        return "ceiling"
    if rep == CANDIDATE:
        return "candidate"
    return "pole"


# --- reporting thresholds (policy shared by plots.py and report.py) --------
# A cell with n < SMALL_N is screening-only (its mean is too noisy to rank on);
# a cell whose coverage (n_scored / n) is below MIN_COVERAGE is not rank-eligible
# (its AC is conditioned on the surviving subset, so it can't be compared on AC
# alone against a full-coverage cell -- the context_exceeded survivorship trap).
SMALL_N = 6
MIN_COVERAGE = 0.80

# --- separation rule (thesis 4.6) -------------------------------------
# Comparisons are PAIRED and the SCENE is the unit of replication: both members
# of a pair are evaluated on the same scene-question instances, so they are
# compared on their per-question difference, averaged within each host scene.
# No confidence interval is computed and interval overlap is never a decision
# rule -- at three scenes and a handful of questions per cell an interval would
# imply a precision this design cannot support. Spreads are descriptive only.
#
# A pair is a CONSISTENT ADVANTAGE when the overall mean of the scene-level
# differences clears PRACTICAL_MARGIN in absolute value, no scene runs against
# that direction, and at least MIN_SCENES_SHOWING of them show it. A scene whose
# difference is exactly zero does not contradict a direction, but neither does it
# supply one -- hence two conditions rather than one.
#
# The margin is fixed for all comparisons and is deliberately coarser than the
# granularity of one question's AC: a question carries 1-4 core facts, so a
# single fact changing hands moves a small cell by more than a finer margin
# would tolerate. It is NOT preregistered -- it was set once preliminary results
# already existed -- so it lives here, applied uniformly to every regenerated
# report, precisely so that no single pair can be graded under a margin picked
# for it.
PRACTICAL_MARGIN = 0.10
MIN_SCENES_SHOWING = 2

# Verdict vocabulary (thesis 4.6, Table 4.5). CAPPED_VERDICT is what a
# declared confound downgrades an otherwise-consistent result to.
VERDICT_CONSISTENT = "consistent advantage"
VERDICT_DIRECTIONAL = "directional"
VERDICT_MIXED = "mixed"
VERDICT_NO_SEPARATION = "no practically meaningful separation"
VERDICT_NOT_LICENSED = "not licensed"
CAPPED_VERDICT = VERDICT_DIRECTIONAL


@dataclass(frozen=True)
class Axis:
    id: str                       # descriptive slug ("spatial_encoding", ...): the
                                  # machine key for filenames (axis_card_<id>.png) and
                                  # AXIS_BY_ID. Axes are named, not lettered -- there is
                                  # no A/B/D/F/G scheme (and thus no confusing C/E gap).
    label: str                    # short human label
    host: str                     # primary host dataset (procthor | 3rscan | gibson)
    ladder: list[str]             # reps in rung/ladder order (poles; floor/ceiling
                                  # are anchored separately by the renderer)
    probe_types: list[str]        # question types this axis is read on
    headline_pair: tuple[str, str] | None = None  # within-axis contrast for the
                                  # paired-delta chart; None = pure ladder, no pair
    note: str = ""                # caveat surfaced in the card
    # What this entry IS, so a reader holding report.md next to the design chapter
    # can tell the five representation axes from the extra cards. "axis": one of
    # the five (thesis 4.2). "companion": the same axis measured on a second host,
    # carded separately because cards are single-host. "exhibit": a graded result
    # that is not an axis contrast at all -- it borrows the card/paired machinery
    # so a finding the design axes do not cover still gets a verdict under the
    # same rules instead of being quoted as an anecdote.
    kind: str = "axis"
    # A DECLARED confound (thesis 4.6): named in the design chapter, not
    # discovered in the results, and it caps the verdict at CAPPED_VERDICT
    # however consistent the scene values are. Empty string = none declared.
    confound: str = ""
    # Which reps the confound attaches to. Empty tuple = the whole axis (the
    # format pair's content superset); otherwise only pairs touching one of
    # these reps are capped (the derived poles' vocabulary coupling).
    confound_reps: tuple[str, ...] = ()
    # What KIND of confound, which decides whether the natural/constructed re-cut
    # can resolve it. "vocabulary": the question's wording mirrors a derived pole's
    # own printed output, so the `natural` subset -- questions a user could have
    # asked without ever seeing that output -- is by construction uncoupled and the
    # cap does not apply there. "content": one member simply carries more content
    # than the other (prose superset of topology_inventory), which no question-style split
    # addresses, so the cap stands on every subset. Only "vocabulary" axes are
    # re-cut (thesis 4.6 / 6.7 name the two axes with a derived pole).
    confound_kind: str = ""

    def confound_for(self, rep_a: str, rep_b: str) -> str:
        """The declared confound capping this pair's verdict, or "".

        Only the axis's OWN contrast can be capped. A comparison against the
        floor or the ceiling is an anchor comparison, not the design decision
        this axis isolates, so neither the format pair's content superset nor a
        derived pole's vocabulary coupling is a confound there.
        """
        if not self.confound:
            return ""
        if FLOOR in (rep_a, rep_b) or CEILING in (rep_a, rep_b):
            return ""
        if not self.confound_reps:
            return self.confound
        if rep_a in self.confound_reps or rep_b in self.confound_reps:
            return self.confound
        return ""


# The order/content mirrors METHODOLOGY 1 and AXIS_CONTRAST. The retracted density
# and source-fidelity axes (see those docs) are intentionally absent -- with named
# axes there is no letter gap to explain. The spatial-encoding axis is a ladder, not
# a single contrast pair, so it carries no headline_pair (its result is read as lift
# over the floor / where the peak sits, not a two-pole delta).
#
# Five entries carry kind="axis" -- the five representation axes of thesis 4.2. The
# rest are companions and exhibits (see Axis.kind); they are cards, not axes, and the
# renderers say so, so the count here never has to be reconciled against that five.
AXES: list[Axis] = [
    Axis("spatial_encoding", "Spatial encoding", "procthor",
         ["inventory", "topology_inventory", "metric_relations", "json_mini"],
         ["connectivity", "proximity", "direction"],
         note="ladder is per-type: only in-scope rungs are drawn; inventory is the "
              "spatial-prior floor, not a competitor. See the Spatial encoding "
              "(metric rung) card for the Gibson metric-rung companion (axis cards "
              "are single-host)."),
    Axis("metric_rung", "Spatial encoding (metric rung)", "gibson",
         ["metric_relations"],
         ["proximity"],
         note="Gibson companion to the spatial-encoding axis: the cleanest metric-rung "
              "exhibit in the study lives on this host, not ProcTHOR (a consistent "
              "advantage over the json_mini ceiling across all three scenes), so it "
              "needs its own card rather than being folded into the main "
              "spatial-encoding card.",
         kind="companion"),
    Axis("format", "Structured vs. prose presentation", "procthor",
         ["topology_inventory", "prose"],
         ["connectivity"],
         headline_pair=("topology_inventory", "prose"),
         note="prose is a content superset of topology_inventory (adds the "
              "object-relation section) -- not a pure syntax flip; account for the "
              "extra content.",
         confound="prose is a content superset of topology_inventory, so the pair "
                  "is not a pure structured/prose flip",
         confound_reps=("prose",),
         confound_kind="content"),
    Axis("reference_frame", "Spatial anchoring", "gibson",
         ["metric_relations", "navigation"],
         # direction only. `route` used to be declared here and was dead: all 12
         # route questions are ProcTHOR's (Gibson has no door graph, which is
         # exactly why this axis is hosted here), so the entry named a probe this
         # axis can never be read on. The ProcTHOR route result is a separate
         # exhibit below and is deliberately NOT a spatial-anchoring reading.
         ["direction"],
         headline_pair=("metric_relations", "navigation"),
         note="pure anchoring isolation only on Gibson; on ProcTHOR navigation also "
              "restricts to doorway moves, so it fuses connectivity (coverage view)."),
    Axis("structure_presentation", "Structure presentation", "procthor",
         ["topology", "room_tree", "graph_digest"],
         ["connectivity"],
         headline_pair=("topology", "graph_digest"),
         note="same door graph, three content-matched presentations (all "
              "connectivity-only): raw adjacency -> drawn tree -> derived "
              "structure. Full topology_inventory (with inventories) stays on the "
              "spatial-encoding/format axes; topology_inventory vs topology reads "
              "as a distractor-content contrast, not part of this ladder.",
         confound="question vocabulary mirrors graph_digest's own computed output "
                  "(hub / bottleneck); read the natural/constructed re-cut and the "
                  "matched within-fact-set comparison",
         confound_reps=("graph_digest",),
         confound_kind="vocabulary"),
    Axis("relation_linearization", "Relation organization", "3rscan",
         ["relations_flat", "relations_subject", "relations_predicate",
          "relations_tree", "relations_digest"],
         ["object_relation", "relation_structure", "relation_aggregate"],
         headline_pair=("relations_subject", "relations_predicate"),
         note="same edge set, five presentations; only diverges on dense 3RScan "
              "(near-trivial on ProcTHOR's on-forest). relations_tree/relations_digest "
              "are lossy derived presentations (narrower scope.py channels), so "
              "object_relation/relation_structure/relation_aggregate are read as one "
              "relation-organization probe family, not three separate axes.",
         confound="question vocabulary mirrors relations_digest's own computed "
                  "output (chain depth / clusters); read the natural/constructed "
                  "re-cut and the matched within-fact-set comparison",
         confound_reps=("relations_digest",),
         confound_kind="vocabulary"),
    # --- exhibits: graded results the five axes do not cover ------------------
    # NAMING: this used to be "route_presentation", which invited the reading that
    # one pole prints routes. NEITHER DOES. Both print the same one-hop doorway
    # adjacency and nothing else -- no path, no hop count, no summed distance. What
    # differs is the FRAMING of that adjacency: navigational/action-oriented ("from
    # X you can walk to Y, 2.9 m north-east") vs relational/locative ("X is 2.9 m
    # south-west of Y"). "route" survives only as the question type it is read on.
    Axis("route_framing", "Navigational vs relational framing (route)", "procthor",
         ["topology_metric", "navigation"],
         ["route"],
         headline_pair=("topology_metric", "navigation"),
         kind="exhibit",
         note="A FACT-MATCHED framing contrast over one-hop doorway adjacency. "
              "Neither pole prints a route: both are doorway-restricted, carry exactly "
              "{connectivity, metric_edges}, and state the same rooms, the same edge "
              "set and the same per-edge distances in the same order, so the multi-hop "
              "search and the addition a route question asks for stay the model's work "
              "on BOTH sides. Fact-for-fact equality is asserted by "
              "tests/test_topology_metric_equivalence.py, not inferred from REP_CAPS, "
              "and the pair is approximately length-matched too (topology_metric is "
              "1.076-1.082x navigation across the three scenes on the rendered "
              "pre-question prompt -- representation-side cost including the "
              "chat/system wrapping, excluding the question, which is identical for "
              "both), so a delta is not a context-length effect. WHAT DIFFERS is not "
              "one thing: which room is the "
              "located figure, the block layout, AND the lexical/semantic framing "
              "(traversal 'you can walk to' vs locative 'is ... of'). Report it as a "
              "framing contrast, not as presentation-only or layout-only -- the "
              "manipulation is a bundle and this design cannot decompose it. DO NOT "
              "attribute a route delta to the figure-ground reversal specifically: an "
              "audit of all 12 route stems found no bearing in any key fact, so the "
              "task never reads the direction. Not a reference-frame result: see the "
              "terminology note on relation_direction, and the reference-frame axis "
              "stays on Gibson. (Supersedes an earlier concatenated baseline that was a "
              "strict channel superset of navigation -- it also carried object "
              "inventories and unconnected-pair metric, and ran 4.5x navigation on the "
              "same measure, so its delta was confounded twice over. That baseline was "
              "excluded from the analysis and its rows removed from the results tree.)"),
    Axis("relation_direction", "Relation direction (direct neighbours)", "procthor",
         ["topology_metric", "navigation"],
         ["direction"],
         headline_pair=("topology_metric", "navigation"),
         kind="exhibit",
         note="The same matched pair read on the family where the relation's direction "
              "IS task-relevant. All 12 ProcTHOR direction stems ask about a directly "
              "connected pair, so both doorway-restricted poles can answer every one, "
              "and the key facts are phrased from the heading room -- which navigation "
              "prints directly and topology_metric states as the converse, so the "
              "latter must invert it. Equal information, unequal work: that is the "
              "contrast. NOT IDENTIFIED, though: it is the same two texts as "
              "route_framing, so the inversion arrives bundled with the layout and "
              "lexical-framing differences, whose contributions could run in either "
              "direction and cannot be separated from it here. A direction delta "
              "reflects the bundled manipulation. It is NOT a bound on the cost of the "
              "inversion in either direction -- calling it an upper bound would assume "
              "the other components can only help, which nothing here establishes. "
              "TERMINOLOGY: this is NOT an "
              "egocentric/allocentric flip. Both "
              "poles print world-frame compass bearings and no ProcTHOR room carries a "
              "facing, so there is no observer orientation to be egocentric about; "
              "what differs is figure-ground assignment (which room is the subject of "
              "a converse-equivalent binary relation) plus framing register. Report it "
              "under that description. The reference-frame axis remains Gibson's, and "
              "this exhibit is not a second reading of it."),
    # --- power check on the matched pair --------------------------------------
    # DECLARED POST-HOC (2026-08-13), after these cells were generated and judged.
    # Unlike json_formatting below, this entry CANNOT claim its outcome was
    # unreadable in advance, and it is not offered as independent confirmation of
    # anything. It exists because route_framing and relation_direction turned out
    # too thin to carry a verdict alone -- 12 stems over 3 scenes is 4 per scene,
    # so one answer changing hands moves a scene delta by 0.125, the same size as
    # the effects those two exhibits grade. The connectivity cells were already
    # generated (topology_metric carries the connectivity channel, so they are in
    # scope) and were sitting unread. Disclose the post-hoc declaration wherever
    # this card is quoted.
    Axis("adjacency_framing", "Framing on plain adjacency (power check)", "procthor",
         ["topology_metric", "navigation"],
         ["connectivity"],
         headline_pair=("topology_metric", "navigation"),
         kind="exhibit",
         note="The SAME matched pair as route_framing and relation_direction, read on "
              "the only well-powered type in this design: 18 connectivity stems per "
              "scene (54 total, 27 fact-sets) against 4 per scene for route and "
              "direction. Its job is to say how much of those two exhibits' spread is "
              "sampling noise, so read it FIRST and read them through it. WHAT IT "
              "TESTS: a connectivity stem asks only which rooms are joined. Both poles "
              "print the same edge set in the same order, so neither the figure-ground "
              "assignment nor the per-edge metric is task-relevant here -- the framing "
              "manipulation has no channel to act through. NO SEPARATION is therefore "
              "the expected and uninteresting outcome, and it is what five of the six "
              "responder directories show. A separation here would NOT strengthen the "
              "other two exhibits; it would mean the manipulation is doing something "
              "they do not attribute to it, and would need explaining before either is "
              "quoted. DECLARED POST-HOC on 2026-08-13, after the cells existed: this "
              "card is a power check on exhibits already run, not a preregistered "
              "contrast, and must be cited as such."),
    Axis("content_verbosity", "Verbosity on content questions", "gibson",
         ["prose"],
         ["containment", "aggregation", "set_logic"],
         kind="exhibit",
         note="Content-only questions: they need the inventory channel and nothing "
              "else, so prose and json_mini are matched on everything the question uses "
              "and json_mini's remaining channels are pure distractor -- which makes "
              "this a clean verbosity reading rather than a format one. Read it that "
              "way: on set_logic and containment prose ties the inventory floor exactly "
              "(see the `vs floor` column), so what separates is json_mini's cost, not "
              "prose's form; only aggregation puts prose above the floor. "
              "metric_relations is held out of the ladder despite scoring well here -- "
              "it carries channels the questions do not need, which is the superset "
              "confound this card exists without."),
    # --- formatting ablation: json_pretty vs the json_mini ceiling -------------
    # Declared 2026-08-09 alongside the ceiling migration, BEFORE any json_mini
    # cells existed, so neither outcome can be read post-hoc. The two members are
    # the same parse() output under two serializations, so they are matched on
    # every channel by construction -- hence no confound (confound_for exempts
    # any pair containing CEILING anyway), and no headline_pair: a single-rung
    # ladder plus the ceiling anchor already yields exactly the one pair, so a
    # bar would restate the card.
    #
    # (That reason used to be joined by a second one -- AXIS_PAIRS dropped `host`,
    # so axis_contrasts.png pooled every host and a bar here would have dragged
    # 3RScan back in. AXIS_PAIRS now carries the host and the pair is filtered to
    # it, so that hazard is gone; the restatement argument above is what still
    # withholds the headline_pair.)
    #
    # 3RScan is deliberately not hosted: json_pretty is CONTEXT_EXCEEDED on two of
    # its three scenes, so its coverage there falls under MIN_COVERAGE and the pair
    # is not rank-eligible. The host filter is the belt; that gate is the braces.
    #
    # probe_types is every non-planning question type on the host -- a stated rule,
    # not a chosen subset, which is what a two-sided null-hypothesis ablation needs
    # (planning has its own report section). All counts clear SMALL_N.
    Axis("json_formatting", "JSON formatting (pretty vs minified)", "procthor",
         ["json_pretty"],
         ["connectivity", "direction", "route", "aggregation", "proximity",
          "set_logic", "containment"],
         kind="exhibit",
         note="Same parse() output under two serializations, so the members carry "
              "identical information and any separation is an accessibility effect of "
              "FORMATTING, not of content. Interpretation is fixed in advance and is "
              "two-sided. json_mini is the canonical full-information serialization "
              "for COST ACCOUNTING; that decision is not a claim about accuracy, so a "
              "json_pretty win does not unsettle it -- it would mean whitespace aids "
              "access to the same facts, and what must then be dropped is any "
              "description of json_mini as an empirical upper bound on accuracy across "
              "serializations (it bounds INFORMATION, a different claim). No "
              "separation, or a json_mini win, is reported plainly with no implied "
              "vindication. Neither direction was predicted."),
    Axis("json_formatting_gibson", "JSON formatting (Gibson)", "gibson",
         ["json_pretty"],
         ["aggregation", "proximity", "set_logic", "direction", "containment"],
         kind="companion",
         note="Gibson companion to the JSON formatting exhibit (cards are "
              "single-host); same construction, same two-sided reading. 3RScan hosts "
              "no companion: json_pretty overflows the responder window on two of its "
              "three scenes, so the pair is not rank-eligible there."),
]

AXIS_BY_ID = {a.id: a for a in AXES}

# Re-derived for plots.axis_contrasts (paired per-question AC delta). Only axes with
# a headline_pair contribute (the spatial-encoding axis is a ladder, so it has none).
# plots.py imports this rather than maintaining its own copy.
#
# `host` and `probe_types` ride along and are NOT optional decoration: a headline
# pair is only defined on the questions its axis declares. Carrying just the two
# pole names loses that, and the consumer then has nothing to filter on but the
# structural scope mask -- which admits every host and every type both poles happen
# to support. That is strictly wider than the axis, and the surplus is not noise: it
# mixed ProcTHOR questions into the Gibson-hosted reference-frame contrast and
# pulled `planning` -- ungraded everywhere -- into two axes, none of it visible on
# the chart. It also erases any distinction between two axes that share a pair and
# differ only in the type they declare: they become one number drawn twice.
# Anything reading AXIS_PAIRS must filter on both fields; the question set is part
# of the comparison's definition, not a display preference.
class AxisPair(NamedTuple):
    label: str                    # bar label (Axis.label)
    a: str                        # first pole (subtrahend: the delta is b - a)
    b: str                        # second pole
    host: str                     # the ONLY host dataset this pair is defined on
    probe_types: tuple[str, ...]  # the ONLY question types it is read on


AXIS_PAIRS = [
    AxisPair(a.label, a.headline_pair[0], a.headline_pair[1], a.host,
             tuple(a.probe_types))
    for a in AXES if a.headline_pair
]


def dataset_of(scene_id: str) -> str:
    """Host dataset of a scene from its committed id (METHODOLOGY 1.2). Gibson
    scenes are named Brinnon/Thrall/Donaldson, so anything not procthor_*/3rscan_*
    falls into gibson. Handles the pooled `scene_id` column (still the bare scene
    id; aggregate_results namespaces only `question_id`)."""
    s = (scene_id or "").lower()
    if s.startswith("procthor"):
        return "procthor"
    if s.startswith("3rscan"):
        return "3rscan"
    return "gibson"


def tier_of(judge_model: str) -> str:
    """Reporting tier from the judge model. Strong cloud judges (Gemini/GPT/Claude)
    produce confirmatory numbers; local screening judges (gemma2/qwen/...) produce
    exploratory ones reliable for ranking but not for reported effect sizes."""
    j = (judge_model or "").lower()
    if any(k in j for k in ("gemini", "gpt", "claude", "anthropic", "openai")):
        return "confirmatory"
    return "screening"
