"""Axis and representation-role registry: maps design axes onto representations and their reporting roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# floor/ceiling anchor each axis card; candidate (synthesis) is reported separately since it answers a different question.
FLOOR = "inventory"
# Minified: as denominator of the cost claim, un-minified whitespace would overstate every derived view's saving.
CEILING = "json_mini"
CANDIDATE = "synthesis"

# Multi-view combos (e.g. topology+metric_relations) are retired and purged; neither has a REP_CAPS entry, so a strict run aborts if one is referenced.


def rep_role(rep: str) -> str:
    """Reporting role of a representation (drives where it is shown)."""
    if rep == FLOOR:
        return "floor"
    if rep == CEILING:
        return "ceiling"
    if rep == CANDIDATE:
        return "candidate"
    return "pole"


# n < SMALL_N is screening-only (too noisy to rank); coverage < MIN_COVERAGE is not rank-eligible (AC is conditioned on a survivor subset -- the context_exceeded survivorship trap).
SMALL_N = 6
MIN_COVERAGE = 0.80

# Paired per-scene deltas, no CI (an interval would imply a precision this design cannot support at 3 scenes).
# Consistent advantage = mean delta clears PRACTICAL_MARGIN, no scene runs against it, >= MIN_SCENES_SHOWING agree. Margin is fixed, not preregistered.
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
    # "axis": one of thesis 4.2's five axes. "companion": same axis on a second host. "exhibit": a graded result outside the five axes.
    kind: str = "axis"
    # Declared confound (thesis 4.6), named in the design chapter; caps the verdict at CAPPED_VERDICT regardless of scene consistency. "" = none declared.
    confound: str = ""
    # Empty tuple = confound applies to the whole axis; else only pairs touching one of these reps are capped.
    confound_reps: tuple[str, ...] = ()
    # "vocabulary": wording mirrors the pole's own output, so the natural-question subset is uncoupled and exempt from the cap. "content": no subset removes it.
    confound_kind: str = ""

    def confound_for(self, rep_a: str, rep_b: str) -> str:
        """Declared confound capping this pair's verdict, or "" (floor/ceiling comparisons are exempt -- they're anchor comparisons, not the axis's own contrast)."""
        if not self.confound:
            return ""
        if FLOOR in (rep_a, rep_b) or CEILING in (rep_a, rep_b):
            return ""
        if not self.confound_reps:
            return self.confound
        if rep_a in self.confound_reps or rep_b in self.confound_reps:
            return self.confound
        return ""


# Retired density/source-fidelity axes are absent (named axes have no letter gap to explain). Five entries carry kind="axis" (thesis 4.2); the rest are companions/exhibits.
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
    Axis("format", "Block vs. sentence rendering", "procthor",
         ["topology_inventory", "narrative"],
         ["connectivity"],
         headline_pair=("topology_inventory", "narrative"),
         note="narrative carries exactly topology_inventory's facts (inventory + "
              "connectivity, same rooms, same order) rendered as sentences instead "
              "of labelled blocks -- a pure rendering flip, asserted by "
              "tests/test_format_axis_equivalence.py rather than claimed in prose. "
              "(`prose`, the original partner, was a content superset -- an "
              "explicit per-room object count plus an object-relation section on "
              "ProcTHOR -- which is why this axis now reads narrative instead; "
              "prose keeps its readings elsewhere: the Gibson content_verbosity "
              "exhibit, the relation-linearization family, planning, and as the "
              "synthesis backbone.)"),
    # RETIRED pairing "metric_relations vs navigation" (never fact-matched, ~1.43x length mismatch) -- a pairing retirement, not a representation one; metric_relations keeps its metric_rung role. Replacement lives at `framing_gibson`, below.
    Axis("structure_presentation", "Graph-structure representation", "procthor",
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
    Axis("relation_linearization", "Relation organization and abstraction", "3rscan",
         ["relations_flat", "relations_subject", "relations_predicate",
          "relations_tree", "relations_digest"],
         ["object_relation", "relation_structure", "relation_aggregate"],
         headline_pair=("relations_subject", "relations_predicate"),
         note="same source relation graph exposed through direct, grouped, "
              "selective hierarchical, and derived views; only diverges on dense 3RScan "
              "(near-trivial on ProcTHOR's on-forest). relations_tree/relations_digest "
              "are lossy derived presentations (narrower scope.py channels), so "
              "object_relation/relation_structure/relation_aggregate are read as one "
              "relation-organization probe family, not three separate axes.",
         confound="question vocabulary mirrors relations_digest's own computed "
                  "output (chain depth / clusters); read the natural/constructed "
                  "re-cut and the matched within-fact-set comparison",
         confound_reps=("relations_digest",),
         confound_kind="vocabulary"),
    # connectivity (18 stems/scene) is the well-powered read; route/direction (4/scene) are underpowered checks on it.
    # route has no bearing in any key fact; direction bundles the figure-ground reversal with framing register (not separable). Never egocentric/allocentric.
    Axis("framing", "Relational vs. navigational framing", "procthor",
         ["topology_metric", "navigation"],
         ["route", "direction", "connectivity"],
         headline_pair=("topology_metric", "navigation"),
         kind="axis",
         note="A FACT-MATCHED, LAYOUT-MATCHED framing contrast over one-hop doorway "
              "adjacency, proven fact-for-fact (not inferred from REP_CAPS) by "
              "tests/test_topology_metric_equivalence.py and structurally by "
              "tests/test_framing_layout_isolation.py: same rooms, same edge set, "
              "same per-edge distances, converse bearings, identical header text and "
              "block shape, no route printed on either side, ~12-13% longer on "
              "navigation due to register wording alone (measured 2026-08-21, up from "
              "an ~8% figure under the pre-layout-fix joined-line rendering). Read "
              "`connectivity` first -- the only well-powered type and a post-hoc "
              "(2026-08-13) power check on `route`/`direction` -- then read those two "
              "through it, each under its OWN interpretation (route: a clean isolating "
              "contrast of framing register, no bearing in any key fact; direction: "
              "the figure-ground reversal is task-relevant but not identified, bundled "
              "with framing register -- layout is matched, not part of the bundle). "
              "Never egocentric/allocentric -- see the full note above. This card's "
              "Gibson companion is `framing_gibson`."),
    Axis("framing_gibson", "Relational vs. navigational framing (Gibson companion)",
         "gibson",
         ["metric_framing", "navigation"],
         ["direction"],
         headline_pair=("metric_framing", "navigation"),
         kind="companion",
         note="The Gibson realization of the framing axis, over K-nearest-neighbour "
              "room geometry rather than doorway adjacency. Fact-matched AND "
              "layout-matched by an independently-derived equivalence test "
              "(tests/test_metric_framing_equivalence.py: both poles checked against "
              "K-NN geometry recomputed fresh from source room positions, not just "
              "against each other -- Gibson has no door graph to serve as independent "
              "ground truth the way ProcTHOR's does, so this is a milder but different "
              "caveat than the content mismatch this companion replaces, not the same "
              "one) and structurally by tests/test_framing_layout_isolation.py: "
              "identical header text and block shape, same room and neighbour order. "
              "Length-matched to within ~1% (measured 2026-08-21) -- tighter than the "
              "ProcTHOR pair, whose longer 'you can walk to' clause adds more overhead "
              "than this pair's shorter proximity wording. Same residual bundle as the "
              "ProcTHOR axis's `direction` reading (figure-ground + register, layout "
              "matched); same terminology discipline (never egocentric/allocentric). "
              "`direction` is the only in-scope type on this host -- no door graph "
              "means no connectivity/route questions exist here -- so there is no "
              "power check available for this reading and 12 questions total is its "
              "ceiling, not a choice. Supersedes the retired `metric_relations` vs "
              "`navigation` pairing (see the retirement comment above `framing`, "
              "up-file) -- `metric_relations` is unretired and keeps its unrelated "
              "spatial-encoding role."),
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
    # Declared BEFORE any json_mini cells existed, so neither outcome can be read
    # post-hoc. The two members are the same parse() output under two
    # serializations, so they are matched on every channel by construction --
    # hence no confound (confound_for exempts any pair containing CEILING
    # anyway), and no headline_pair: a single-rung ladder plus the ceiling anchor
    # already yields exactly the one pair, so a bar would restate the card.
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

# host/probe_types ride along deliberately: filtering on pole names alone would admit any question both poles happen to support, wider than the axis actually declares.
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
    """Host dataset from a scene id; anything not procthor_*/3rscan_* falls into gibson (Gibson scenes are named, not prefixed)."""
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
