"""axes.py — axis + representation-role registry (the reporting backbone).

Single source of truth for *how the design axes map onto representations*: which
reps form each axis ladder, on which host dataset, probing which question types,
plus each rep's reporting role (floor / ceiling / candidate / combo / pole).

This is distinct from `scope.py`: scope says which rep *can answer* which type
(the structural mask); this says how those reps are *grouped and ordered for
reporting* (the axis story). `plots.py` and `report.py` both read it so the same
axis ladder is told the same way — ladder order, floor/ceiling anchored — in every
artifact. Keeping it here (not duplicated in plots.py) means a new parser joins the
story in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- reporting roles -------------------------------------------------------
# floor and ceiling anchor every axis card (they bound the band a pole sits in).
# candidate (synthesis) and combos answer *different* questions than the axis
# poles, so they are reported in their own tables, never mixed into a card.
FLOOR = "inventory"
CEILING = "json"
CANDIDATE = "synthesis"


def rep_role(rep: str) -> str:
    """Reporting role of a representation (drives where it is shown)."""
    if "+" in rep:
        return "combo"
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
    # than the other (prose superset of topology), which no question-style split
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
         ["inventory", "topology", "metric_relations", "json"],
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
              "advantage over json across all three scenes), so it needs its own card "
              "rather than being folded into the main spatial-encoding card.",
         kind="companion"),
    Axis("format", "Format", "procthor",
         ["topology", "prose"],
         ["connectivity"],
         headline_pair=("topology", "prose"),
         note="prose is a content superset of topology (adds the object-relation "
              "section) -- not a pure syntax flip; account for the extra content.",
         confound="prose is a content superset of topology, so the pair is not a "
                  "pure format flip",
         confound_reps=("prose",),
         confound_kind="content"),
    Axis("reference_frame", "Reference frame", "gibson",
         ["metric_relations", "navigation"],
         # direction only. `route` used to be declared here and was dead: all 12
         # route questions are ProcTHOR's (Gibson has no door graph, which is
         # exactly why this axis is hosted here), so the entry named a probe this
         # axis can never be read on. The ProcTHOR route result is a separate
         # exhibit below and is deliberately NOT a reference-frame reading.
         ["direction"],
         headline_pair=("metric_relations", "navigation"),
         note="pure-frame isolation only on Gibson; on ProcTHOR navigation also "
              "restricts to doorway moves, so it fuses connectivity (coverage view)."),
    Axis("structure_presentation", "Structure presentation", "procthor",
         ["topology_edges_only", "room_tree", "graph_digest"],
         ["connectivity"],
         headline_pair=("topology_edges_only", "graph_digest"),
         note="same door graph, three content-matched presentations (all "
              "connectivity-only): raw adjacency -> drawn tree -> derived "
              "structure. Full topology (with inventories) stays on the spatial-"
              "encoding/format axes; topology vs topology_edges_only reads as a "
              "distractor-content contrast, not part of this ladder.",
         confound="question vocabulary mirrors graph_digest's own computed output "
                  "(hub / bottleneck); read the natural/constructed re-cut and the "
                  "matched within-fact-set comparison",
         confound_reps=("graph_digest",),
         confound_kind="vocabulary"),
    Axis("relation_linearization", "Relation linearization", "3rscan",
         ["relations_flat", "relations_subject", "relations_predicate",
          "relations_tree", "relations_digest"],
         ["object_relation", "relation_structure", "relation_aggregate"],
         headline_pair=("relations_subject", "relations_predicate"),
         note="same edge set, five presentations; only diverges on dense 3RScan "
              "(near-trivial on ProcTHOR's on-forest). relations_tree/relations_digest "
              "are lossy derived presentations (narrower scope.py channels), so "
              "object_relation/relation_structure/relation_aggregate are read as one "
              "relation-linearization probe family, not three separate axes.",
         confound="question vocabulary mirrors relations_digest's own computed "
                  "output (chain depth / clusters); read the natural/constructed "
                  "re-cut and the matched within-fact-set comparison",
         confound_reps=("relations_digest",),
         confound_kind="vocabulary"),
    # --- exhibits: graded results the five axes do not cover ------------------
    Axis("route_presentation", "Route presentation", "procthor",
         ["topology+metric_relations", "navigation"],
         ["route"],
         headline_pair=("topology+metric_relations", "navigation"),
         kind="exhibit",
         note="NOT a reference-frame result -- that axis stays on Gibson, because on "
              "ProcTHOR navigation restricts to doorway moves and so fuses "
              "connectivity; this contrast is presentation-with-frame, not vantage "
              "alone. The baseline is a COMBO by necessity: a route question needs "
              "connectivity and metric_edges together and no single pole carries both "
              "(topology has no metric, metric_relations has no connectivity), so the "
              "doors-with-distances combination is navigation's only content-matched "
              "allocentric counterpart. It is a strict channel superset of navigation, "
              "and it LOSES -- the extra content cannot explain the gap, so nothing is "
              "capped. Probed on route only: navigation states bearings solely along "
              "connections, so admitting `direction` would readmit the content gap this "
              "study hosts the reference-frame axis on Gibson to avoid. Question "
              "wording shares framing with navigation ('walk', 'connected rooms' -- "
              "topology says 'connect' too), but no representation prints a route or a "
              "total distance, so the multi-hop search and the sum are the model's."),
    Axis("content_verbosity", "Verbosity on content questions", "gibson",
         ["prose"],
         ["containment", "aggregation", "set_logic"],
         kind="exhibit",
         note="Content-only questions: they need the inventory channel and nothing "
              "else, so prose and json are matched on everything the question uses and "
              "json's remaining channels are pure distractor -- which makes this a "
              "clean verbosity reading rather than a format one. Read it that way: on "
              "set_logic and containment prose ties the inventory floor exactly (see "
              "the `vs floor` column), so what separates is json's cost, not prose's "
              "form; only aggregation puts prose above the floor. metric_relations is "
              "held out of the ladder despite scoring well here -- it carries channels "
              "the questions do not need, which is the superset confound this card "
              "exists without."),
]

AXIS_BY_ID = {a.id: a for a in AXES}

# Re-derived for plots.axis_contrasts (paired per-question AC delta). Each entry is
# (label, first_pole, second_pole); only axes with a headline_pair contribute (the
# spatial-encoding axis is a ladder, so it has none). plots.py imports this rather
# than maintaining its own copy.
AXIS_PAIRS = [
    (a.label, a.headline_pair[0], a.headline_pair[1])
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
