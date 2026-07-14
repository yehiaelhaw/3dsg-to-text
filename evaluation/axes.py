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


@dataclass(frozen=True)
class Axis:
    id: str                       # "A", "B", ...
    label: str                    # short human label
    host: str                     # primary host dataset (procthor | 3rscan | gibson)
    ladder: list[str]             # reps in rung/ladder order (poles; floor/ceiling
                                  # are anchored separately by the renderer)
    probe_types: list[str]        # question types this axis is read on
    headline_pair: tuple[str, str] | None = None  # within-axis contrast for the
                                  # paired-delta chart; None = pure ladder, no pair
    note: str = ""                # caveat surfaced in the card


# The order/content mirrors METHODOLOGY 1 and AXIS_CONTRAST. Axis C and E are
# retracted (see those docs) and intentionally absent. Axis A is a ladder, not a
# single contrast pair, so it carries no headline_pair (its result is read as lift
# over the floor / where the peak sits, not a two-pole delta).
AXES: list[Axis] = [
    Axis("A", "Spatial encoding", "procthor",
         ["inventory", "topology", "metric_relations", "json"],
         ["connectivity", "proximity", "direction"],
         note="ladder is per-type: only in-scope rungs are drawn; inventory is the "
              "spatial-prior floor, not a competitor. See Axis A2 for the Gibson "
              "metric-rung companion card (axis cards are single-host)."),
    Axis("A2", "Spatial encoding (metric rung)", "gibson",
         ["metric_relations"],
         ["proximity"],
         note="Gibson companion to Axis A: the cleanest metric-rung exhibit in the "
              "study (no CI overlap vs json) lives on this host, not ProcTHOR, so it "
              "needs its own card rather than being folded into Axis A."),
    Axis("B", "Format", "procthor",
         ["topology", "prose"],
         ["connectivity"],
         headline_pair=("topology", "prose"),
         note="prose is a content superset of topology (adds the object-relation "
              "section) -- not a pure syntax flip; account for the extra content."),
    Axis("D", "Reference frame", "gibson",
         ["metric_relations", "navigation"],
         ["direction", "route"],
         headline_pair=("metric_relations", "navigation"),
         note="pure-frame isolation only on Gibson; on ProcTHOR navigation also "
              "restricts to doorway moves, so it fuses connectivity (coverage view)."),
    Axis("F", "Structure presentation", "procthor",
         ["topology_edges_only", "room_tree", "graph_digest"],
         ["connectivity"],
         headline_pair=("topology_edges_only", "graph_digest"),
         note="same door graph, three content-matched presentations (all "
              "connectivity-only): raw adjacency -> drawn tree -> derived "
              "structure. Full topology (with inventories) stays on axes A/B; "
              "topology vs topology_edges_only reads as a distractor-content "
              "contrast, not part of this ladder."),
    Axis("G", "Relation linearization", "3rscan",
         ["relations_flat", "relations_subject", "relations_predicate",
          "relations_tree", "relations_digest"],
         ["object_relation", "relation_structure", "relation_aggregate"],
         headline_pair=("relations_subject", "relations_predicate"),
         note="same edge set, five presentations; only diverges on dense 3RScan "
              "(near-trivial on ProcTHOR's on-forest). relations_tree/relations_digest "
              "are lossy derived presentations (narrower scope.py channels), so "
              "object_relation/relation_structure/relation_aggregate are read as one "
              "axis-G probe family, not three separate axes."),
]

AXIS_BY_ID = {a.id: a for a in AXES}

# Re-derived for plots.axis_contrasts (paired per-question AC delta). Each entry is
# (label, first_pole, second_pole); only axes with a headline_pair contribute (A is
# a ladder). plots.py imports this rather than maintaining its own copy.
AXIS_PAIRS = [
    (f"{a.id} {a.label}", a.headline_pair[0], a.headline_pair[1])
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
