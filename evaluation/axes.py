"""Axis and representation-role registry: maps design axes onto representations and their reporting roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# Reference anchors used on every axis; synthesis is reported separately.
NON_SPATIAL_ANCHOR = "inventory"
FULL_RECORD_ANCHOR = "json_mini"  # Minified; formatting whitespace inflates apparent savings
CANDIDATE = "synthesis"

# Excluded from thesis-facing paired/anchor/ladder/plot rendering.
# Not part of the final evaluated taxonomy; raw rows are untouched.
NOT_EVALUATED = {"prose", "synthesis"}


def rep_role(rep: str) -> str:
    """Reporting role used by the rendering/reporting layer."""
    if rep == NON_SPATIAL_ANCHOR:
        return "non_spatial_anchor"
    if rep == FULL_RECORD_ANCHOR:
        return "full_record_anchor"
    if rep == CANDIDATE:
        return "candidate"
    return "pole"


# Fewer than MIN_OBSERVATIONS or coverage below MIN_COVERAGE fails the reporting gate.
# The coverage gate avoids accuracy estimates based only on non-overflow survivors.
MIN_OBSERVATIONS = 6
MIN_COVERAGE = 0.80

# Scene is the replication unit; with only three scenes, no CI is reported.
# Consistent advantage: |mean delta| >= PRACTICAL_MARGIN, no reversed scene,
# and at least MIN_SCENES_SHOWING scenes in that direction. The margin is predeclared.
PRACTICAL_MARGIN = 0.10
MIN_SCENES_SHOWING = 2

# Declared confounds cap otherwise-consistent results at directional.
VERDICT_CONSISTENT = "consistent advantage"
VERDICT_DIRECTIONAL = "directional"
VERDICT_MIXED = "mixed"
VERDICT_NO_SEPARATION = "no practically meaningful separation"
VERDICT_NOT_LICENSED = "not licensed"
CAPPED_VERDICT = VERDICT_DIRECTIONAL


@dataclass(frozen=True)
class Axis:
    id: str                       # machine key for filenames and AXIS_BY_ID
    label: str                    # report/card label
    host: str                     # host dataset for this reading
    ladder: list[str]             # representations in ladder order
    probe_types: list[str]        # question types licensed for this reading
    headline_pair: tuple[str, str] | None = None  # paired-delta contrast; None for a pure ladder
    note: str = ""                # short interpretation caveat surfaced in the card
    # Shared group key for other readings of the same underlying axis; they nest
    # as subsections under one heading instead of each getting a top-level one.
    family: str = ""
    # True for a non-axis reading with no `family`; nests under a shared
    # "Further readings" heading so it's never miscounted as a sixth axis.
    secondary: bool = False
    # Declared confound; caps an otherwise-consistent verdict at CAPPED_VERDICT.
    confound: str = ""
    # Empty = whole axis; otherwise only pairs touching these representations.
    confound_reps: tuple[str, ...] = ()
    # "vocabulary" allows the natural-question exemption; "content" does not.
    confound_kind: str = ""

    def confound_for(self, rep_a: str, rep_b: str) -> str:
        """Return the declared confound for this pair, if any.

        Reference-anchor comparisons are exempt.
        """
        if not self.confound:
            return ""
        if NON_SPATIAL_ANCHOR in (rep_a, rep_b) or FULL_RECORD_ANCHOR in (rep_a, rep_b):
            return ""
        if not self.confound_reps:
            return self.confound
        if rep_a in self.confound_reps or rep_b in self.confound_reps:
            return self.confound
        return ""


# Five thesis axes; remaining entries are further readings grouped under one of them
# (see `family`).
AXES: list[Axis] = [
    Axis("spatial_encoding", "Spatial encoding", "procthor",
         ["inventory", "topology_inventory", "metric_relations", "json_mini"],
         ["connectivity", "proximity", "direction"],
         note="Per-question-type ladder; anchors are reference only, not performance bounds."),
    Axis("metric_rung", "Spatial encoding (metric rung)", "gibson",
         ["metric_relations"],
         ["proximity"],
         note="Gibson reading of the spatial-encoding metric rung on `proximity`, "
              "using room-level metric geometry.",
         family="Spatial encoding"),
    Axis("format", "Formatting", "procthor",
         ["topology_inventory", "narrative"],
         ["connectivity"],
         headline_pair=("topology_inventory", "narrative"),
         note="`topology_inventory` and `narrative` are fact-for-fact equivalent and "
              "differ only in rendering."),
    Axis("json_formatting", "Formatting (JSON)", "procthor",
         ["json_pretty"],
         ["connectivity", "direction", "route", "aggregation", "proximity",
          "set_logic", "containment"],
         family="Formatting",
         note="Whitespace-only, run separately on full-record anchor to avoid content-difference confusion."),
    Axis("json_formatting_gibson", "Formatting (JSON, Gibson)", "gibson",
         ["json_pretty"],
         ["aggregation", "proximity", "set_logic", "direction", "containment"],
         family="Formatting",
         note="Gibson reading of the same formatting ablation. 3RScan has no reading "
              "here because `json_pretty` fails the coverage gate there."),
    Axis("structure", "Graph-structure representation", "procthor",
         ["topology", "room_tree", "graph_digest"],
         ["connectivity"],
         headline_pair=("topology", "graph_digest"),
         note="Same connectivity graph as lists, tree, or derived facts; `topology` used to avoid inventory distractors.",
         confound="vocabulary mirrors graph_digest's computed output; see re-cut analyses",
         confound_reps=("graph_digest",),
         confound_kind="vocabulary"),
    Axis("relation_linearization", "Relation organization and abstraction", "3rscan",
         ["relations_flat", "relations_subject", "relations_predicate",
          "relations_tree", "relations_digest"],
         ["object_relation", "relation_structure", "relation_aggregate"],
         headline_pair=("relations_subject", "relations_predicate"),
         note="Same object-relation graph as listed, grouped, or derived views; some are lossy.",
         confound="vocabulary mirrors relations_digest's output; see re-cut analyses",
         confound_reps=("relations_digest",),
         confound_kind="vocabulary"),
    Axis("framing", "Relational vs. navigational framing", "procthor",
         ["topology_metric", "navigation"],
         ["route", "direction", "connectivity"],
         headline_pair=("topology_metric", "navigation"),
         note="Fact/layout-matched over same geometry. Each question type uses independent channel; read separately."),
    Axis("framing_gibson", "Relational vs. navigational framing (Gibson)", "gibson",
         ["metric_framing", "navigation"],
         ["direction"],
         headline_pair=("metric_framing", "navigation"),
         family="Relational vs. navigational framing",
         note="K-NN room pairs; `direction` only (no door graph on Gibson)."),
]

AXIS_BY_ID = {a.id: a for a in AXES}

# Carry host/probe_types so pair scope cannot widen beyond the declared axis.
class AxisPair(NamedTuple):
    label: str                    # report label
    a: str                        # first pole; delta is b - a
    b: str                        # second pole
    host: str                     # only host on which this pair is defined
    probe_types: tuple[str, ...]  # only question types on which it is read


AXIS_PAIRS = [
    AxisPair(a.label, a.headline_pair[0], a.headline_pair[1], a.host,
             tuple(a.probe_types))
    for a in AXES if a.headline_pair
]


def dataset_of(scene_id: str) -> str:
    """Map a scene ID to its host dataset.

    Unprefixed scene names fall back to Gibson.
    """
    s = (scene_id or "").lower()
    if s.startswith("procthor"):
        return "procthor"
    if s.startswith("3rscan"):
        return "3rscan"
    return "gibson"
