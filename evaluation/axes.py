"""Axis and representation-role registry: maps design axes onto representations and their reporting roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# Reference anchors used on every axis; synthesis is reported separately.
NON_SPATIAL_ANCHOR = "inventory"
# Minified json_mini is the cost denominator; formatting whitespace would inflate apparent savings.
FULL_RECORD_ANCHOR = "json_mini"
CANDIDATE = "synthesis"


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
         note="Read the ladder per question type: only in-scope rungs are compared. "
              "`inventory` is the non-spatial anchor and `json_mini` the full-record "
              "anchor, not a performance bound."),
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
              "differ only in rendering; `prose` is excluded because it contains "
              "additional information."),
    Axis("json_formatting", "Formatting (JSON)", "procthor",
         ["json_pretty"],
         ["connectivity", "direction", "route", "aggregation", "proximity",
          "set_logic", "containment"],
         family="Formatting",
         note="Same whitespace-only manipulation as the rendering axis above, run "
              "separately on the full-record anchor rather than pooled with it: "
              "`json_pretty` and `json_mini` contain identical keys, ordering, and "
              "values, so pairing either against `topology_inventory`/`narrative` "
              "would mix in a genuine content difference. `json_mini` remains the "
              "full-record anchor rather than an accuracy bound."),
    Axis("json_formatting_gibson", "Formatting (JSON, Gibson)", "gibson",
         ["json_pretty"],
         ["aggregation", "proximity", "set_logic", "direction", "containment"],
         family="Formatting",
         note="Gibson reading of the same formatting ablation. 3RScan has no reading "
              "here because `json_pretty` fails the coverage gate there."),
    Axis("structure_presentation", "Graph-structure representation", "procthor",
         ["topology", "room_tree", "graph_digest"],
         ["connectivity"],
         headline_pair=("topology", "graph_digest"),
         note="The same connectivity graph is exposed as neighbour lists, a tree, or "
              "derived structural facts. `topology` is used instead of "
              "`topology_inventory` to avoid inventory distractors and token load.",
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
         note="The same object-relation graph is exposed through listed/grouped, "
              "selectively drawn, and derived views. `relations_tree` and "
              "`relations_digest` are lossy; `relations_subject` caps some grouped lists.",
         confound="question vocabulary mirrors relations_digest's own computed "
                  "output (chain depth / clusters); read the natural/constructed "
                  "re-cut and the matched within-fact-set comparison",
         confound_reps=("relations_digest",),
         confound_kind="vocabulary"),
    Axis("framing", "Relational vs. navigational framing", "procthor",
         ["topology_metric", "navigation"],
         ["route", "direction", "connectivity"],
         headline_pair=("topology_metric", "navigation"),
         note="Fact- and layout-matched over the same doorway edges and geometry. "
              "Framing jointly changes figure-ground assignment and "
              "locative/navigational register, so it is not an egocentric/allocentric "
              "manipulation. Route tests framing, direction makes the converse relation "
              "task-relevant, and connectivity is diagnostic only."),
    Axis("framing_gibson", "Relational vs. navigational framing (Gibson)", "gibson",
         ["metric_framing", "navigation"],
         ["direction"],
         headline_pair=("metric_framing", "navigation"),
         family="Relational vs. navigational framing",
         note="Gibson reading using the same K-nearest-neighbour room pairs and "
              "metrics in locative and navigational form. Equivalence is checked against "
              "independently recomputed K-NN geometry; `direction` is the only in-scope "
              "type because Gibson has no door graph."),
    Axis("content_verbosity", "Verbosity on content questions", "gibson",
         ["prose"],
         ["containment", "aggregation", "set_logic"],
         secondary=True,
         note="For `containment`, `aggregation`, and `set_logic`, `prose` and "
              "`json_mini` contain the information the questions require; `json_mini` "
              "adds unused channels, so this reading tests verbosity/content load rather "
              "than formatting."),
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
