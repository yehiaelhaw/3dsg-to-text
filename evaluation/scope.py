"""scope.py — which representations can answer which question types.

A representation only carries certain information channels (connectivity, metric,
...); a question type needs certain channels to be answerable at all. A rep is
*in scope* for a type when it covers what the type needs. This is the single source
of truth shared by the runner (which skips out-of-scope cells so they are never
computed) and plots.py (which would otherwise mask them after the fact).

The model is fail-open: an unknown representation defaults to all channels, and an
undeclared question type is never filtered — so new parsers/types are run and shown
until their scope is declared here.
"""

from __future__ import annotations

# Information channels each representation's text carries. Combinations ("a+b")
# take the union of their parts. Unknown reps default to all channels (shown).
#
# The metric channel is split in two: "metric" is *pairwise* distance/bearing data
# (between arbitrary rooms), "metric_edges" is distance/bearing only along room
# connections. `navigation` carries only the latter — it cannot answer questions
# about distances between unconnected rooms (e.g. bathroom-to-bathroom proximity
# when no two bathrooms are adjacent), so it must not be in scope for them.
REP_CAPS: dict[str, set[str]] = {
    "inventory":        {"inventory"},
    "topology":         {"inventory", "connectivity"},
    "graph_digest":     {"connectivity"},
    "room_tree":        {"connectivity"},
    "prose":            {"inventory", "connectivity", "object_relations"},
    "metric_relations": {"inventory", "metric", "metric_edges"},
    # density-axis (C) twin of metric_relations: same channels, exhaustive all-pairs.
    "metric_relations_full": {"inventory", "metric", "metric_edges"},
    "navigation":       {"connectivity", "metric_edges"},
    # relation-linearization family: five presentations of one object-relation graph.
    "relations_flat":      {"object_relations"},
    "relations_subject":   {"object_relations"},
    "relations_predicate": {"object_relations"},
    "relations_tree":      {"object_relations"},
    "relations_digest":    {"object_relations"},
    "json":             {"inventory", "connectivity", "metric", "metric_edges", "object_relations"},
}
ALL_CAPS = {"inventory", "connectivity", "metric", "metric_edges", "object_relations"}

# What each question type needs to be answerable at all. Spatial family needs a
# spatial channel; the general-reasoning family only needs room/object content
# (inventory), so the discriminating variable there is format/density, not encoding.
# proximity needs pairwise metric data; direction/route questions only ask about
# bearings along connections, so edge-level metric suffices.
TYPE_NEEDS: dict[str, set[str]] = {
    # spatial family
    "connectivity":    {"connectivity"},
    "proximity":       {"metric"},
    "direction":       {"metric_edges"},
    "route":           {"connectivity", "metric_edges"},
    "object_relation": {"object_relations"},
    # general-reasoning family (content only)
    "containment":     {"inventory"},
    "aggregation":     {"inventory"},
    "set_logic":       {"inventory"},
    "planning":        {"inventory"},
}


def caps(rep: str) -> set[str]:
    """Channels a representation carries; union across the parts of a "a+b" combo."""
    out: set[str] = set()
    for part in rep.split("+"):
        out |= REP_CAPS.get(part, ALL_CAPS)
    return out


def in_scope(rep: str, qtype: str | None) -> bool:
    """True if `rep` carries everything question type `qtype` needs (fail-open)."""
    need = TYPE_NEEDS.get(qtype or "")
    if not need:            # undeclared type -> never filter
        return True
    return need <= caps(rep)
