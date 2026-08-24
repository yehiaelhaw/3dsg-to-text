"""Which representations can answer which question types: a rep is in-scope for a type when it covers the channels the type needs."""

from __future__ import annotations

# "+" combinations union capabilities; unknown reps fail open to ALL_CAPS.
# metric = arbitrary room-pair geometry; metric_edges = connected-edge geometry.
# Relation tiers: raw, support, derived. relations_digest is curated; some
# per-subject raw lists are capped without creating a separate tier.

ALL_CAPS = {"inventory", "connectivity", "metric", "metric_edges",
            "object_relations_raw", "object_relations_support", "object_relations_derived"}

REP_CAPS: dict[str, set[str]] = {
    "inventory":          {"inventory"},
    "topology_inventory": {"inventory", "connectivity"},
    "topology":           {"connectivity"},
    "graph_digest":       {"connectivity"},
    "room_tree":          {"connectivity"},
    "topology_metric":    {"connectivity", "metric_edges"},
    "narrative":          {"inventory", "connectivity"},
    "prose":              {"inventory", "connectivity",
                           "object_relations_raw", "object_relations_support",
                           "object_relations_derived"},
    "metric_relations":   {"inventory", "metric", "metric_edges"},
    "navigation":         {"connectivity", "metric_edges"},
    "metric_framing":     {"metric_edges"},
    "relations_flat":      {"object_relations_raw", "object_relations_support", "object_relations_derived"},
    "relations_subject":   {"object_relations_raw", "object_relations_support", "object_relations_derived"},
    "relations_predicate": {"object_relations_raw", "object_relations_support", "object_relations_derived"},
    "relations_tree":      {"object_relations_support"},
    "relations_digest":    {"object_relations_derived"},
    "json_mini":          set(ALL_CAPS),
    "json_pretty":        set(ALL_CAPS),
    "synthesis":          {"inventory", "connectivity", "metric", "metric_edges",
                           "object_relations_raw", "object_relations_support",
                           "object_relations_derived"},
}

# Need forms: set = AND, list = OR alternatives, dict = host-specific.
TYPE_NEEDS: dict[str, set[str] | list[set[str]] | dict[str, set[str] | list[set[str]]]] = {
    "connectivity":    {"connectivity"},
    "proximity":       {"metric"},
    "direction":       {"metric_edges"},
    "route":           {"connectivity", "metric_edges"},
    "object_relation": {"object_relations_raw"},
    "relation_structure": [{"object_relations_support"}, {"object_relations_derived"}],
    "relation_aggregate": {"object_relations_derived"},
    "containment":     {"inventory"},
    "aggregation":     {"inventory"},
    "set_logic":       {"inventory"},
    # Planning uses inventory on ProcTHOR/Gibson and raw object relations on 3RScan.
    "planning": {
        "procthor": {"inventory"},
        "gibson":   {"inventory"},
        "3rscan":   {"object_relations_raw"},
    },
}


def caps(rep: str) -> set[str]:
    """Union capabilities across "+" components."""
    out: set[str] = set()
    for part in rep.split("+"):
        out |= REP_CAPS.get(part, ALL_CAPS)
    return out


def validate_declared(reps: set[str], qtypes: set[str | None]) -> None:
    """Reject undeclared reps/types before final runs instead of failing open."""
    problems: list[str] = []
    for rep in sorted(reps):
        for part in rep.split("+"):
            if part not in REP_CAPS:
                problems.append(f"representation {part!r} (from {rep!r}) has no REP_CAPS entry")
    for qt in sorted(qtypes, key=lambda t: t or ""):
        if not qt:
            problems.append("a question has no question_type (never filtered under fail-open)")
        elif qt not in TYPE_NEEDS:
            problems.append(f"question type {qt!r} has no TYPE_NEEDS entry")
    if problems:
        raise ValueError(
            "strict_scope: undeclared names would run fail-open:\n  - "
            + "\n  - ".join(problems)
            + "\nDeclare them in evaluation/scope.py (REP_CAPS / TYPE_NEEDS) "
              "or run with strict_scope=False."
        )


def in_scope(rep: str, qtype: str | None, host: str | None = None) -> bool:
    """Return whether `rep` satisfies `qtype`; undeclared types/hosts fail open."""
    need = TYPE_NEEDS.get(qtype or "")
    if not need:  # undeclared type
        return True
    if isinstance(need, dict):
        if host is None:
            raise ValueError(
                f"in_scope(qtype={qtype!r}) has a host-dependent requirement "
                f"({sorted(need)}) -- pass host=..."
            )
        need = need.get(host)
        if need is None:  # undeclared host
            return True
    c = caps(rep)
    if isinstance(need, list):
        return any(n <= c for n in need)
    return need <= c
