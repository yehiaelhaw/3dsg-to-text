"""Which representations can answer which question types: a rep is in-scope for a type when it covers the channels the type needs."""

from __future__ import annotations

# Information channels each representation's text carries. Combinations ("a+b")
# take the union of their parts. Unknown reps default to all channels (shown).
#
# The metric channel is split in two: "metric" is *pairwise* distance/bearing data
# (between arbitrary rooms), "metric_edges" is distance/bearing only along room
# connections. `navigation` carries only the latter — it cannot answer questions
# about distances between unconnected rooms (e.g. bathroom-to-bathroom proximity
# when no two bathrooms are adjacent), so it must not be in scope for them.
#
# object_relations splits into three tiers: "raw" (arbitrary triples, exact for flat/predicate, per-object-capped for subject/prose), "support" (containment forest, drawn exhaustively), "derived" (digest's precomputed summary -- a curated top-N, NOT an exhaustive enumeration, so an arbitrary non-highlighted fact must stay under "raw" only).

# Every channel there is. Defined before REP_CAPS so the full-information views can
# reference it instead of restating the list: "identical content ⇒ identical
# channels" then holds structurally, not by three copies staying in sync.
ALL_CAPS = {"inventory", "connectivity", "metric", "metric_edges", "object_relations",
            "object_relations_raw", "object_relations_support", "object_relations_derived"}

REP_CAPS: dict[str, set[str]] = {
    "inventory":        {"inventory"},
    "topology_inventory": {"inventory", "connectivity"},
    # topology_inventory minus the inventories: the content-matched raw-adjacency
    # pole of the structure-presentation axis (room_tree/graph_digest are
    # connectivity-only, so the raw pole is too).
    "topology":         {"connectivity"},
    "graph_digest":     {"connectivity"},
    "room_tree":        {"connectivity"},
    # Channel set deliberately IDENTICAL to navigation's (metric_edges only, no inventory) -- any addition here would un-match the fact-matched pair.
    "topology_metric":  {"connectivity", "metric_edges"},
    # The content-matched prose pole of the format axis: `topology_inventory`'s
    # facts (inventory + connectivity only, nothing else), rendered as sentences.
    # Channel set is deliberately IDENTICAL to topology_inventory's -- the two
    # differ only in rendering, never in what they know. See
    # parsers/narrative_parser.py and tests/test_format_axis_equivalence.py.
    "narrative":        {"inventory", "connectivity"},
    "prose":            {"inventory", "connectivity", "object_relations",
                          "object_relations_raw", "object_relations_support", "object_relations_derived"},
    "metric_relations": {"inventory", "metric", "metric_edges"},
    "navigation":       {"connectivity", "metric_edges"},
    # NOTE: `connectivity` is declared host-invariantly even though Gibson renders K-NN proximity here, not real doorway adjacency -- harmless only because no `connectivity`-typed question is authored on Gibson. Do NOT copy this pattern into a new rep; see `metric_framing` below, which declares only what it actually carries.
    #
    # Gibson counterpart to `navigation` on `direction`: deliberately NARROWER than navigation's declaration (metric_edges only, no connectivity), since Gibson has no door graph.
    "metric_framing":   {"metric_edges"},
    # flat/predicate print the edge set in full; subject is capped per-object (see the "raw" tier note above). All three declared at all three tiers, as scored.
    "relations_flat":      {"object_relations", "object_relations_raw", "object_relations_support", "object_relations_derived"},
    "relations_subject":   {"object_relations", "object_relations_raw", "object_relations_support", "object_relations_derived"},
    "relations_predicate": {"object_relations", "object_relations_raw", "object_relations_support", "object_relations_derived"},
    # relations_tree draws only the support/containment forest (exhaustively) --
    # no raw enumeration of proximity/directional/comparative pairs, no cluster
    # or attribute-clique content at all.
    "relations_tree":      {"object_relations", "object_relations_support"},
    # relations_digest is a derived summary -- no raw triple enumeration, and its
    # support-chain coverage is curated (see note above), so it gets only the
    # "derived" tier, not "support".
    "relations_digest":    {"object_relations", "object_relations_derived"},
    # Same parse() output, two serializations -- reference ALL_CAPS so they can't drift apart. The bare name `json` has no entry and no alias (deliberately): an alias would let one cell exist under two resume-cache keys and double a rep group.
    "json_mini":        set(ALL_CAPS),
    "json_pretty":      set(ALL_CAPS),
    # object_relations_derived is earned, not assumed: synthesis_parser folds in relations_digest's derived content explicitly rather than this being an over-claim. Written out rather than referencing ALL_CAPS, since this entry's channel list should be auditable directly.
    "synthesis":        {"inventory", "connectivity", "metric", "metric_edges", "object_relations",
                          "object_relations_raw", "object_relations_support", "object_relations_derived"},
}

# Spatial family needs a spatial channel; general-reasoning only needs inventory (discriminator = format/density, not encoding). proximity needs pairwise metric; direction/route need only edge-level metric.
# Values may be set[str] (all required) or list[set[str]] (any alternative suffices).
TYPE_NEEDS: dict[str, set[str] | list[set[str]]] = {
    # spatial family
    "connectivity":    {"connectivity"},
    "proximity":       {"metric"},
    "direction":       {"metric_edges"},
    "route":           {"connectivity", "metric_edges"},
    # arbitrary specific-pair object-relation lookups (proximity/directional/
    # comparative pairs, or a support fact not in relations_digest's curated
    # highlights) -- excludes relations_tree and relations_digest, neither of
    # which states arbitrary triples.
    "object_relation": {"object_relations_raw"},
    # Answerable via relations_tree's exhaustive forest OR relations_digest's guaranteed highlights -- verify against the actual printed text before tagging, since digest's content is curated, not exhaustive.
    "relation_structure": [{"object_relations_support"}, {"object_relations_derived"}],
    # digest-only aggregate content: proximity-cluster composition, shared-
    # attribute-clique membership, relation-type census. relations_tree carries
    # none of this.
    "relation_aggregate": {"object_relations_derived"},
    # general-reasoning family (content only)
    "containment":     {"inventory"},
    "aggregation":     {"inventory"},
    "set_logic":       {"inventory"},
    # planning: room-level on Gibson/ProcTHOR (needs inventory) or object-level on
    # 3RScan (needs the raw enumeration -- these questions ask about specific
    # proximity/support pairs that relations_tree/relations_digest don't reliably
    # state) — disjunctive OR.
    "planning":        [{"inventory"}, {"object_relations_raw"}],
}


def caps(rep: str) -> set[str]:
    """Channels a representation carries; union across the parts of a "a+b" combo."""
    out: set[str] = set()
    for part in rep.split("+"):
        out |= REP_CAPS.get(part, ALL_CAPS)
    return out


def validate_declared(reps: set[str], qtypes: set[str | None]) -> None:
    """Fail-closed check for final runs: raise if any representation part or
    question type would fall through to the fail-open defaults.

    A combo ("a+b") is checked part by part. A missing/None question type counts as
    undeclared: fail-open never filters it, so in a strict run every question must
    carry a type with a TYPE_NEEDS entry.
    """
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


def in_scope(rep: str, qtype: str | None) -> bool:
    """True if `rep` carries everything question type `qtype` needs (fail-open).

    If TYPE_NEEDS[qtype] is a list, any alternative need-set suffices (OR).
    """
    need = TYPE_NEEDS.get(qtype or "")
    if not need:            # undeclared type -> never filter
        return True
    c = caps(rep)
    if isinstance(need, list):
        return any(n <= c for n in need)
    return need <= c
