"""scope.py — which representations can answer which question types.

A representation only carries certain information channels (connectivity, metric,
...); a question type needs certain channels to be answerable at all. A rep is
*in scope* for a type when it covers what the type needs. This is the single source
of truth shared by the runner (which skips out-of-scope cells so they are never
computed) and plots.py (which would otherwise mask them after the fact).

The model is fail-open: an unknown representation defaults to all channels, and an
undeclared question type is never filtered — so new parsers/types are run and shown
until their scope is declared here. That is the right default for exploration, but
it silently scores structurally-impossible cells when a name is missing (this bit
once: relations_tree/relations_digest were scored on arbitrary-pair questions before
their channels were split). Final scored runs should therefore set
`EvalConfig.strict_scope=True`, which calls `validate_declared` before generating
and aborts on any undeclared representation or question type.

TYPE_NEEDS values may be a plain set[str] (all channels required) or a
list[set[str]] (any one alternative suffices — disjunctive OR). The OR form is
used when a question type is grounded in different channels across datasets, e.g.
planning: room-level inventory on Gibson/ProcTHOR, object_relations on 3RScan.
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
#
# The object-relation channel is split in three, because the five relations_*
# views do NOT all carry the same granularity despite linearizing the same edge
# set (relations_tree and relations_digest are lossy derived presentations, the
# object-level analogs of graph_digest/room_tree on the structure-presentation axis):
#   - "object_relations_raw": every individual triple is stated (support,
#     proximity, directional, comparative) -- arbitrary specific-pair lookups
#     are answerable. relations_flat/subject/predicate carry this; relations_tree
#     and relations_digest do not (see below).
#   - "object_relations_support": the support/containment sub-graph specifically,
#     drawn *exhaustively* -- relations_tree's entire content (every parent/child
#     edge, so any support/containment question is answerable, not just curated
#     highlights).
#   - "object_relations_derived": relations_digest's precomputed summary content
#     -- deepest-nesting depth + the single deepest chain (guaranteed correct,
#     since it is explicitly "deepest first"), the printed top-N receptacle
#     counts, full proximity-cluster membership, full shared-attribute-clique
#     membership, and the relation-type census. This is NOT a general "digest
#     can answer any support question" flag: relations_digest's chain list is a
#     curated top-N, not an exhaustive enumeration (e.g. on 3rscan_02b33dfb it
#     omits several real 2-hop chains like basket[26]->bath cabinet[16]), so
#     arbitrary non-highlighted support facts stay under object_relations_raw
#     only. Any new question relying on relations_digest for a specific fact
#     must still be verified against the actual printed text before being
#     admitted, same as always (CLAUDE.md "Writing a QA dataset for a scene").

# Every channel there is. Defined before REP_CAPS so the full-information views can
# reference it instead of restating the list: "identical content ⇒ identical
# channels" then holds structurally, not by three copies staying in sync.
ALL_CAPS = {"inventory", "connectivity", "metric", "metric_edges", "object_relations",
            "object_relations_raw", "object_relations_support", "object_relations_derived"}

# Names that were once valid and are now errors. `json` was split on 2026-08-09 into
# json_pretty (the historical pretty serialization) and json_mini (the minified
# ceiling). There is deliberately NO alias: the resume cache keys on
# (question_id, representation, repetition), so a silent alias would let one logical
# cell exist under two keys and quietly double a rep group in every mean.
RETIRED_REPS: dict[str, str] = {
    "json": "split into 'json_pretty' (historical pretty serialization) and "
            "'json_mini' (the minified ceiling) on 2026-08-09",
}

REP_CAPS: dict[str, set[str]] = {
    "inventory":        {"inventory"},
    "topology":         {"inventory", "connectivity"},
    # topology minus the inventories: the content-matched raw-adjacency pole of the
    # structure-presentation axis (room_tree/graph_digest are connectivity-only, so
    # the raw pole is too).
    "topology_edges_only": {"connectivity"},
    "graph_digest":     {"connectivity"},
    "room_tree":        {"connectivity"},
    "prose":            {"inventory", "connectivity", "object_relations",
                          "object_relations_raw", "object_relations_support", "object_relations_derived"},
    "metric_relations": {"inventory", "metric", "metric_edges"},
    "navigation":       {"connectivity", "metric_edges"},
    # relation-linearization family: five presentations of one object-relation graph.
    # flat/subject/predicate state every triple, so they carry all three tiers.
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
    # The two serializations of one parse() output. They are the same object printed
    # two ways, so they carry exactly the same channels -- expressed by referencing
    # ALL_CAPS rather than restating it, so the two entries cannot drift apart.
    # json_mini is the ceiling (axes.CEILING); json_pretty is the raw pole of the
    # json_formatting ablation. The bare name `json` is retired -- see RETIRED_REPS.
    "json_mini":        set(ALL_CAPS),
    "json_pretty":      set(ALL_CAPS),
    # synthesized best-of-axes default: prose backbone + derived connectivity
    # (graph_digest) + salient metric (metric_relations) + derived object-relation
    # structure (relations_digest), so it carries every channel on a fully-equipped
    # scene. Egocentric routing is deliberately left to navigation, but synthesis
    # still carries metric_edges (per-room bearings), so it stays in scope for
    # direction/route -- the gap to navigation there is a result to measure, not a
    # cell to mask.
    #
    # object_relations_derived history: this channel was an over-claim from
    # 2026-07-14 to (same day) its fix. Briefly, synthesis's relations content was
    # raw enumeration only (prose's section, reused verbatim) with none of
    # relations_digest's guaranteed derived content (deepest-chain, top-N
    # receptacles, proximity-cluster membership, relation-type census) -- yet the
    # channel was claimed anyway. Root-caused to `parsers/synthesis_parser.py`
    # (recovered after being briefly missing from disk) never having imported
    # `relations_digest_parser`. Fixed by extracting `object_relations_digest`
    # (relations_digest_parser.py, mirrors `graph_digest_parser.connectivity_digest`
    # -- body without the head line, byte-identical to the standalone view) and
    # folding it into `synthesis_parser.parse`, appended after the raw section
    # (verified byte-identical `relations_digest` output; verified the fold-in
    # does not restate prose's own "Shared attributes" clique rollup -- that
    # trailing block is cut from the raw tier since the derived section always
    # supplies it whenever both are present). Channel restored the same day.
    # scene_contexts/*/synthesis.txt regenerated for all 9 scenes 2026-07-14.
    # NB: `prose` also declares this channel. This used to be flagged here as an
    # unreviewed over-claim of the same kind; re-checked 2026-08-06, it is not one,
    # and the two cases are not the same kind of thing. The channel is declared by
    # every view that states the triples exhaustively, because raw enumeration is
    # what makes the derived content derivable -- that is why relations_flat/
    # subject/predicate carry it (above), and prose enumerates the same way. The
    # synthesis defect was a BUILD defect, not a scope-convention question: the
    # candidate's whole claim is to carry each channel in its winning form, and the
    # assembled document was silently missing a component it advertised, so the fix
    # belonged in the parser rather than in this table. Checked against results
    # rather than left as an argument: on 3RScan relation_aggregate -- the type that
    # needs this channel and nothing else -- prose scores 0.76, level with
    # relations_predicate and above relations_flat (0.72) and relations_subject
    # (0.67), i.e. it derives the aggregate content as well as the enumerating views
    # whose declaration is not in question. Declaration stands; REP_CAPS unchanged,
    # which also keeps the published relation-linearization numbers interpretable.
    # Written out rather than referencing ALL_CAPS: this entry's channel list was
    # contested (see the object_relations_derived history above), so an auditor
    # should be able to read the declaration itself, not a constant.
    "synthesis":        {"inventory", "connectivity", "metric", "metric_edges", "object_relations",
                          "object_relations_raw", "object_relations_support", "object_relations_derived"},
}

# What each question type needs to be answerable at all. Spatial family needs a
# spatial channel; the general-reasoning family only needs room/object content
# (inventory), so the discriminating variable there is format/density, not encoding.
# proximity needs pairwise metric data; direction/route questions only ask about
# bearings along connections, so edge-level metric suffices.
#
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
    # support/containment identification (a specific parent/child edge or a
    # chain's depth) -- answerable either by relations_tree's exhaustive drawn
    # forest, or by relations_digest where the fact is one of its guaranteed
    # highlights (the deepest chain + its depth, or a printed top-N receptacle).
    # Only tag a question this way once both are verified against the actual
    # printed text -- relations_digest does not get object_relations_support,
    # so a question needing an arbitrary non-highlighted chain must stay
    # "object_relation" (raw only) even though relations_tree could answer it.
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

    A combo ("a+b") is checked part-by-part. A missing/None question type counts
    as undeclared: fail-open never filters it, so in a strict run every question
    must carry a type with a TYPE_NEEDS entry.
    """
    problems: list[str] = []
    for rep in sorted(reps):
        for part in rep.split("+"):
            if part in RETIRED_REPS:
                problems.append(
                    f"representation {part!r} (from {rep!r}) is RETIRED: {RETIRED_REPS[part]}. "
                    f"Name the replacement explicitly -- there is no alias, on purpose"
                )
            elif part not in REP_CAPS:
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
