"""relations_predicate -- the object-relation graph grouped by relation type.

The edge-centric pole of the relation-linearization axis: instead of gathering edges
under each object (`relations_subject`), it gathers them under each *predicate*, so
all ``standing on`` edges sit in one bucket, all ``close by`` in another. Same edge
set as `relations_flat`/`relations_subject`; the organizing key is the relation type.
This is the cleanest within-axis contrast -- group-by-object vs group-by-predicate,
identical content, and neither obviously better a priori -- so it is the headline
pair of the axis.

Grouping by type also lets each predicate render in its natural shape, which is where
this view earns its keep on dense 3DSSG data:
  * symmetric equivalence cliques (``same material``) collapse to one group per clique
    via connected components -- the O(k^2) -> O(k) collapse that makes `relations_flat`
    explode;
  * symmetric proximity (``close by``) lists each unordered pair once;
  * everything directed (support, directional, ``... than``) lists subject -> object.

Runs anywhere object relations are annotated; refuses on scenes that carry none.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict

from _base import run_parser, NotApplicable
from _format import sort_key, undirected_components
from _relations import resolve_labels, classify_predicate
from utils.capabilities import has_object_relations
from utils.models import Building

# Read order: spatial buckets first, low-value comparative/attribute buckets last.
_CLASS_ORDER = {"support": 0, "proximity": 1, "directional": 2, "other": 3, "attribute": 4}


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("relations_predicate needs annotated object relations; this scene has none")

    _, lbl = resolve_labels(building)

    by_pred: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in building.object_relations:
        by_pred[r.predicate].append((r.subject_id, r.object_id))

    def bucket_lines(pred: str, edges: list[tuple[str, str]]) -> list[str]:
        # No per-bucket count: this view lists every edge in full (nothing truncated),
        # so a count would only restate what is already shown -- and a bare "(14)" after
        # a predicate also mimics the "[14]" id syntax. Counts that disclose an omission
        # live where they matter (relations_tree's trailer, relations_subject's "+N more")
        # and per-type totals live in relations_digest.
        if pred.startswith("same "):
            # Symmetric + transitive: one group per equivalence clique.
            comps = [sorted(set(m), key=sort_key) for m in undirected_components(edges)]
            comps.sort(key=lambda g: (-len(g), sort_key(g[0])))
            return [f"{pred}:"] + [f"  {{{', '.join(lbl(o) for o in g)}}}" for g in comps]
        if classify_predicate(pred) == "proximity":
            # Symmetric, not transitive: list each unordered pair once.
            pairs = sorted({tuple(sorted(e, key=sort_key)) for e in edges},
                           key=lambda e: (sort_key(e[0]), sort_key(e[1])))
            shown = "; ".join(f"{{{lbl(a)}, {lbl(b)}}}" for a, b in pairs)
            return [f"{pred}: {shown}"]
        # Directed: subject -> object.
        pairs = sorted(set(edges), key=lambda e: (sort_key(e[0]), sort_key(e[1])))
        shown = "; ".join(f"{lbl(a)} -> {lbl(b)}" for a, b in pairs)
        return [f"{pred}: {shown}"]

    ordered = sorted(by_pred, key=lambda p: (_CLASS_ORDER[classify_predicate(p)], -len(by_pred[p]), p))

    head = (
        f"{building.name} -- object relations grouped by relation type "
        f"({len(ordered)} relation types)."
    )
    lines = [head, ""]
    for pred in ordered:
        lines.extend(bucket_lines(pred, by_pred[pred]))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize object relations grouped into per-predicate buckets")
