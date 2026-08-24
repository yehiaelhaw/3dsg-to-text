"""relations_flat — the deliberately structureless floor pole of the relation-linearization
axis; every object-relation edge on its own line as a raw (subject, relation, object)
triple, no grouping or derivation."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import sort_key
from _relations import resolve_labels
from scene_graph.capabilities import has_object_relations
from scene_graph.models import Building


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("relations_flat needs annotated object relations; this scene has none")

    rels = building.object_relations
    _, lbl = resolve_labels(building)

    head = (
        f"{building.name} -- {len(rels)} object relations, one per line as "
        "(subject, relation, object) triples; unordered, every edge stated separately."
    )
    lines = [head, ""]
    for r in sorted(rels, key=lambda r: (sort_key(r.subject_id), r.predicate, sort_key(r.object_id))):
        lines.append(f"({lbl(r.subject_id)}, {r.predicate}, {lbl(r.object_id)})")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize object relations as a flat list of atomized triples")
