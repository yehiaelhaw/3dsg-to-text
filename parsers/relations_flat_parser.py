"""relations_flat -- the object-relation graph as a flat list of atomized triples.

The deliberately structureless pole of the relation-linearization axis: every edge
on its own line, no grouping, no collapsing, no derivation. It is the baseline the
other `relations_*` views lift over -- if grouping by object (`relations_subject`),
by predicate (`relations_predicate`), drawing the support forest (`relations_tree`),
or pre-computing its consequences (`relations_digest`) helps an LLM, the gain shows
as a delta against this floor. Its cost is intentional: on a dense 3DSSG scene the
symmetric attribute cliques (every wall ``same material`` as every other) emit O(k^2)
lines here, exactly the token blow-up the grouped views exist to avoid.

Runs anywhere object relations are annotated (3RScan dense; ProcTHOR sparse); refuses
on scenes that carry none.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import sort_key
from _relations import resolve_labels
from utils.capabilities import has_object_relations
from utils.models import Building


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
