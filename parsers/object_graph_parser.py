"""object_graph — annotated object-object edges.

Serializes the hand-annotated 3DSSG semantic graph: an object inventory plus the
labelled relations between objects (``standing on``, ``next to``, ``attached
to`` …). This is the only representation in the study built on *annotated*
topology rather than *computed* topology, so it runs only where those relations
exist (3RScan); it refuses elsewhere (Gibson, ProcTHOR).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import obj_label, sort_key, relation_lines
from utils.capabilities import has_object_relations
from utils.models import Building


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("object_graph needs annotated object relations; this scene has none")

    objects = [o for room in building.rooms.values() for o in room.objects]
    objects.sort(key=lambda o: ((o.category or "object"), sort_key(o.id)))
    inventory = ", ".join(obj_label(o.category, o.id) for o in objects)

    lines = [
        f"{building.name} — {len(objects)} objects: {inventory}.",
        "",
        "Object relations (most-connected first):",
    ]
    lines.extend(relation_lines(building))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the annotated object-object relation graph")
