"""relations_subject — the node-centric pole of the relation-linearization axis; groups the
object-relation graph by subject object, one line per object. Comparative/shared-attribute
predicates are capped at 3 objects per subject (with "(+N more)"); the cap leaves
shared-attribute cliques recoverable by unioning lines but drops comparative edges outright."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict

from _base import run_parser, NotApplicable
from _format import sort_key, is_attribute_predicate
from _relations import resolve_labels
from scene_graph.capabilities import has_object_relations
from scene_graph.models import Building


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("relations_subject needs annotated object relations; this scene has none")

    _, lbl = resolve_labels(building)

    by_subject: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in building.object_relations:
        by_subject[r.subject_id][r.predicate].append(r.object_id)

    def spatial_deg(sid: str) -> int:
        return sum(len(v) for p, v in by_subject[sid].items() if not is_attribute_predicate(p))

    def total_deg(sid: str) -> int:
        return sum(len(v) for v in by_subject[sid].values())

    def render(sid: str, attr_cap: int = 3) -> str:
        preds = by_subject[sid]
        spatial = sorted(p for p in preds if not is_attribute_predicate(p))
        comparative = sorted(p for p in preds if is_attribute_predicate(p))
        parts = [f"{pred} {', '.join(lbl(o) for o in preds[pred])}" for pred in spatial]
        for pred in comparative:
            objs = preds[pred]
            shown = ", ".join(lbl(o) for o in objs[:attr_cap])
            extra = len(objs) - attr_cap
            parts.append(f"{pred} {shown}" + (f" (+{extra} more)" if extra > 0 else ""))
        return f"{lbl(sid)}: {'; '.join(parts)}."

    # Most-connected objects first (spatial degree, then total, then id) so the
    # hubs of the relation graph lead -- same ordering prose uses within a room.
    subjects = sorted(
        by_subject,
        key=lambda sid: (-spatial_deg(sid), -total_deg(sid), sort_key(sid)),
    )

    head = (
        f"{building.name} -- object relations grouped by object "
        f"({len(subjects)} objects have outgoing relations)."
    )
    lines = [head, ""]
    lines.extend(render(sid) for sid in subjects)

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize object relations grouped by subject object")
