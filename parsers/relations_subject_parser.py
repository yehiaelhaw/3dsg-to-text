"""relations_subject -- the object-relation graph grouped by object (node-local).

Each object is stated once, with all of its outgoing relations gathered under it:
``chair [5]: standing on floor [1]; close by table [7], lamp [3].`` This is the
node-centric pole of the relation-linearization axis. It draws on the same source
edge set as `relations_flat` and `relations_predicate`, and the grouping key (here,
the subject) is the intended difference, so a delta against those largely isolates
whether node-local grouping aids reasoning.

It is NOT, however, a complete restatement of that edge set, so do not describe the
pair as content-identical. The `attr_cap` below prints only the first 3 objects of
each comparative/shared-attribute predicate per subject, with a ``(+N more)`` count
standing in for the rest: 109 / 146 / 414 triples go unprinted on the current primary
trio, 02b33dfb / 1d2f8518 / 0cac762f. Two different consequences, worth keeping apart:
shared-attribute (``same ...``) membership survives the cap, because each member
prints up to 3 others and the printed fragments stay connected within the group, so
the full clique is recoverable by unioning lines (checked for all 29 cliques across
the three scenes); comparative (``... than``) edges truncated in *both* directions
are simply absent from the document (0 / 12 / 117 triples). No evaluated question's
key facts were found to turn on one of the absent comparative edges -- an absence of
demonstrated impact, not a proof of none.

It is the de-prosed twin of `prose`'s "Spatial relations by room:" section -- same
content and ordering, but stripped of the room headers and the "is" narration, so a
`prose` vs `relations_subject` contrast reads on the *format* axis (sentences vs
compact list) over identical relation content. Spatial relations lead; comparative
attribute cliques (``same material``) are capped per object so they do not drown the
spatial edges.

Runs anywhere object relations are annotated; refuses on scenes that carry none.
"""

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
