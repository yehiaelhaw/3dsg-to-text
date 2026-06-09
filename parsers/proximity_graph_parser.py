"""proximity_graph — object-object proximity computed from coordinates.

The coordinate-derived counterpart to `object_graph`'s hand-annotated 3DSSG
edges. Both serialize the same objects in the same subject-grouped format; they
differ only in where the edges come from — geometry here, human labels there.
That makes `object_graph` vs `proximity_graph` the clean test of the
source-fidelity axis: does ground-truth annotation beat what a parser can recover
from coordinates alone?

Each object is linked to its `NEAREST_K` closest neighbours by 3D centroid
distance (``close by``). The parser deliberately stops there: it does **not** try
to recover typed or support relations (``standing on``, ``attached to``). Those
need a reliable gravity axis and contact reasoning that bounding-box centroids in
a single real-world scan do not robustly provide — attempting them yields
nonsensical edges (a wall "on" a towel). The gap between this proximity-only view
and `object_graph`'s rich typed edges is precisely what the comparison measures.

Gated on having object positions; in practice this is the object-level 3RScan scene.
"""

import sys
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import obj_label, sort_key
from _geometry import distance_3d
from utils.capabilities import has_object_positions
from utils.models import Building, SceneObject

NEAREST_K = 4


def _all_objects(building: Building) -> list[SceneObject]:
    return [o for room in building.rooms.values() for o in room.objects if o.position is not None]


def parse(building: Building) -> str:
    if not has_object_positions(building):
        raise NotApplicable(
            "proximity_graph needs >=2 objects with positions; this scene has none"
        )

    objects = _all_objects(building)
    label = {o.id: obj_label(o.category, o.id) for o in objects}

    # Each object's NEAREST_K closest neighbours by 3D centroid distance.
    by_subject: dict[str, list[str]] = {}
    for a in objects:
        ranked = sorted(
            ((distance_3d(a.position, b.position), b) for b in objects if b is not a),
            key=lambda t: t[0],
        )[:NEAREST_K]
        by_subject[a.id] = [label[b.id] for _, b in ranked]

    inventory = ", ".join(
        label[o.id]
        for o in sorted(objects, key=lambda o: ((o.category or "object"), sort_key(o.id)))
    )
    lines = [
        f"{building.name} — {len(objects)} objects: {inventory}.",
        "",
        "Object proximity (computed from coordinates):",
    ]
    for sid in sorted(by_subject, key=sort_key):
        lines.append(f"{label[sid]} — close by {', '.join(by_subject[sid])}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize object-object proximity computed from coordinates")
