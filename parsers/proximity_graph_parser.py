"""proximity_graph — object-object proximity computed from coordinates.

The coordinate-derived counterpart to `object_graph`'s hand-annotated 3DSSG
edges. Both serialize the same objects; they differ only in where the edges come
from — geometry here, human labels there. That makes `object_graph` vs
`proximity_graph` the clean test of the source-fidelity axis: does ground-truth
annotation beat what a parser can recover from coordinates alone?

Objects are grouped by room. Within each room, each object is related to its
NEAREST_K closest same-room neighbours by 3D centroid distance. Each
relationship is expressed as an allocentric direction (compass bearing and/or
above/below) derived from the position offset; ``close by`` is the fallback
when the offset is too small to orient reliably (floor distance < MIN_FLOOR_DIST
and vertical gap < MIN_VERT_GAP).

The parser deliberately does not try to recover support or containment relations
(``standing on``, ``attached to``). Those need contact reasoning and a reliable
gravity axis that bounding-box centroids do not robustly provide. The gap between
this coordinate-derived view and `object_graph`'s rich typed edges is precisely
what Axis E measures.

Gated on having object positions; runs on 3RScan and ProcTHOR.
"""

import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import obj_label, room_label, sort_key
from _geometry import compass, floor_plane, vertical_axis, distance_3d
from utils.capabilities import has_object_positions
from utils.models import Building, Room, SceneObject

NEAREST_K      = 4
MIN_FLOOR_DIST = 0.3   # metres; below this compass direction is noise
MIN_VERT_GAP   = 0.3   # metres; below this no above/below qualifier


def _positioned(room: Room) -> list[SceneObject]:
    return [o for o in room.objects if o.position is not None]


def _short_label(o: SceneObject) -> str:
    return obj_label(o.category, o.short_id or o.id)


def _direction_phrase(
    a: SceneObject, b: SceneObject, up: int, fu: int, fv: int
) -> str:
    """Allocentric direction from a to b: compass bearing and/or above/below."""
    du    = b.position[fu] - a.position[fu]
    dv    = b.position[fv] - a.position[fv]
    dvert = b.position[up] - a.position[up]

    fd        = math.hypot(du, dv)
    has_floor = fd        >= MIN_FLOOR_DIST
    has_vert  = abs(dvert) >= MIN_VERT_GAP

    if has_floor and has_vert:
        vert = "above" if dvert > 0 else "below"
        return f"{compass(du, dv)} of and {vert}"
    if has_floor:
        return f"{compass(du, dv)} of"
    if has_vert:
        return "above" if dvert > 0 else "below"
    return "close by"


def _room_block(room: Room, up: int, fu: int, fv: int, indent: str) -> list[str]:
    objs = _positioned(room)
    if len(objs) < 2:
        return []

    label = {o.id: _short_label(o) for o in objs}
    lines = [f"{indent}{room_label(room)}:"]

    for a in sorted(objs, key=lambda o: sort_key(o.short_id or o.id)):
        neighbors = sorted(
            ((distance_3d(a.position, b.position), b) for b in objs if b is not a),
            key=lambda t: t[0],
        )[:NEAREST_K]

        rel_lines = [
            f"{indent}    {_direction_phrase(a, b, up, fu, fv)} {label[b.id]}"
            for _, b in neighbors
        ]
        if rel_lines:
            lines.append(f"{indent}  {label[a.id]}:")
            lines.extend(rel_lines)

    return lines


def parse(building: Building) -> str:
    if not has_object_positions(building):
        raise NotApplicable(
            "proximity_graph needs >=2 objects with positions; this scene has none"
        )

    all_pos = [
        o.position
        for room in building.rooms.values()
        for o in room.objects
        if o.position is not None
    ]
    up     = vertical_axis(all_pos)
    fu, fv = floor_plane(all_pos)

    n_objects = sum(len(_positioned(r)) for r in building.rooms.values())
    lines = [
        f"{building.name} -- {n_objects} objects in {len(building.rooms)} rooms.",
        "",
        "Object proximity by room (allocentric directions computed from coordinates):",
    ]

    floors = sorted({r.floor for r in building.rooms.values() if r.floor})
    rooms_sorted = sorted(building.rooms.values(), key=lambda r: sort_key(r.id))

    if len(floors) > 1:
        for fl in floors:
            fl_rooms = [r for r in rooms_sorted if r.floor == fl]
            fl_label = f"Floor {fl} ({len(fl_rooms)} rooms):"
            blocks = []
            for room in fl_rooms:
                block = _room_block(room, up, fu, fv, indent="  ")
                if block:
                    blocks.extend(block)
                    blocks.append("")
            if blocks:
                lines.append(fl_label)
                lines.extend(blocks)
        no_floor = [r for r in rooms_sorted if not r.floor]
        for room in no_floor:
            block = _room_block(room, up, fu, fv, indent="")
            if block:
                lines.extend(block)
                lines.append("")
    else:
        for room in rooms_sorted:
            block = _room_block(room, up, fu, fv, indent="")
            if block:
                lines.extend(block)
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize object-object proximity computed from coordinates")
