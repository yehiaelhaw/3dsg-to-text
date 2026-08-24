"""narrative — the content-matched prose pole of the format axis; carries exactly
topology_inventory's facts (same rooms, order, connectivity, per-room inventory) as
sentences instead of labelled blocks, reusing prose_parser's `_room_line` but without
prose's object count or relations section."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_type_summary, sort_key as _id_key
from prose_parser import _room_line, _degree
from scene_graph.capabilities import has_room_connectivity
from scene_graph.models import Building


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("narrative needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    connectivity = building.connectivity
    total_objects = sum(len(r.objects) for r in rooms.values())

    lines = []

    # -- Head: same facts as prose's head, same wording (incl. floor count, which
    # is a no-op on ProcTHOR -- its scenes carry no floor values). --
    room_count = len(rooms)
    floors = sorted({r.floor for r in rooms.values() if r.floor})
    floor_desc = (
        f" across {len(floors)} floor{'s' if len(floors) != 1 else ''}"
        if floors else ""
    )
    lines.append(
        f"{building.name} contains {room_count} rooms{floor_desc} and {total_objects} objects."
    )
    type_str = room_type_summary(rooms.values())
    if type_str:
        lines.append(f"Room types: {type_str}.")
    connection_count = sum(len(v) for v in connectivity.values()) // 2
    if connection_count:
        lines.append(
            f"Rooms are connected by {connection_count} room "
            f"connection{'s' if connection_count != 1 else ''} (doorways or open passages)."
        )
    lines.append("")

    # -- Body: one sentence per room, topology_inventory's order and content,
    # no object count, no floor grouping (topology_inventory never groups). --
    def sort_key(r) -> tuple:
        return (-_degree(connectivity, rooms, r.id), _id_key(r.id), -len(r.objects))

    for room in sorted(rooms.values(), key=sort_key):
        lines.append(_room_line(room, connectivity, rooms, show_count=False))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the room connectivity graph as matched prose (format-axis pole)")
