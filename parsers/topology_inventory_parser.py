"""topology_inventory — the connectivity rung of the spatial-encoding ladder (room
adjacency + per-room contents, no coordinates/distances) and the structured pole of the
format axis; matched fact-for-fact against narrative's prose rendering of the same content."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, object_inventory, room_type_summary, sort_key
from scene_graph.capabilities import has_room_connectivity
from scene_graph.models import Building, Room


def _degree(connectivity: dict[str, list[str]], rooms: dict[str, Room], rid: str) -> int:
    """Number of distinct neighbouring rooms reachable through a connection."""
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("topology_inventory needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    connectivity = building.connectivity
    total_objects = sum(len(r.objects) for r in rooms.values())
    connection_count = sum(len(v) for v in connectivity.values()) // 2

    # Shared ordering with prose: degree desc, ties by id.
    ordered = sorted(
        rooms.values(),
        key=lambda r: (-_degree(connectivity, rooms, r.id), sort_key(r.id)),
    )

    # -- Head: same facts as prose's head --
    head = (
        f"{building.name} — {len(rooms)} rooms, {total_objects} objects, "
        f"{connection_count} room connections (doorways or open passages)."
    )
    type_str = room_type_summary(rooms.values())
    if type_str:
        head += f" Room types: {type_str}."
    lines = [head, ""]

    # -- Per-room blocks: connectivity + contents, same order as prose --
    for room in ordered:
        neighbor_ids = sorted(
            {n for n in connectivity.get(room.id, []) if n in rooms},
            key=lambda nid: (-_degree(connectivity, rooms, nid), sort_key(nid)),
        )
        neighbors = ", ".join(room_label(rooms[nid]) for nid in neighbor_ids) or "(none)"
        lines.append(room_label(room))
        lines.append(f"  connects to: {neighbors}")
        lines.append(f"  contains: {object_inventory(room)}")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the room connectivity graph (no metric data)")
