"""topology — connectivity-first serialization (structured).

Foregrounds the room adjacency graph and deliberately drops all metric data:
which room opens into which, and what each room contains, with no coordinates,
distances, or directions. This is the connectivity rung of the spatial-encoding
ladder (compare `metric_relations`, which keeps distance but not doors).

It is also the structured half of the format axis: `topology` and `prose` carry
the *same* facts in the *same* order — rooms degree-first, identical connectivity
and category inventory — and differ only in rendering (labelled blocks here, prose
sentences there). Any score delta between them is attributable to format alone.

Runs only where a room connection graph (doors + open-plan passages) exists
(ProcTHOR); refuses elsewhere (Gibson, 3RScan).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, object_inventory, room_type_summary, sort_key
from utils.capabilities import has_room_connectivity
from utils.models import Building, Room


def _degree(connectivity: dict[str, list[str]], rooms: dict[str, Room], rid: str) -> int:
    """Number of distinct neighbouring rooms reachable through a connection."""
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("topology needs a room connectivity graph; this scene has none")

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
