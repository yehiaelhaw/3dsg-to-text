"""topology_edges_only — the room adjacency graph alone (structured).

`topology` minus the per-room inventories: exactly the door graph, stated
room-by-room, with no object lists, no coordinates, no distances. Same room
order (degree desc, ties by id), same labels, same "connects to" blocks — the
only difference from `topology` is the dropped inventory content.

Axis role: the content-matched raw-adjacency pole of the structure-presentation
axis. `room_tree` and `graph_digest` are deliberately connectivity-only, so
the raw pole must be too — otherwise a structure-presentation delta could be caused
by `topology`'s inventory content (distractor text + token load) rather than by
how the same graph is presented. Full `topology` keeps its spatial-encoding and
format roles; the
side pair (`topology` vs `topology_edges_only`) additionally reads as a free
does-irrelevant-content-hurt contrast on connectivity questions.

Runs only where a room connection graph (doors + open-plan passages) exists
(ProcTHOR); refuses elsewhere (Gibson, 3RScan).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from utils.capabilities import has_room_connectivity
from utils.models import Building, Room


def _degree(connectivity: dict[str, list[str]], rooms: dict[str, Room], rid: str) -> int:
    """Number of distinct neighbouring rooms reachable through a connection."""
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("topology_edges_only needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    connectivity = building.connectivity
    connection_count = sum(len(v) for v in connectivity.values()) // 2

    # Same ordering as topology/prose: degree desc, ties by id.
    ordered = sorted(
        rooms.values(),
        key=lambda r: (-_degree(connectivity, rooms, r.id), sort_key(r.id)),
    )

    # Head matches room_tree's content level: rooms + connections, nothing else.
    lines = [
        f"{building.name} — {len(rooms)} rooms, {connection_count} room "
        f"connections (doorways or open passages).",
        "",
    ]

    for room in ordered:
        neighbor_ids = sorted(
            {n for n in connectivity.get(room.id, []) if n in rooms},
            key=lambda nid: (-_degree(connectivity, rooms, nid), sort_key(nid)),
        )
        neighbors = ", ".join(room_label(rooms[nid]) for nid in neighbor_ids) or "(none)"
        lines.append(room_label(room))
        lines.append(f"  connects to: {neighbors}")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the room connectivity graph alone (no inventory, no metric data)")
