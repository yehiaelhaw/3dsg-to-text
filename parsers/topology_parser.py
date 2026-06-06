"""topology — connectivity-first serialization.

Foregrounds the room adjacency graph and deliberately drops all metric data.
This is the connectivity end of design-rationale Axis A: it tests whether
preserving "which room opens into which" — at the expense of coordinates and
distances — improves an LLM's spatial reasoning. Runs only where a door graph
exists (ProcTHOR); refuses elsewhere (Gibson, 3RScan).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from utils.capabilities import has_room_connectivity
from utils.models import Building, Room


def _degree(connectivity: dict[str, list[str]], rooms: dict[str, Room], rid: str) -> int:
    """Number of distinct neighbouring rooms reachable through a door."""
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def _content_lines(room: Room) -> list[str]:
    """Bulleted object inventory by category (no positions, no metrics)."""
    if not room.objects:
        return ["  - (no objects)"]
    counts = Counter(o.category or "object" for o in room.objects)
    return [
        f"  - {cat} (×{n})" if n > 1 else f"  - {cat}"
        for cat, n in sorted(counts.items(), key=lambda x: -x[1])
    ]


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("topology needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    connectivity = building.connectivity

    # Most-connected room first (GraphInsight importance ordering); ties by id.
    ordered = sorted(
        rooms.values(),
        key=lambda r: (-_degree(connectivity, rooms, r.id), sort_key(r.id)),
    )
    door_count = sum(len(v) for v in connectivity.values()) // 2

    lines = [
        f"{building.name} — {len(rooms)} rooms connected by {door_count} doors.",
        "",
        "Room connectivity (most-connected first):",
    ]
    for room in ordered:
        neighbor_ids = sorted(
            {n for n in connectivity.get(room.id, []) if n in rooms},
            key=lambda nid: (-_degree(connectivity, rooms, nid), sort_key(nid)),
        )
        neighbors = ", ".join(room_label(rooms[nid]) for nid in neighbor_ids) or "(none)"
        lines.append(f"{room_label(room)} — connected to: {neighbors}")

    lines.append("")
    lines.append("Room contents:")
    for room in ordered:
        lines.append(f"{room_label(room)}:")
        lines.extend(_content_lines(room))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the room connectivity graph (no metric data)")
