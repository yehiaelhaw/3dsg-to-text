import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import (
    room_label as _room_label,
    object_inventory as _object_inventory,
    relation_lines,
    sort_key as _id_key,
)
from utils.models import Building, Room


def _degree(connectivity: dict | None, rooms: dict, rid: str) -> int:
    """Distinct doorway-reachable neighbours — matches topology's degree."""
    if not connectivity:
        return 0
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def _room_line(room: Room, connectivity: dict | None, rooms: dict) -> str:
    label = _room_label(room)
    obj_str = _object_inventory(room)
    parts = [label]

    if connectivity:
        # Same neighbour ordering as topology: degree desc, ties by id.
        neighbor_ids = sorted(
            {n for n in connectivity.get(room.id, []) if n in rooms},
            key=lambda nid: (-_degree(connectivity, rooms, nid), _id_key(nid)),
        )
        if neighbor_ids:
            neighbor_labels = [_room_label(rooms[nid]) for nid in neighbor_ids]
            parts.append(f"connects to {', '.join(neighbor_labels)}")

    n = len(room.objects)
    parts.append(f"contains {n} object{'s' if n != 1 else ''}: {obj_str}")
    return "; ".join(parts) + "."


def _relations_lines(building: Building) -> list[str]:
    rel = relation_lines(building, indent="  ")
    return ["", "Spatial relations:", *rel] if rel else []


def parse(building: Building) -> str:
    rooms = building.rooms
    connectivity = building.connectivity
    floors = sorted({r.floor for r in rooms.values() if r.floor})

    lines = []

    # -- Head --
    room_count = len(rooms)
    total_objects = sum(len(r.objects) for r in rooms.values())

    if room_count == 1:
        room = next(iter(rooms.values()))
        cat = room.category or "room"
        lines.append(f"{building.name} is a single {cat} containing {total_objects} objects.")
    else:
        floor_desc = (
            f" across {len(floors)} floor{'s' if len(floors) != 1 else ''}"
            if floors else ""
        )
        lines.append(
            f"{building.name} contains {room_count} rooms{floor_desc} and {total_objects} objects."
        )

    room_types = Counter(r.category for r in rooms.values() if r.category)
    if room_types:
        type_str = ", ".join(
            f"{t} ({n})" for t, n in sorted(room_types.items(), key=lambda x: -x[1])
        )
        lines.append(f"Room types: {type_str}.")

    if connectivity:
        connection_count = sum(len(v) for v in connectivity.values()) // 2
        if connection_count:
            lines.append(
                f"Rooms are connected by {connection_count} room "
                f"connection{'s' if connection_count != 1 else ''} (doorways or open passages)."
            )

    lines.append("")

    # -- Body: rooms -- (same ordering as topology: degree desc, ties by id,
    # then object count as a final tiebreaker for the no-connectivity case)
    def sort_key(r: Room) -> tuple:
        return (-_degree(connectivity, rooms, r.id), _id_key(r.id), -len(r.objects))

    if len(floors) > 1:
        for floor in floors:
            floor_rooms = sorted([r for r in rooms.values() if r.floor == floor], key=sort_key)
            lines.append(f"Floor {floor} ({len(floor_rooms)} rooms):")
            for room in floor_rooms:
                lines.append("  " + _room_line(room, connectivity, rooms))
            lines.append("")
        no_floor = sorted([r for r in rooms.values() if not r.floor], key=sort_key)
        for room in no_floor:
            lines.append(_room_line(room, connectivity, rooms))
    else:
        for room in sorted(rooms.values(), key=sort_key):
            lines.append(_room_line(room, connectivity, rooms))

    # -- Spatial relations (3DSSG only) --
    lines.extend(_relations_lines(building))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a 3D scene to natural language prose")
