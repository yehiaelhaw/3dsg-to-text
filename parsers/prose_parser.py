import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import room_label as _room_label, object_inventory as _object_inventory, relation_lines
from utils.models import Building, Room


def _room_line(room: Room, connectivity: dict | None, rooms: dict) -> str:
    label = _room_label(room)
    obj_str = _object_inventory(room)
    parts = [label]

    if connectivity:
        neighbors = [nid for nid in sorted(connectivity.get(room.id, [])) if nid in rooms]
        if neighbors:
            neighbor_labels = [_room_label(rooms[nid]) for nid in neighbors]
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
        door_count = sum(len(v) for v in connectivity.values()) // 2
        if door_count:
            lines.append(f"Rooms are connected by {door_count} door{'s' if door_count != 1 else ''}.")

    lines.append("")

    # -- Body: rooms --
    def sort_key(r: Room) -> tuple:
        degree = len(connectivity.get(r.id, [])) if connectivity else 0
        return (-degree, -len(r.objects))

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
