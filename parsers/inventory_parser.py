"""inventory — the non-spatial anchor of the spatial-encoding axis; room + object containment only, no
coordinates, distances, directions, or connectivity. Universal, runs on every scene."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import room_label, object_inventory, room_type_summary, sort_key
from scene_graph.models import Building


def parse(building: Building) -> str:
    rooms = list(building.rooms.values())
    total_objects = sum(len(r.objects) for r in rooms)

    lines: list[str] = []

    # -- Head: counts + room-type summary, no spatial facts --
    if len(rooms) == 1:
        room = rooms[0]
        cat = room.category or "room"
        lines.append(f"{building.name} — single {cat} with {total_objects} objects.")
    else:
        lines.append(f"{building.name} — {len(rooms)} rooms, {total_objects} objects.")
        type_str = room_type_summary(rooms)
        if type_str:
            lines.append(f"Room types: {type_str}.")

    lines.append("")

    # -- Body: one line per room, objects by category, most-populated first --
    for room in sorted(rooms, key=lambda r: (-len(r.objects), sort_key(r.id))):
        lines.append(f"{room_label(room)} — contains: {object_inventory(room)}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a room/object inventory with no spatial structure")
