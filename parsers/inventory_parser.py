"""inventory — the no-spatial control.

Lists each room and the objects it contains, with no spatial structure at all:
no coordinates, no distances, no directions, no connectivity. It is the floor of
the spatial-encoding axis — the baseline that answers "what is in room X" and
"how many bathrooms" purely from containment plus the model's world knowledge.

Comparing every spatial representation against `inventory` isolates the value of
spatial structure itself: if a representation does not beat `inventory`, the
spatial information it adds was not used. Universal — runs on every scene.
"""

import sys
import os
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import room_label, object_inventory, sort_key
from utils.models import Building


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
        room_types = Counter(r.category for r in rooms if r.category)
        if room_types:
            type_str = ", ".join(
                f"{t} ({n})" for t, n in sorted(room_types.items(), key=lambda x: -x[1])
            )
            lines.append(f"Room types: {type_str}.")

    lines.append("")

    # -- Body: one line per room, objects by category, most-populated first --
    for room in sorted(rooms, key=lambda r: (-len(r.objects), sort_key(r.id))):
        lines.append(f"{room_label(room)} — contains: {object_inventory(room)}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a room/object inventory with no spatial structure")
