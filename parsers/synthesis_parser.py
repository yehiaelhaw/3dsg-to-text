"""synthesis — the best-of-axes default representation; merges (not concatenates) the prose
room inventory with graph_digest's connectivity, metric_relations' salient distances, and
relations_digest's derived object-relation structure into one document, each section
appearing only when the underlying capability is present."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import room_type_summary, sort_key as _id_key
from graph_digest_parser import connectivity_digest
from prose_parser import _room_line, _relations_by_room, _degree as _prose_degree
from metric_relations_parser import (
    _floor_groups,
    _grouped_relation_lines,
    NEAREST_K as _METRIC_K,
)
from relations_digest_parser import object_relations_digest
from scene_graph.capabilities import has_room_connectivity, has_multiroom_layout, has_object_relations
from scene_graph.models import Building, Room


def _head(building: Building) -> list[str]:
    """Compact orientation line(s): scale, room types, connection count."""
    rooms = building.rooms
    room_count = len(rooms)
    total_objects = sum(len(r.objects) for r in rooms.values())
    floors = sorted({r.floor for r in rooms.values() if r.floor})

    lines: list[str] = []
    if room_count == 1:
        room = next(iter(rooms.values()))
        cat = room.category or "room"
        lines.append(f"{building.name} is a single {cat} containing {total_objects} objects.")
    else:
        floor_desc = (
            f" across {len(floors)} floor{'s' if len(floors) != 1 else ''}" if floors else ""
        )
        lines.append(
            f"{building.name} contains {room_count} rooms{floor_desc} and {total_objects} objects."
        )

    type_str = room_type_summary(rooms.values())
    if type_str:
        lines.append(f"Room types: {type_str}.")

    if has_room_connectivity(building):
        cc = sum(len(v) for v in building.connectivity.values()) // 2
        if cc:
            lines.append(
                f"Rooms are connected by {cc} room connection{'s' if cc != 1 else ''} "
                "(doorways or open passages)."
            )
    return lines


def _room_sort_key(building: Building):
    """Prose's room ordering: doorway degree desc, then id, then object count."""
    connectivity = building.connectivity
    rooms = building.rooms
    return lambda r: (-_prose_degree(connectivity, rooms, r.id), _id_key(r.id), -len(r.objects))


def _room_body(building: Building, sort_key) -> list[str]:
    """Per-room inventory (+ local adjacency) lines, reusing prose's `_room_line`."""
    rooms = building.rooms
    connectivity = building.connectivity
    floors = sorted({r.floor for r in rooms.values() if r.floor})

    lines = ["Rooms:"]
    if len(floors) > 1:
        for floor in floors:
            floor_rooms = sorted([r for r in rooms.values() if r.floor == floor], key=sort_key)
            lines.append(f"Floor {floor} ({len(floor_rooms)} rooms):")
            for room in floor_rooms:
                lines.append("  " + _room_line(room, connectivity, rooms))
            lines.append("")
        for room in sorted([r for r in rooms.values() if not r.floor], key=sort_key):
            lines.append("  " + _room_line(room, connectivity, rooms))
    else:
        for room in sorted(rooms.values(), key=sort_key):
            lines.append("  " + _room_line(room, connectivity, rooms))
    return lines


def _metric_block(building: Building) -> list[str]:
    """Salient per-room distance+bearing lines, reusing metric_relations' `_grouped_relation_lines`
    without its head/inventory (the prose backbone already carries those). [] if empty."""
    groups = _floor_groups(building)
    multi = len(groups) > 1

    body: list[str] = []
    for floor, rooms in groups:
        if len(rooms) < 2:
            continue
        if multi:
            label = f"Floor {floor}" if floor is not None else "Unassigned floor"
            body.append(f"{label} ({len(rooms)} rooms):")
        body.extend(_grouped_relation_lines(rooms))

    if not body:
        return []
    header = (
        f"Salient metric relations (each room's {_METRIC_K} nearest neighbours, "
        "nearest first):"
    )
    return [header, *body]


def parse(building: Building) -> str:
    lines = _head(building)

    if has_room_connectivity(building):
        lines.append("")
        lines.append("Connectivity (derived from the room connection graph; no metric data):")
        lines.extend(connectivity_digest(building))

    sort_key = _room_sort_key(building)
    lines.append("")
    lines.extend(_room_body(building, sort_key))

    if has_multiroom_layout(building):
        metric = _metric_block(building)
        if metric:
            lines.append("")
            lines.extend(metric)

    # Trailing "Shared attributes" clique rollup is cut here since the derived section
    # below restates the same cliques under its own header.
    raw_relations = _relations_by_room(building, sort_key)
    header_idx = next(
        (i for i, l in enumerate(raw_relations) if l.strip().startswith("Shared attributes (")),
        None,
    )
    if header_idx is not None:
        raw_relations = raw_relations[:header_idx - 1]  # drop blank line + header + groups
    lines.extend(raw_relations)

    # Guarded separately (not folded into _relations_by_room) so it degrades independently.
    if has_object_relations(building):
        lines.append("")
        lines.append(
            "Derived object-relation structure (support depth, receptacles, "
            "proximity clusters, shared attributes, relation census):"
        )
        lines.extend(object_relations_digest(building))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the synthesized best-of-axes representation")
