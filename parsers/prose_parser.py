import sys
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from _format import (
    room_label as _room_label,
    object_inventory as _object_inventory,
    room_type_summary as _room_type_summary,
    obj_label as _obj_label,
    sort_key as _id_key,
    is_attribute_predicate,
    undirected_components,
)
from utils.models import Building, Room


def _degree(connectivity: dict | None, rooms: dict, rid: str) -> int:
    """Distinct doorway-reachable neighbours — matches topology_inventory's degree."""
    if not connectivity:
        return 0
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def _room_line(room: Room, connectivity: dict | None, rooms: dict,
               show_count: bool = True) -> str:
    """One room as a sentence. `show_count` prints the explicit object tally.

    The tally is prose's own (`contains 22 objects: ...`); `topology_inventory`
    states only the list. `narrative` passes False so the format pair is a pure
    rendering flip -- see parsers/narrative_parser.py. The default keeps `prose`
    and `synthesis` byte-identical.
    """
    label = _room_label(room)
    obj_str = _object_inventory(room)
    parts = [label]

    if connectivity:
        # Same neighbour ordering as topology_inventory: degree desc, ties by id.
        neighbor_ids = sorted(
            {n for n in connectivity.get(room.id, []) if n in rooms},
            key=lambda nid: (-_degree(connectivity, rooms, nid), _id_key(nid)),
        )
        if neighbor_ids:
            neighbor_labels = [_room_label(rooms[nid]) for nid in neighbor_ids]
            parts.append(f"connects to {', '.join(neighbor_labels)}")

    if show_count:
        n = len(room.objects)
        parts.append(f"contains {n} object{'s' if n != 1 else ''}: {obj_str}")
    else:
        parts.append(f"contains: {obj_str}")
    return "; ".join(parts) + "."


def _relations_by_room(building: Building, room_sort_key) -> list[str]:
    """Object relations grouped by room, using natural-language 'is' phrasing."""
    if not building.object_relations:
        return []

    obj_room: dict[str, str] = {}
    id_to_label: dict[str, str] = {}
    for rid, room in building.rooms.items():
        for obj in room.objects:
            obj_room[obj.id] = rid
            id_to_label[obj.id] = _obj_label(obj.category, obj.short_id or obj.id)

    def lbl(oid: str) -> str:
        return id_to_label.get(oid, f"object [{oid}]")

    by_subject: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    same_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in building.object_relations:
        if rel.predicate.startswith("same "):
            same_edges[rel.predicate].append((rel.subject_id, rel.object_id))
        else:
            by_subject[rel.subject_id][rel.predicate].append(rel.object_id)

    def spatial_deg(sid: str) -> int:
        return sum(len(v) for p, v in by_subject[sid].items() if not is_attribute_predicate(p))

    def total_deg(sid: str) -> int:
        return sum(len(v) for v in by_subject[sid].values())

    def render(sid: str, attr_cap: int = 3) -> str:
        preds = by_subject[sid]
        spatial = sorted(p for p in preds if not is_attribute_predicate(p))
        comparative = sorted(p for p in preds if is_attribute_predicate(p))
        parts = [f"{pred} {', '.join(lbl(o) for o in preds[pred])}" for pred in spatial]
        for pred in comparative:
            objs = preds[pred]
            shown = ", ".join(lbl(o) for o in objs[:attr_cap])
            extra = len(objs) - attr_cap
            parts.append(f"{pred} {shown}" + (f" (+{extra} more)" if extra > 0 else ""))
        return f"  {lbl(sid)} is {'; '.join(parts)}." if parts else ""

    lines = ["", "Spatial relations by room:"]
    found = False
    for room in sorted(building.rooms.values(), key=room_sort_key):
        subjects = sorted(
            [sid for sid in by_subject if obj_room.get(sid) == room.id],
            key=lambda sid: (-spatial_deg(sid), -total_deg(sid), _id_key(sid)),
        )
        room_lines = [l for l in (render(sid) for sid in subjects) if l]
        if room_lines:
            found = True
            lines.append(f"{_room_label(room)}:")
            lines.extend(room_lines)

    if not found:
        return []

    # Symmetric+transitive "same X" edges: one statement per equivalence group.
    group_lines = []
    for pred in sorted(same_edges):
        attr = pred[len("same "):]
        comps = [sorted(set(m), key=_id_key) for m in undirected_components(same_edges[pred])]
        comps.sort(key=lambda g: _id_key(g[0]))
        for g in comps:
            group_lines.append(f"  Same {attr}: {', '.join(lbl(o) for o in g)}.")
    if group_lines:
        lines += ["", "  Shared attributes (each line lists items sharing that property):"]
        lines.extend(group_lines)

    return lines


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

    type_str = _room_type_summary(rooms.values())
    if type_str:
        lines.append(f"Room types: {type_str}.")

    if connectivity:
        connection_count = sum(len(v) for v in connectivity.values()) // 2
        if connection_count:
            lines.append(
                f"Rooms are connected by {connection_count} room "
                f"connection{'s' if connection_count != 1 else ''} (doorways or open passages)."
            )

    lines.append("")

    # -- Body: rooms -- (same ordering as topology_inventory: degree desc, ties by id,
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
    lines.extend(_relations_by_room(building, sort_key))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a 3D scene to natural language prose")
