"""topology_metric — the locative-framing pole of the navigation/topology_metric pair; states the
same door-graph edges and per-edge distances as navigation, but as "is D bearing of B" rather
than "you can walk to B, D bearing"."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from _geometry import floor_plane, compass, plane_distance
from scene_graph.capabilities import has_multiroom_layout, has_room_connectivity, has_floors
from scene_graph.models import Building, Room


def _floor_groups(building: Building) -> list[tuple[str | None, list[Room]]]:
    """Identical grouping to navigation_parser, so the two views split floors alike."""
    placed = [r for r in building.rooms.values() if r.position is not None]
    if has_floors(building):
        floors = sorted({r.floor for r in placed if r.floor is not None})
        groups = [(f, [r for r in placed if r.floor == f]) for f in floors]
        loose = [r for r in placed if r.floor is None]
        if loose:
            groups.append((None, loose))
        return groups
    return [(None, placed)]


def _degree(connectivity, rooms, rid) -> int:
    return len({n for n in connectivity.get(rid, []) if n in rooms})


def _placement(a: Room, b: Room, axes) -> tuple[float, str]:
    """(distance, bearing of `a` relative to `b`) -- `a` is the located figure; sign of the
    delta is the only computational difference from navigation_parser's `b - a`."""
    du = a.position[axes[0]] - b.position[axes[0]]
    dv = a.position[axes[1]] - b.position[axes[1]]
    return plane_distance(a.position, b.position, axes), compass(du, dv)


def parse(building: Building) -> str:
    if not has_multiroom_layout(building):
        raise NotApplicable(
            "topology_metric needs >=2 rooms with positions; this scene has none"
        )
    if not has_room_connectivity(building):
        raise NotApplicable(
            "topology_metric needs a room connection graph; this scene has none"
        )

    connectivity = building.connectivity
    rooms_by_id = building.rooms
    groups = _floor_groups(building)
    placed_count = sum(len(rs) for _, rs in groups)

    lines: list[str] = []
    # Room count and edge semantics only, no edge total (matches navigation's head line).
    lines.append(
        f"{building.name} — {placed_count} rooms. Room connections (doorways or open "
        f"passages), with each room placed relative to each of its connected neighbours."
    )

    multi = len(groups) > 1
    for floor, rooms in groups:
        positions = [r.position for r in rooms]
        axes = floor_plane(positions)
        # Same room order as navigation (degree desc, then id) so ordering is not
        # a second thing that differs between the two views.
        ordered = sorted(
            rooms, key=lambda r: (-_degree(connectivity, rooms_by_id, r.id), sort_key(r.id))
        )

        lines.append("")
        if multi:
            label = f"Floor {floor}" if floor is not None else "Unassigned floor"
            lines.append(f"{label} ({len(rooms)} rooms):")

        for room in ordered:
            neighbor_rooms = [
                rooms_by_id[n] for n in connectivity.get(room.id, [])
                if n in rooms_by_id and rooms_by_id[n] in rooms
            ]
            # Nearest first, matching navigation's ordering of the same list.
            targets = sorted(
                (_placement(room, b, axes) + (b,) for b in neighbor_rooms),
                key=lambda t: t[0],
            )
            if not targets:
                lines.append(f"{room_label(room)} — no connected rooms (isolated).")
                continue
            # "connected rooms" keeps the connectivity channel explicit, not just proximity.
            lines.append(f"{room_label(room)} — connected rooms:")
            for d, bearing, b in targets:
                lines.append(f"  is {d:.1f} m {bearing} of {room_label(b)}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the door graph with per-edge metric, in locative "
                      "(map) framing: each room placed relative to its connected neighbours")
