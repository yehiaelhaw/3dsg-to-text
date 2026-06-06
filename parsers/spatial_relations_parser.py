"""spatial_relations — inter-room metric relations (distance + cardinal direction).

Pre-computes, for every room pair on a floor, the metric relation an LLM would
otherwise have to derive from raw coordinates: an in-plane distance and an 8-way
compass bearing. This is the metric end of design-rationale Axis A and runs on
any scene with a multi-room metric layout (Gibson, ProcTHOR); it refuses on
single-room scenes (3RScan).

Coordinate note: the two datasets put "up" on different axes (Gibson is z-up;
ProcTHOR room centroids are x/z-plane with a constant-0 y). The parser is
dataset-agnostic, so rather than hard-code a floor plane it drops the
lowest-variance axis within each floor group — the up-axis falls out
automatically. Bearings are therefore only emitted between rooms on the same
floor, which is also what makes a compass direction meaningful.
"""

import sys
import os
import math
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, object_inventory
from utils.capabilities import has_multiroom_layout, has_floors
from utils.models import Building, Room

# Pairs farther apart than this (in metres, in-plane) are omitted to cap the
# O(n²) blow-up and the Lost-in-the-Middle dispersion penalty. The study scenes
# are small, so the default is effectively "all same-floor pairs"; lower it to
# restrict output to local adjacency.
MAX_DISTANCE_M = 1e9

# Bearings, counter-clockwise from +u (east), one per 45° sector.
_COMPASS = [
    "east", "north-east", "north", "north-west",
    "west", "south-west", "south", "south-east",
]


def _vertical_axis(positions: list[tuple[float, float, float]]) -> int:
    """Index of the axis to treat as 'up' — the one with the least spread."""
    variances = []
    for axis in range(3):
        vals = [p[axis] for p in positions]
        mean = sum(vals) / len(vals)
        variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
    return min(range(3), key=lambda a: variances[a])


def _compass(du: float, dv: float) -> str:
    """8-way bearing of the in-plane vector (du, dv); +u is east, +v is north."""
    sector = round(math.atan2(dv, du) / (math.pi / 4)) % 8
    return _COMPASS[sector]


def _floor_groups(building: Building) -> list[tuple[str | None, list[Room]]]:
    """Rooms with a position, grouped by floor when the scene has floors."""
    placed = [r for r in building.rooms.values() if r.position is not None]
    if has_floors(building):
        floors = sorted({r.floor for r in placed if r.floor is not None})
        groups = [(f, [r for r in placed if r.floor == f]) for f in floors]
        loose = [r for r in placed if r.floor is None]
        if loose:
            groups.append((None, loose))
        return groups
    return [(None, placed)]


def _relation_lines(rooms: list[Room]) -> list[str]:
    """One line per room pair within MAX_DISTANCE_M, nearest first."""
    positions = [r.position for r in rooms]
    up = _vertical_axis(positions)
    u, v = [a for a in range(3) if a != up]  # floor-plane axes, ascending

    pairs = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            du = a.position[u] - b.position[u]
            dv = a.position[v] - b.position[v]
            dist = math.hypot(du, dv)
            if dist <= MAX_DISTANCE_M:
                pairs.append((dist, a, b, du, dv))

    pairs.sort(key=lambda p: p[0])
    lines = []
    for dist, a, b, du, dv in pairs:
        bearing = _compass(du, dv)  # direction of a relative to b
        lines.append(f"{room_label(a)} is {dist:.1f} m {bearing} of {room_label(b)}.")
    return lines


def _inventory_lines(rooms: list[Room]) -> list[str]:
    """One line per room: its objects by category (no coordinates)."""
    ordered = sorted(rooms, key=lambda r: -len(r.objects))
    return [f"{room_label(r)} — contains: {object_inventory(r)}." for r in ordered]


def parse(building: Building) -> str:
    if not has_multiroom_layout(building):
        raise NotApplicable(
            "spatial_relations needs >=2 rooms with positions; this scene has none"
        )

    groups = _floor_groups(building)
    placed_count = sum(len(rs) for _, rs in groups)

    lines: list[str] = []

    # -- Head --
    floor_desc = f", across {len(groups)} floors" if has_floors(building) else ""
    head = f"{building.name} — {placed_count} rooms{floor_desc}."
    room_types = Counter(r.category for _, rs in groups for r in rs if r.category)
    if room_types:
        type_str = ", ".join(
            f"{t} ({n})" for t, n in sorted(room_types.items(), key=lambda x: -x[1])
        )
        head += f" Room types: {type_str}."
    lines.append(head)

    # -- Per floor group --
    multi = len(groups) > 1
    for floor, rooms in groups:
        lines.append("")
        if multi:
            label = f"Floor {floor}" if floor is not None else "Unassigned floor"
            lines.append(f"{label} ({len(rooms)} rooms):")

        lines.extend(_inventory_lines(rooms))

        if len(rooms) >= 2:
            lines.append("")
            lines.append("Spatial relations:")
            lines.extend(_relation_lines(rooms))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize inter-room metric relations (distance + cardinal direction)")
