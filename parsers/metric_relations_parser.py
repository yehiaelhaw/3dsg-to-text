"""metric_relations — salient inter-room metric relations (distance + direction).

Pre-computes the metric relation an LLM would otherwise derive from raw
coordinates: an in-plane distance and an 8-way compass bearing between rooms.
This is the metric rung of the spatial-encoding ladder (compare
`topology_inventory`, which keeps only connectivity) -- see axes.metric_rung
for its Gibson companion card, the cleanest metric-rung exhibit in the study.

Density is controlled, not exhaustive: instead of every O(n^2) room pair, each
room lists only its `NEAREST_K` closest same-floor neighbours. Relations are
grouped by room -- each room heads a block of its nearest neighbours, nearest
first -- using the same floor-grouping and neighbour-ranking rule as
`navigation`'s proximity mode. Unlike that view, though, this one ALSO carries
a full per-room object inventory and a room-type census (see `_inventory_lines`
below), so it is not a bare geometry restatement and not a content-matched
counterpart to anything: an earlier design paired it directly with `navigation`
as if the two differed only in framing, and that pairing has been retired
because it does not hold -- the inventory and census are real extra content,
not a vantage change. The actual fact-matched relational-framing counterpart to
`navigation` on Gibson is `metric_framing` (metric_framing_parser.py), a
separate, purpose-built view carrying none of this view's extra content. This
view keeps its inventory and census unchanged, because they are what the
spatial-encoding ladder needs it for. Listing is directed: a pair appears
under both of its rooms, not de-duplicated.

`NEAREST_K` stays at the salient default of 4 -- the exhaustive all-pairs variant
this knob used to produce (`metric_relations_full`, the retracted density axis's
pole) is
retired: the axis was near-null (-0.043 AC) and the questions probing it were
partly circular (a global-extremum fact, like the farthest pair in the scene, is
answerable only by the exhaustive pole, so scoring the salient pole on it just
measures a missing-data guess, not a density effect).

Runs on any multi-room metric layout (Gibson, ProcTHOR); refuses on single-room
scenes (3RScan). The floor plane is detected per group (see `_geometry`), so
bearings are only emitted between rooms on the same floor.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, object_inventory, room_type_summary, sort_key
from _geometry import floor_plane, compass, plane_distance
from utils.capabilities import has_multiroom_layout, has_floors
from utils.models import Building, Room

# Each room keeps its K nearest same-floor neighbours. Set very high (>= room
# count) to recover the exhaustive all-pairs variant for the density ablation.
NEAREST_K = 4


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


def _grouped_relation_lines(rooms: list[Room]) -> list[str]:
    """Per-room blocks: each room heads its NEAREST_K nearest neighbours, nearest
    first. Directed -- a pair appears under both of its rooms -- and allocentric
    (the heading room is the subject: "is 3.2 m north-west of corridor [13]")."""
    positions = [r.position for r in rooms]
    u, v = floor_plane(positions)

    lines: list[str] = []
    for a in sorted(rooms, key=lambda r: sort_key(r.id)):
        ranked = sorted(
            ((plane_distance(a.position, b.position, (u, v)), b) for b in rooms if b is not a),
            key=lambda t: t[0],
        )[:NEAREST_K]
        if not ranked:
            continue
        lines.append(f"{room_label(a)}:")
        for d, b in ranked:
            du = a.position[u] - b.position[u]  # direction of a relative to b
            dv = a.position[v] - b.position[v]
            lines.append(f"  is {d:.1f} m {compass(du, dv)} of {room_label(b)}.")
    return lines


def _inventory_lines(rooms: list[Room]) -> list[str]:
    """One line per room: its objects by category (no coordinates)."""
    ordered = sorted(rooms, key=lambda r: (-len(r.objects), sort_key(r.id)))
    return [f"{room_label(r)} — contains: {object_inventory(r)}." for r in ordered]


def parse(building: Building) -> str:
    if not has_multiroom_layout(building):
        raise NotApplicable(
            "metric_relations needs >=2 rooms with positions; this scene has none"
        )

    groups = _floor_groups(building)
    placed_count = sum(len(rs) for _, rs in groups)
    # Once NEAREST_K reaches the largest group's neighbour count nothing is
    # dropped, so the view is the exhaustive all-pairs variant (density axis).
    exhaustive = NEAREST_K >= max((len(rs) for _, rs in groups), default=0) - 1

    lines: list[str] = []

    # -- Head --
    floor_desc = f", across {len(groups)} floors" if has_floors(building) else ""
    head = f"{building.name} — {placed_count} rooms{floor_desc}."
    type_str = room_type_summary(r for _, rs in groups for r in rs)
    if type_str:
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
            if exhaustive:
                note = "Spatial relations grouped by room (all neighbours, nearest first):"
            else:
                note = (
                    f"Spatial relations grouped by room (each room's {NEAREST_K} nearest "
                    "neighbours, nearest first):"
                )
            lines.append(note)
            lines.extend(_grouped_relation_lines(rooms))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize salient inter-room metric relations (distance + direction)")
