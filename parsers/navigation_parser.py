"""navigation — the navigational-framing pole of the relational/navigational framing axis
(traversal register "you can walk to", matched against topology_metric on ProcTHOR and
metric_framing on Gibson, both in locative register); falls back to honestly-labelled
proximity when no room connection graph exists."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from _geometry import floor_plane, compass, plane_distance
from scene_graph.capabilities import has_multiroom_layout, has_room_connectivity, has_floors
from scene_graph.models import Building, Room

# In the door-free (proximity) mode, how many nearest neighbours each room lists.
NEAREST_K = 4


def _floor_groups(building: Building) -> list[tuple[str | None, list[Room]]]:
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


def _bearing(a: Room, b: Room, axes) -> tuple[float, str]:
    du = b.position[axes[0]] - a.position[axes[0]]  # direction of b relative to a (navigational framing)
    dv = b.position[axes[1]] - a.position[axes[1]]
    return plane_distance(a.position, b.position, axes), compass(du, dv)


def parse(building: Building) -> str:
    if not has_multiroom_layout(building):
        raise NotApplicable(
            "navigation needs >=2 rooms with positions; this scene has none"
        )

    connectivity = building.connectivity if has_room_connectivity(building) else None
    rooms_by_id = building.rooms
    groups = _floor_groups(building)
    placed_count = sum(len(rs) for _, rs in groups)

    lines: list[str] = []
    mode = "room connections (doorways or open passages)" if connectivity else "proximity"
    head = f"{building.name} — {placed_count} rooms. Routes by {mode}."
    lines.append(head)

    multi = len(groups) > 1
    for floor, rooms in groups:
        positions = [r.position for r in rooms]
        axes = floor_plane(positions)

        if connectivity:
            ordered = sorted(
                rooms, key=lambda r: (-_degree(connectivity, rooms_by_id, r.id), sort_key(r.id))
            )
        else:
            ordered = sorted(rooms, key=lambda r: sort_key(r.id))

        lines.append("")
        if multi:
            label = f"Floor {floor}" if floor is not None else "Unassigned floor"
            lines.append(f"{label} ({len(rooms)} rooms):")

        for room in ordered:
            if connectivity:
                neighbor_rooms = [
                    rooms_by_id[n] for n in connectivity.get(room.id, [])
                    if n in rooms_by_id and rooms_by_id[n] in rooms
                ]
                targets = sorted(
                    (_bearing(room, b, axes) + (b,) for b in neighbor_rooms),
                    key=lambda t: t[0],
                )
                if not targets:
                    # Same wording as topology_metric's isolated line -- this fact is
                    # framing-neutral, so there is no reason for the two texts to differ.
                    lines.append(f"{room_label(room)} — no connected rooms (isolated).")
                    continue
                # Header text matches topology_metric's; the framing register lives only
                # in the per-neighbour line below.
                lines.append(f"{room_label(room)} — connected rooms:")
                for d, br, b in targets:
                    lines.append(f"  you can walk to {room_label(b)}, {d:.1f} m {br}.")
            else:
                others = [b for b in rooms if b is not room]
                ranked = sorted(
                    (_bearing(room, b, axes) + (b,) for b in others),
                    key=lambda t: t[0],
                )[:NEAREST_K]
                if not ranked:
                    # Matches metric_framing's handling of a floor with <2 rooms: no
                    # block at all, rather than a header over an empty list.
                    continue
                # Header text matches metric_framing's; item line stays honestly proximity-framed
                # since Gibson has no door graph to claim traversal from.
                lines.append(f"{room_label(room)} -- nearest neighbours:")
                for d, br, b in ranked:
                    lines.append(f"  {room_label(b)}, {d:.1f} m {br} away.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a navigational-framing route view (where you can go from each room)")
