"""navigation — the navigational-framing route view.

Describes the scene the way a person moves through it: from each room, where you
can go and in which direction. It is the NAVIGATIONAL-framing pole of the
"relational vs. navigational framing" axis (traversal register: "you can walk
to", proximity register: "nearby"). Its matched relational-framing counterpart is
`topology_metric` on the ProcTHOR door graph, and `metric_framing` on Gibson's
K-nearest-neighbour geometry — both restate exactly this view's edges and
distances with the figure-ground assignment reversed and a locative register
("is ... of") instead; see either parser's docstring for why that reversal is
not an egocentric/allocentric flip.

BLOCK LAYOUT IS MATCHED TO THE COUNTERPART, ON PURPOSE
-------------------------------------------------------
Until 2026-08-21 this view joined a room's neighbours into one semicolon-joined
sentence ("From X you can walk to: A (..); B (..)."), while `topology_metric`/
`metric_framing` used a header line plus one indented clause per neighbour. That
was an accidental structural confound, not a framing choice -- a design review
found it and this file was rewritten to use the SAME block shape as its
counterpart: identical header text ("X — connected rooms:" / "X -- nearest
neighbours:", framing-neutral, structural), then one indented line per neighbour,
nearest first, same room order, same numeric precision. The framing register now
lives entirely in the per-neighbour clause ("you can walk to B, D bearing." vs
`topology_metric`'s "is D bearing of B."), which is where it belongs -- see
tests/test_framing_layout_isolation.py, which canonicalizes away the lexical
register and asserts the remaining document structure is identical.

It is also the only single non-`json` representation that fuses connectivity
with metric direction, so it is the natural single view for navigation
questions.

Two modes, chosen from the data:
- With a room connection graph (ProcTHOR): list each room's reachable neighbours
  (through doorways or open passages). These are genuine moves — "you can walk to".
- Without one (Gibson): list each room's nearest same-floor neighbours. There is
  no traversability guarantee, so these are framed honestly as proximity — named
  plainly with no traversal claim, unlike the connectivity-mode clause above.

Runs on any multi-room metric layout; refuses on single-room scenes (3RScan).
"""

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
                # Header text is IDENTICAL to topology_metric's -- "connected rooms" is
                # a structural fact, not a framing choice, so it carries no register.
                # The framing register (traversal vs locative) lives entirely in the
                # per-neighbour line below, matching topology_metric's block shape:
                # one header line, then one indented clause per neighbour, nearest first.
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
                # Header text is IDENTICAL to metric_framing's -- same reasoning as the
                # connectivity-mode header above. The item line stays honestly
                # proximity-framed (no traversal claim: Gibson has no door graph), and
                # differs from metric_framing's "is ... of" locative anchor by naming
                # the neighbour first rather than anchoring it to the heading room.
                lines.append(f"{room_label(room)} -- nearest neighbours:")
                for d, br, b in ranked:
                    lines.append(f"  {room_label(b)}, {d:.1f} m {br} away.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize a navigational-framing route view (where you can go from each room)")
