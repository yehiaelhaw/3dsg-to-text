"""topology_metric — the door graph with per-edge metric, in locative (map) framing.

The matched counterpart to `navigation`. Both views state the SAME edge set with
the SAME distances; they differ in which room is the located figure and which is
the reference ground, and in the framing verb:

    navigation:       From Bedroom [6] you can walk to: Bathroom [7] (2.9 m north-east); ...
    topology_metric:  Bedroom [6] — connected rooms:
                        is 2.9 m south-west of Bathroom [7].

A NOTE ON TERMINOLOGY (deliberately not "egocentric vs allocentric")
--------------------------------------------------------------------
Both views print WORLD-FRAME compass bearings computed by `_geometry.compass`,
and no ProcTHOR room carries a facing or heading -- there is no observer
orientation anywhere in the data to be egocentric with respect to. The contrast
here is therefore NOT a coordinate-frame transform. It is exactly two things:

  1. FIGURE-GROUND ASSIGNMENT (which room is the subject of the relation).
     `navigation` locates the neighbour against the heading room; this view
     locates the heading room against the neighbour. The two statements are
     converses of one binary spatial relation -- `NE(b, a)` iff `SW(a, b)` -- so
     they are informationally equivalent by construction, which the equivalence
     test asserts fact-for-fact rather than assuming.
  2. FRAMING REGISTER: traversal ("you can walk to") vs locative ("is ... of").

Calling that an egocentric/allocentric flip would overclaim: reversing the
argument order of a symmetric-invertible relation is not a change of reference
frame. `metric_framing` vs `navigation` on Gibson has the same property (see
metric_framing_parser.py's identical note) -- that pair is the Gibson companion
to this one, matched fact-for-fact by tests/test_metric_framing_equivalence.py.
(An earlier design paired `metric_relations` directly with `navigation` on
Gibson on this same "pure frame flip" premise; that pairing has been retired
because `metric_relations` additionally carries a per-room object inventory and
room-type census `navigation` lacks, so it was not in fact matched.
`metric_relations` itself is unchanged and still serves the spatial-encoding
ladder.)

NEITHER VIEW PRINTS A ROUTE
---------------------------
Both state ONE-HOP doorway adjacency and stop there: no path, no hop count, no
summed distance, no reachability grouping. The multi-hop search and the addition
that a route question asks for are the model's work on both sides. The exhibit
that reads them on `route` is therefore named for the FRAMING of that adjacency
(navigational/action-oriented vs relational/locative), not for "route
presentation" -- which would imply one side hands the answer over.

HOW THE TWO QUESTION FAMILIES MUST BE READ DIFFERENTLY
------------------------------------------------------
  * `route` -- an audit of all 12 ProcTHOR route stems found NO bearing in any
    key fact: routes need the edge set, the per-edge distances and room labels
    only. So on route this pair is a fact-matched test of FRAMING over identical
    task-relevant facts, and a route delta must NOT be attributed to the
    figure-ground reversal, which the task does not read. Nor should it be called
    a presentation- or layout-only effect: the manipulation bundles figure-ground
    assignment, block layout and lexical/semantic register ("you can walk to" is
    an action, "is ... of" is a location), and this design cannot decompose them.
  * `direction` -- all 12 ProcTHOR direction stems ask about a DIRECTLY CONNECTED
    pair, so both doorway-restricted views can answer every one, and the key
    facts are phrased from the heading room. There the reversal IS task-relevant:
    this view must invert the printed relation to answer. That is a reasoning
    asymmetry over equal information, and it is the clean matched contrast --
    but it is NOT identified: the inversion arrives inside the same bundle as
    the layout and lexical-framing differences, and since those could help or
    hurt, a direction delta reflects the bundled manipulation and is not a bound
    on the inversion's own cost in either direction.

WHY THIS PARSER EXISTS
----------------------
The contrast needs a counterpart carrying exactly `navigation`'s channels:
connectivity AND per-edge metric. No other single view does -- `topology_inventory`
has no metric, `metric_relations` has no connectivity (and its metric is arbitrary-pair,
not edge-restricted). The gap used to be filled by concatenating two views, which
bought the channels at the price of a strict superset: the concatenation also
carried object inventories (stated twice, once per part) and metric between
UNCONNECTED rooms, so any delta against `navigation` confounded the contrast with
that surplus. This is a designed schema instead of a concatenation, matched to
`navigation` fact-for-fact.

WHAT IT DELIBERATELY DOES NOT CARRY
-----------------------------------
Every omission is an item `navigation` also lacks; including any would hand this
view an informational advantage and destroy the match:
  * the scene's total edge count (`topology`/`graph_digest` print it);
  * reachability groups, hubs, bottlenecks, hop counts (`graph_digest`'s tier);
  * any multi-hop path or summed distance -- the search and the addition stay the
    model's work, which is what a route question exists to measure;
  * object inventories (no route or direction stem in the corpus references an
    object -- verified against all 24 ProcTHOR stems);
  * distance/bearing between rooms that are NOT directly connected (that is the
    `metric` channel; carrying it would change this view's scope, not its wording).

Gate: needs BOTH a room connection graph and positions, so it runs on ProcTHOR
only. Gibson has no doors (its pair stays `metric_relations` vs `navigation`) and
3RScan is single-room.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from _geometry import floor_plane, compass, plane_distance
from utils.capabilities import has_multiroom_layout, has_room_connectivity, has_floors
from utils.models import Building, Room


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
    """(distance, bearing of `a` relative to `b`) -- `a` is the located figure.

    The delta's sign is the ONLY computational difference from navigation_parser,
    which takes `b.position - a.position` to locate the neighbour instead. The
    two bearings are exact converses (negating an in-plane vector shifts the
    8-way sector by exactly 4), and distance is symmetric, so both views round to
    byte-identical metres.
    """
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
    # States the room count and the edge SEMANTICS but not the edge TOTAL --
    # navigation's head line does the same, and the total is a derived fact this
    # view must not carry alone.
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
            # "connected rooms" is load-bearing, not decoration: without it these
            # blocks read as proximity and the connectivity channel -- the one a
            # route question needs -- is silently lost. It states no count.
            lines.append(f"{room_label(room)} — connected rooms:")
            for d, bearing, b in targets:
                lines.append(f"  is {d:.1f} m {bearing} of {room_label(b)}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the door graph with per-edge metric, in locative "
                      "(map) framing: each room placed relative to its connected neighbours")
