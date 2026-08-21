"""metric_framing — the matched Gibson companion to `navigation`, in relational
(locative) framing.

The Gibson realization of the "relational vs. navigational framing" axis
(compare `topology_metric`, its ProcTHOR sibling on the door graph). Both
views state the SAME K-nearest-neighbour room geometry, in the SAME block
layout (identical header text, one indented clause per neighbour, nearest
first, same room order -- see tests/test_framing_layout_isolation.py); they
differ only in which room is the located figure and which is the reference
ground, and in the framing register:

    navigation:      Bedroom [6] -- nearest neighbours:
                       Bathroom [7], 2.9 m north-east away.
    metric_framing:  Bedroom [6] -- nearest neighbours:
                       is 2.9 m south-west of Bathroom [7].

A NOTE ON TERMINOLOGY (deliberately not "egocentric vs allocentric")
--------------------------------------------------------------------
As with `topology_metric` on ProcTHOR: both views print WORLD-FRAME compass
bearings, and no Gibson room carries a facing or heading, so there is no
observer orientation here to be egocentric with respect to. The contrast is
NOT a coordinate-frame transform. It is exactly two things:

  1. FIGURE-GROUND ASSIGNMENT. `navigation` locates the neighbour against the
     heading room; this view locates the heading room against the neighbour.
     The two statements are converses of one binary spatial relation, so they
     are informationally equivalent by construction -- asserted fact-for-fact
     by tests/test_metric_framing_equivalence.py, not assumed.
  2. FRAMING REGISTER: proximity-traversal ("nearest rooms") vs locative
     ("is ... of").

Call this RELATIONAL framing vs NAVIGATIONAL framing. Never egocentric vs
allocentric -- see `topology_metric_parser.py`'s identical note.

WHY THIS PARSER EXISTS
-----------------------
The original Gibson companion paired `metric_relations` directly against
`navigation`, on the premise that the two differed only in anchoring. They do
not: `metric_relations` additionally prints a full per-room object inventory
(see `metric_relations_parser.py`) and a room-type census that `navigation`
lacks entirely -- an undisclosed channel, and a ~1.43x length mismatch on the
primary scenes, found during a design review of the framing axis.
`metric_relations` keeps that inventory unchanged, because it also serves the
spatial-encoding ladder, where it belongs (see `axes.metric_rung`). This
parser is a new, purpose-built pole carrying EXACTLY `navigation`'s channels
and nothing else -- the same relationship `topology_metric` has to
`navigation` on ProcTHOR, and the reason this file exists rather than
`metric_relations` being edited in place.

Deliberately DUPLICATES its floor-grouping and neighbour-ranking logic rather
than importing from `metric_relations_parser.py` or `navigation_parser.py`:
every parser in this study is self-contained, so a future edit to a sibling
file can never silently change this one's guaranteed-matched output. Compare
`topology_metric_parser.py`'s identical duplication of `navigation_parser`'s
`_floor_groups`.

WHAT IT DELIBERATELY DOES NOT CARRY
------------------------------------
Every omission is an item `navigation` also lacks; including any would hand
this view an informational advantage and destroy the match:
  * per-room object inventory (`metric_relations`'s distinguishing extra);
  * the room-type census line (`metric_relations`'s "Room types: ..." head
    fragment);
  * any derived fact (edge counts, clustering, hop counts);
  * distance/bearing to rooms beyond the K nearest (that is the arbitrary-pair
    `metric` channel, not `metric_edges` -- see REP_CAPS).

Gate: needs >=2 rooms with positions (`has_multiroom_layout`), same as
`metric_relations`/`navigation`. Not host-restricted in code -- the study
uses it on Gibson only, but nothing here assumes the absence of a door graph.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from _geometry import floor_plane, compass, plane_distance
from utils.capabilities import has_multiroom_layout, has_floors
from utils.models import Building, Room

# Same K as `metric_relations`/`navigation`'s proximity mode -- this pole must
# reproduce their neighbour selection exactly, not choose its own value.
NEAREST_K = 4


def _floor_groups(building: Building) -> list[tuple[str | None, list[Room]]]:
    """Identical grouping to metric_relations_parser/navigation_parser, so all
    three views split floors alike."""
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
    """Per-room blocks: each room heads its NEAREST_K nearest same-floor
    neighbours, nearest first, in locative framing. Ranking is by ascending
    planar distance, same rule and room-iteration order as
    `metric_relations`/`navigation`'s proximity mode, so ties (if any) break
    identically -- verified empirically for the three evaluated scenes rather
    than assumed (no exact-distance tie falls at the K=4 boundary on
    Brinnon/Thrall/Donaldson; see tests/test_metric_framing_equivalence.py)."""
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
        lines.append(f"{room_label(a)} -- nearest neighbours:")
        for d, b in ranked:
            du = a.position[u] - b.position[u]  # direction of a relative to b
            dv = a.position[v] - b.position[v]
            lines.append(f"  is {d:.1f} m {compass(du, dv)} of {room_label(b)}.")
    return lines


def parse(building: Building) -> str:
    if not has_multiroom_layout(building):
        raise NotApplicable(
            "metric_framing needs >=2 rooms with positions; this scene has none"
        )

    groups = _floor_groups(building)
    placed_count = sum(len(rs) for _, rs in groups)

    lines: list[str] = []
    # Room count only -- no room-type census (`metric_relations`'s "Room
    # types: ..." fragment), matching `navigation`'s head line exactly.
    lines.append(
        f"{building.name} -- {placed_count} rooms. Each room's position stated "
        f"relative to its {NEAREST_K} nearest same-floor neighbours."
    )

    multi = len(groups) > 1
    for floor, rooms in groups:
        lines.append("")
        if multi:
            label = f"Floor {floor}" if floor is not None else "Unassigned floor"
            lines.append(f"{label} ({len(rooms)} rooms):")

        if len(rooms) >= 2:
            lines.extend(_relation_lines(rooms))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize Gibson K-NN room geometry in relational "
                      "(locative) framing: each room placed relative to its "
                      "nearest same-floor neighbours")
