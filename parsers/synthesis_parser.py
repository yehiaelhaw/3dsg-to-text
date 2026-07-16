"""synthesis - the best-of-axes default representation.

Derived from the screening per-question-type winners (experiments/results/
qwen2.5-14b/): each evaluation axis's strongest serialization is composed into
ONE compact natural-language document. This is a *merge*, not a concatenation --
naive `a+b+c` combos lose to plain prose and bloat tokens, so the merge keeps a
single head, a single per-room inventory, states connectivity once as derived
facts, keeps the metric salient, and states object relations at both the raw
and derived tier, once each:

  * prose backbone (NL room inventory + object-relation section) -- the format
    that wins aggregation / object_relation / set_logic. Reused verbatim via
    `prose_parser._room_line` and `prose_parser._relations_by_room`.
  * graph_digest's derived connectivity facts (reachability groups, hubs,
    bottlenecks, multi-step distances) -- wins connectivity. Folded in via
    `graph_digest_parser.connectivity_digest`, so the facts are byte-identical to
    the standalone `graph_digest` view. The per-room line keeps the *local* 1-hop
    adjacency ("connects to ...") the digest omits, so the two are complementary
    (local adjacency + global derived structure), not a restatement.
  * metric_relations' salient per-room distance+bearing (its NEAREST_K nearest
    neighbours) -- wins proximity / containment. Only the relation lines are
    taken; the prose backbone already carries the inventory the metric parser
    would otherwise repeat. That dropped duplicate inventory (plus the dropped
    second head) is what keeps this smaller than the losing `prose+graph_digest+
    metric_relations` concatenation.
  * relations_digest's derived object-relation facts (support depth,
    receptacles, proximity clusters, shared-attribute cliques, relation census)
    -- wins relation_structure / relation_aggregate on 3RScan. Folded in via
    `relations_digest_parser.object_relations_digest`, byte-identical to the
    standalone `relations_digest` view, appended after prose's raw per-object
    section the same way graph_digest's global facts sit alongside topology's
    local adjacency: raw triples (specific-pair lookups) + derived structure
    (chain-depth/cluster/census lookups), not a restatement. Added 2026-07-14 --
    until then this section was missing and `synthesis` silently under-performed
    `relations_digest` on both derived-tier types with no way to close the gap;
    see `evaluation/scope.py`'s `synthesis` entry for the scope-model side of
    that correction.

Egocentric routing (`navigation`) is deliberately *excluded*. Its edge on
direction/route comes from the first-person reference frame, which cannot be
folded into an allocentric document without becoming navigation; it stays the
per-type override a router applies for routing questions.

Universal: each section appears only when its capability is present, so the rep
degrades gracefully -- to prose on a single-room 3RScan scene, to inventory +
metric on a door-free Gibson scene, to the full document on ProcTHOR.
"""

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
from utils.capabilities import has_room_connectivity, has_multiroom_layout, has_object_relations
from utils.models import Building, Room


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
    """Salient per-room distance+bearing lines, reusing metric_relations'
    `_grouped_relation_lines` (identical to the standalone metric view, minus its
    head and the inventory the prose backbone already carries). [] if empty."""
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

    # Object relations, raw tier (prose's section): prepends its own blank line +
    # header, or contributes nothing on scenes without object relations. Its
    # trailing "Shared attributes" clique rollup is cut when the derived section
    # below is also present -- both run the identical undirected_components
    # grouping over the same "same X" edges, so keeping both would restate the
    # same cliques twice under two different headers.
    raw_relations = _relations_by_room(building, sort_key)
    header_idx = next(
        (i for i, l in enumerate(raw_relations) if l.strip().startswith("Shared attributes (")),
        None,
    )
    if header_idx is not None:
        raw_relations = raw_relations[:header_idx - 1]  # drop blank line + header + groups
    lines.extend(raw_relations)

    # Object relations, derived tier: same facts as the standalone relations_digest
    # view (including the shared-attribute cliques cut from the raw tier above).
    # Guarded separately (not folded into _relations_by_room) so it degrades the
    # same way the connectivity digest does on scenes without the channel.
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
