"""metric_framing — the locative-framing pole of the Gibson navigation/metric_framing pair
(the K-nearest-neighbour analogue of topology_metric's door-graph pairing on ProcTHOR)."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from _geometry import floor_plane, compass, plane_distance
from scene_graph.capabilities import has_multiroom_layout, has_floors
from scene_graph.models import Building, Room

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
    """Per-room blocks of its NEAREST_K nearest same-floor neighbours, nearest first, ranked by
    ascending planar distance to match metric_relations/navigation's proximity mode."""
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
