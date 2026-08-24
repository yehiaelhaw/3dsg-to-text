"""json — shared library building the scene-graph dict and both its serializations; not a
runnable parser itself (the shipped views are json_pretty_parser and json_mini_parser)."""

import json
import sys
import os
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scene_graph.loaders import REGISTRY, load
from scene_graph.models import Building

_NDIGITS = 4


def _roundf(v) -> float | None:
    return round(float(v), _NDIGITS) if v is not None else None


def _round3(t) -> list[float] | None:
    return [round(float(v), _NDIGITS) for v in t] if t is not None else None


def _compact(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != []}


def _object_to_dict(obj) -> dict[str, Any]:
    return _compact({
        "id": obj.id,
        "category": obj.category,
        "position": _round3(obj.position),
        "size": _round3(obj.size),
        "affordances": obj.affordances or None,
        "material": obj.material or None,
        "visual_texture": obj.visual_texture,
        "tactile_texture": obj.tactile_texture,
    })


def _sort_key(obj_id: str):
    try:
        return (0, int(obj_id))
    except ValueError:
        return (1, obj_id)


def parse(building: Building) -> dict[str, Any]:
    rooms = []
    for room in sorted(building.rooms.values(), key=lambda r: _sort_key(r.id)):
        rooms.append(_compact({
            "id": room.id,
            "category": room.category,
            "floor": room.floor,
            "position": _round3(room.position),
            "size": _round3(room.size),
            "floor_area": _roundf(room.floor_area),
            "volume": _roundf(room.volume),
            "objects": [_object_to_dict(o) for o in sorted(room.objects, key=lambda o: _sort_key(o.id))],
        }))

    relations = None
    if building.object_relations:
        relations = [
            {"subject": r.subject_id, "predicate": r.predicate, "object": r.object_id}
            for r in building.object_relations
        ]

    return _compact({
        "building": _compact({
            "name": building.name,
            "size": _round3(building.size),
            "connectivity": building.connectivity,
        }),
        "rooms": rooms,
        "relations": relations,
    })


def to_json_string(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


# `separators` is load-bearing: dropping `indent` alone leaves json.dumps' default
# `(", ", ": ")` and only recovers part of the saving.
def to_json_mini_string(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
