from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.graph_loader import load_3DSceneGraph, Building

_NDIGITS = 4


def _as_scalar(value):
    """Convert a numpy/scalar value to a plain JSON-compatible Python value."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_float_list(values) -> list[float]:
    """Convert a numeric iterable to rounded Python floats."""
    return [round(float(v), _NDIGITS) for v in values]


def _as_int(value) -> int | None:
    """Convert a numpy/scalar value to int, preserving None."""
    return None if value is None else int(value)


def _object_to_dict(obj) -> dict[str, Any]:
    """Serialize a scene-graph object to a dictionary."""
    return {
        "id": int(obj.id),
        "class": str(obj.class_),
        "parent_room": _as_int(obj.parent_room),
        "location_xyz": _as_float_list(obj.location),
        "size_xyz_m": _as_float_list(obj.size),
        "action_affordances": [str(a) for a in obj.action_affordance] if obj.action_affordance else [],
    }


def parse(building: Building) -> dict[str, Any]:
    """Convert a Building into a JSON-serializable dictionary of raw coordinates."""
    room_to_objects: dict[int, list] = {int(rid): [] for rid in building.room}
    unassigned_objects = []

    for obj in building.object.values():
        parent_room = _as_int(obj.parent_room)
        if parent_room is not None and parent_room in room_to_objects:
            room_to_objects[parent_room].append(obj)
        else:
            unassigned_objects.append(obj)

    rooms = []
    for room_id, room in sorted(building.room.items(), key=lambda item: int(item[0])):
        room_id = int(room_id)
        rooms.append({
            "id": room_id,
            "scene_category": str(room.scene_category),
            "floor_number": _as_scalar(room.floor_number),
            "location_xyz": _as_float_list(room.location),
            "size_xyz_m": _as_float_list(room.size),
            "objects": [
                _object_to_dict(obj)
                for obj in sorted(room_to_objects.get(room_id, []), key=lambda o: int(o.id))
            ],
        })

    return {
        "building": {
            "name": str(building.name),
            "size_xyz_m": _as_float_list(building.size),
            "total_rooms": len(building.room),
            "total_objects": len(building.object),
        },
        "rooms": rooms,
        "unassigned_objects": [
            _object_to_dict(obj)
            for obj in sorted(unassigned_objects, key=lambda o: int(o.id))
        ],
    }


def to_json_string(data: dict[str, Any], indent: int = 2) -> str:
    """Serialize parser output to a JSON string."""
    return json.dumps(data, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parse a 3D Scene Graph into raw-coordinates JSON"
    )
    ap.add_argument("--model", required=True, help="Gibson model name, e.g. Silas")
    ap.add_argument("--path", required=True, help="Path to folder containing the .npz file")
    ap.add_argument("--output", default=None, help="Optional path to save the output JSON file")
    args = ap.parse_args()

    print(f"Loading {args.model} from {args.path} ...")
    building, _ = load_3DSceneGraph(args.model, args.path)

    json_text = to_json_string(parse(building))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_text)
        print(f"Saved to {args.output}")
    else:
        print(json_text)