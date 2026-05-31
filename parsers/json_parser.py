import argparse
import json
import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loaders import load_gibson
from utils.models import Building

_NDIGITS = 4


def _round3(t) -> list[float] | None:
    return [round(float(v), _NDIGITS) for v in t] if t is not None else None


def _object_to_dict(obj) -> dict[str, Any]:
    return {
        "id": obj.id,
        "category": obj.category,
        "center_xyz": _round3(obj.position),
        "dimensions_xyz_m": _round3(obj.size),
        "affordances": obj.affordances or [],
    }


def parse(building: Building) -> dict[str, Any]:
    total_objects = sum(len(r.objects) for r in building.rooms.values())
    rooms = []
    for room in sorted(building.rooms.values(), key=lambda r: r.id):
        rooms.append({
            "id": room.id,
            "category": room.category,
            "floor": room.floor_number,
            "center_xyz": _round3(room.position),
            "dimensions_xyz_m": _round3(room.size),
            "objects": [_object_to_dict(o) for o in sorted(room.objects, key=lambda o: int(o.id))],
        })

    return {
        "building": {
            "name": building.name,
            "dimensions_xyz_m": _round3(building.size),
            "total_rooms": len(building.rooms),
            "total_objects": total_objects,
        },
        "rooms": rooms,
    }


def to_json_string(data: dict[str, Any], indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse a 3D Scene Graph into raw-coordinates JSON")
    ap.add_argument("--model", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"Loading {args.model} from {args.path} ...")
    building = load_gibson(args.model, args.path)
    json_text = to_json_string(parse(building))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_text)
        print(f"Saved to {args.output}")
    else:
        print(json_text)
