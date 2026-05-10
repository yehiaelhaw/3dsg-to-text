from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.graph_loader import load_3DSceneGraph, Building


def parse_raw_coordinates(building: Building) -> str:
    """Convert a Building into a human-readable raw-coordinates text block."""
    lines = []

    # Building header
    lines.append(f"Building: {building.name}")
    lines.append(
        f"Size (x/y/z metres): {building.size[0]:.2f} x "
        f"{building.size[1]:.2f} x {building.size[2]:.2f}"
    )
    lines.append(f"Total rooms: {len(building.room)}")
    lines.append(f"Total objects: {len(building.object)}")
    lines.append("")

    # Map each room to its objects
    room_to_objects: dict[int, list] = {rid: [] for rid in building.room}
    unassigned_objects = []

    for obj in building.object.values():
        if obj.parent_room is not None and obj.parent_room in building.room:
            room_to_objects[obj.parent_room].append(obj)
        else:
            unassigned_objects.append(obj)

    # Rooms
    for room_id, room in sorted(building.room.items()):
        loc, sz = room.location, room.size
        lines.append(
            f"Room {room_id} [{room.scene_category}] (Floor {room.floor_number})"
        )
        lines.append(f"  Location (x/y/z): ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})")
        lines.append(f"  Size (x/y/z metres): {sz[0]:.2f} x {sz[1]:.2f} x {sz[2]:.2f}")

        objects_in_room = room_to_objects.get(room_id, [])
        if objects_in_room:
            lines.append(f"  Objects ({len(objects_in_room)}):")
            for obj in sorted(objects_in_room, key=lambda o: o.id):
                lines.append(_format_object(obj, indent=4))
        else:
            lines.append("  Objects: none")
        lines.append("")

    # Unassigned objects
    if unassigned_objects:
        lines.append("Objects with no assigned room:")
        for obj in sorted(unassigned_objects, key=lambda o: o.id):
            lines.append(_format_object(obj, indent=2))
        lines.append("")

    return "\n".join(lines)


def _format_object(obj, indent: int = 4) -> str:
    """Format a single object as a descriptive one-liner."""
    pad = " " * indent
    loc, sz = obj.location, obj.size
    affordances = ", ".join(obj.action_affordance) if obj.action_affordance else "none"
    return (
        f"{pad}- {obj.class_} (id={int(obj.id)})"
        f"  location: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})"
        f"  size: {sz[0]:.2f} x {sz[1]:.2f} x {sz[2]:.2f} m"
        f"  affordances: [{affordances}]"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parse a 3D Scene Graph into raw-coordinates text"
    )
    ap.add_argument("--model", required=True, help="Gibson model name, e.g. Silas")
    ap.add_argument("--path", required=True, help="Path to folder containing the .npz file")
    ap.add_argument("--output", default=None, help="Optional path to save the output text file")
    args = ap.parse_args()

    print(f"Loading {args.model} from {args.path} ...")
    building, _ = load_3DSceneGraph(args.model, args.path)

    text = parse_raw_coordinates(building)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {args.output}")
    else:
        print(text)