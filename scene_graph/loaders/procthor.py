import gzip
import json
import os
import warnings
from collections import defaultdict

from scene_graph.loaders.base import DatasetLoader
from scene_graph.models import Building, ObjectRelation, Room, SceneObject


def _room_id(room_str: str) -> str:
    return room_str.split("|")[1]


def _obj_category(obj_id: str) -> str:
    return obj_id.split("|")[0]


def _obj_room(obj_id: str) -> str:
    """Room number encoded in a ProcTHOR object id: the 2nd segment, except for
    surface children (``Type|surface|room|index``) where it's the 3rd -- reading
    the second-to-last segment instead misfiles asset-group members.
    """
    parts = obj_id.split("|")
    return parts[2] if parts[1] == "surface" else parts[1]


def _centroid(polygon: list[dict]) -> tuple[float, float, float]:
    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    return (sum(xs) / len(xs), 0.0, sum(zs) / len(zs))


def _wall_footprint(wall: dict) -> frozenset:
    """Floor-plan signature of a wall segment; the two rooms sharing a boundary
    each carry their own wall entry with matching (x, z) endpoints, so this
    rounded footprint pairs them up.
    """
    return frozenset((round(p["x"], 3), round(p["z"], 3)) for p in wall["polygon"])


def _components(adjacency: dict[str, list[str]]) -> int:
    """Number of connected components in the room graph."""
    seen: set[str] = set()
    count = 0
    for start in adjacency:
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node])
    return count


def _make_obj(obj: dict) -> SceneObject:
    pos = obj["position"]
    return SceneObject(
        id=obj["id"],
        category=_obj_category(obj["id"]),
        position=(float(pos["x"]), float(pos["y"]), float(pos["z"])),
    )


class ProcTHORLoader(DatasetLoader):
    def load(self, scene_id: str, data_path: str) -> Building:
        """Load a ProcTHOR scene. scene_id format: 'train:0', 'val:5', 'test:10'."""
        split, idx_str = scene_id.split(":")
        idx = int(idx_str)

        gz_path = os.path.join(data_path, f"{split}.jsonl.gz")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    house = json.loads(line)
                    break
            else:
                raise IndexError(f"Index {idx} out of range in {gz_path}")

        return self._parse(house, name=scene_id)

    def _parse(self, house: dict, name: str) -> Building:
        rooms: dict[str, Room] = {}
        for rdata in house["rooms"]:
            rid = _room_id(rdata["id"])
            rooms[rid] = Room(
                id=rid,
                category=rdata["roomType"],
                position=_centroid(rdata["floorPolygon"]),
            )

        loaded: set[str] = set()

        def add_obj(o: dict) -> bool:
            try:
                rid = _obj_room(o["id"])
            except (ValueError, IndexError):
                return False
            if rid not in rooms:
                return False
            rooms[rid].objects.append(_make_obj(o))
            loaded.add(o["id"])
            return True

        # Object relations come from the generator's structure, not id text alone:
        # `children` entries become "on" edges; `Type|room|group|member` ids
        # co-place furniture as one asset group, but nested members (faucet on
        # sink) already have an "on" edge, so group edges link only top-level
        # members to the anchor.
        relations: list[ObjectRelation] = []
        groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        for obj in house["objects"]:
            if add_obj(obj):
                parts = obj["id"].split("|")
                if len(parts) == 4 and parts[1] != "surface" and parts[3].isdigit():
                    groups[(parts[1], parts[2])].append((int(parts[3]), obj["id"]))
            for child in obj.get("children", []):
                if add_obj(child) and obj["id"] in loaded:
                    relations.append(ObjectRelation(child["id"], "on", obj["id"]))
        for members in groups.values():
            if len(members) > 1:
                members.sort()
                anchor = members[0][1]
                relations.extend(
                    ObjectRelation(mid, "arranged with", anchor) for _, mid in members[1:]
                )

        # Compact per-scene label ids (raw ids are noisy pipe strings), deterministic
        # so every parser run mints the same numbering; regenerate all contexts together.
        counter = 1
        for rid in sorted(rooms, key=int):
            for obj in rooms[rid].objects:
                obj.short_id = str(counter)
                counter += 1

        # Room adjacency = doors + open-plan passages (a wall segment flagged
        # `empty` on both sides); doors alone leave ~28% of houses spuriously
        # disconnected.
        connectivity: dict[str, list[str]] = {rid: [] for rid in rooms}

        def connect(r0: str, r1: str) -> None:
            if r0 != r1 and r0 in rooms and r1 in rooms and r1 not in connectivity[r0]:
                connectivity[r0].append(r1)
                connectivity[r1].append(r0)

        for door in house["doors"]:
            connect(_room_id(door["room0"]), _room_id(door["room1"]))

        sides: dict[frozenset, list[tuple[str, bool]]] = defaultdict(list)
        for wall in house.get("walls", []):
            parts = wall.get("roomId", "").split("|")
            if len(parts) > 1 and parts[1] in rooms:
                sides[_wall_footprint(wall)].append((parts[1], bool(wall.get("empty"))))
        for owners in sides.values():
            owner_rooms = {rid for rid, _ in owners}
            if len(owner_rooms) == 2 and all(empty for _, empty in owners):
                connect(*sorted(owner_rooms))

        if len(rooms) > 1 and _components(connectivity) > 1:
            warnings.warn(
                f"ProcTHOR scene {name}: room connectivity graph is disconnected "
                f"({_components(connectivity)} components). ProcTHOR houses are fully "
                "traversable, so this likely means an extraction gap (missed door or "
                "open-plan passage)."
            )

        return Building(
            name=name,
            rooms=rooms,
            connectivity=connectivity,
            object_relations=relations or None,
        )
