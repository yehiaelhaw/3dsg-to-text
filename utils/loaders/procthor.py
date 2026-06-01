import gzip
import json
import os

from utils.loaders.base import DatasetLoader
from utils.models import Building, Room, SceneObject


def _room_id(room_str: str) -> str:
    return room_str.split("|")[1]


def _obj_category(obj_id: str) -> str:
    return obj_id.split("|")[0]


def _obj_room(obj_id: str) -> str:
    return obj_id.split("|")[-2]


def _centroid(polygon: list[dict]) -> tuple[float, float, float]:
    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    return (sum(xs) / len(xs), 0.0, sum(zs) / len(zs))


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

        for obj in house["objects"]:
            try:
                rid = _obj_room(obj["id"])
                if rid in rooms:
                    rooms[rid].objects.append(_make_obj(obj))
            except (ValueError, IndexError):
                pass
            for child in obj.get("children", []):
                try:
                    rid = _obj_room(child["id"])
                    if rid in rooms:
                        rooms[rid].objects.append(_make_obj(child))
                except (ValueError, IndexError):
                    pass

        connectivity: dict[str, list[str]] = {rid: [] for rid in rooms}
        for door in house["doors"]:
            r0 = _room_id(door["room0"])
            r1 = _room_id(door["room1"])
            if r0 != r1 and r0 in rooms and r1 in rooms:
                connectivity[r0].append(r1)
                connectivity[r1].append(r0)

        return Building(
            name=name,
            rooms=rooms,
            floor_count=1,
            connectivity=connectivity,
        )


def load_procthor(scene_id: str, data_path: str) -> Building:
    return ProcTHORLoader().load(scene_id, data_path)
