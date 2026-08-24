import os
import numpy as np
from scene_graph.loaders.base import DatasetLoader
from scene_graph.models import Building, Room, SceneObject


def _tuple3(arr) -> tuple[float, float, float] | None:
    if arr is None:
        return None
    try:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    except (TypeError, IndexError, ValueError):
        return None


def _str_list(val) -> list[str] | None:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, list) and len(val) > 0:
        return [str(v) for v in val]
    return None


class GibsonLoader(DatasetLoader):
    def load(self, scene_id: str, data_path: str) -> Building:
        npz_path = os.path.join(data_path, f"3DSceneGraph_{scene_id}.npz")
        data = np.load(npz_path, allow_pickle=True)["output"].item()
        bdata = data["building"]

        rooms_objects: dict[str, list[SceneObject]] = {}
        for object_id in np.unique(bdata["object_inst_segmentation"]):
            if object_id == 0:
                continue
            odata = data["object"][object_id]
            parent_room = str(int(odata.get("parent_room") or 0))
            obj = SceneObject(
                id=str(int(object_id)),
                category=str(odata["class_"]) if odata.get("class_") is not None else None,
                position=_tuple3(odata.get("location")),
                size=_tuple3(odata.get("size")),
                affordances=_str_list(odata.get("action_affordance")),
                material=_str_list(odata.get("material")),
                visual_texture=odata.get("visual_texture"),
                tactile_texture=odata.get("tactile_texture"),
            )
            rooms_objects.setdefault(parent_room, []).append(obj)

        rooms: dict[str, Room] = {}
        for room_id in np.unique(bdata["room_inst_segmentation"]):
            if room_id == 0:
                continue
            rid = str(int(room_id))
            rdata = data["room"][room_id]
            rooms[rid] = Room(
                id=rid,
                category=str(rdata["scene_category"]) if rdata.get("scene_category") is not None else None,
                position=_tuple3(rdata.get("location")),
                objects=rooms_objects.get(rid, []),
                floor=str(rdata["floor_number"]) if rdata.get("floor_number") is not None else None,
                size=_tuple3(rdata.get("size")),
                floor_area=float(rdata["floor_area"]) if rdata.get("floor_area") is not None else None,
                volume=float(rdata["volume"]) if rdata.get("volume") is not None else None,
            )

        return Building(
            name=str(bdata["name"]),
            rooms=rooms,
            size=_tuple3(bdata.get("size")),
        )
