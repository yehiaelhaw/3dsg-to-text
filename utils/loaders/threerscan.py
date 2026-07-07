import json
import os

from utils.loaders.base import DatasetLoader
from utils.models import Building, ObjectRelation, Room, SceneObject


def _tuple3(lst) -> tuple[float, float, float] | None:
    if lst is None:
        return None
    try:
        return (float(lst[0]), float(lst[1]), float(lst[2]))
    except (TypeError, IndexError, ValueError):
        return None


def _round3(lst) -> tuple[float, float, float] | None:
    t = _tuple3(lst)
    if t is None:
        return None
    return (round(t[0], 4), round(t[1], 4), round(t[2], 4))


class ThreeRScanLoader(DatasetLoader):
    def load(self, scene_id: str, data_path: str) -> Building:
        dssg_path = os.path.normpath(os.path.join(data_path, "..", "3DSSG"))

        # Step 1 — geometry from semseg.v2.json
        semseg_path = os.path.join(data_path, scene_id, "semseg.v2.json")
        with open(semseg_path, encoding="utf-8") as f:
            semseg = json.load(f)
        semseg_by_id: dict[str, dict] = {
            str(seg["objectId"]): seg for seg in semseg["segGroups"]
        }

        # Step 2 — rich metadata from 3DSSG objects.json
        meta_by_id: dict[str, dict] = {}
        obj_path = os.path.join(dssg_path, "objects.json")
        if os.path.isfile(obj_path):
            with open(obj_path, encoding="utf-8") as f:
                obj_data = json.load(f)
            for entry in obj_data["scans"]:
                if entry["scan"] == scene_id:
                    meta_by_id = {o["id"]: o for o in entry["objects"]}
                    break

        # Step 3 — build SceneObjects by merging geometry + metadata
        objects: list[SceneObject] = []
        for obj_id, seg in semseg_by_id.items():
            meta = meta_by_id.get(obj_id, {})
            category = meta.get("label") or seg["label"]
            affordances = meta.get("affordances") or None
            objects.append(SceneObject(
                id=obj_id,
                category=category,
                position=_round3(seg["obb"]["centroid"]),
                size=_round3(seg["obb"]["axesLengths"]),
                affordances=affordances,
            ))

        # Step 4 — single room for the whole scan
        room = Room(id=scene_id, objects=objects)

        # Step 5 — semantic relations from 3DSSG relationships.json
        relations: list[ObjectRelation] | None = None
        rel_path = os.path.join(dssg_path, "relationships.json")
        if os.path.isfile(rel_path):
            with open(rel_path, encoding="utf-8") as f:
                rel_data = json.load(f)
            for entry in rel_data["scans"]:
                if entry["scan"] == scene_id:
                    relations = [
                        ObjectRelation(str(r[0]), r[3], str(r[1]))
                        for r in entry["relationships"]
                        # "same object type" is label-inferable (the category is
                        # already printed on both endpoints), unlike the other
                        # "same ..." predicates (color/material/texture/shape/
                        # state/symmetry/as), which encode attributes not present
                        # in the label. Dropped at load so every downstream view
                        # (relations_*, prose, synthesis, json) is unaffected, not
                        # just relations_digest where it was first noticed.
                        if r[3] != "same object type"
                    ]
                    break

        return Building(
            name=scene_id,
            rooms={scene_id: room},
            object_relations=relations,
        )
