"""Scene registry for experiment execution and aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Scene:
    scene_id: str                                  # == scene_contexts/<scene_id>/ dir
    dataset_path: str
    representations: Optional[list[str]] = None    # None -> auto-discover singles


def _qa(scene_id: str) -> str:
    return f"experiments/qa/{scene_id}.jsonl"


# Explicit ProcTHOR representation set; relation-linearization is evaluated on 3RScan.
_PROCTHOR_REPS = [
    "inventory", "topology_inventory", "topology", "room_tree", "graph_digest",
    # Full-record representations used for the formatting comparison.
    "prose", "metric_relations", "navigation", "json_mini", "json_pretty", "synthesis",
    # Content-matched navigation counterpart; see test_topology_metric_equivalence.py.
    "topology_metric",
    # Content-matched topology_inventory counterpart; see test_format_axis_equivalence.py.
    "narrative",
]

SCENES: list[Scene] = [
    Scene("procthor_train1", _qa("procthor_train1"), _PROCTHOR_REPS),
    Scene("procthor_train232", _qa("procthor_train232"), _PROCTHOR_REPS),
    Scene("procthor_train314", _qa("procthor_train314"), _PROCTHOR_REPS),

    # 3RScan scenes span sparse, medium, and dense relation graphs.
    Scene("3rscan_02b33dfb", _qa("3rscan_02b33dfb"), None),
    Scene("3rscan_1d2f8518", _qa("3rscan_1d2f8518"), None),
    Scene("3rscan_0cac762f", _qa("3rscan_0cac762f"), None),

    # Gibson scenes span a floor-area gradient; representations auto-discover.
    Scene("Brinnon", _qa("Brinnon"), None),
    Scene("Thrall", _qa("Thrall"), None),
    Scene("Donaldson", _qa("Donaldson"), None),
]

BY_ID: dict[str, Scene] = {s.scene_id: s for s in SCENES}


class UnregisteredScene(KeyError):
    """Raised when a scene has no registry entry."""


def check_registered(scene_id: str) -> None:
    """Fail closed if this scene has no registry entry."""
    if scene_id not in BY_ID:
        raise UnregisteredScene(
            f"{scene_id!r} is not in experiments/scenes.py. Register it there "
            f"before it can be pooled."
        )
