"""Scene registry for experiment execution and aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Scene roles: confirmatory, operational stress, and sensitivity analysis.
PRIMARY = "primary"
STRESS = "stress"
SENSITIVITY = "sensitivity"
ROLES = (PRIMARY, STRESS, SENSITIVITY)


@dataclass(frozen=True)
class Scene:
    scene_id: str                                  # == scene_contexts/<scene_id>/ dir
    dataset_path: str
    representations: Optional[list[str]] = None    # None -> auto-discover singles
    role: str = PRIMARY                            # primary | stress | sensitivity

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"{self.scene_id}: unknown scene role {self.role!r} "
                f"(expected one of {', '.join(ROLES)})")


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

    # 3RScan primary scenes span sparse, medium, and dense relation graphs.
    Scene("3rscan_02b33dfb", _qa("3rscan_02b33dfb"), None, role=PRIMARY),
    Scene("3rscan_1d2f8518", _qa("3rscan_1d2f8518"), None, role=PRIMARY),
    Scene("3rscan_0cac762f", _qa("3rscan_0cac762f"), None, role=PRIMARY),

    # Non-primary 3RScan scenes for stress and sensitivity analyses.
    Scene("3rscan_7f30f36c", _qa("3rscan_7f30f36c"), None, role=STRESS),
    Scene("3rscan_d7d40d62", _qa("3rscan_d7d40d62"), None, role=SENSITIVITY),

    # Gibson primary scenes span a floor-area gradient; representations auto-discover.
    Scene("Brinnon", _qa("Brinnon"), None),
    Scene("Thrall", _qa("Thrall"), None),
    Scene("Donaldson", _qa("Donaldson"), None),
]

BY_ID: dict[str, Scene] = {s.scene_id: s for s in SCENES}
PRIMARY_SCENE_IDS: frozenset[str] = frozenset(
    s.scene_id for s in SCENES if s.role == PRIMARY)


class UnregisteredScene(KeyError):
    """Raised when a scene has no registry entry."""


def role_of(scene_id: str) -> str:
    """Declared role of a registered scene. Fails closed on an unknown id."""
    try:
        return BY_ID[scene_id].role
    except KeyError:
        raise UnregisteredScene(
            f"{scene_id!r} is not in experiments/scenes.py. Register it with an "
            f"explicit role ({', '.join(ROLES)}) before it can be pooled."
        ) from None


def is_primary(scene_id: str) -> bool:
    """True iff this scene may enter a pooled primary aggregate."""
    return role_of(scene_id) == PRIMARY
