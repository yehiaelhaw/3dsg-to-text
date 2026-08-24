"""The scene registry consumed by the evaluation matrix: each Scene names its
scene_contexts/ dir, QA file, rep set, and pooling `role` (primary/stress/
sensitivity) -- the single source of truth run_experiments.py and
aggregate_results.py both read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# PRIMARY: the confirmatory experiment (all axis cards/verdicts/figures).
# STRESS: an operational limit (context overflow), reported as token fit only.
# SENSITIVITY: how much the verdict set depends on scene choice, reported separately.
# The pooling layer treats both non-primary roles identically (only `!= PRIMARY` matters).
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
    return f"experiments/qa/{scene_id}/keyfact-qa.jsonl"


# Explicit rep list pins the full structure-presentation ladder (topology ->
# room_tree -> graph_digest, connectivity-only). relations_* is excluded --
# linearization is near-trivial on ProcTHOR's forest (3RScan carries that axis).
# metric_relations_full is retired (near-null axis, -0.043, half its questions
# were circular).
_PROCTHOR_REPS = [
    "inventory", "topology_inventory", "topology", "room_tree", "graph_digest",
    # json_mini is the full-record anchor; json_pretty is the same content pretty-printed (the
    # formatting ablation's raw pole). The bare name `json` is retired -- name
    # explicitly.
    "prose", "metric_relations", "navigation", "json_mini", "json_pretty", "synthesis",
    # Matched counterpart to `navigation` on route/direction (door graph + per-edge
    # metric) -- proven fact-for-fact in test_topology_metric_equivalence.py.
    "topology_metric",
    # Content-matched counterpart to topology_inventory (prose sentences, same
    # channels) -- proven fact-for-fact in test_format_axis_equivalence.py.
    "narrative",
]

# Combos are retired: neither `topology+metric_relations` nor
# `graph_digest+metric_relations` has a REP_CAPS entry any more, so declaring one
# aborts a strict run. The loader still parses the concatenation syntax; no scene
# uses it.


SCENES: list[Scene] = [
    # ProcTHOR (spatial-encoding / format / structure-presentation axes). Three
    # 10-room trees (all connectivity graphs are trees).
    Scene("procthor_train1", _qa("procthor_train1"), _PROCTHOR_REPS),
    Scene("procthor_train232", _qa("procthor_train232"), _PROCTHOR_REPS),
    Scene("procthor_train314", _qa("procthor_train314"), _PROCTHOR_REPS),

    # 3RScan (relation-linearization axis): auto-discover; no combos, since
    # concatenating two relations_* views would mix poles of the same axis.
    #
    # Primary trio is a sparse -> medium -> dense gradient in relation count
    # (321/647/1304), selected under gates frozen before any response was
    # generated. 1d2f8518 is the medium scene: zero relation contradictions, the
    # G1-G5/P1-P4 argmin of the medium band.
    Scene("3rscan_02b33dfb", _qa("3rscan_02b33dfb"), None, role=PRIMARY),
    Scene("3rscan_1d2f8518", _qa("3rscan_1d2f8518"), None, role=PRIMARY),
    Scene("3rscan_0cac762f", _qa("3rscan_0cac762f"), None, role=PRIMARY),

    # Non-primary; kept for the STRESS/SENSITIVITY reporting artifacts (see ROLES
    # above). Both fail G1 (json_mini context headroom): 7f30f36c overflows
    # outright (STRESS); d7d40d62 fits under the 12% headroom bar (SENSITIVITY).
    # Real 3rscan_* ids, so axes.dataset_of() still resolves them to the host.
    Scene("3rscan_7f30f36c", _qa("3rscan_7f30f36c"), None, role=STRESS),
    Scene("3rscan_d7d40d62", _qa("3rscan_d7d40d62"), None, role=SENSITIVITY),

    # Gibson (spatial-encoding/reference-frame axes): auto-discover the 6 single
    # files; no door graph so no combos. Trio spans a floor-area gradient
    # (Brinnon 35 > Thrall > Donaldson 27 rooms).
    Scene("Brinnon", _qa("Brinnon"), None),
    Scene("Thrall", _qa("Thrall"), None),
    Scene("Donaldson", _qa("Donaldson"), None),
]

BY_ID: dict[str, Scene] = {s.scene_id: s for s in SCENES}
PRIMARY_SCENE_IDS: frozenset[str] = frozenset(
    s.scene_id for s in SCENES if s.role == PRIMARY)


class UnregisteredScene(KeyError):
    """A scene id with no entry in SCENES; raised rather than defaulted because any
    default role (primary or not) would silently corrupt the aggregate.
    """


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
