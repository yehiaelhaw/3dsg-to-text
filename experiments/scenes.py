"""scenes.py -- the scene registry consumed by the evaluation matrix.

Each Scene names a scene_contexts/<scene_id>/ directory, the QA file authored for
it under experiments/scripts/<scene_id>/, and the representation set to score
(None -> auto-discover every single file present; an explicit list is needed only
to add "a+b" combos or to pin/exclude specific views). This is the single source
of truth for the per-scene rep set -- run_experiments.py reads it, so the list
lives here and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Scene:
    scene_id: str                                  # == scene_contexts/<scene_id>/ dir
    dataset_path: str
    representations: Optional[list[str]] = None    # None -> auto-discover singles


def _qa(scene_id: str) -> str:
    return f"experiments/scripts/{scene_id}/keyfact-qa.jsonl"


# ProcTHOR rep set (shared across its scenes for gradient comparability): the
# explicit list adds the two orthogonal combos and pins the full structure-
# presentation ladder (topology_edges_only -> room_tree -> graph_digest; all
# connectivity-only, so full topology's inventories can't masquerade as a
# presentation effect). relations_* are deliberately excluded
# -- relation linearization is near-trivial on ProcTHOR's on-forest (its deep-dive
# lives on 3RScan). metric_relations_full (the retracted density axis's exhaustive
# pole) is retired: the axis was near-null (-0.043) and half its questions were
# circular (only the exhaustive pole could answer the farthest-pair facts it was
# scored on).
_PROCTHOR_REPS = [
    "inventory", "topology", "topology_edges_only", "room_tree", "graph_digest",
    # json_mini is the ceiling; json_pretty is the same content pretty-printed, kept
    # as the formatting ablation's raw pole (axes.json_formatting). The bare name
    # `json` is retired -- scope.RETIRED_REPS aborts any run that still names it.
    "prose", "metric_relations", "navigation", "json_mini", "json_pretty", "synthesis",
    "topology+metric_relations", "graph_digest+metric_relations",
]


SCENES: list[Scene] = [
    # ProcTHOR (spatial-encoding / format / structure-presentation axes). Three
    # 10-room trees (all connectivity graphs are trees).
    Scene("procthor_train1", _qa("procthor_train1"), _PROCTHOR_REPS),
    Scene("procthor_train232", _qa("procthor_train232"), _PROCTHOR_REPS),
    Scene("procthor_train314", _qa("procthor_train314"), _PROCTHOR_REPS),

    # 3RScan (relation-linearization axis): auto-discover the 9 single files. No
    # combos -- concatenating two relations_* views would mix poles of the same
    # axis. The trio spans the
    # density gradient (02b33dfb 355 rels < d7d40d62 < 7f30f36c 3971 rels).
    Scene("3rscan_02b33dfb", _qa("3rscan_02b33dfb"), None),
    Scene("3rscan_d7d40d62", _qa("3rscan_d7d40d62"), None),
    Scene("3rscan_7f30f36c", _qa("3rscan_7f30f36c"), None),

    # Gibson (spatial-encoding / reference-frame axes): auto-discover the 6 single
    # files. No door graph, so no
    # orthogonal channel to cross with the metric views -> no combos. The trio
    # spans a floor-area gradient (Brinnon 35 rooms > Thrall > Donaldson 27).
    Scene("Brinnon", _qa("Brinnon"), None),
    Scene("Thrall", _qa("Thrall"), None),
    Scene("Donaldson", _qa("Donaldson"), None),
]
