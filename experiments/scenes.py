"""scenes.py -- the scene registry consumed by the evaluation matrix.

Each Scene names a scene_contexts/<scene_id>/ directory, the QA file authored for
it under experiments/scripts/<scene_id>/, and the representation set to score
(None -> auto-discover every single file present; an explicit list is needed only
to pin or exclude specific views). This is the single source
of truth for the per-scene rep set -- run_experiments.py reads it, so the list
lives here and nowhere else.

No scene declares an "a+b" combo any more, and the two that once existed are named
in scope.RETIRED_REPS, so declaring one again aborts the run (see the retirement
note below the ProcTHOR list). The loader still parses the syntax; that path is now
unexercised by the study.
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
# explicit list pins the full structure-presentation ladder
# (topology_edges_only -> room_tree -> graph_digest; all
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
    # The matched counterpart to `navigation` on route/direction: the door graph
    # with per-edge metric, in locative framing. Carries exactly navigation's
    # channels {connectivity, metric_edges} -- proven fact-for-fact, not just by
    # REP_CAPS, in evaluation/tests/test_topology_metric_equivalence.py.
    "topology_metric",
]

# COMBOS ARE RETIRED AND PURGED (retired 2026-08-11, rows deleted 2026-08-12).
#
# `topology+metric_relations` and `graph_digest+metric_relations` were scored on
# all three ProcTHOR scenes under the Gemini judge. Those 242 rows have been
# DELETED from experiments/results/ (one raw copy sits outside the active tree, in
# experiments/backups/combo-purge-2026-08-12/). No reporting, plotting or test path
# is combo-aware any more, and both names are declared in scope.RETIRED_REPS, so
# re-declaring one aborts the run instead of quietly re-entering the analysis.
#
# Why they went: the only comparison that REQUIRED a concatenation was the
# ProcTHOR route baseline, because no single view carried connectivity and
# per-edge metric together. `topology_metric` now does, by design and matched to
# `navigation` fact-for-fact, so the requirement is gone. The combo's other
# question -- "can salient views match the json_mini ceiling?" -- is already owned
# by `synthesis`, which is a curated single document rather than a concatenation,
# so no distinct research question is left that concatenation answers.
#
# The concatenation MECHANISM stays in scene_loader/scope -- it is small and
# tested, and removing it would be a load-path change with no user. This list is
# simply where it stops being used.


SCENES: list[Scene] = [
    # ProcTHOR (spatial-encoding / format / structure-presentation axes). Three
    # 10-room trees (all connectivity graphs are trees).
    Scene("procthor_train1", _qa("procthor_train1"), _PROCTHOR_REPS),
    Scene("procthor_train232", _qa("procthor_train232"), _PROCTHOR_REPS),
    Scene("procthor_train314", _qa("procthor_train314"), _PROCTHOR_REPS),

    # 3RScan (relation-linearization axis): auto-discover the 9 single files. No
    # combos -- concatenating two relations_* views would mix poles of the same
    # axis. Relation density rises across the trio, but not evenly: after the
    # label-inferable `same object type` filter applied at load the counts are
    # 02b33dfb 321, d7d40d62 1501, 7f30f36c 1541 -- one comparatively sparse scene
    # and two substantially denser ones that sit close together. (The 355 / 3971
    # figures this comment used to quote were pre-filter, and overstated the spread.)
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
