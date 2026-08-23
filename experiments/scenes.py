"""scenes.py -- the scene registry consumed by the evaluation matrix.

Each Scene names a scene_contexts/<scene_id>/ directory, the QA file authored for
it under experiments/qa/<scene_id>/, and the representation set to score
(None -> auto-discover every single file present; an explicit list is needed only
to pin or exclude specific views). This is the single source
of truth for the per-scene rep set -- run_experiments.py reads it, so the list
lives here and nowhere else.

No scene declares an "a+b" combo any more, and the two that once existed are named
in scope.RETIRED_REPS, so declaring one again aborts the run (see the retirement
note below the ProcTHOR list). The loader still parses the syntax; that path is now
unexercised by the study.

Each Scene also carries a `role` -- the pool it may enter when results are POOLED.
It is a property of the study design, not of the data, so it lives here and not in a
directory name: `aggregate_results.py` reads it, `axes.dataset_of()` never sees it,
and a non-primary scene keeps its ordinary `3rscan_*` directory so it still resolves
to its host dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- scene roles -----------------------------------------------------------
# PRIMARY   -- carries the confirmatory experiment. Every axis card, paired table,
#              verdict, headline count and thesis figure is computed over these and
#              only these.
# STRESS    -- kept to demonstrate an OPERATIONAL limit (a representation that cannot
#              fit the responder context at all). Reported as token fit/overflow only,
#              never as accuracy.
# SENSITIVITY -- kept so that "how much does the verdict set depend on which scenes
#              were chosen?" can be answered from already-cached results, in its own
#              section, outside the primary verdicts.
#
# The two non-primary roles are distinguished for the reader; the pooling layer treats
# them identically, because the only property that matters there is `!= PRIMARY`.
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


# ProcTHOR rep set (shared across its scenes for gradient comparability): the
# explicit list pins the full structure-presentation ladder
# (topology -> room_tree -> graph_digest; all
# connectivity-only, so full topology_inventory's inventories can't masquerade as a
# presentation effect). relations_* are deliberately excluded
# -- relation linearization is near-trivial on ProcTHOR's on-forest (its deep-dive
# lives on 3RScan). metric_relations_full (the retracted density axis's exhaustive
# pole) is retired: the axis was near-null (-0.043) and half its questions were
# circular (only the exhaustive pole could answer the farthest-pair facts it was
# scored on).
_PROCTHOR_REPS = [
    "inventory", "topology_inventory", "topology", "room_tree", "graph_digest",
    # json_mini is the ceiling; json_pretty is the same content pretty-printed, kept
    # as the formatting ablation's raw pole (axes.json_formatting). The bare name
    # `json` is retired -- scope.RETIRED_REPS aborts any run that still names it.
    "prose", "metric_relations", "navigation", "json_mini", "json_pretty", "synthesis",
    # The matched counterpart to `navigation` on route/direction: the door graph
    # with per-edge metric, in locative framing. Carries exactly navigation's
    # channels {connectivity, metric_edges} -- proven fact-for-fact, not just by
    # REP_CAPS, in evaluation/tests/test_topology_metric_equivalence.py.
    "topology_metric",
    # The content-matched counterpart to topology_inventory on the format axis:
    # its facts rendered as prose sentences instead of labelled blocks. Carries
    # exactly topology_inventory's channels {inventory, connectivity} -- proven
    # fact-for-fact in evaluation/tests/test_format_axis_equivalence.py. `prose`
    # (above) was the original partner but is a content superset, not a twin.
    "narrative",
]

# COMBOS ARE RETIRED AND PURGED.
#
# `topology+metric_relations` and `graph_digest+metric_relations` were scored on
# all three ProcTHOR scenes under the Gemini judge. Those 242 rows have been
# DELETED from experiments/results/ (one raw copy is kept under
# experiments/backups/). No reporting, plotting or test path is combo-aware any
# more, and both names are declared in scope.RETIRED_REPS, so re-declaring one
# aborts the run instead of quietly re-entering the analysis.
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

    # 3RScan (relation-linearization axis): auto-discover the single files. No
    # combos -- concatenating two relations_* views would mix poles of the same
    # axis.
    #
    # The primary trio is an even sparse -> medium -> dense gradient in post-filter
    # relation count (321 -> 647 -> 1304; consecutive ratios 2.016x / 2.015x), selected
    # under gates frozen before any response was generated. It
    # replaced the old trio, whose two dense scenes sat close together (1501 vs 1541)
    # and whose json_mini ceiling could not be scored on either of them.
    #
    # Medium scene amended (still pre-responder): 38770ca1 -> 1d2f8518.
    # 38770ca1's door-state relations were
    # internally contradictory (all 7 door pairs asserted both "more open" and "more
    # closed" between the same pair) -- rank 13/1335 worst in the corpus under the new
    # G5 gate. 1d2f8518 is the G1-G5/P1-P4 argmin of the medium band, zero
    # contradictions, confirmed by mechanical re-run, not by any responder result.
    Scene("3rscan_02b33dfb", _qa("3rscan_02b33dfb"), None, role=PRIMARY),
    Scene("3rscan_1d2f8518", _qa("3rscan_1d2f8518"), None, role=PRIMARY),
    Scene("3rscan_0cac762f", _qa("3rscan_0cac762f"), None, role=PRIMARY),

    # Non-primary, kept for the two reporting artifacts defined in
    # Both fail G1 (json_mini context headroom):
    # 7f30f36c overflows the window outright (32,907 / 36,783 tokens against a 32,512
    # budget), d7d40d62 fits with 2,202 tokens = 6.7% of num_ctx, under the 12% bar.
    # They keep their ordinary 3rscan_* ids on purpose -- role is carried here, never
    # in a directory name, so axes.dataset_of() still resolves them to the 3rscan host.
    Scene("3rscan_7f30f36c", _qa("3rscan_7f30f36c"), None, role=STRESS),
    Scene("3rscan_d7d40d62", _qa("3rscan_d7d40d62"), None, role=SENSITIVITY),

    # Gibson (spatial-encoding / reference-frame axes): auto-discover the 6 single
    # files. No door graph, so no
    # orthogonal channel to cross with the metric views -> no combos. The trio
    # spans a floor-area gradient (Brinnon 35 rooms > Thrall > Donaldson 27).
    Scene("Brinnon", _qa("Brinnon"), None),
    Scene("Thrall", _qa("Thrall"), None),
    Scene("Donaldson", _qa("Donaldson"), None),
]

BY_ID: dict[str, Scene] = {s.scene_id: s for s in SCENES}
PRIMARY_SCENE_IDS: frozenset[str] = frozenset(
    s.scene_id for s in SCENES if s.role == PRIMARY)


class UnregisteredScene(KeyError):
    """A scene id with no entry in SCENES.

    Raised rather than defaulted, because every default is wrong here: treating an
    unknown scene as primary lets a stress scene into the headline numbers the moment
    someone forgets to register it, and treating it as non-primary silently drops a
    real scene out of the aggregate. Both failures are invisible in the output.
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
