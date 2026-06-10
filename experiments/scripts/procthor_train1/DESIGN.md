# ProcTHOR (train:1) — axis-aligned QA

19 reasoning questions in **two families**, each placed on a design axis so the representations
*discriminate*. The `question_type` field is the axis tag. Run with
`python -m experiments.scripts.procthor_train1.run`.

- **Spatial family** — needs a spatial channel (doors / distance). `inventory` is a *no-information
  control* here, so these axes appear in `value_of_spatial_structure.png` (lift over that control).
- **General-reasoning family** — needs only room/object *content*, which every content-carrying rep
  has. The spatial-encoding axis drops out; the discriminating variable is **format (B) / density (C)**:
  does the model reason better over the same content as JSON vs prose vs a topology list vs flat
  inventory? Here `inventory` is a legitimate format, not a floor, so these are compared head-to-head on
  absolute score (no lift chart).

## Which representations can answer which type (the scope map)

A question is only a *fair comparison* among the representations that carry the information it needs.
The reps that structurally cannot answer score ~0 — that confirms the axis, but read the **delta among
the in-scope reps** for the actual finding.

| family | `question_type` | ids | In-scope representations | Axis it probes |
|---|---|---|---|---|
| spatial | `connectivity` | 3, 4, 5, 6 | topology, prose, navigation, json | A connectivity rung + **B: topology vs prose** |
| spatial | `proximity` | 7, 8, 9 | metric_relations, navigation, json | A metric rung |
| spatial | `direction` | 10, 11 | metric_relations, navigation, json | **D: allocentric vs egocentric** |
| spatial | `route` | 12, 13 | navigation, json; **topology+metric_relations** (combo) | D + the combination headline |
| general | `containment` | 1, 2 | inventory, topology, prose, metric_relations, json | content filter; **B/C: format/density** |
| general | `aggregation` | 14, 15 | inventory, topology, prose, metric_relations, json | counting / argmax over rooms; **B/C** |
| general | `set_logic` | 16, 17 | inventory, topology, prose, metric_relations, json | intersection / rule-filter; **B/C** |
| general | `planning` | 18, 19 | inventory, topology, prose, metric_relations, json | multi-condition feasibility / inference; **B/C** |

`navigation` is out-of-scope for every general type (it carries routes, not object inventory).

Notes:
- Q6 is the sharp one: `metric_relations` says Bathroom [7]/[9] are 3.2 m apart (close) — a model leaning
  on it may wrongly answer "yes, walk straight across". Only connectivity reveals there is no direct door.
- Q9's weight-1 fact ("matches also correspond to doorways") needs *both* metric and connectivity, so
  full credit favours `json` or the `topology+metric_relations` combo; the metric reps still earn the
  weight-3 core.
- Q12 (`route`) is the headline for combinations: the room *path* needs connectivity, the per-leg
  *heading* needs metric. `navigation` has both; each single lean view is partial; the
  `topology+metric_relations` combo should reconstruct it.

## How to read the output

`aggregate.csv` groups by **(representation x question_type)**. Read it **row-band by row-band**: within
one `question_type`, scan the `answer_correctness_mean` across representations. Example claims it can
support:
- `connectivity`: if `topology` ~= `json`, raw coordinates do not help connectivity reasoning; if
  `prose` ~= `topology`, format (NL vs structured) barely matters.
- `direction`: if `navigation` > `metric_relations`, the egocentric frame is easier than the allocentric
  map for the *same* geometry.
- `route`: if `topology+metric_relations` ~= `json` (and each single view < both), complementary salient
  views rival the dense ceiling — *form beats raw availability*.
- general family (`aggregation`/`set_logic`/`planning`/`containment`): all content reps carry the same
  facts, so any spread is a **pure format/density effect** — e.g. if `prose` > `json`, natural language
  supports reasoning over content better than nested JSON; if `inventory` ties the richer reps, the extra
  spatial scaffolding neither helps nor hurts general reasoning. This is the other half of the
  "operational trade-offs" question.

## Ground-truth provenance

Every answer was verified against the concrete files in `scene_contexts/procthor_train1/`:
- connectivity / paths / components -> `topology.txt`
- distances / "nearest" / isolation -> `metric_relations.txt`
- egocentric headings / left-right -> `navigation.txt`
- object inventories (containment) -> `inventory.txt` / `prose.txt`

Key structural facts: three disconnected components — {Bedroom 6 hub + Bedrooms 8/10, Bathrooms 7/9/11,
LivingRoom 4}, the closed {Kitchen 2, LivingRoom 3} pair, and isolated LivingRoom 5. Bedroom 6 is the
articulation point of the large component.
