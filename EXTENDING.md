# Extending this repository

This project separates cleanly into a few extension points: new **serializations**
(`parsers/`), new **source datasets** (`scene_graph/loaders/`), and new **responder/judge
models** (`experiments/models.py`, `evaluation/llm/`). This doc is for someone who forks the
repo and wants to plug something new into one of those points without reading the whole
codebase first. See [README.md](README.md) for how to run the pipeline once it's wired up.

## Adding a new serialization (parser)

A parser is a standalone CLI that turns a `Building` (the domain model in
[scene_graph/models.py](scene_graph/models.py)) into a plain-text string. Every existing one
lives in [parsers/](parsers/) as `<name>_parser.py`; that filename stem is the representation's
name everywhere else in the codebase (`scene_contexts/<scene>/<name>.txt`,
`results.csv` rows, `evaluation/scope.py`, `evaluation/axes.py`).

Minimal shape, copied from [parsers/inventory_parser.py](parsers/inventory_parser.py):

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import room_label, sort_key       # shared formatting helpers
from scene_graph.models import Building

def parse(building: Building) -> str:
    # ... build and return the serialized string ...
    return text

if __name__ == "__main__":
    run_parser(parse, "One-line description of what this serialization does")
```

`run_parser` ([parsers/_base.py](parsers/_base.py)) supplies the CLI (`--dataset`, `--model`,
`--path`, `--output`) and calls `parse(building)`. Your `parse` function is the only thing you
write.

**If your serialization needs data a scene might not have** (a door graph, object relations,
multiple rooms), check for it with a predicate from
[scene_graph/capabilities.py](scene_graph/capabilities.py) (`has_room_connectivity`,
`has_object_relations`, `has_multiroom_layout`, `has_floors`) and raise `NotApplicable(...)`.
`run_parser` catches it, prints `skipped: ...`, and exits 0 — so the parser can be run
unconditionally over every scene without special-casing incompatible ones.

**Shared helpers**, so you don't reinvent formatting:
- [parsers/_format.py](parsers/_format.py) — `room_label`, `obj_label`, `object_inventory`,
  `room_type_summary`, `sort_key` (stable ordering), `undirected_components`.
- [parsers/_geometry.py](parsers/_geometry.py) — `vertical_axis`, `floor_plane`, `compass`,
  `plane_distance`, `distance_3d`.
- [parsers/_relations.py](parsers/_relations.py) — `classify_predicate`, `resolve_labels`,
  `support_forest`, `render_tree`, for anything working with `Building.object_relations`.

Files prefixed `_` are helper modules, not serializations — the evaluation code never treats
them as representations.

**Generate output for a scene:**

```bash
python parsers/your_parser.py --dataset procthor --model "train:1" \
    --path dataset/ProcTHOR-10K/ --output scene_contexts/procthor_train1/your_rep.txt
```

**Wire it into the evaluation framework.** Skipping these steps doesn't error — it just
produces silently wrong results, so don't treat them as optional:

1. [evaluation/scope.py](evaluation/scope.py) — add a `REP_CAPS["your_rep"]` entry declaring
   which capability channels your text actually contains (`inventory`, `connectivity`,
   `metric`, `metric_edges`, `object_relations_raw/support/derived`). This is what keeps a
   question from being asked against a representation that doesn't contain the fact needed to
   answer it. An undeclared rep fails **open** (treated as having every capability) unless
   `strict_scope=True`, in which case `validate_declared` refuses to run — don't rely on the
   fail-open behavior for a real run.
2. `experiments/scenes.py` — if a scene has an explicit `representations` list (ProcTHOR does,
   via `_PROCTHOR_REPS`), add your rep name there. Scenes with `representations=None`
   auto-discover every file present in their `scene_contexts/<scene>/` directory
   ([evaluation/scene_loader.py](evaluation/scene_loader.py)), so nothing to do there.
3. [evaluation/axes.py](evaluation/axes.py) — **only if** your new representation is one pole of
   a comparison you want reported as a named axis/exhibit. This module encodes this thesis's
   specific comparisons (which reps form a ladder, which pair is the headline delta, declared
   confounds); a rep can be fully functional in the pipeline without ever appearing here. Most
   forks adding a serialization for their own purposes can skip this file entirely and read
   results straight out of `results.csv` / `aggregate.csv`.

## Adding a new source dataset

Datasets are loaded through the `DatasetLoader` ABC in
[scene_graph/loaders/base.py](scene_graph/loaders/base.py):

```python
class DatasetLoader(ABC):
    @abstractmethod
    def load(self, scene_id: str, data_path: str) -> Building: ...
```

Implement one (see [scene_graph/loaders/gibson.py](scene_graph/loaders/gibson.py),
`procthor.py`, or `threerscan.py` for reference — increasing complexity in that order), building
a `Building` out of the dataclasses in `scene_graph/models.py`:

- `Building(name, rooms, size?, connectivity?, object_relations?)`
- `Room(id, position?, category?, objects, floor?, size?, floor_area?, volume?)`
- `SceneObject(id, category, position, short_id?, size?, affordances?, material?, ...)`
- `ObjectRelation(subject_id, predicate, object_id)`

Only `Building.name` and `Building.rooms` are required; every other field is `None`/empty when
your dataset doesn't have that kind of data. `scene_graph/capabilities.py`'s predicates key off
exactly these optional fields, so parsers automatically skip themselves on scenes your loader
can't populate — you don't need to touch every parser.

Register it in [scene_graph/loaders/\_\_init\_\_.py](scene_graph/loaders/__init__.py):

```python
from scene_graph.loaders.yourdataset import YourDatasetLoader
REGISTRY["yourdataset"] = YourDatasetLoader
```

That string becomes the `--dataset` value every parser CLI accepts. Then register each scene
you want evaluated in `experiments/scenes.py` (`Scene(scene_id, dataset_path, representations)`)
and add a mapping for it in `evaluation/axes.py`'s `dataset_of()` if scene IDs need host
detection (currently prefix-based: `procthor*`, `3rscan*`, else Gibson).

## Adding a different model

There are two independent things that might be called "adding a model":

### A new profile for an existing backend (most common)

If the model is already served by Ollama, Gemini, or OpenAI, just add a
`ModelProfile` in [experiments/models.py](experiments/models.py):

```python
ModelProfile("your-model-name", "ollama", "your-model:tag",
             {"host": _HOST_A, "num_ctx": 32768, "temperature": 0})
```

`name` becomes the results directory (`experiments/results/<name>/`) and the `--models` CLI
value; `backend` selects the provider in `evaluation/llm/factory.py`; `options` is passed
straight through to the provider (Ollama options, or `api_key`/other kwargs for cloud
backends). Reasoning models that emit a `<think>` trace are handled automatically by
`OllamaProvider` (the trace is stripped from the judged answer but still counted in
`completion_tokens`) — see [evaluation/llm/ollama.py](evaluation/llm/ollama.py).

Then run it:

```bash
python -m experiments.run_experiments --models your-model-name --scenes Brinnon
```

### A new backend (a provider not yet supported)

Implement `LLMProvider` ([evaluation/llm/base.py](evaluation/llm/base.py)):

```python
class LLMProvider(ABC):
    def __init__(self, model: str, options: dict) -> None: ...

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult: ...

    def unload(self) -> None:   # optional; default no-op
        ...
```

`generate()` must return a `GenerationResult(text, prompt_tokens, completion_tokens,
latency_ms)` — real token counts from the API/server if it reports them (see
[evaluation/llm/openai.py](evaluation/llm/openai.py) for the simplest example), since those
counts feed directly into the cost-axis reporting. Override `unload()` only for local backends
that hold GPU memory the two-phase runner needs to free between responder and judge generation
(see `OllamaProvider.unload`); remote/API backends don't need it.

Register the backend name in [evaluation/llm/factory.py](evaluation/llm/factory.py):

```python
if backend == "yourbackend":
    from evaluation.llm.yourbackend import YourBackendProvider
    return YourBackendProvider(model, options)
```

That `backend` string is what `ModelProfile.backend` and `--judge-backend` refer to. A backend
can serve as a responder, a judge, or both — `experiments/run_experiments.py` and
`--judge-backend`/`--judge-model` treat responder and judge identically through the same
`LLMProvider` interface.

## Adding a question type

Question types (`connectivity`, `proximity`, `object_relation`, ...) are declared in
`evaluation/scope.py`'s `TYPE_NEEDS`, mapping a type to the capability channel(s) a
representation must have to be in-scope for it (a `set` = AND, a `list[set]` = OR
alternatives, a `dict` = host-specific needs — see `planning` for an example). Add your type
there, then author questions of that type in `experiments/qa/<scene>.jsonl`
(`evaluation/dataset.py` reads this format; each question needs `question_type` set for
`scope.py` filtering to apply — an unset type is never filtered).

## After wiring something in

Verify a new loader/backend/parser with a small real run:

```bash
python -m experiments.run_experiments --plan                        # sanity-check the matrix
python -m experiments.run_experiments --models <name> --scenes <one scene>
```
