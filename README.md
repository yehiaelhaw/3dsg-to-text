# 3D Scene Graph Serialization — evaluation codebase

Research code for a bachelor thesis on how the choice of *textual serialization* affects an
LLM's ability to answer questions about a 3D scene graph.

The pipeline has three stages:

1. **Parse** — a scene graph from one of three source datasets is loaded into a common data
   model and written out as plain text in one of ~20 serializations
   (`parsers/` → `scene_contexts/<scene>/<representation>.txt`).
2. **Evaluate** — a *responder* LLM answers authored questions with one serialization as its
   only context; a *judge* LLM then scores each answer against hand-written key facts
   (`evaluation/` → `experiments/results/<model>/<scene>/results.csv`).
3. **Report** — the CSVs are pooled into tables and charts
   (`evaluation/report.py`, `plots.py`).

This README covers the code only. The research design — why these serializations, why these
questions, what the results mean — is the subject of the thesis in `thesis/`.

Forking this to add your own serialization, dataset, or model? See
[EXTENDING.md](EXTENDING.md).

## Layout

```
scene_graph/         Domain model. Building / Room / SceneObject / ObjectRelation
                     dataclasses, capability predicates over them, and one loader
                     per source dataset (Gibson, ProcTHOR, 3RScan).

parsers/             One module per serialization, each a standalone CLI. Files
                     prefixed with _ are shared helpers, not serializations.

evaluation/          The evaluation framework.
  config.py          EvalConfig — everything one run needs
  core.py            Question / Response / EvalRecord + the results.csv schema
  dataset.py         reads the authored qa/<scene>.jsonl files
  scene_loader.py    reads scene_contexts/
  scope.py           which serializations can answer which question types
  axes.py            how serializations group into comparisons, for reporting
  runner.py          the two-phase loop: generate all answers, then judge them
  judging.py         judge prompts, scoring, and verdict parsing
  results.py         CSV writing and aggregation
  report.py          the numeric report (report.md)
  plots.py           charts
  token_count.py     exact pre-call prompt token counts for Ollama models
  llm/               provider abstraction: ollama, gemini, openai

experiments/         Run configuration and drivers.
  models.py          responder profiles (which model, which host, which options)
  scenes.py          scene registry — which scenes, which QA file, which reps
  run_experiments.py runs the responder x scene matrix
  aggregate_results.py  pools per-scene results and redraws the charts
  qa/<scene>.jsonl   the authored questions and their key facts

thesis/              LaTeX sources.
```

### Directories that are not in version control

The code refers to these; they are generated, downloaded, or local-only.

| Path | What it is |
| --- | --- |
| `dataset/` | the source datasets (Gibson, ProcTHOR-10K, 3RScan/3DSSG), downloaded separately — see each subfolder's `download.txt` |
| `scene_contexts/` | parser output — the serializations the responder actually reads |
| `experiments/results/` | run output: `results.csv`, `aggregate.csv`, `responses.jsonl`, `report.md`, charts |
| `experiments/backups/` | timestamped pre-edit copies of QA data and affected results |
| `evaluation/.token_cache/` | tokenizer vocabularies pulled from the Ollama server, ~2 MB per model |
| `.env` | API keys |

## Setup

```bash
pip install -r requirements.txt   # Python 3.10+ (developed on 3.11)
cp .env.example .env              # then fill in the keys you need
```

A local [Ollama](https://ollama.com) server provides the responder and the screening judge;
`GEMINI_API_KEY` is needed only for confirmatory judging. Model hosts are set per profile in
`experiments/models.py`.

## Running

Everything runs as a module from the repository root.

**Generate a scene context.** Each parser is its own CLI and writes one file:

```bash
python parsers/topology_parser.py \
    --dataset procthor --model "train:1" --path dataset/ProcTHOR-10K/ \
    --output scene_contexts/procthor_train1/topology.txt
```

`--dataset` is one of `gibson`, `procthor`, `3rscan`. A parser whose required data is absent
from the scene (no door graph, no object relations) exits cleanly without writing, so a whole
directory of parsers can be run over any scene.

**Run the evaluation matrix** — every profile in `models.py` against every scene in
`scenes.py`. Runs are resumable: finished cells are skipped, so an interrupted run can simply
be restarted.

```bash
python -m experiments.run_experiments --plan          # print the matrix, run nothing
python -m experiments.run_experiments --models qwen2.5-14b --scenes Brinnon
python -m experiments.run_experiments --models qwen2.5-14b --scenes Brinnon \
    --score-only --judge-backend gemini --judge-model gemini-2.5-flash
```

| Flag | Meaning |
| --- | --- |
| `--models` | comma-separated `ModelProfile` names (default: all in `models.py`) |
| `--scenes` | comma-separated scene ids (default: all in `scenes.py`) |
| `--representations` | comma-separated rep names, overriding each scene's default set |
| `--question-types` | comma-separated question types to run (default: every in-scope type) |
| `--plan` | print the planned matrix and exit; nothing runs |
| `--generate-only` | generate responses only; judge later with `--score-only` |
| `--score-only` | re-judge cached `responses.jsonl` instead of generating (needs the two below) |
| `--judge-backend` / `--judge-model` | override the judge, e.g. `gemini` / `gemini-2.5-flash` |
| `--faithfulness` | also compute the faithfulness metric (off by default) |
| `--no-report` | skip plots/`report.md` per scene (CSVs still written); rebuild later with `aggregate_results` |

`--generate-only` and `--score-only` are mutually exclusive.

**Aggregate and report:**

```bash
python -m experiments.aggregate_results                 # every model found
python -m experiments.aggregate_results --models qwen2.5-14b
```

Both write into `experiments/results/`; `aggregate_results` also regenerates `report.md` and
the charts.

| Flag | Meaning |
| --- | --- |
| `--models` | comma-separated model directory names under `experiments/results/` (default: all) |
| `--diagnostics` | also draw the latency diagnostic chart (off by default; confounded, not the cost axis) |
| `--allow-partial` | write a per-dataset group even with fewer than 3 scenes, instead of skipping it |
