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
3. **Report** — the CSVs are pooled into tables, charts and the thesis figures
   (`evaluation/report.py`, `plots.py`, `thesis_figures.py`).

This README covers the code only. The research design — why these serializations, why these
questions, what the results mean — is the subject of the thesis in `thesis/`.

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
  dataset.py         reads the authored keyfact-qa.jsonl files
  scene_loader.py    reads scene_contexts/
  scope.py           which serializations can answer which question types
  axes.py            how serializations group into comparisons, for reporting
  runner.py          the two-phase loop: generate all answers, then judge them
  judging.py         judge prompts, scoring, and verdict parsing
  results.py         CSV writing and aggregation
  report.py          the numeric report (report.md)
  plots.py           charts
  thesis_figures.py  page-sized renders + LaTeX tables for the thesis
  token_count.py     exact pre-call prompt token counts for Ollama models
  llm/               provider abstraction: ollama, gemini, openai
  tests/             checks; see "Tests" below

experiments/         Run configuration and drivers.
  models.py          responder profiles (which model, which host, which options)
  scenes.py          scene registry — which scenes, which QA file, which reps
  run_experiments.py runs the responder x scene matrix
  aggregate_results.py  pools per-scene results and redraws the charts
  qa/<scene>/keyfact-qa.jsonl   the authored questions and their key facts

thesis/              LaTeX sources.
```

### Directories that are not in version control

The code refers to these; they are generated, downloaded, or local-only.

| Path | What it is |
| --- | --- |
| `dataset/` | the source datasets (Gibson, ProcTHOR-10K, 3RScan/3DSSG), downloaded separately |
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
python -m experiments.run_experiments --list          # print the matrix, run nothing
python -m experiments.run_experiments --models qwen2.5-14b --scenes Brinnon
python -m experiments.run_experiments --models qwen2.5-14b --scenes Brinnon \
    --score-only --judge-backend gemini --judge-model gemini-2.5-flash
```

`--score-only` re-judges the cached answers in `responses.jsonl` with a different judge — no
regeneration. Also useful: `--reps`, `--types`, `--generate-only`.

**Aggregate and report:**

```bash
python -m experiments.aggregate_results                 # every model found
python -m experiments.aggregate_results --models qwen2.5-14b
```

Both write into `experiments/results/`; `aggregate_results` also regenerates `report.md` and
the charts.

## Tests

The files in `evaluation/tests/` are **standalone scripts, not a pytest suite**. Each is run
on its own and prints `PASS` or raises:

```bash
python -m evaluation.tests.test_scope_smoke
python -m evaluation.tests.test_reporting_smoke
```

Most run offline. These five call live LLMs, so they cost money or need a running Ollama
server: `test_runner_smoke`, `test_results_smoke`, `test_ollama_smoke`, `test_gemini_smoke`,
`test_context_based_smoke`. `test_token_count_replay` needs the Ollama server too, to read
back the tokenizer it replays against.

Six offline ones need `dataset/` present, because they check live parser output against
freshly loaded source geometry rather than against a fixture:
`test_procthor_loader_smoke`, `test_format_axis_equivalence`, `test_framing_layout_isolation`,
`test_metric_framing_equivalence`, `test_topology_metric_equivalence`,
`test_structure_presentation_rename`.
