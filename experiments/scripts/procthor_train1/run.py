import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from evaluation.config import EvalConfig
from evaluation.results import save
from evaluation.runner import iter_records

config = EvalConfig(
    dataset_path="experiments/scripts/procthor_train1/keyfact-qa.jsonl",
    scene_contexts_dir="scene_contexts",
    output_dir="experiments/results/procthor_train1_keyfacts_3repeations",

    responder_backend="ollama",
    responder_model="qwen2.5:14b",
    responder_options={"num_ctx": 32768},

    judge_backend="ollama",
    judge_model="qwen2.5:14b",

    # Explicit so the multi-view combination is included (auto-discovery only
    # finds single files). This is the full ProcTHOR experiment matrix.
    representations=[
        "inventory",
        "topology",
        "graph_digest",
        "prose",
        "metric_relations",
        "navigation",
        "json",
        "topology+metric_relations",        # headline combo: raw doors + distances
        "graph_digest+metric_relations",    # axis F twin: derived doors + distances
    ],
    repetitions=3,
)

save(iter_records(config), config)
