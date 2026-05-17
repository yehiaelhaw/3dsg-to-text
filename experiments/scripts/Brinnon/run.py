import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from evaluation.config import EvalConfig
from evaluation.results import save
from evaluation.runner import run

config = EvalConfig(
    dataset_path="experiments/scripts/Brinnon/qa.jsonl",
    scene_contexts_dir="scene_contexts",
    output_dir="experiments/results/Brinnon",

    responder_backend="ollama",
    responder_model="qwen2.5:7b",
    responder_options={"num_ctx": 32768},

    judge_backend="gemini",
    judge_model="gemini-2.5-flash",

    representations=None,
    repetitions=1,
)

save(run(config), config)
