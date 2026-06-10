import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from evaluation.config import EvalConfig
from evaluation.results import save
from evaluation.runner import iter_records

config = EvalConfig(
    dataset_path="experiments/scripts/procthor_train1/keyfact-qa.jsonl",
    scene_contexts_dir="scene_contexts",
    output_dir="experiments/results/procthor_train1_qwen14b_gemma2_3rep",

    responder_backend="ollama",
    responder_model="qwen2.5:14b",
    responder_options={"num_ctx": 12288, "temperature": 0.7},  # 12k window holds the ~9.9k-tok json prompt + answer

    judge_backend="ollama",
    judge_model="gemma2:9b",
    judge_options={"temperature": 0.0},

    # faithfulness feeds the judge the full context, but the ~9.9k-tok json
    # overflows gemma2's 8k window -> unreliable. answer_correctness needs only
    # the short answer, so it's fine here. (Re-enable faithfulness with Gemini.)
    compute_faithfulness=False,

    # Explicit list to include the a+b combos (auto-discovery finds singles only).
    representations=[
        "inventory",
        "topology",
        "graph_digest",
        "prose",
        "metric_relations",
        "navigation",
        "json",
        "topology+metric_relations",
        "graph_digest+metric_relations",
    ],
    repetitions=3,
)

save(iter_records(config), config)
