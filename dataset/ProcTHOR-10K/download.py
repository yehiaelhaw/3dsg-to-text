# Dataset: ProcTHOR-10K -- 10,000 procedurally generated houses
# Source:  https://github.com/allenai/procthor-10k  (via the `prior` package)
# Place:   handled automatically -- copies train/val/test .jsonl.gz into this folder
#
# Usage: pip install prior && python download_dataset.py

import os
import shutil
import pathlib

if "HOME" not in os.environ:
    os.environ["HOME"] = os.environ["USERPROFILE"]

import prior

OUTPUT_DIR = pathlib.Path(__file__).parent.resolve()
PRIOR_ROOT = pathlib.Path.home() / ".prior" / "datasets" / "allenai" / "procthor-10k"


def main() -> None:
    print("Downloading ProcTHOR-10K via prior (cached to ~/.prior) ...")
    dataset = prior.load_dataset("procthor-10k")
    print(f"Loaded: train={len(dataset['train'])}  val={len(dataset['val'])}  test={len(dataset['test'])}")

    gz_files = list(PRIOR_ROOT.rglob("*.jsonl.gz"))
    if not gz_files:
        raise FileNotFoundError(f"No .jsonl.gz files found under {PRIOR_ROOT}")

    for src in gz_files:
        dst = OUTPUT_DIR / src.name
        if dst.exists():
            print(f"  Already exists, skipping: {dst.name}")
            continue
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} ({src.stat().st_size // 1024} KB) -> {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
