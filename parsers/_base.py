import argparse
import sys
from pathlib import Path
from typing import Callable

from scene_graph.loaders import REGISTRY, load
from scene_graph.models import Building


class NotApplicable(Exception):
    """Raised by a parser when the Building lacks the topology it requires."""
    pass


def run_parser(parse_fn: Callable[[Building], str], description: str) -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--dataset", required=True, choices=list(REGISTRY))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"Loading {args.model} from {args.path} ...")
    building = load(args.dataset, args.model, args.path)

    try:
        text = parse_fn(building)
    except NotApplicable as e:
        print(f"skipped: {e}")
        sys.exit(0)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print(text)
