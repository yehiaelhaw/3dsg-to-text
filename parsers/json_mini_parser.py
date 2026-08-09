"""json_mini — the full scene graph as JSON, minified. The study's ceiling anchor.

The same `parse()` output as `json_pretty`, losslessly reserialized without
indentation or separator padding. Not byte-for-byte identical to it, obviously —
that is the entire point — but identical in keys, key order, values and included
content, so the pair differs in formatting alone. Round-tripping proves it:
``json.dumps(json.loads(pretty), separators=(",", ":"), ensure_ascii=False)``
reproduces this file exactly (asserted in evaluation/tests/test_json_representations.py).

Why this one is the ceiling. The ceiling's token cost is the denominator of the
headline cost claim, so charging it for semantically meaningless whitespace
overstates the saving every derived view appears to deliver. Minifying is a
lossless cost-accounting correction, not a change in what the ceiling contains.
Measured saving is **32–45% of tokens**, host- and tokenizer-dependent (~44% on
dense 3RScan, ~33% on ProcTHOR, and smaller throughout on mistral-nemo's tekken
tokenizer) — never quote a single number, and never without naming the responder.
The ~48% figure that used to sit here was a *character* count.

This is explicitly not a coverage fix. `3rscan_7f30f36c` still exceeds the 32,512
budget when minified (32,899 tok on qwen2.5, 36,775 on mistral-nemo) and stays a
genuine CONTEXT_EXCEEDED outcome. Minifying does recover `3rscan_d7d40d62`, which
is a secondary consequence, not the rationale.

Shares `parse` with json_parser by import (the same move synthesis_parser makes
for its digests), so "same content, different formatting" is guaranteed by
construction rather than by convention.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from json_parser import parse, to_json_mini_string

if __name__ == "__main__":
    run_parser(lambda b: to_json_mini_string(parse(b)),
               "Serialize a 3D scene to structured JSON, minified")
