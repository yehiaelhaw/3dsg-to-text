"""json_mini — the full scene graph as JSON, minified.

Byte-for-byte the same object as `json`: the same `parse()` builds it, so the two
views cannot drift in keys, values, ordering or included content. The only
difference is that this one is serialized without indentation or separator
padding, which on the measured corpus is ~48% of the pretty form's tokens.

That matters because `json` is the study's ceiling anchor, and the ceiling's token
cost is the denominator of the headline cost claim -- charging it for whitespace
overstates that denominator. It also decides whether the ceiling fits at all: the
pretty form overflows the 32k responder window on the two dense 3RScan scenes.

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
