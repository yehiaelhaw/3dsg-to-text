"""json_pretty — the full scene graph as JSON, indented. The formatting ablation pole.

The same `parse()` output as `json_mini`, serialized with `indent=2`. The pair is
content-identical by construction and differs in whitespace alone, which is what
makes it a clean ablation: any accuracy separation between them is an effect of
formatting on the *accessibility* of information, never of the information itself.

Axis role: the raw pole of the `json_formatting` axis (ProcTHOR exhibit, Gibson
companion). 3RScan is out of scope for it — `json_pretty` is CONTEXT_EXCEEDED on
two of that host's three scenes, so its coverage there falls below MIN_COVERAGE
and the pair is not rank-eligible.

**The shipped scene_contexts/<scene>/json_pretty.json files were NOT produced by
this module.** They are the historical `json.json` artifacts, renamed in place on
2026-08-09, and they are what every pre-migration run actually consumed. This
module exists so the two views cannot drift going forward, and as a reproducibility
check: run it to a scratch path and diff against the shipped file after normalizing
newlines. Do not regenerate the shipped file from it. `_base.run_parser` writes via
`Path.write_text` with no `newline=`, so on Windows it translates \\n to \\r\\n — every
historical json.json is 100% CRLF on disk while `json.dumps(indent=2)` emits bare
\\n. The difference cancels on read (`scene_loader.load` uses `read_text`), which is
precisely why it would have been invisible had we rebuilt instead of renamed.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from json_parser import parse, to_json_string

if __name__ == "__main__":
    run_parser(lambda b: to_json_string(parse(b)),
               "Serialize a 3D scene to structured JSON, indented")
