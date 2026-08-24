"""json_mini — the full-record anchor of the format axis; the same content as json_pretty
(shares its `parse()` by import) reserialized without indentation, as a cost-accounting
correction so the anchor isn't charged for meaningless whitespace tokens."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from json_parser import parse, to_json_mini_string

if __name__ == "__main__":
    run_parser(lambda b: to_json_mini_string(parse(b)),
               "Serialize a 3D scene to structured JSON, minified")
