"""json_pretty — the raw (indented) pole of the json_formatting axis; content-identical to
json_mini via shared parse(), differing only in whitespace. Shipped
scene_contexts/*/json_pretty.json files are the renamed historical json.json artifacts, not
this module's output — do not regenerate the shipped files from it."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser
from json_parser import parse, to_json_string

if __name__ == "__main__":
    run_parser(lambda b: to_json_string(parse(b)),
               "Serialize a 3D scene to structured JSON, indented")
