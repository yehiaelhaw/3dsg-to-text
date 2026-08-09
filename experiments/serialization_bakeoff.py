"""serialization_bakeoff.py -- is minified JSON the cheapest *lossless* ceiling?

The ceiling representation is the complete source graph, and its token cost is the
denominator of the headline cost claim. `json_mini` earned that slot by removing
`indent=2` whitespace, which is pure syntax and costs nothing semantically. This
script asks the next question: is JSON itself the cheapest *syntax* for the same
object, or would some other standard textual serialization carry the identical
content for fewer tokens?

Rules of the comparison (they are what make it a ceiling question rather than a
compression question). A candidate is ADMISSIBLE only if it keeps:
  * every field, object, room, relation and value of `json_parser.parse()`;
  * the original field names, spelled out;
  * the nesting (objects inside rooms, not flattened into a side table);
  * direct readability -- no legend, no key dictionary, no positional records
    whose meaning has to be recovered from a schema stated elsewhere.
Candidates that break a rule are still measured, but reported as DISQUALIFIED, so
the table shows what the rules cost rather than hiding it.

Everything is offline. No generation calls:
  * the parsed graph is read back from scene_contexts/<scene>/json.json, which is
    `parse()` output verbatim (asserted against json_mini.json, which is the same
    object minified);
  * token counts come from evaluation.token_count -- the tokenizer read out of the
    GGUF blob the server serves, replay-verified equal to prompt_eval_count on
    13,320 recorded cells. Specs are loaded from evaluation/.token_cache/ so no
    SSH tunnel is needed.

Reads nothing but scene_contexts/ and experiments/scripts/; writes nothing.

Usage:
    python -m experiments.serialization_bakeoff
    python -m experiments.serialization_bakeoff --scene 3rscan_7f30f36c
    python -m experiments.serialization_bakeoff --dump-samples DIR
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

import tomllib
import yaml

from evaluation import token_count
from evaluation.runner import _CTX_RESERVE, _RESPONDER_PROMPT
from experiments.models import MODEL_PROFILES

CONTEXTS = Path("scene_contexts")
QA_ROOT = Path("experiments/scripts")
BASELINE = "json_mini"

# One profile per distinct (tokenizer, chat template, default system prompt).
# qwen2.5-7b/14b/32b share all three, so they are one row, reported under 14b.
REPORT_MODELS = ["qwen2.5-14b", "mistral-nemo-12b", "deepseek-r1-14b"]


# --------------------------------------------------------------------------- #
# Serializers
# --------------------------------------------------------------------------- #

def s_json_mini(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def s_json_pretty(d: dict) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False)


_WIDE = 1 << 30  # never line-wrap; a wrapped line is a token we did not intend to spend


def s_yaml_block(d: dict) -> str:
    return yaml.safe_dump(d, default_flow_style=False, sort_keys=False,
                          allow_unicode=True, width=_WIDE, indent=2)


def s_yaml_flow(d: dict) -> str:
    return yaml.safe_dump(d, default_flow_style=True, sort_keys=False,
                          allow_unicode=True, width=_WIDE)


class _FlowList(list):
    """A list PyYAML must emit inline, whatever the document default is."""


def _mark_leaf_lists(v: Any) -> Any:
    """Coordinate/affordance lists -> flow; structural lists stay block."""
    if isinstance(v, dict):
        return {k: _mark_leaf_lists(x) for k, x in v.items()}
    if isinstance(v, list):
        if v and not any(isinstance(x, (dict, list)) for x in v):
            return _FlowList(v)
        return [_mark_leaf_lists(x) for x in v]
    return v


yaml.SafeDumper.add_representer(
    _FlowList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", list(data), flow_style=True),
)


def s_yaml_hybrid(d: dict) -> str:
    return yaml.safe_dump(_mark_leaf_lists(d), default_flow_style=False, sort_keys=False,
                          allow_unicode=True, width=_WIDE, indent=2)


# --- minimal flow YAML: the cheapest spelling of YAML that is still YAML ----- #
# `{a: 1,b: 2}` is legal: the space after ':' is required before a plain scalar in
# flow context, the space after ',' is not. Strings that would resolve to another
# type (or that flow punctuation would break) get single-quoted; everything else
# rides bare, which is where YAML can undercut JSON.

@lru_cache(maxsize=None)
def _plain_ok(s: str) -> bool:
    """Would `s` survive unquoted inside a flow sequence, as itself?"""
    if s == "" or s != s.strip():
        return False
    try:
        return yaml.safe_load("[" + s + ",_]") == [s, "_"]
    except yaml.YAMLError:
        return False


def _y_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return s if _plain_ok(s) else "'" + s.replace("'", "''") + "'"


def _y_min(v: Any) -> str:
    if isinstance(v, dict):
        return "{" + ",".join(f"{_y_scalar(k)}: {_y_min(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_y_min(x) for x in v) + "]"
    return _y_scalar(v)


def s_yaml_flow_min(d: dict) -> str:
    return _y_min(d) + "\n"


# --- JSON5: JSON with the key quotes dropped -------------------------------- #

_IDENT = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _j5(v: Any) -> str:
    if isinstance(v, dict):
        parts = []
        for k, x in v.items():
            key = k if _IDENT.match(k) else json.dumps(k, ensure_ascii=False)
            parts.append(f"{key}:{_j5(x)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_j5(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def s_json5(d: dict) -> str:
    return _j5(d)


# --- TOML ------------------------------------------------------------------- #

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _t_key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _t_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ",".join(_t_val(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def _t_emit(out: list[str], prefix: str, obj: dict, sep: str) -> None:
    """Scalars/arrays first, then nested tables -- TOML requires that order."""
    tables: list[tuple[str, Any]] = []
    for k, v in obj.items():
        if isinstance(v, dict) or (isinstance(v, list) and v and isinstance(v[0], dict)):
            tables.append((k, v))
        else:
            out.append(f"{_t_key(k)}{sep}{_t_val(v)}")
    for k, v in tables:
        path = f"{prefix}.{_t_key(k)}" if prefix else _t_key(k)
        if isinstance(v, dict):
            out.append(f"[{path}]")
            _t_emit(out, path, v, sep)
        else:
            for item in v:
                out.append(f"[[{path}]]")
                _t_emit(out, path, item, sep)


def _s_toml(d: dict, sep: str) -> str:
    out: list[str] = []
    _t_emit(out, "", d, sep)
    return "\n".join(out) + "\n"


def s_toml(d: dict) -> str:
    return _s_toml(d, "=")


def s_toml_spaced(d: dict) -> str:
    return _s_toml(d, " = ")


# --- XML -------------------------------------------------------------------- #
# Faithful: every scalar field is an attribute, every list is a child element with
# one <item> per entry. Packing a list into a delimited attribute string would be
# the very schema-decoding move the rules forbid, so it is not done.

def _x_emit(out: list[str], tag: str, obj: dict) -> None:
    attrs, kids = [], []
    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            kids.append((k, v))
        else:
            attrs.append(f" {k}={quoteattr(str(v))}")
    out.append(f"<{tag}{''.join(attrs)}" + (">" if kids else "/>"))
    for k, v in kids:
        if isinstance(v, dict):
            _x_emit(out, k, v)
        elif v and isinstance(v[0], dict):
            out.append(f"<{k}>")
            for item in v:
                _x_emit(out, k.rstrip("s") or "item", item)
            out.append(f"</{k}>")
        else:
            out.append(f"<{k}>" + "".join(f"<item>{escape(str(x))}</item>" for x in v) + f"</{k}>")
    if kids:
        out.append(f"</{tag}>")


def s_xml(d: dict) -> str:
    out: list[str] = []
    _x_emit(out, "scene", d)
    return "".join(out) + "\n"


# --- DISQUALIFIED reference points ------------------------------------------ #

def s_csv_tables(d: dict) -> str:
    """Header row + positional rows. Field names survive, but only once, so every
    value has to be matched back to a column the model must hold in mind -- the
    schema-reconstruction burden the rules exclude. Measured as a floor only."""
    lines = ["[building]", ",".join(d["building"]), ",".join(str(v) for v in d["building"].values())]
    ocols = ["room", "id", "category", "position", "size", "affordances",
             "material", "visual_texture", "tactile_texture"]
    rcols = [k for k in ("id", "category", "floor", "position", "size", "floor_area", "volume")]
    lines += ["[rooms]", ",".join(rcols)]
    for r in d["rooms"]:
        lines.append(",".join(_csv_cell(r.get(c)) for c in rcols))
    lines += ["[objects]", ",".join(ocols)]
    for r in d["rooms"]:
        for o in r.get("objects", []):
            lines.append(",".join([_csv_cell(r["id"])] + [_csv_cell(o.get(c)) for c in ocols[1:]]))
    if d.get("relations"):
        lines += ["[relations]", "subject,predicate,object"]
        for rel in d["relations"]:
            lines.append(f"{rel['subject']},{rel['predicate']},{rel['object']}")
    return "\n".join(lines) + "\n"


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    return str(v)


# --------------------------------------------------------------------------- #
# Round-trip verification
# --------------------------------------------------------------------------- #

def _strict_eq(a: Any, b: Any, path: str = "$") -> str | None:
    """Deep equality that also compares types. Returns a path on mismatch.

    Type-checking is the point: Python says 1 == 1.0 and would let a serializer
    that turned the string id "1" into an int, or a float into an int, pass as
    lossless. Those are exactly the losses these formats risk.
    """
    if type(a) is not type(b) and not (isinstance(a, str) and isinstance(b, str)):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        if list(a) != list(b):
            return f"{path}: key ORDER differs"
        for k in a:
            if (m := _strict_eq(a[k], b[k], f"{path}.{k}")):
                return m
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if (m := _strict_eq(x, y, f"{path}[{i}]")):
                return m
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _l_json(t: str):
    return json.loads(t)


def _l_yaml(t: str):
    return yaml.safe_load(t)


def _l_toml(t: str):
    return tomllib.loads(t)


def _l_json5(t: str):
    import json5
    return json5.loads(t)


def _l_xml(t: str):
    def node(el, want_list_of: str | None = None):
        d: dict[str, Any] = dict(el.attrib)
        for child in el:
            if child.tag == "item":
                continue
            items = [c for c in child if c.tag == "item"]
            if items:
                d[child.tag] = [c.text or "" for c in items]
            elif len(child) and all(c.tag != "item" for c in child):
                d[child.tag] = [node(c) for c in child]
            else:
                d[child.tag] = node(child)
        return d
    return node(ET.fromstring(t))


# --------------------------------------------------------------------------- #
# Candidate table
# --------------------------------------------------------------------------- #

class Candidate:
    def __init__(self, name, fn, loader, note, admissible=True, coerces=None):
        self.name, self.fn, self.loader, self.note = name, fn, loader, note
        self.admissible, self.coerces = admissible, coerces

    def verify(self, data: dict) -> str:
        """'' if the text reloads to a strictly identical object, else the reason."""
        if self.loader is None:
            return "no parser -- structural equivalence not machine-checkable"
        try:
            back = self.loader(self.fn(data))
        except Exception as exc:                              # noqa: BLE001
            return f"reload failed: {type(exc).__name__}: {exc}"
        if self.coerces:
            back = self.coerces(back)
        return _strict_eq(data, back) or ""


CANDIDATES: list[Candidate] = [
    Candidate("json_mini",     s_json_mini,     _l_json,  "baseline: JSON, no whitespace"),
    Candidate("json",          s_json_pretty,   _l_json,  "JSON, indent=2 (the old ceiling)"),
    Candidate("json5",         s_json5,         _l_json5, "JSON with unquoted keys (JSON5)"),
    Candidate("yaml_flow_min", s_yaml_flow_min, _l_yaml,  "YAML flow style, minimal padding"),
    Candidate("yaml_flow",     s_yaml_flow,     _l_yaml,  "YAML flow style, PyYAML default padding"),
    Candidate("yaml_hybrid",   s_yaml_hybrid,   _l_yaml,  "YAML block, inline coordinate lists"),
    Candidate("yaml_block",    s_yaml_block,    _l_yaml,  "YAML block style throughout"),
    Candidate("toml",          s_toml,          _l_toml,  "TOML array-of-tables, key=value"),
    Candidate("toml_spaced",   s_toml_spaced,   _l_toml,  "TOML array-of-tables, key = value"),
    # XML's data model has no scalar types: an attribute or text node is a string,
    # so -1.2778 comes back as "-1.2778" and the float/string distinction that
    # separates a coordinate from an id survives only if a schema is supplied
    # alongside. That is the reload failure reported below, and it is a property
    # of the format, not of the reader written here.
    Candidate("xml",           s_xml,           _l_xml,   "XML, attributes + <item> lists"),
    Candidate("csv_tables",    s_csv_tables,    None,
              "DISQUALIFIED: positional columns, header stated once",
              admissible=False),
]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def scenes() -> list[str]:
    return sorted(p.name for p in CONTEXTS.iterdir() if p.is_dir())


def parsed_graph(scene_id: str) -> dict:
    """`json_parser.parse()` output for a scene, read back from disk.

    json_mini is the same object at a different indent setting, so json.json is a
    faithful record of it. Only 3rscan_7f30f36c has a json_mini.json on disk (the
    check_json_mini_fit pilot); where it exists it is asserted byte-for-byte
    against the re-minified object, which proves the reconstruction for the rest.
    """
    pretty = (CONTEXTS / scene_id / "json.json").read_text(encoding="utf-8")
    data = json.loads(pretty)
    on_disk = CONTEXTS / scene_id / "json_mini.json"
    if on_disk.exists():
        mini = on_disk.read_text(encoding="utf-8")
        if json.dumps(data, separators=(",", ":"), ensure_ascii=False) != mini:
            raise SystemExit(f"{scene_id}: json.json and json_mini.json disagree; "
                             f"regenerate scene_contexts before trusting this bake-off")
    return data


def all_questions(scene_id: str) -> list[str]:
    path = QA_ROOT / scene_id / "keyfact-qa.jsonl"
    return [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def longest_question(scene_id: str) -> str:
    """Worst case for the fit test: the longest stem asked of this scene."""
    return max(all_questions(scene_id), key=len)


# --------------------------------------------------------------------------- #
# Sizers, offline
# --------------------------------------------------------------------------- #

def offline_sizer(model_tag: str) -> Callable[[str], int]:
    """token_count.make_sizer without the server: the cached spec is the spec.

    make_sizer() keys its disk cache by the digest reported by /api/tags, so it
    needs the host up. Here the cached file IS the record of what that host was
    serving, so it is loaded directly and the tokenizer built by the same code.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_tag)
    matches = sorted((token_count._CACHE_DIR).glob(f"{safe}__*.json.gz"))
    if not matches:
        raise SystemExit(f"no cached tokenizer spec for {model_tag} in "
                         f"{token_count._CACHE_DIR}; run once with the tunnel up")
    with gzip.open(matches[-1], "rt", encoding="utf-8") as fh:
        spec = json.load(fh)
    render, expected_pre = token_count._family_for(model_tag)
    tok = token_count._build(spec, expected_pre)
    system, bos = spec["system"], (1 if spec["add_bos"] else 0)
    return lambda prompt: len(tok.encode(render(prompt, system)).ids) + bos


def prompt_for(context: str, question: str) -> str:
    # byte-identical to runner._generate_one
    return _RESPONDER_PROMPT.format(context=context.strip(), question=question.strip())


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", action="append", help="restrict to these scenes")
    ap.add_argument("--models", nargs="*", default=REPORT_MODELS)
    ap.add_argument("--dump-samples", metavar="DIR",
                    help="write the first 40 lines of every candidate for the "
                         "largest scene, so readability can be judged by eye")
    args = ap.parse_args()

    scene_ids = args.scene or scenes()
    profiles = {p.name: p for p in MODEL_PROFILES}

    graphs = {s: parsed_graph(s) for s in scene_ids}
    questions = {s: longest_question(s) for s in scene_ids}
    texts = {s: {c.name: c.fn(graphs[s]) for c in CANDIDATES} for s in scene_ids}

    # --- losslessness, checked on every scene, before any token is counted ---
    print("=" * 100)
    print("LOSSLESSNESS  reload each candidate and compare to parse() output, types included")
    print("=" * 100)
    verdicts: dict[str, str] = {}
    for c in CANDIDATES:
        fails = [(s, m) for s in scene_ids if (m := c.verify(graphs[s]))]
        verdicts[c.name] = "" if not fails else fails[0][1]
        mark = "OK  " if not fails else "FAIL"
        detail = "" if not fails else f"   [{fails[0][0]}] {fails[0][1]}"
        print(f"  {mark}  {c.name:<14} {len(scene_ids) - len(fails)}/{len(scene_ids)} scenes{detail}")

    # --- per-model token tables ---
    biggest = max(scene_ids, key=lambda s: len(texts[s][BASELINE]))
    for model in args.models:
        prof = profiles[model]
        sizer = offline_sizer(prof.model)
        limit = prof.options["num_ctx"] - _CTX_RESERVE

        print()
        print("=" * 100)
        print(f"{model}   ({prof.model})   effective prompt budget "
              f"{prof.options['num_ctx']:,} - {_CTX_RESERVE} = {limit:,} tokens")
        print("=" * 100)
        print(f"\n  WORST SCENE: {biggest} (largest json_mini), longest question stem\n")
        print(f"  {'candidate':<15} {'rep tok':>9} {'prompt tok':>11} {'vs json_mini':>13} "
              f"{'fits':>6}  {'lossless':>8}  note")
        print("  " + "-" * 96)

        base_prompt = sizer(prompt_for(texts[biggest][BASELINE], questions[biggest]))
        scaffold = sizer(prompt_for("", ""))
        rows = []
        for c in CANDIDATES:
            rep_tok = sizer(prompt_for(texts[biggest][c.name], "")) - scaffold
            p_tok = sizer(prompt_for(texts[biggest][c.name], questions[biggest]))
            rows.append((c, rep_tok, p_tok))
        for c, rep_tok, p_tok in sorted(rows, key=lambda r: r[2]):
            delta = (p_tok - base_prompt) / base_prompt * 100
            tag = ("yes" if p_tok < limit else "NO")
            loss = "yes" if not verdicts[c.name] else "NO"
            print(f"  {c.name:<15} {rep_tok:>9,} {p_tok:>11,} {delta:>+12.1f}% {tag:>6}  "
                  f"{loss:>8}  {c.note}")

        # every scene, so a win on one scene is not mistaken for a win
        print(f"\n  ALL SCENES: prompt tokens, % vs json_mini, and (fits?)\n")
        head = f"  {'scene':<20} {'json_mini':>10}"
        cand_cols = [c for c in CANDIDATES if c.name != BASELINE]
        head += "".join(f"{c.name:>15}" for c in cand_cols)
        print(head)
        print("  " + "-" * (30 + 15 * len(cand_cols)))
        for s in sorted(scene_ids, key=lambda s: -len(texts[s][BASELINE])):
            base = sizer(prompt_for(texts[s][BASELINE], questions[s]))
            cells = []
            for c in cand_cols:
                n = sizer(prompt_for(texts[s][c.name], questions[s]))
                pct = (n - base) / base * 100
                cells.append(f"{pct:>+13.1f}%" + ("!" if n >= limit else " "))
            print(f"  {s:<20} {base:>10,}" + "".join(cells))
        print("\n  '!' = does not fit the effective budget."
              "  Percentages are of the full responder prompt, not the context alone.")

        # A single worst-case stem gives a yes/no; the question set gives the
        # margin. Where the verdict is 'NO' by a percent or two it matters whether
        # that is every question or only the long ones.
        print(f"\n  FIT MARGIN on {biggest}, over all {len(all_questions(biggest))} "
              f"question stems (budget {limit:,})\n")
        print(f"  {'candidate':<15} {'min':>9} {'max':>9} {'stems fitting':>15}")
        print("  " + "-" * 52)
        stems = all_questions(biggest)
        for c, _rep, _p in sorted(rows, key=lambda r: r[2]):
            counts = [sizer(prompt_for(texts[biggest][c.name], q)) for q in stems]
            n_fit = sum(1 for n in counts if n < limit)
            print(f"  {c.name:<15} {min(counts):>9,} {max(counts):>9,} "
                  f"{n_fit:>10}/{len(stems)}")

    if args.dump_samples:
        out = Path(args.dump_samples)
        out.mkdir(parents=True, exist_ok=True)
        for c in CANDIDATES:
            (out / f"{biggest}.{c.name}.txt").write_text(
                texts[biggest][c.name][:4000], encoding="utf-8")
        print(f"\nwrote {len(CANDIDATES)} samples (first 4000 chars) to {out}")


if __name__ == "__main__":
    main()
