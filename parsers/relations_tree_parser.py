"""relations_tree — the object-level analogue of room_tree; draws the support/containment
forest as an indented tree instead of a flat edge list. Only support/containment edges are
drawn — everything else (proximity, directional, attribute) is quantified in a trailer, not
hidden, so a delta against relations_flat/relations_subject stays attributable to presentation."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter, defaultdict

from _base import run_parser, NotApplicable
from _format import sort_key
from _relations import resolve_labels, classify_predicate, support_forest, render_tree
from scene_graph.capabilities import has_object_relations
from scene_graph.models import Building

_CLASS_LABEL = {
    "proximity": "proximity (close by / next to)",
    "directional": "directional (left / right / front / behind)",
    "attribute": "comparative / shared-attribute (same ... / ... than)",
    "other": "other",
}


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("relations_tree needs annotated object relations; this scene has none")

    _, lbl = resolve_labels(building)
    children, roots, back, edge_pred = support_forest(building.object_relations)

    head = (
        f"{building.name} -- support / containment structure drawn as an indented tree. "
        "Each object is listed under the one it sits on or is fixed to, and the "
        "parenthesised relation says how it attaches to that object above it -- e.g. "
        "'wall [2] (attached to)' under 'floor [1]' means wall [2] is attached to floor [1]."
    )
    lines = [head, ""]

    drawn_roots = [r for r in roots if children.get(r)]
    if not drawn_roots:
        lines.append("No support or containment relations to draw in this scene.")
    else:
        # Each tree is identified by its (unindented) root; a blank line separates
        # them, so no group header is needed.
        for root in drawn_roots:
            lines.extend(render_tree(children, root, lbl, edge_label=lambda c: edge_pred[c]))
            lines.append("")

    if back:
        lines.append("Additional support links not drawn (a second base for an object, "
                     "or one that would close a loop):")
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for child, parent, pred in back:
            grouped[(child, pred)].append(parent)
        for child, pred in sorted(grouped, key=lambda k: (sort_key(k[0]), k[1])):
            parents = ", ".join(lbl(p) for p in sorted(grouped[(child, pred)], key=sort_key))
            lines.append(f"  {lbl(child)} is also {pred} {parents}")
        lines.append("")

    # Quantify every non-support edge so the drawing's omissions are explicit.
    other = Counter()
    for r in building.object_relations:
        cls = classify_predicate(r.predicate)
        if cls != "support":
            other[cls] += 1
    if other:
        parts = ", ".join(
            f"{other[c]} {_CLASS_LABEL[c]}" for c in ("proximity", "directional", "attribute", "other") if other[c]
        )
        lines.append(f"Relations not shown in this support view: {parts}.")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Draw the support/containment relations as an indented ASCII tree")
