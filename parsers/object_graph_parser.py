"""object_graph — given (not computed) object-object edges.

Serializes relations the dataset itself supplies, as opposed to
`proximity_graph`'s coordinate-computed edges — the source-fidelity axis. On
3RScan these are the hand-annotated 3DSSG labels (``standing on``, ``next to``,
``attached to`` …); on ProcTHOR they come from the generator's placement
structure (``on`` for surface/attached children, ``arranged with`` for asset-group
siblings) — same axis, coarser vocabulary, and findings must be framed as
generator-given rather than annotated. Refuses where no relations exist (Gibson).

Two render modes, chosen by the shape of the relation graph, never by dataset
name. When every subject has exactly one outgoing edge and the graph is acyclic
(a containment forest — ProcTHOR), objects nest as room -> group -> parts using
category labels and counts only; raw ids are generator noise and stay out of the
context. Dense annotated graphs (3RScan) are not forests and keep the flat
subject-grouped edge list, where the short ids disambiguate.
"""

import sys
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _base import run_parser, NotApplicable
from _format import obj_label, room_label, sort_key, relation_lines
from utils.capabilities import has_object_relations
from utils.models import Building, ObjectRelation


def _outgoing(building: Building) -> dict[str, ObjectRelation] | None:
    """``subject_id -> relation`` when the relations form a containment forest
    (each subject has exactly one outgoing edge, no cycles); None otherwise."""
    out: dict[str, ObjectRelation] = {}
    for rel in building.object_relations:
        if rel.subject_id in out:
            return None
        out[rel.subject_id] = rel
    for start in out:
        seen: set[str] = set()
        node = start
        while node in out:
            if node in seen:
                return None
            seen.add(node)
            node = out[node].object_id
    return out


def _phrase(predicate: str) -> str:
    return {"on": "on it", "arranged with": "arranged with it"}.get(predicate, f"{predicate} it")


def _hierarchy_lines(building: Building, out: dict[str, ObjectRelation]) -> list[str]:
    children: dict[str, list[str]] = defaultdict(list)
    for rel in building.object_relations:
        children[rel.object_id].append(rel.subject_id)
    by_id = {o.id: o for room in building.rooms.values() for o in room.objects}

    def cat(oid: str) -> str:
        obj = by_id.get(oid)
        return obj.category if obj and obj.category else "object"

    def sid(oid: str) -> str:
        obj = by_id.get(oid)
        return (obj.short_id or obj.id) if obj else oid

    def agg_label(c: str, ids: list[str]) -> str:
        ids = sorted(ids, key=sort_key)
        if len(ids) > 1:
            return f"{c} (×{len(ids)}) [{', '.join(ids)}]"
        return f"{c} [{ids[0]}]"

    def render(oid: str, indent: str) -> list[str]:
        kids = children.get(oid, [])
        leaves = [k for k in kids if not children.get(k)]
        branches = [k for k in kids if children.get(k)]
        lines = []
        ids_by_kind: dict[tuple[str, str], list[str]] = defaultdict(list)
        for k in leaves:
            ids_by_kind[(cat(k), out[k].predicate)].append(sid(k))
        for (c, pred), ids in sorted(ids_by_kind.items()):
            lines.append(f"{indent}{agg_label(c, ids)} {_phrase(pred)}")
        for k in sorted(branches, key=lambda k: (cat(k), sort_key(sid(k)))):
            lines.append(f"{indent}{cat(k)} [{sid(k)}] {_phrase(out[k].predicate)}:")
            lines.extend(render(k, indent + "    "))
        return lines

    lines = []
    for room in sorted(building.rooms.values(), key=lambda r: sort_key(r.id)):
        roots = [o.id for o in room.objects if o.id not in out]
        grouped = [oid for oid in roots if children.get(oid)]
        singles = [oid for oid in roots if not children.get(oid)]
        lines.append(f"{room_label(room)}:")
        for oid in sorted(grouped, key=lambda i: (cat(i), sort_key(sid(i)))):
            lines.append(f"  {cat(oid)} [{sid(oid)}]:")
            lines.extend(render(oid, "    "))
        ids_by_cat: dict[str, list[str]] = defaultdict(list)
        for i in singles:
            ids_by_cat[cat(i)].append(sid(i))
        for c, ids in sorted(ids_by_cat.items()):
            lines.append(f"  {agg_label(c, ids)}")
        lines.append("")
    return lines


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("object_graph needs given object relations; this scene has none")

    n_objects = sum(len(room.objects) for room in building.rooms.values())
    out = _outgoing(building)
    if out is not None:
        lines = [
            f"{building.name} — {n_objects} objects in {len(building.rooms)} rooms.",
            "",
            "Objects by room; nesting shows what each object sits on or is grouped with:",
            "",
            *_hierarchy_lines(building, out),
        ]
        return "\n".join(lines).rstrip() + "\n"

    objects = [o for room in building.rooms.values() for o in room.objects]
    objects.sort(key=lambda o: ((o.category or "object"), sort_key(o.id)))
    inventory = ", ".join(obj_label(o.category, o.id) for o in objects)

    lines = [
        f"{building.name} — {n_objects} objects: {inventory}.",
        "",
        "Object relations (most-connected first):",
    ]
    lines.extend(relation_lines(building))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize the given object-object relation graph")
