"""Shared object-relation backbone for the `relations_*` parsers.

The five `relations_*` views (`relations_flat`, `relations_subject`,
`relations_predicate`, `relations_tree`, `relations_digest`) are all
linearizations of one common object-relation graph (`Building.object_relations`),
so a score delta between them is a delta over a shared source graph rather than
over separately-sourced content. That is weaker than "one fixed printed edge set",
and deliberately so: `relations_tree` and `relations_digest` are avowedly lossy,
and `relations_subject` applies a bounded per-object cap to attribute predicates
(see its own docstring), so only `relations_flat` and `relations_predicate` print
the edge set in full. This module holds the one backbone they share: label
resolution, a dataset-agnostic predicate classifier, and the support forest used
by the drawn-tree and digest views.

Dataset-agnostic by design: predicates are bucketed by *structure*, not by a
hard-coded per-dataset vocabulary. 3DSSG (3RScan) supplies a dense ~40-predicate
graph; ProcTHOR supplies a sparse `on` / `arranged with` forest. The same code
serializes both -- it just degrades to a near-trivial drawing on the sparse case.
"""

from collections import defaultdict

from _format import obj_label, sort_key, is_attribute_predicate

# Structural predicate classes. is_attribute_predicate (``same ...`` / ``... than``)
# is checked first, so comparative verticals like "higher than"/"lower than" land
# in `attribute`, not `directional`. Unlisted predicates fall through to `other`
# (kept as generic directed edges) so a new vocabulary is never silently dropped.
#
# SUPPORT edges are read child -> parent: the triple "subject PREDICATE object"
# means the subject rests in/on/within the object, so the *object* is the base
# (parent) and the subject is what it carries (child). This holds for 3DSSG
# ("cushion supported by sofa") and ProcTHOR ("child on receptacle") alike.
SUPPORT = {
    "on", "supported by", "standing on", "lying on", "sitting on", "hanging on",
    "standing in", "lying in", "hanging in", "inside", "build in", "cover",
    "leaning against", "part of", "belonging to", "attached to",
}
PROXIMITY = {"close by", "next to", "near", "beside", "arranged with"}
DIRECTIONAL = {"left", "right", "front", "behind", "above", "below"}


def classify_predicate(pred: str) -> str:
    """One of: support | proximity | directional | attribute | other.

    `attribute` (symmetric ``same ...`` cliques and comparative ``... than``
    edges) is tested first; these are the low-value O(k^2) edges that drown the
    spatial ones if treated like the rest.
    """
    if is_attribute_predicate(pred):
        return "attribute"
    if pred in SUPPORT:
        return "support"
    if pred in PROXIMITY:
        return "proximity"
    if pred in DIRECTIONAL:
        return "directional"
    return "other"


def resolve_labels(building):
    """`(obj_room, lbl)` where `lbl(object_id)` -> ``"chair [5]"`` for any id.

    Prefers the loader's compact `short_id` (ProcTHOR) and falls back to the raw
    id (3DSSG objectIds). Unknown ids -- relation endpoints with no loaded object
    -- degrade to ``"object [<id>]"`` rather than raising.
    """
    obj_room: dict[str, str] = {}
    label: dict[str, str] = {}
    for rid, room in building.rooms.items():
        for obj in room.objects:
            obj_room[obj.id] = rid
            label[obj.id] = obj_label(obj.category, obj.short_id or obj.id)

    def lbl(oid: str) -> str:
        return label.get(oid, f"object [{oid}]")

    return obj_room, lbl


def support_forest(relations):
    """Build the support/containment forest from the support-class edges.

    Returns `(children, roots, back_edges, edge_pred)`:
      * `children[parent]` -> ordered child ids (the drawn branches),
      * `roots` -> base nodes with no parent (floor, walls, free-standing bases),
      * `back_edges` -> `(child, parent, predicate)` triples not drawn: a child's
        *second* base (kept to one parent per child so the drawing is a tree) or an
        edge whose inclusion would close a cycle. Listed, never silently dropped, so
        a `relations_tree` delta stays interpretable.
      * `edge_pred[child]` -> the predicate of the child's drawn up-edge to its
        chosen parent (``attached to``, ``hanging on``, ...). Support is *typed* --
        a ceiling is attached to a wall, a towel hangs on a door -- so the drawing
        names the relation on each branch rather than flattening all of them to a
        single "rests on".

    Deterministic: one parent per child (lowest id wins; lowest predicate name on a
    tie), children sorted by id.
    """
    parents_all: dict[str, set[str]] = defaultdict(set)
    preds: dict[tuple[str, str], set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for r in relations:
        if classify_predicate(r.predicate) != "support":
            continue
        child, parent = r.subject_id, r.object_id
        if child == parent:
            continue
        parents_all[child].add(parent)
        preds[(child, parent)].add(r.predicate)
        nodes.add(child)
        nodes.add(parent)

    def pred_of(child: str, parent: str) -> str:
        return sorted(preds[(child, parent)])[0]

    chosen: dict[str, str] = {}
    back: list[tuple[str, str, str]] = []
    for child in sorted(parents_all, key=sort_key):
        ranked = sorted(parents_all[child], key=sort_key)
        chosen[child] = ranked[0]
        back.extend((child, extra, pred_of(child, extra)) for extra in ranked[1:])

    # Break any cycle deterministically: walking up from a node must terminate.
    for child in sorted(list(chosen), key=sort_key):
        seen, cur = {child}, child
        while cur in chosen:
            cur = chosen[cur]
            if cur in seen:
                back.append((child, chosen[child], pred_of(child, chosen[child])))
                del chosen[child]
                break
            seen.add(cur)

    children: dict[str, list[str]] = defaultdict(list)
    edge_pred: dict[str, str] = {}
    for child, parent in chosen.items():
        children[parent].append(child)
        edge_pred[child] = pred_of(child, parent)
    for parent in children:
        children[parent].sort(key=sort_key)

    roots = sorted([n for n in nodes if n not in chosen], key=sort_key)
    back.sort(key=lambda e: (sort_key(e[0]), sort_key(e[1])))
    return children, roots, back, edge_pred


def render_tree(children: dict[str, list[str]], root: str, label_of, edge_label=None) -> list[str]:
    """Draw one rooted tree as indented rows (the `tree`-command shape).

    Same pure-ASCII glyphs as `room_tree` ("|-- " / "`-- " / "|   ") so the nesting
    is carried by indentation. If `edge_label(child)` is given, the relation tying a
    child to its parent is named in parentheses on the branch -- so the drawing keeps
    *which* kind of support each edge is, not just that one exists.
    """
    rows = [label_of(root)]

    def walk(node: str, prefix: str) -> None:
        kids = children.get(node, [])
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            lab = edge_label(k) if edge_label else ""
            suffix = f" ({lab})" if lab else ""
            rows.append(prefix + ("`-- " if last else "|-- ") + label_of(k) + suffix)
            walk(k, prefix + ("    " if last else "|   "))

    walk(root, "")
    return rows
