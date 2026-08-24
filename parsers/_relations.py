"""Shared object-relation backbone for the `relations_*` parsers: label resolution, a
dataset-agnostic predicate classifier, and the support forest used by the drawn-tree and
digest views, so a score delta between those parsers is over one shared source graph."""

from collections import defaultdict

from _format import obj_label, sort_key, is_attribute_predicate

# SUPPORT edges are read child -> parent: "subject PREDICATE object" means the subject
# rests in/on/within the object, so the object is the base (parent).
SUPPORT = {
    "on", "supported by", "standing on", "lying on", "sitting on", "hanging on",
    "standing in", "lying in", "hanging in", "inside", "build in", "cover",
    "leaning against", "part of", "belonging to", "attached to",
}
PROXIMITY = {"close by", "next to", "near", "beside", "arranged with"}
DIRECTIONAL = {"left", "right", "front", "behind", "above", "below"}


def classify_predicate(pred: str) -> str:
    """One of: support | proximity | directional | attribute | other. `attribute` is
    tested first so comparative verticals ("higher than") don't land in `directional`."""
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
    """`(obj_room, lbl)` where `lbl(object_id)` -> ``"chair [5]"`` for any id; unknown ids
    degrade to ``"object [<id>]"`` rather than raising."""
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
    """Build the support/containment forest from the support-class edges. Returns
    `(children, roots, back_edges, edge_pred)`, where `back_edges` are the edges not drawn
    (a child's second base, or one that would close a cycle) -- listed, never dropped.
    Deterministic: one parent per child (lowest id wins, then lowest predicate name)."""
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
    """Draw one rooted tree as indented rows (the `tree`-command shape, same glyphs as
    `room_tree`). If `edge_label(child)` is given, it's named in parentheses on the branch."""
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
