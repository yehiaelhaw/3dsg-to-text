"""Shared formatting helpers for the natural-language parsers.

These keep label and inventory rendering consistent across `prose`,
`spatial_relations`, `topology`, and `object_graph`.
"""

from collections import Counter, defaultdict

from utils.models import Building, Room


def sort_key(id_str: str):
    """Sort numeric ids numerically, falling back to lexicographic for the rest."""
    try:
        return (0, int(id_str))
    except ValueError:
        return (1, id_str)


def room_label(room: Room) -> str:
    """`"kitchen [16]"`, or `"room [16]"` when the category is unknown."""
    return f"{room.category} [{room.id}]" if room.category else f"room [{room.id}]"


def obj_label(category: str | None, obj_id: str) -> str:
    """`"chair [5]"`, or `"object [5]"` when the category is unknown."""
    return f"{category} [{obj_id}]" if category else f"object [{obj_id}]"


def object_inventory(room: Room) -> str:
    """Object categories by descending frequency: `"chair (×4), table (×2), floor"`."""
    if not room.objects:
        return "no objects"
    counts = Counter(o.category or "object" for o in room.objects)
    parts = []
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        parts.append(f"{cat} (×{n})" if n > 1 else cat)
    return ", ".join(parts)


def is_attribute_predicate(predicate: str) -> bool:
    """True for low-value symmetric/comparative edges (``same color``, ``brighter than``).

    These form large symmetric cliques (every wall is ``same material`` as every
    other wall) that drown the high-value spatial/support edges (``standing on``,
    ``close by``) if left unchecked.
    """
    return predicate.startswith("same ") or predicate.endswith(" than")


def undirected_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components (size >= 2) over undirected edges, via union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    return [members for members in groups.values() if len(members) >= 2]


def relation_lines(building: Building, indent: str = "", attr_cap: int = 3) -> list[str]:
    """Object-object relations grouped by subject, salient (spatial) edges first.

    One line per subject: ``chair [5] — standing on floor [1]; next to table [2].``
    Spatial/support predicates are listed in full and first; comparative attribute
    edges (``brighter than`` …) come last, each capped to ``attr_cap`` objects with
    a ``(+N more)`` tail so they cannot swamp the useful edges. Subjects are ordered
    by spatial-edge degree (then total degree, then id), so genuinely connected
    objects lead rather than wall cliques.

    ``same X`` predicates are symmetric **and** transitive (an equivalence relation:
    ``same color``, ``same material`` …), so restating them per subject duplicates
    the same fact in both directions and inflates the densest objects (every wall is
    ``same material`` as every other wall). They are pulled out of the per-subject
    lines and stated **once per group** as connected components, in a trailing
    ``Shared attributes`` block: ``Same color: wall [2], wall [3], wall [7].``
    (Caveat: components assume transitivity, so a group can assert A~C from annotated
    A~B and B~C; sound for genuine equivalence attributes, but not a literal edge.)
    Symmetric-but-not-transitive edges (``close by``) and directional ones stay
    per-subject. Returns ``[]`` when the scene carries no annotated relations.
    """
    if not building.object_relations:
        return []

    id_to_label: dict[str, str] = {}
    for room in building.rooms.values():
        for obj in room.objects:
            id_to_label[obj.id] = obj_label(obj.category, obj.short_id or obj.id)

    def lbl(oid: str) -> str:
        return id_to_label.get(oid, f"object [{oid}]")

    by_subject: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    same_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in building.object_relations:
        if rel.predicate.startswith("same "):
            same_edges[rel.predicate].append((rel.subject_id, rel.object_id))
        else:
            by_subject[rel.subject_id][rel.predicate].append(rel.object_id)

    def spatial_degree(sid: str) -> int:
        return sum(len(o) for p, o in by_subject[sid].items() if not is_attribute_predicate(p))

    def total_degree(sid: str) -> int:
        return sum(len(o) for o in by_subject[sid].values())

    ordered = sorted(
        by_subject.keys(),
        key=lambda sid: (-spatial_degree(sid), -total_degree(sid), sort_key(sid)),
    )

    lines = []
    for sid in ordered:
        preds = by_subject[sid]
        spatial = sorted(p for p in preds if not is_attribute_predicate(p))
        comparative = sorted(p for p in preds if is_attribute_predicate(p))

        parts = [f"{pred} {', '.join(lbl(o) for o in preds[pred])}" for pred in spatial]
        for pred in comparative:
            objs = preds[pred]
            shown = ", ".join(lbl(o) for o in objs[:attr_cap])
            extra = len(objs) - attr_cap
            parts.append(f"{pred} {shown}" + (f" (+{extra} more)" if extra > 0 else ""))

        if parts:
            lines.append(f"{indent}{lbl(sid)} — {'; '.join(parts)}.")

    # Symmetric+transitive "same X" edges: state each shared-attribute group once.
    group_lines = []
    for pred in sorted(same_edges):
        attr = pred[len("same "):]
        comps = [sorted(set(m), key=sort_key) for m in undirected_components(same_edges[pred])]
        comps.sort(key=lambda g: sort_key(g[0]))
        for g in comps:
            group_lines.append(f"{indent}Same {attr}: {', '.join(lbl(o) for o in g)}.")

    if group_lines:
        lines.append("")
        lines.append(f"{indent}Shared attributes (each line lists items sharing that property):")
        lines.extend(group_lines)

    return lines
