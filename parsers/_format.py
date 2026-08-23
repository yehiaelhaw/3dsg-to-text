"""Shared formatting helpers for the natural-language parsers.

These keep label and inventory rendering consistent across `prose`,
`topology_inventory`, and `inventory`.
"""

from collections import Counter, defaultdict

from scene_graph.models import Room


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


def room_type_summary(rooms) -> str:
    """Room categories by descending frequency, same `×` convention as objects:
    `"bathroom (×5), kitchen (×3), lobby"`. Empty string when no room is typed."""
    counts = Counter(r.category for r in rooms if r.category)
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
