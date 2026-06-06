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


def relation_lines(building: Building, indent: str = "") -> list[str]:
    """Object-object relations grouped by subject, subjects ordered by edge degree.

    One line per subject: ``chair [5] — standing on floor [1]; next to table [2].``
    Predicates within a subject are sorted; neighbourhoods are listed in insertion
    order. Returns ``[]`` when the scene carries no annotated relations.
    """
    if not building.object_relations:
        return []

    id_to_label: dict[str, str] = {}
    for room in building.rooms.values():
        for obj in room.objects:
            id_to_label[obj.id] = obj_label(obj.category, obj.id)

    by_subject: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rel in building.object_relations:
        obj_l = id_to_label.get(rel.object_id, f"object [{rel.object_id}]")
        by_subject[rel.subject_id][rel.predicate].append(obj_l)

    ordered = sorted(
        by_subject.keys(),
        key=lambda sid: -sum(len(v) for v in by_subject[sid].values()),
    )

    lines = []
    for sid in ordered:
        subj_l = id_to_label.get(sid, f"object [{sid}]")
        parts = [f"{pred} {', '.join(objs)}" for pred, objs in sorted(by_subject[sid].items())]
        lines.append(f"{indent}{subj_l} — {'; '.join(parts)}.")
    return lines
