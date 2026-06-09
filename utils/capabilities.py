from utils.models import Building


def has_multiroom_layout(b: Building) -> bool:
    return sum(1 for r in b.rooms.values() if r.position is not None) >= 2


def has_room_connectivity(b: Building) -> bool:
    return bool(b.connectivity) and any(len(v) > 0 for v in b.connectivity.values())


def has_floors(b: Building) -> bool:
    return len({r.floor for r in b.rooms.values() if r.floor is not None}) >= 2


def has_object_relations(b: Building) -> bool:
    return bool(b.object_relations)


def has_object_positions(b: Building) -> bool:
    return sum(
        1 for r in b.rooms.values() for o in r.objects if o.position is not None
    ) >= 2
