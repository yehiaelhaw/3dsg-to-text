from dataclasses import dataclass, field


@dataclass
class SceneObject:
    id: str
    category: str | None
    position: tuple[float, float, float]
    short_id: str | None = None                # compact per-scene label id; minted by the loader where raw ids are noisy (ProcTHOR pipe strings)
    size: tuple[float, float, float] | None = None
    affordances: list[str] | None = None       # Gibson: action_affordance; ProcTHOR: primary+secondary properties
    material: list[str] | None = None          # Gibson only
    visual_texture: str | None = None          # Gibson only
    tactile_texture: str | None = None         # Gibson only


@dataclass
class Room:
    id: str
    position: tuple[float, float, float] | None = None
    category: str | None = None
    objects: list[SceneObject] = field(default_factory=list)
    floor: str | None = None                   # Gibson: 'A'/'B'/etc; absent for ProcTHOR (single-floor)
    size: tuple[float, float, float] | None = None
    floor_area: float | None = None            # Gibson only
    volume: float | None = None                # Gibson only


@dataclass
class ObjectRelation:
    subject_id: str
    predicate: str
    object_id: str


@dataclass
class Building:
    name: str
    rooms: dict[str, Room]
    size: tuple[float, float, float] | None = None

    connectivity: dict[str, list[str]] | None = None   # ProcTHOR: doors + open-plan passages; Gibson: None
    object_relations: list[ObjectRelation] | None = None  # 3DSSG annotations; ProcTHOR: generator structure (on / arranged with)
