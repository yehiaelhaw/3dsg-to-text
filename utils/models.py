from dataclasses import dataclass, field


@dataclass
class SceneObject:
    id: str
    category: str
    position: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    affordances: list[str] | None = None       # Gibson: action_affordance; ProcTHOR: primary+secondary properties
    material: list[str] | None = None          # Gibson only
    visual_texture: str | None = None          # Gibson only
    tactile_texture: str | None = None         # Gibson only


@dataclass
class Room:
    id: int
    category: str
    position: tuple[float, float, float]
    objects: list[SceneObject] = field(default_factory=list)
    floor_number: str | None = None            # Gibson: 'A'/'B'/etc; ProcTHOR: '0' (always single-floor)
    size: tuple[float, float, float] | None = None
    floor_area: float | None = None            # Gibson only
    volume: float | None = None                # Gibson only


@dataclass
class Building:
    name: str
    rooms: dict[int, Room]
    size: tuple[float, float, float] | None = None
    floor_count: int | None = None
    connectivity: dict[int, list[int]] | None = None   # ProcTHOR: door graph; Gibson: None
