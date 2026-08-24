"""Shared geometry helpers for the metric parsers. The floor plane is detected per scene
(least-spread axis = vertical) rather than hard-coded, since datasets disagree on which
axis is "up"."""

import math

# 8-way rose, counter-clockwise from +u (east), one label per 45 deg sector.
COMPASS = [
    "east", "north-east", "north", "north-west",
    "west", "south-west", "south", "south-east",
]


def vertical_axis(positions: list[tuple[float, float, float]]) -> int:
    """Index of the least-spread axis — treated as 'up' and dropped from the plane."""
    variances = []
    for axis in range(3):
        vals = [p[axis] for p in positions]
        mean = sum(vals) / len(vals)
        variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
    return min(range(3), key=lambda a: variances[a])


def floor_plane(positions: list[tuple[float, float, float]]) -> tuple[int, int]:
    """The two floor-plane axis indices (u, v), ascending, for these positions."""
    up = vertical_axis(positions)
    u, v = (a for a in range(3) if a != up)
    return u, v


def compass(du: float, dv: float) -> str:
    """8-way bearing of the in-plane vector (du, dv); +u is east, +v is north."""
    sector = round(math.atan2(dv, du) / (math.pi / 4)) % 8
    return COMPASS[sector]


def plane_distance(a: tuple, b: tuple, axes: tuple[int, int]) -> float:
    """Euclidean distance between a and b within the floor plane `axes`."""
    u, v = axes
    return math.hypot(a[u] - b[u], a[v] - b[v])


def distance_3d(a: tuple, b: tuple) -> float:
    """Full 3D Euclidean distance between two positions."""
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
