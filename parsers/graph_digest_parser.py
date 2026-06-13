"""graph_digest — derived global structure of the room connectivity graph.

Every other connectivity parser (`topology`, `prose`, `navigation`) states *local*
adjacency: "Bedroom [6] connects to ...". The reader must then traverse that list to
answer global questions — which rooms are mutually reachable, which room is a
bottleneck, how many connections separate two rooms — exactly the transitive-closure
reasoning LLMs are weakest at. This parser pre-computes those facts instead:

  * reachability groups (connected components),
  * hub rooms (highest connection degree),
  * bottleneck rooms (articulation points) and what they cut off,
  * multi-step distances (shortest-path connection counts, with the route).

It opens a new evaluation axis — *raw adjacency* (`topology`) vs *derived structure*
(here) — orthogonal to the format axis (`topology` vs `prose`). It is deliberately
metric-free (connection counts, not metres; keeps it on the connectivity rung, not
the metric one) and carries no per-room object inventory (that is `topology`'s job;
omitting it is what stops this from being a sixth restatement of the same primitives).

Runs only where a room connection graph exists (ProcTHOR); refuses elsewhere.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from utils.capabilities import has_room_connectivity
from utils.models import Building, Room


def _adjacency(connectivity: dict[str, list[str]], rooms: dict[str, Room]) -> dict[str, set[str]]:
    """Undirected neighbour sets, restricted to rooms that actually exist."""
    adj: dict[str, set[str]] = {rid: set() for rid in rooms}
    for rid, neighbors in connectivity.items():
        if rid not in rooms:
            continue
        for n in neighbors:
            if n in rooms and n != rid:
                adj[rid].add(n)
                adj[n].add(rid)
    return adj


def _components(adj: dict[str, set[str]]) -> list[list[str]]:
    """Connected components as lists of room ids, via flood fill."""
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def _bfs(adj: dict[str, set[str]], src: str) -> tuple[dict[str, int], dict[str, str]]:
    """Shortest-path hop distance and parent pointers from `src` (BFS)."""
    dist = {src: 0}
    parent: dict[str, str] = {}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u], key=sort_key):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return dist, parent


def _path(parent: dict[str, str], src: str, dst: str) -> list[str]:
    """Reconstruct the room-id path src..dst from BFS parent pointers."""
    route = [dst]
    while route[-1] != src:
        route.append(parent[route[-1]])
    route.reverse()
    return route


def _articulation_points(adj: dict[str, set[str]]) -> dict[str, list[list[str]]]:
    """Rooms whose removal splits their group, mapped to the pieces left behind.

    Brute force (graphs here are tiny): drop each room, and if its component breaks
    into two or more pieces, record those pieces. The room is then a bottleneck — the
    only way between the pieces.
    """
    base = len(_components(adj))
    cuts: dict[str, list[list[str]]] = {}
    for node in adj:
        sub = {u: (adj[u] - {node}) for u in adj if u != node}
        pieces = _components(sub)
        if len(pieces) > base:
            # The pieces that were joined only through `node` are the ones whose
            # rooms all sat in node's original component. Keep those.
            neighbours = adj[node]
            split = [p for p in pieces if neighbours & set(p)]
            cuts[node] = sorted(split, key=lambda p: (len(p), sort_key(p[0])))
    return cuts


def _degree(adj: dict[str, set[str]], rid: str) -> int:
    return len(adj[rid])


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("graph_digest needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    adj = _adjacency(building.connectivity, rooms)
    connection_count = sum(len(v) for v in adj.values()) // 2

    def label(rid: str) -> str:
        return room_label(rooms[rid])

    def by_degree(ids):
        return sorted(ids, key=lambda r: (-_degree(adj, r), sort_key(r)))

    head = f"{building.name} — {len(rooms)} rooms, {connection_count} room connections " \
           "(doorways or open passages). Connectivity digest (derived from the room " \
           "connection graph; no metric data)."
    lines = [head, ""]

    # -- Reachability groups (connected components) --
    comps = sorted(_components(adj), key=lambda c: (-len(c), sort_key(c[0])))
    if len(comps) == 1:
        lines.append("Reachability: every room is reachable from every other (one connected group).")
    else:
        lines.append(
            f"Reachability: the rooms split into {len(comps)} separate groups — "
            "you cannot walk from one group to another."
        )
    lines.append("")
    for i, comp in enumerate(comps, 1):
        members = ", ".join(label(r) for r in by_degree(comp))
        if len(comp) == 1:
            lines.append(f"Group {i} (1 room): {members} — isolated, no connections.")
        else:
            lines.append(f"Group {i} ({len(comp)} rooms): {members}")
    lines.append("")

    # -- Hubs (highest door degree) --
    ranked = by_degree([r for r in rooms if _degree(adj, r) > 0])
    if ranked:
        top = _degree(adj, ranked[0])
        hubs = [r for r in ranked if _degree(adj, r) == top]
        hub_str = ", ".join(f"{label(r)} ({_degree(adj, r)} connections)" for r in hubs)
        lines.append(f"Hub rooms (most connections): {hub_str}.")

    # -- Bottlenecks (articulation points) --
    cuts = _articulation_points(adj)
    if cuts:
        lines.append("Bottlenecks (removing the room cuts its group apart):")
        for node in by_degree(cuts.keys()):
            pieces = cuts[node]
            piece_strs = []
            for p in pieces:
                piece_strs.append(", ".join(label(r) for r in by_degree(p)))
            joined = "  |  ".join(f"{{{s}}}" for s in piece_strs)
            lines.append(f"  {label(node)} — without it these become mutually unreachable: {joined}")
    else:
        lines.append("Bottlenecks: none — no single room is a sole connector.")
    lines.append("")

    # -- Multi-step distances (shortest-path door counts, with the route) --
    pair_lines: list[tuple[int, str, str, str]] = []
    for comp in comps:
        if len(comp) < 3:
            continue  # 1- and 2-room groups have nothing multi-step
        for src in comp:
            dist, parent = _bfs(adj, src)
            for dst, d in dist.items():
                if d >= 2 and sort_key(src) < sort_key(dst):
                    route = _path(parent, src, dst)
                    via = ", ".join(label(r) for r in route[1:-1])
                    pair_lines.append((d, src, dst, via))
    if pair_lines:
        # Farthest pairs first: the diameter is the least obvious fact.
        pair_lines.sort(key=lambda t: (-t[0], sort_key(t[1]), sort_key(t[2])))
        lines.append("Distance between rooms in steps (connections to cross; only multi-step pairs):")
        for d, src, dst, via in pair_lines:
            lines.append(f"  {label(src)} -> {label(dst)}: {d} steps (via {via})")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize derived global structure of the room connectivity graph")
