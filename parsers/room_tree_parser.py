"""room_tree — the drawn middle rung of the structure-presentation axis (topology_inventory ->
room_tree -> graph_digest); connectivity only, no metric or object inventory, so score deltas
isolate presentation. Assumes acyclic connectivity (true of all evaluated ProcTHOR scenes) --
`_center`'s leaf-peel does not terminate on a cycle."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque

from _base import run_parser, NotApplicable
from _format import room_label, sort_key
from scene_graph.capabilities import has_room_connectivity
from scene_graph.models import Building, Room


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


def _center(adj: dict[str, set[str]], comp: list[str]) -> str:
    """Graph center of a component: peel leaf layers until 1-2 nodes remain, so the drawn
    tree is rooted shallow and balanced; ties break to the higher-degree room, then by id."""
    comp_set = set(comp)
    induced = lambda n: len(adj[n] & comp_set)
    if len(comp) <= 2:
        return sorted(comp, key=lambda n: (-induced(n), sort_key(n)))[0]
    deg = {n: induced(n) for n in comp}
    leaves = [n for n in comp if deg[n] <= 1]
    count = len(comp)
    while count > 2:
        nxt: list[str] = []
        for leaf in leaves:
            count -= 1
            for nb in adj[leaf] & comp_set:
                if deg[nb] > 0:
                    deg[nb] -= 1
                    if deg[nb] == 1:
                        nxt.append(nb)
        leaves = nxt
    return sorted(leaves, key=lambda n: (-induced(n), sort_key(n)))[0]


def _spanning_tree(adj, root, comp):
    """BFS spanning tree rooted at `root`: ordered children + the leftover (back) edges,
    which would close a loop if drawn, so they're returned separately instead of dropped."""
    comp_set = set(comp)
    children: dict[str, list[str]] = {n: [] for n in comp}
    seen = {root}
    q = deque([root])
    tree_edges: set[frozenset] = set()
    while q:
        u = q.popleft()
        for v in sorted(adj[u] & comp_set, key=sort_key):
            if v not in seen:
                seen.add(v)
                children[u].append(v)
                tree_edges.add(frozenset((u, v)))
                q.append(v)
    back: set[frozenset] = set()
    for u in comp:
        for v in adj[u] & comp_set:
            e = frozenset((u, v))
            if e not in tree_edges:
                back.add(e)
    back_pairs = [tuple(sorted(e, key=sort_key)) for e in back]
    return children, back_pairs


def _render(children, root, label_of) -> list[str]:
    """Draw one rooted tree as indented text rows (the `tree`-command shape)."""
    rows = [label_of(root)]

    def walk(node: str, prefix: str) -> None:
        kids = children.get(node, [])
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            rows.append(prefix + ("`-- " if last else "|-- ") + label_of(k))
            walk(k, prefix + ("    " if last else "|   "))

    walk(root, "")
    return rows


def parse(building: Building) -> str:
    if not has_room_connectivity(building):
        raise NotApplicable("room_tree needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    adj = _adjacency(building.connectivity, rooms)
    connection_count = sum(len(v) for v in adj.values()) // 2
    label_of = lambda rid: room_label(rooms[rid])

    head = (
        f"{building.name} — {len(rooms)} rooms, {connection_count} room connections "
        "(doorways or open passages). Connectivity tree: rooms drawn as an indented "
        "tree; each room is listed under a room it directly connects to (indentation "
        "and branch lines show the connection)."
    )
    lines = [head, ""]

    comps = sorted(_components(adj), key=lambda c: (-len(c), sort_key(min(c, key=sort_key))))
    multi = len(comps) > 1
    all_back: list[tuple[str, str]] = []

    for i, comp in enumerate(comps, 1):
        if multi:
            lines.append(f"Group {i} ({len(comp)} rooms):")
        if len(comp) == 1:
            lines.append(f"{label_of(comp[0])}  (isolated, no connections)")
            lines.append("")
            continue
        root = _center(adj, comp)
        children, back = _spanning_tree(adj, root, comp)
        all_back.extend(back)
        lines.extend(_render(children, root, label_of))
        lines.append("")

    if all_back:
        lines.append("Additional connections (these close loops, so they are not drawn as branches above):")
        for u, v in sorted(all_back, key=lambda e: (sort_key(e[0]), sort_key(e[1]))):
            lines.append(f"  {label_of(u)} -- {label_of(v)}")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Draw the room connectivity graph as an indented ASCII tree")
