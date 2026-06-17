"""ascii_map -- room connectivity drawn as a 2D ASCII tree (a floor-plan sketch).

The thesis bottleneck is that flattening a graph into a 1D token string loses the
native 2D spatial adjacency a human reads off a map at a glance. Every other
connectivity view here restates *local* adjacency in linear form -- `topology` as a
labelled list ("Bedroom [6] connects to ..."), `prose` as sentences, `graph_digest`
as pre-computed global facts. This parser instead lays the room graph out in two
dimensions and draws the connections as branches, so adjacency is carried by
*position on the page* rather than by a list the reader must traverse.

Axis role: it is the visual middle rung of the structure-presentation axis (F) --
  topology (raw adjacency list) -> ascii_map (adjacency drawn) -> graph_digest
  (adjacency's consequences stated)
-- and a third point on the format axis (B): structured list vs natural language vs
spatial diagram. Like `graph_digest` it is deliberately connectivity-only (no metric
data, no per-room object inventory): a map of doors, nothing else, so any score delta
against `topology`/`graph_digest` is attributable to the *presentation* of the same
connectivity, not to extra content.

Layout: each connected component is rooted at its graph center and drawn as an
indented tree (the `tree`-command shape: each room listed under the room it connects
up to, with branch glyphs and indentation carrying the connection). Deterministic;
children ordered by id. The indented form never collides and stays bounded in width
no matter how wide or deep the graph is. A scene with no cycles (the common ProcTHOR
case) renders as a single faithful tree. Any connection that would close a loop is
omitted from the drawing and listed afterwards as a back-edge, so the map never
implies a false adjacency nor hides a real one.

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


def _center(adj: dict[str, set[str]], comp: list[str]) -> str:
    """Graph center of a component: peel leaf layers until 1-2 nodes remain.

    Rooting at the center keeps the drawn tree shallow and balanced; ties break to
    the higher-degree room (the hub), then by id, so the layout is deterministic.
    """
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
    """BFS spanning tree rooted at `root`: ordered children + the leftover edges.

    Tree edges become drawn branches; every other in-component edge would close a
    loop, so it is returned as a back-edge to be listed (never silently dropped).
    """
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
    """Draw one rooted tree as indented text rows (the `tree`-command shape).

    Each child sits on its own line under its parent, prefixed by the branch glyphs
    that show the connection: "|-- " for a child with siblings still to come, "`-- "
    for the last child, and the running prefix carries "|   " / "    " so deeper
    levels stay vertically aligned under the right ancestor.
    """
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
        raise NotApplicable("ascii_map needs a room connectivity graph; this scene has none")

    rooms = building.rooms
    adj = _adjacency(building.connectivity, rooms)
    connection_count = sum(len(v) for v in adj.values()) // 2
    label_of = lambda rid: room_label(rooms[rid])

    head = (
        f"{building.name} — {len(rooms)} rooms, {connection_count} room connections "
        "(doorways or open passages). Connectivity map: rooms drawn as an indented "
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
    run_parser(parse, "Draw the room connectivity graph as a 2D ASCII map")
