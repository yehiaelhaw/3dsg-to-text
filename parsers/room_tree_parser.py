"""room_tree -- room connectivity drawn as an indented connectivity tree.

The thesis bottleneck is that flattening a graph into a 1D token string loses the
adjacency a reader takes in at a glance. Every other connectivity view here restates
*local* adjacency in linear form -- `topology` as a labelled list ("Bedroom [6]
connects to ..."), `prose` as sentences, `graph_digest` as pre-computed global facts.
This parser instead *draws* the room graph: each room is listed under a room it
connects to, with indentation and branch glyphs carrying the connection, so adjacency
is read off the tree's shape rather than from a list the reader must traverse.

(It was originally conceived as a 2D floor-plan-style map; that idea was dropped --
laying rooms out on a grid is ambiguous and lossy -- in favour of the indented
`tree`-command shape, which is unambiguous and bounded in width. The name reflects
the tree, not the abandoned map.)

Axis role: it is the drawn middle rung of the structure-presentation axis (F) --
  topology (raw adjacency list) -> room_tree (adjacency drawn) -> graph_digest
  (adjacency's consequences stated)
-- and a third point on the format axis (B): structured list vs natural language vs
drawn tree. Like `graph_digest` it is deliberately connectivity-only (no metric data,
no per-room object inventory): the door graph, nothing else, so any score delta
against `topology`/`graph_digest` is attributable to the *presentation* of the same
connectivity, not to extra content. It is the room-level analogue of `relations_tree`
(the object-level support forest drawn the same way).

Layout: each connected component is rooted at its graph center and drawn as an
indented tree (the `tree`-command shape: each room listed under the room it connects
up to, with branch glyphs and indentation carrying the connection). Deterministic;
children ordered by id. The indented form never collides and stays bounded in width
no matter how wide or deep the graph is. A scene with no cycles renders as a single
faithful tree. A connection that would close a loop is intended to be omitted from
the drawing and listed afterwards as a back-edge, so that the tree implies no false
adjacency and hides no real one.

Validated on tree-structured connectivity only. All three evaluated ProcTHOR scenes
are acyclic (10 rooms, 9 connections each), so every shipped drawing is a lossless
spanning tree -- and so the back-edge path above has never actually run. It should
not be described as exercised or as robust on arbitrary graphs: `_center` peels leaf
layers until one or two nodes remain, and a component containing a cycle can reach a
state with no leaf left to peel, at which point the loop makes no further progress.
A latent limitation of the frozen implementation, untriggered by the evaluated
scenes; left as-is because those scenes are the ones the reported results come from.

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

    Assumes an acyclic component, which is what every evaluated ProcTHOR scene
    supplies. The peel does not terminate on a component holding a cycle: once the
    tree fringe is consumed no node has induced degree <= 1, so the layer is empty
    and the remaining count stops falling. Never reached by the evaluated scenes.
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
