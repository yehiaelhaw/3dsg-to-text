"""relations_digest — the object-level analogue of graph_digest; derived consequences of the
object-relation graph (support depth, receptacles, proximity clusters, shared-attribute
cliques, relation census). Support chains, receptacles, and proximity clusters are
display-capped and only the cluster block states its true total, so a printed count is not
the scene's true count."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

from _base import run_parser, NotApplicable
from _format import sort_key, undirected_components
from _relations import resolve_labels, classify_predicate, support_forest
from scene_graph.models import Building
from scene_graph.capabilities import has_object_relations


def _root_to_leaf_paths(children: dict[str, list[str]], roots: list[str]) -> list[list[str]]:
    """Every root->leaf path in the support forest."""
    paths: list[list[str]] = []

    def dfs(node: str, acc: list[str]) -> None:
        kids = children.get(node, [])
        if not kids:
            paths.append(acc)
            return
        for k in kids:
            dfs(k, acc + [k])

    for r in roots:
        dfs(r, [r])
    return paths


def object_relations_digest(building: Building) -> list[str]:
    """The derived-structure section without the head line, split out so `synthesis` can fold
    these exact facts into its own document instead of carrying only raw triples."""
    rels = building.object_relations
    _, lbl = resolve_labels(building)
    children, roots, _, edge_pred = support_forest(rels)

    def render_chain(path: list[str]) -> str:
        """Path (root..leaf) rendered leaf-first, so each named hop reads forward."""
        rev = path[::-1]
        s = lbl(rev[0])
        for i in range(len(rev) - 1):
            s += f" ({edge_pred[rev[i]]}) {lbl(rev[i + 1])}"
        return s

    census = Counter(classify_predicate(r.predicate) for r in rels)
    lines: list[str] = []

    # -- Support depth: the multi-hop stacking chains, read leaf -> base --
    paths = _root_to_leaf_paths(children, roots)
    if not children:
        lines.append("No support or containment relations in this scene.")
    else:
        chains = sorted((p for p in paths if len(p) >= 3),
                        key=lambda p: (-len(p), sort_key(p[-1])))
        if chains:
            max_depth = max(len(p) - 1 for p in paths)
            lines.append(f"Deepest support nesting: {max_depth} levels.")
            lines.append("Support chains (deepest first):")
            for p in chains[:8]:
                lines.append(f"  {render_chain(p)}")
        else:
            lines.append("Support: every item sits directly on its base (nothing stacked deeper).")
    lines.append("")

    # -- Receptacles: objects that carry the most others --
    receptacles = sorted(children, key=lambda p: (-len(children[p]), sort_key(p)))
    if receptacles:
        top = ", ".join(
            f"{lbl(p)} ({len(children[p])} object{'s' if len(children[p]) != 1 else ''})"
            for p in receptacles[:5]
        )
        lines.append(f"Main receptacles (carry the most objects): {top}.")
        lines.append("")

    # -- Proximity clusters: mutually close-by groups --
    prox_edges = [(r.subject_id, r.object_id) for r in rels if classify_predicate(r.predicate) == "proximity"]
    if prox_edges:
        clusters = [sorted(set(m), key=sort_key) for m in undirected_components(prox_edges)]
        clusters.sort(key=lambda g: (-len(g), sort_key(g[0])))
        lines.append(f"Proximity clusters ({len(clusters)} group{'s' if len(clusters) != 1 else ''} of objects near each other):")
        for g in clusters[:8]:
            lines.append(f"  {{{', '.join(lbl(o) for o in g)}}}")
        lines.append("")

    # -- Shared-attribute cliques (same color / material / ...) --
    same_edges: dict[str, list[tuple[str, str]]] = {}
    for r in rels:
        if r.predicate.startswith("same "):
            same_edges.setdefault(r.predicate, []).append((r.subject_id, r.object_id))
    if same_edges:
        lines.append("Shared attributes (each group lists items sharing that property):")
        for pred in sorted(same_edges):
            comps = [sorted(set(m), key=sort_key) for m in undirected_components(same_edges[pred])]
            comps.sort(key=lambda g: (-len(g), sort_key(g[0])))
            for g in comps:
                lines.append(f"  {pred}: {', '.join(lbl(o) for o in g)}")
        lines.append("")

    # -- Census by structural class --
    order = [("support", "support/containment"), ("proximity", "proximity"),
             ("directional", "directional"), ("attribute", "comparative/shared-attribute"),
             ("other", "other")]
    parts = [f"{census[c]} {name}" for c, name in order if census[c]]
    lines.append(f"Relation census: {', '.join(parts)}.")

    return lines


def parse(building: Building) -> str:
    if not has_object_relations(building):
        raise NotApplicable("relations_digest needs annotated object relations; this scene has none")

    rels = building.object_relations
    head = (
        f"{building.name} -- object-relation digest (derived from {len(rels)} relations; "
        "facts pre-computed, not the raw edge list)."
    )
    lines = [head, ""]
    lines.extend(object_relations_digest(building))

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    run_parser(parse, "Serialize derived consequences of the object-relation graph")
