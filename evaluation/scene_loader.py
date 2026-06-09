"""scene_loader.py — Load scene context text for a given scene and representation."""

from __future__ import annotations

from pathlib import Path


def list_representations(scene_contexts_dir: Path, scene_id: str) -> list[str]:
    """Return all available representation names for a scene (stems of files in its directory)."""
    scene_dir = Path(scene_contexts_dir) / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    return sorted(p.stem for p in scene_dir.iterdir() if p.is_file())


def load(scene_contexts_dir: Path, scene_id: str, representation: str) -> str:
    """Return the scene context string for scene_id / representation.

    A representation is either a single file stem, or a multi-view combination
    written as ``"a+b"`` (e.g. ``"topology+metric_relations"``). For a combination
    each part is loaded and the parts are concatenated under per-view headers so
    the responder can tell the views apart. Combinations are opt-in via
    ``EvalConfig.representations``; auto-discovery only lists single files.

    Looks for scene_contexts_dir/<scene_id>/<part>.<any extension> per part.
    Raises FileNotFoundError if a part has no matching file.
    """
    if "+" in representation:
        parts = representation.split("+")
        blocks = []
        for part in parts:
            text = load(scene_contexts_dir, scene_id, part).strip()
            blocks.append(f"=== {part.upper()} ===\n{text}")
        return "\n\n".join(blocks) + "\n"

    scene_dir = Path(scene_contexts_dir) / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    matches = list(scene_dir.glob(f"{representation}.*"))
    if not matches:
        raise FileNotFoundError(
            f"No file for representation '{representation}' in {scene_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous representation '{representation}' in {scene_dir}: {matches}"
        )

    return matches[0].read_text(encoding="utf-8")
