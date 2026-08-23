from scene_graph.loaders.base import DatasetLoader
from scene_graph.loaders.gibson import GibsonLoader
from scene_graph.loaders.procthor import ProcTHORLoader
from scene_graph.loaders.threerscan import ThreeRScanLoader
from scene_graph.models import Building

REGISTRY: dict[str, type[DatasetLoader]] = {
    "gibson": GibsonLoader,
    "procthor": ProcTHORLoader,
    "3rscan": ThreeRScanLoader,
}


def load(dataset: str, scene_id: str, data_path: str) -> Building:
    cls = REGISTRY.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(REGISTRY)}")
    return cls().load(scene_id, data_path)


__all__ = ["DatasetLoader", "REGISTRY", "load"]
