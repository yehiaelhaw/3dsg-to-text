from utils.loaders.base import DatasetLoader
from utils.loaders.gibson import GibsonLoader
from utils.loaders.procthor import ProcTHORLoader
from utils.loaders.threerscan import ThreeRScanLoader
from utils.models import Building

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
