from abc import ABC, abstractmethod
from utils.models import Building


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, scene_id: str, data_path: str) -> Building:
        ...
