"""base.py — Abstract base for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text:              str
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        float


class LLMProvider(ABC):
    def __init__(self, model: str, options: dict) -> None:
        self.model = model
        self.options = options

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        ...

    def unload(self) -> None:
        """Release any held resources (e.g. free GPU VRAM). Default no-op.

        Local backends override this so the two-phase runner can evict the
        responder before the judge loads, avoiding co-residence on small GPUs.
        """
        return None
