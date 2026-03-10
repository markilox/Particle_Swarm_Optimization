from abc import ABC, abstractmethod
from typing import Any


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, swarm: Any, objective) -> None:
        """Evaluate swarm particles and update per-particle best values."""
        raise NotImplementedError
