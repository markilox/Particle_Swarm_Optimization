from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pso.core.swarm import Swarm


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, swarm: "Swarm", objective) -> None:
        """Evaluate swarm particles and update per-particle best values."""
        raise NotImplementedError
