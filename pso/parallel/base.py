from abc import ABC, abstractmethod


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, swarm, objective) -> None:
        """Evaluate swarm particles and update per-particle best values."""
        raise NotImplementedError
