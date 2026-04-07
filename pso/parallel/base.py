from abc import ABC, abstractmethod


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, swarm, objective) -> None:
        """Evaluate swarm particles and update per-particle best values."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the evaluator (e.g. worker pools).

        No-op for evaluators that do not hold resources.
        Called automatically by run_experiment() after optimize() completes.
        """

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
