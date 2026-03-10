from typing import Any

from parallel.base import FitnessEvaluator


class SequentialEvaluator(FitnessEvaluator):
    def evaluate(self, swarm: Any, objective) -> None:
        for particle in swarm.particles:
            particle.evaluate(objective)
