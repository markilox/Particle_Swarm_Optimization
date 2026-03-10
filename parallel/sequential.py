from typing import TYPE_CHECKING

from parallel.base import FitnessEvaluator

if TYPE_CHECKING:
    from pso.core.swarm import Swarm


class SequentialEvaluator(FitnessEvaluator):
    def evaluate(self, swarm: "Swarm", objective) -> None:
        for particle in swarm.particles:
            particle.evaluate(objective)
