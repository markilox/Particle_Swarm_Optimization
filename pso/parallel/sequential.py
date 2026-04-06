from pso.parallel.base import FitnessEvaluator


class SequentialEvaluator(FitnessEvaluator):
    def evaluate(self, swarm, objective) -> None:
        for particle in swarm.particles:
            particle.evaluate(objective)
