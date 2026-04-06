
from concurrent.futures import ThreadPoolExecutor

from pso.parallel.base import FitnessEvaluator


class ThreadingEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int | None = None) -> None:
        # None lets ThreadPoolExecutor choose based on CPU count
        self.max_workers = max_workers

    def evaluate(self, swarm, objective) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # distribute each particle across threads and wait for all to finish
            list(executor.map(lambda p: p.evaluate(objective), swarm.particles))
