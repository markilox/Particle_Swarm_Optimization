
from concurrent.futures import ThreadPoolExecutor

from pso.parallel.base import FitnessEvaluator


class ThreadingEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int | None = None) -> None:
        # The pool is created once here and reused across all iterations,
        # avoiding thread creation/destruction overhead every iteration.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def evaluate(self, swarm, objective) -> None:
        # distribute each particle across threads and wait for all to finish
        list(self._executor.map(lambda p: p.evaluate(objective), swarm.particles))

    def close(self) -> None:
        """Shut down the thread pool. Called automatically by run_experiment()."""
        self._executor.shutdown(wait=True)
