
from concurrent.futures import ProcessPoolExecutor

from pso.parallel.base import FitnessEvaluator


def _evaluate_single(args: tuple) -> float:
    """Worker: receives (position, objective) and returns the fitness value.

    Must be a module-level function (not a class method) to be picklable.
    The main process applies personal best updates once all results are received.
    """
    position, objective = args
    return float(objective(position))


class MultiprocessingEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int | None = None, chunksize: int = 1) -> None:
        # max_workers=None lets ProcessPoolExecutor choose based on available cores
        # The pool is created once here and reused across all iterations,
        # avoiding the overhead of spawning/destroying workers every iteration.
        self.chunksize = chunksize
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    def evaluate(self, swarm, objective) -> None:
        particles = swarm.particles

        # build argument list: (position, objective) per particle
        args = [(p.position, objective) for p in particles]

        # executor.map distributes args across workers and collects results in order
        # chunksize groups iterable elements to reduce IPC overhead
        values = list(self._executor.map(_evaluate_single, args, chunksize=self.chunksize))

        # update personal best in the main process
        # (workers operate on copies; changes do not propagate back automatically)
        for particle, value in zip(particles, values):
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()

    def close(self) -> None:
        """Shut down the worker pool. Called automatically by run_experiment()."""
        self._executor.shutdown(wait=True)
