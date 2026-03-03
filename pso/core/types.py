import numpy as np


class StopCriteria:
    def __init__(
        self,
        max_iterations: int,
        tolerance: float | None = None,
        stagnation_iterations: int | None = None,
        min_delta: float = 0.0,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.stagnation_iterations = stagnation_iterations
        self.min_delta = min_delta


class BoxBounds:
    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = lower
        self.upper = upper


class PSOConfig:
    def __init__(
        self,
        dimension: int,
        swarm_size: int = 30,
        inertia_weight: float = 0.7,
        cognitive_weight: float = 1.5,
        social_weight: float = 1.5,
        stop: StopCriteria | None = None,
        seed: int | None = None,
    ) -> None:
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.stop = stop if stop is not None else StopCriteria(max_iterations=200)
        self.seed = seed


class IterationMetrics:
    def __init__(
        self,
        iteration: int,
        best_fitness: float,
        eval_time_s: float,
        update_time_s: float,
    ) -> None:
        self.iteration = iteration
        self.best_fitness = best_fitness
        self.eval_time_s = eval_time_s
        self.update_time_s = update_time_s


class OptimizationResult:
    def __init__(
        self,
        best_position: np.ndarray,
        best_value: float,
        history: list[IterationMetrics],
        total_time_s: float,
        total_eval_time_s: float,
        total_update_time_s: float,
        converged_iteration: int | None,
    ) -> None:
        self.best_position = best_position
        self.best_value = best_value
        self.history = history
        self.total_time_s = total_time_s
        self.total_eval_time_s = total_eval_time_s
        self.total_update_time_s = total_update_time_s
        self.converged_iteration = converged_iteration
