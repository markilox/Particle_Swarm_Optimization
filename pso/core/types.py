# Central data structures for the PSO package.
# Keeping all types here avoids circular imports: BoxBounds is needed by
# both bounds.py and particle.py, and cannot live in either without creating
# a cross-dependency. All other modules import from here;
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StopCriteria:
    # The three criteria are checked independently; the first one met stops the run.
    # stagnation_iterations counts consecutive iterations with improvement < min_delta.
    max_iterations: int
    tolerance: float | None = None            # stop if best_fitness <= tolerance
    stagnation_iterations: int | None = None  # stop if no improvement for N iterations
    min_delta: float = 0.0                    # minimum improvement to reset the stagnation counter


class BoxBounds: # a == b  # → [True, True]  no True/False
    # Both arrays must have shape (dimension,). Validated in PSO._validate_config.
    # Kept as a regular class to avoid __eq__ issues with numpy arrays.
    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.lower = lower
        self.upper = upper


@dataclass
class PSOConfig:
    # w=0.7, c1=c2=1.5 are the standard default values from PSO literature.
    # If stop is None, 200 iterations are used with no other stopping criterion.
    dimension: int
    swarm_size: int = 30
    inertia_weight: float = 0.7    # w: controls exploration vs exploitation
    cognitive_weight: float = 1.5  # c1: attraction toward the particle's own best
    social_weight: float = 1.5     # c2: attraction toward the global best of the swarm
    seed: int | None = None        # None means non-reproducible run
    stop: StopCriteria = field(default_factory=lambda: StopCriteria(max_iterations=200))


@dataclass
class IterationMetrics:
    # One instance is added to OptimizationResult.history per iteration.
    # eval_time_s and update_time_s are measured separately to analyze parallelism overhead.
    iteration: int
    best_fitness: float
    eval_time_s: float
    update_time_s: float


@dataclass
class OptimizationResult:
    # converged_iteration is None if the run reached max_iterations without triggering
    # the tolerance or stagnation criteria.
    best_position: np.ndarray
    best_value: float
    history: list[IterationMetrics]  # one entry per iteration, used for convergence plots
    total_time_s: float
    total_eval_time_s: float
    total_update_time_s: float
    converged_iteration: int | None
