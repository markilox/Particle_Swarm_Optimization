import json
import logging
import time

import numpy as np

from pso.core.bounds import ClampBoundsPolicy
from pso.core.particle import Particle
from pso.core.swarm import Swarm
from pso.core.topology import GlobalBestTopology
from pso.core.types import BoxBounds, IterationMetrics, OptimizationResult, PSOConfig, StopCriteria


class PSO:
    def __init__(
        self,
        objective,
        bounds: BoxBounds,
        config: PSOConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.objective = objective
        self.bounds = bounds
        self.config = config
        self.logger = logger or logging.getLogger("pso")
        self.rng = np.random.default_rng(config.seed)

        self._validate_config()

    def optimize(self) -> OptimizationResult:
        swarm = self._initialize_swarm()
        history: list[IterationMetrics] = []
        total_eval_time_s = 0.0
        total_update_time_s = 0.0
        start_total = time.perf_counter()

        converged_iteration: int | None = None
        no_improve_count = 0
        prev_best = swarm.global_best_value

        for iteration in range(1, self.config.stop.max_iterations + 1):
            iter_eval_time = 0.0
            iter_update_time = 0.0

            t0_update = time.perf_counter()
            self._update_particles(swarm)
            iter_update_time = time.perf_counter() - t0_update
            total_update_time_s += iter_update_time

            t0_eval = time.perf_counter()
            self._evaluate_particles(swarm)
            iter_eval_time = time.perf_counter() - t0_eval
            total_eval_time_s += iter_eval_time

            history.append(
                IterationMetrics(
                    iteration=iteration,
                    best_fitness=swarm.global_best_value,
                    eval_time_s=iter_eval_time,
                    update_time_s=iter_update_time,
                )
            )
            self._log_iteration(iteration, swarm.global_best_value, iter_eval_time, iter_update_time)

            improvement = prev_best - swarm.global_best_value
            if improvement <= self.config.stop.min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = swarm.global_best_value

            if self._should_stop(swarm.global_best_value, no_improve_count):
                converged_iteration = iteration
                break

        total_time_s = time.perf_counter() - start_total
        return OptimizationResult(
            best_position=swarm.global_best_position.copy(),
            best_value=swarm.global_best_value,
            history=history,
            total_time_s=total_time_s,
            total_eval_time_s=total_eval_time_s,
            total_update_time_s=total_update_time_s,
            converged_iteration=converged_iteration,
        )

    def _initialize_swarm(self) -> Swarm:
        particles: list[Particle] = []
        global_best_value = float("inf")
        global_best_position: np.ndarray | None = None

        span = self.bounds.upper - self.bounds.lower
        for _ in range(self.config.swarm_size):
            position = self.rng.uniform(self.bounds.lower, self.bounds.upper)
            velocity = self.rng.uniform(-span, span) * 0.1
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_value=float("inf"),
            )
            value = particle.evaluate(self.objective)
            particles.append(particle)

            if value < global_best_value:
                global_best_value = value
                global_best_position = position.copy()

        if global_best_position is None:
            raise RuntimeError("Swarm initialization failed.")

        return Swarm(
            particles=particles,
            global_best_position=global_best_position,
            global_best_value=global_best_value,
        )

    def _update_particles(self, swarm: Swarm) -> None:
        for i, particle in enumerate(swarm.particles):
            social_best = GlobalBestTopology.social_best(swarm, i)
            particle.update_velocity(
                rng=self.rng,
                social_best=social_best,
                inertia_weight=self.config.inertia_weight,
                cognitive_weight=self.config.cognitive_weight,
                social_weight=self.config.social_weight,
            )
            particle.update_position(bounds=self.bounds, bounds_policy=ClampBoundsPolicy.apply)

    def _evaluate_particles(self, swarm: Swarm) -> None:
        for particle in swarm.particles:
            particle.evaluate(self.objective)
        swarm.update_global_best()

    def _validate_config(self) -> None:
        if self.config.dimension <= 0:
            raise ValueError("dimension must be > 0")
        if self.config.swarm_size <= 0:
            raise ValueError("swarm_size must be > 0")
        if self.config.stop.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.bounds.lower.shape != (self.config.dimension,) or self.bounds.upper.shape != (self.config.dimension,):
            raise ValueError("bounds must match dimension")
        if np.any(self.bounds.lower >= self.bounds.upper):
            raise ValueError("each lower bound must be < upper bound")

    def _should_stop(self, best_value: float, no_improve_count: int) -> bool:
        tolerance = self.config.stop.tolerance
        stagnation_limit = self.config.stop.stagnation_iterations
        if tolerance is not None and best_value <= tolerance:
            return True
        if stagnation_limit is not None and no_improve_count >= stagnation_limit:
            return True
        return False

    def _log_iteration(self, iteration: int, best_value: float, eval_time_s: float, update_time_s: float) -> None:
        record = {
            "event": "iteration_end",
            "iteration": iteration,
            "best_fitness": best_value,
            "eval_time_s": eval_time_s,
            "update_time_s": update_time_s,
        }
        self.logger.info(json.dumps(record))
