from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pso.core.types import BoxBounds


class Particle:
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        best_position: np.ndarray,
        best_value: float,
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.best_position = best_position
        self.best_value = best_value

    def update_velocity(
        self,
        rng: np.random.Generator,
        social_best: np.ndarray,
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
    ) -> None:
        r1 = rng.random(self.position.shape[0])
        r2 = rng.random(self.position.shape[0])
        cognitive = cognitive_weight * r1 * (self.best_position - self.position)
        social = social_weight * r2 * (social_best - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive + social

    def update_position(
        self,
        bounds: "BoxBounds",
        bounds_policy,
    ) -> None:
        new_position = self.position + self.velocity
        self.position, self.velocity = bounds_policy(new_position, self.velocity, bounds)

    def evaluate(self, objective) -> float:
        value = float(objective(self.position))
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()
        return value
