import numpy as np

from pso.core.particle import Particle


class Swarm:
    def __init__(
        self,
        particles: list[Particle],
        global_best_position: np.ndarray,
        global_best_value: float,
    ) -> None:
        self.particles = particles
        self.global_best_position = global_best_position
        self.global_best_value = global_best_value

    def update_global_best(self) -> None:
        for particle in self.particles:
            if particle.best_value < self.global_best_value:
                self.global_best_value = particle.best_value
                self.global_best_position = particle.best_position.copy()
