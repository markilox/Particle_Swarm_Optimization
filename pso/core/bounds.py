import numpy as np

from pso.core.types import BoxBounds


class ClampBoundsPolicy:
    """Clamp strategy for box constraints: clip positions inside [lower, upper]."""

    @staticmethod # no needs to access the object, works only with the given parameters
    def apply(position: np.ndarray, velocity: np.ndarray, bounds: BoxBounds) -> tuple[np.ndarray, np.ndarray]:
        new_position = np.clip(position, bounds.lower, bounds.upper)
        velocity = velocity.copy()
        for i in range(len(position)):
            if new_position[i] != position[i]:  # particle hit a boundary in this dimension
                velocity[i] = 0.0               # stop movement in that direction
        return new_position, velocity
