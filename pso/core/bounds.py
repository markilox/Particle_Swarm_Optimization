import numpy as np

from pso.core.types import BoxBounds


class ClampBoundsPolicy:
    """Clamp strategy for box constraints: clip positions inside [lower, upper]."""

    @staticmethod
    def apply(position: np.ndarray, velocity: np.ndarray, bounds: BoxBounds) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(position, bounds.lower, bounds.upper)
        velocity = velocity.copy()
        hit_mask = clipped != position
        velocity[hit_mask] = 0.0
        return clipped, velocity
