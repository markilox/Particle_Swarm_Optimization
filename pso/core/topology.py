import numpy as np

from pso.core.swarm import Swarm


class GlobalBestTopology:
    def social_best(self, swarm: Swarm, particle_index: int) -> np.ndarray:
        return swarm.global_best_position
