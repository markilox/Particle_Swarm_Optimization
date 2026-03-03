import numpy as np

from pso.core.swarm import Swarm


class GlobalBestTopology:
    @staticmethod
    def social_best(swarm: Swarm, _: int) -> np.ndarray:
        return swarm.global_best_position
