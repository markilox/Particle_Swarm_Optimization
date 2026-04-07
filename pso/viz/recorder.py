"""Swarm trajectory recorder for visualization.

SwarmRecorder is passed as an on_iteration callback to PSO.optimize().
Captures one snapshot per iteration with minimal overhead: only copies arrays.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotFrame:
    """Swarm state at a single iteration."""
    iteration: int
    positions: np.ndarray              # shape (swarm_size, dimension)
    global_best_position: np.ndarray   # shape (dimension,)
    global_best_value: float


class SwarmRecorder:
    """Accumulates SnapshotFrames on each call.

    Usage:
        recorder = SwarmRecorder(every_n=1)
        result = pso.optimize(on_iteration=recorder)
        frames = recorder.frames
    """

    def __init__(self, every_n: int = 1) -> None:
        """
        Args:
            every_n: save a snapshot only every N iterations.
                     every_n=1 saves all (default).
                     every_n=5 reduces frame count for long animations.
        """
        self.every_n = every_n
        self.frames: list[SnapshotFrame] = []

    def __call__(
        self,
        iteration: int,
        positions: np.ndarray,
        global_best_position: np.ndarray,
        global_best_value: float,
    ) -> None:
        # Only record this frame if it falls on the configured interval
        if iteration % self.every_n == 0:
            self.frames.append(
                SnapshotFrame(
                    iteration=iteration,
                    positions=positions.copy(),               # copy to avoid mutation across iterations
                    global_best_position=global_best_position.copy(),
                    global_best_value=global_best_value,
                )
            )

    def __len__(self) -> int:
        return len(self.frames)
