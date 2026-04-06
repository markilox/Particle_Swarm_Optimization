"""Grabador de trayectorias del enjambre para visualización.

SwarmRecorder se pasa como callback on_iteration a PSO.optimize().
Captura un snapshot por iteración con coste mínimo: solo copia arrays.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotFrame:
    """Estado del enjambre en una iteración."""
    iteration: int
    positions: np.ndarray        # shape (swarm_size, dimension)
    global_best_position: np.ndarray  # shape (dimension,)
    global_best_value: float


class SwarmRecorder:
    """Acumula SnapshotFrame en cada llamada.

    Uso:
        recorder = SwarmRecorder(every_n=1)
        result = pso.optimize(on_iteration=recorder)
        frames = recorder.frames
    """

    def __init__(self, every_n: int = 1) -> None:
        """
        Args:
            every_n: guardar snapshot solo cada N iteraciones.
                     every_n=1 guarda todas (por defecto).
                     every_n=5 reduce el número de frames en animaciones largas.
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
        if iteration % self.every_n == 0:
            self.frames.append(
                SnapshotFrame(
                    iteration=iteration,
                    positions=positions.copy(),
                    global_best_position=global_best_position.copy(),
                    global_best_value=global_best_value,
                )
            )

    def __len__(self) -> int:
        return len(self.frames)
