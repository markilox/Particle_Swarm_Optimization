"""Plots estáticos: curva de convergencia y snapshot 2D del enjambre."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes

from pso.core.types import IterationMetrics, BoxBounds
from pso.viz.recorder import SnapshotFrame


def plot_convergence(
    history: list[IterationMetrics],
    ax: Axes | None = None,
    title: str = "Convergencia",
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Curva de mejor fitness por iteración.

    Args:
        history: lista de IterationMetrics de OptimizationResult.history.
        ax: Axes existente; si es None se crea una figura nueva.
        title: título del gráfico.
        log_scale: usar escala logarítmica en el eje Y.

    Returns:
        Figure que contiene el plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    iterations = [m.iteration for m in history]
    fitnesses = [m.best_fitness for m in history]

    ax.plot(iterations, fitnesses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mejor fitness")
    ax.set_title(title)
    if log_scale and min(fitnesses) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_swarm_2d(
    frame: SnapshotFrame,
    objective,
    bounds: BoxBounds,
    ax: Axes | None = None,
    resolution: int = 200,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Snapshot estático del enjambre 2D sobre el contorno de la función objetivo.

    Args:
        frame: SnapshotFrame con posiciones del enjambre.
        objective: callable (array 1D → float) para evaluar la función.
        bounds: BoxBounds con límites de las dos primeras dimensiones.
        ax: Axes existente; si es None se crea una figura nueva.
        resolution: resolución de la cuadrícula del contorno (resolution × resolution).
        title: título del gráfico; si es None usa "Iteración {n}".
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    x = np.linspace(bounds.lower[0], bounds.upper[0], resolution)
    y = np.linspace(bounds.lower[1], bounds.upper[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [objective(np.array([X[i, j], Y[i, j]])) for j in range(resolution)]
        for i in range(resolution)
    ])

    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.4, alpha=0.5)

    positions = frame.positions
    ax.scatter(positions[:, 0], positions[:, 1], c="white", s=20, zorder=3, label="Partículas")
    ax.scatter(
        frame.global_best_position[0],
        frame.global_best_position[1],
        c="red", s=100, marker="*", zorder=4, label=f"Mejor ({frame.global_best_value:.4e})",
    )

    ax.set_xlim(bounds.lower[0], bounds.upper[0])
    ax.set_ylim(bounds.lower[1], bounds.upper[1])
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_title(title or f"Iteración {frame.iteration}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
