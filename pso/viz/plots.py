"""Static plots: convergence curve and 2D swarm snapshot."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes

from pso.core.types import IterationMetrics, BoxBounds
from pso.viz.recorder import SnapshotFrame


def plot_convergence(
    history: list[IterationMetrics],
    ax: Axes | None = None,
    title: str = "Convergence",
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Best fitness curve per iteration.

    Args:
        history: list of IterationMetrics from OptimizationResult.history.
        ax: existing Axes; if None a new figure is created.
        title: plot title.
        log_scale: use logarithmic scale on the Y axis.

    Returns:
        Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    iterations = [m.iteration for m in history]
    fitnesses = [m.best_fitness for m in history]

    ax.plot(iterations, fitnesses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title(title)
    # Only apply log scale when all values are positive
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
    """Static 2D swarm snapshot overlaid on the objective function contour.

    Args:
        frame: SnapshotFrame with swarm positions.
        objective: callable (1D array → float) to evaluate the function.
        bounds: BoxBounds with limits for the first two dimensions.
        ax: existing Axes; if None a new figure is created.
        resolution: contour grid resolution (resolution × resolution).
        title: plot title; defaults to "Iteration {n}" if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Build the contour grid by evaluating the objective on a meshgrid
    x = np.linspace(bounds.lower[0], bounds.upper[0], resolution)
    y = np.linspace(bounds.lower[1], bounds.upper[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [objective(np.array([X[i, j], Y[i, j]])) for j in range(resolution)]
        for i in range(resolution)
    ])

    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.4, alpha=0.5)

    # Plot all particles and highlight the global best
    positions = frame.positions
    ax.scatter(positions[:, 0], positions[:, 1], c="white", s=20, zorder=3, label="Particles")
    ax.scatter(
        frame.global_best_position[0],
        frame.global_best_position[1],
        c="red", s=100, marker="*", zorder=4, label=f"Best ({frame.global_best_value:.4e})",
    )

    ax.set_xlim(bounds.lower[0], bounds.upper[0])
    ax.set_ylim(bounds.lower[1], bounds.upper[1])
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_title(title or f"Iteration {frame.iteration}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
