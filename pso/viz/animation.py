"""Generación de animaciones de la evolución del enjambre.

Soporta dimensión 2 (contorno + enjambre + convergencia) y
dimensión 3 (scatter 3D + convergencia). Para d > 3 genera solo
la curva de convergencia.

Formato de salida:
  - GIF: requiere Pillow (pip install pillow). Sin dependencias externas.
  - MP4: requiere ffmpeg instalado en el sistema.

La función principal animate_swarm detecta el formato por la extensión del
fichero de salida y elige el writer adecuado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana; debe ir antes de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from pso.core.types import BoxBounds, IterationMetrics
from pso.viz.recorder import SnapshotFrame


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _make_contour_grid(objective, bounds: BoxBounds, resolution: int):
    """Construye la cuadrícula de contorno para d=2 (caro, se hace una vez)."""
    x = np.linspace(bounds.lower[0], bounds.upper[0], resolution)
    y = np.linspace(bounds.lower[1], bounds.upper[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: float(objective(np.array([xi, yi]))))(X, Y)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Animación 2D
# ---------------------------------------------------------------------------

def _animate_2d(
    frames: list[SnapshotFrame],
    history: list[IterationMetrics],
    objective,
    bounds: BoxBounds,
    output_path: Path,
    fps: int,
    resolution: int,
    writer_name: str,
) -> None:
    X, Y, Z = _make_contour_grid(objective, bounds, resolution)
    fitnesses = [m.best_fitness for m in history]
    iterations_all = [m.iteration for m in history]

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
    ax_swarm = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])

    # --- panel izquierdo: contorno ---
    ax_swarm.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax_swarm.contour(X, Y, Z, levels=10, colors="white", linewidths=0.4, alpha=0.5)
    ax_swarm.set_xlim(bounds.lower[0], bounds.upper[0])
    ax_swarm.set_ylim(bounds.lower[1], bounds.upper[1])
    ax_swarm.set_xlabel("x₀")
    ax_swarm.set_ylabel("x₁")

    scatter_particles = ax_swarm.scatter([], [], c="white", s=20, zorder=3, label="Partículas")
    scatter_best = ax_swarm.scatter([], [], c="red", s=120, marker="*", zorder=4, label="Mejor global")
    title_swarm = ax_swarm.set_title("")
    ax_swarm.legend(fontsize=8, loc="upper right")

    # --- panel derecho: convergencia ---
    use_log = min(fitnesses) > 0
    line_conv, = ax_conv.plot([], [], color="steelblue", linewidth=1.5)
    ax_conv.set_xlim(iterations_all[0], iterations_all[-1])
    ymin = min(fitnesses) * 0.5 if min(fitnesses) > 0 else min(fitnesses) - abs(min(fitnesses)) * 0.1
    ymax = max(fitnesses) * 1.5
    ax_conv.set_ylim(ymin, ymax)
    if use_log:
        ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iteración")
    ax_conv.set_ylabel("Mejor fitness")
    ax_conv.set_title("Convergencia")
    ax_conv.grid(True, alpha=0.3)
    vline = ax_conv.axvline(x=iterations_all[0], color="red", linewidth=1, alpha=0.6)

    fig.tight_layout()

    # mapa: iteration → índice en history (para la línea de convergencia)
    iter_to_hist_idx = {m.iteration: i for i, m in enumerate(history)}

    def update(frame_idx: int):
        frame = frames[frame_idx]
        pos = frame.positions

        scatter_particles.set_offsets(pos[:, :2])
        scatter_best.set_offsets(frame.global_best_position[:2].reshape(1, 2))
        title_swarm.set_text(
            f"Iteración {frame.iteration}   |   mejor = {frame.global_best_value:.4e}"
        )

        # convergencia hasta esta iteración
        hist_idx = iter_to_hist_idx.get(frame.iteration, len(history) - 1)
        line_conv.set_data(
            iterations_all[: hist_idx + 1],
            fitnesses[: hist_idx + 1],
        )
        vline.set_xdata([frame.iteration, frame.iteration])
        return scatter_particles, scatter_best, title_swarm, line_conv, vline

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )

    _save_animation(anim, output_path, fps, writer_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animación 3D
# ---------------------------------------------------------------------------

def _animate_3d(
    frames: list[SnapshotFrame],
    history: list[IterationMetrics],
    output_path: Path,
    fps: int,
    writer_name: str,
) -> None:
    fitnesses = [m.best_fitness for m in history]
    iterations_all = [m.iteration for m in history]

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_conv = fig.add_subplot(gs[1])

    # límites del espacio 3D
    all_pos = np.concatenate([f.positions for f in frames], axis=0)
    lims = [(all_pos[:, d].min(), all_pos[:, d].max()) for d in range(3)]

    ax3d.set_xlabel("x₀")
    ax3d.set_ylabel("x₁")
    ax3d.set_zlabel("x₂")
    for d, (lo, hi) in enumerate(lims):
        pad = (hi - lo) * 0.05
        [ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim][d](lo - pad, hi + pad)

    scatter3d = ax3d.scatter([], [], [], c="steelblue", s=20, alpha=0.7, label="Partículas")
    best3d = ax3d.scatter([], [], [], c="red", s=120, marker="*", zorder=5, label="Mejor global")
    title3d = ax3d.set_title("")
    ax3d.legend(fontsize=8)

    # convergencia
    use_log = min(fitnesses) > 0
    line_conv, = ax_conv.plot([], [], color="steelblue", linewidth=1.5)
    ax_conv.set_xlim(iterations_all[0], iterations_all[-1])
    ymin = min(fitnesses) * 0.5 if min(fitnesses) > 0 else min(fitnesses) - abs(min(fitnesses)) * 0.1
    ymax = max(fitnesses) * 1.5
    ax_conv.set_ylim(ymin, ymax)
    if use_log:
        ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iteración")
    ax_conv.set_ylabel("Mejor fitness")
    ax_conv.set_title("Convergencia")
    ax_conv.grid(True, alpha=0.3)
    vline = ax_conv.axvline(x=iterations_all[0], color="red", linewidth=1, alpha=0.6)

    fig.tight_layout()

    iter_to_hist_idx = {m.iteration: i for i, m in enumerate(history)}

    def update(frame_idx: int):
        frame = frames[frame_idx]
        pos = frame.positions

        scatter3d._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        bp = frame.global_best_position
        best3d._offsets3d = ([bp[0]], [bp[1]], [bp[2]])
        title3d.set_text(
            f"Iteración {frame.iteration}   |   mejor = {frame.global_best_value:.4e}"
        )

        hist_idx = iter_to_hist_idx.get(frame.iteration, len(history) - 1)
        line_conv.set_data(
            iterations_all[: hist_idx + 1],
            fitnesses[: hist_idx + 1],
        )
        vline.set_xdata([frame.iteration, frame.iteration])
        return scatter3d, best3d, line_conv, vline

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )

    _save_animation(anim, output_path, fps, writer_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convergencia sola (d > 3)
# ---------------------------------------------------------------------------

def _animate_convergence_only(
    frames: list[SnapshotFrame],
    history: list[IterationMetrics],
    output_path: Path,
    fps: int,
    writer_name: str,
) -> None:
    fitnesses = [m.best_fitness for m in history]
    iterations_all = [m.iteration for m in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    use_log = min(fitnesses) > 0
    line_conv, = ax.plot([], [], color="steelblue", linewidth=1.5)
    ax.set_xlim(iterations_all[0], iterations_all[-1])
    ymin = min(fitnesses) * 0.5 if min(fitnesses) > 0 else min(fitnesses) - abs(min(fitnesses)) * 0.1
    ymax = max(fitnesses) * 1.5
    ax.set_ylim(ymin, ymax)
    if use_log:
        ax.set_yscale("log")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mejor fitness")
    ax.set_title("Convergencia PSO")
    ax.grid(True, alpha=0.3)

    iter_to_hist_idx = {m.iteration: i for i, m in enumerate(history)}

    def update(frame_idx: int):
        frame = frames[frame_idx]
        hist_idx = iter_to_hist_idx.get(frame.iteration, len(history) - 1)
        line_conv.set_data(
            iterations_all[: hist_idx + 1],
            fitnesses[: hist_idx + 1],
        )
        return (line_conv,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )

    _save_animation(anim, output_path, fps, writer_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writer helper
# ---------------------------------------------------------------------------

def _save_animation(
    anim: animation.FuncAnimation,
    output_path: Path,
    fps: int,
    writer_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    else:
        raise ValueError(f"Formato no soportado: '{suffix}'. Usa .gif o .mp4")

    anim.save(str(output_path), writer=writer)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def animate_swarm(
    frames: list[SnapshotFrame],
    history: list[IterationMetrics],
    dimension: int,
    output_path: str | Path,
    objective=None,
    bounds: BoxBounds | None = None,
    fps: int = 10,
    resolution: int = 150,
) -> Path:
    """Genera una animación de la evolución del enjambre y la guarda en disco.

    Args:
        frames: lista de SnapshotFrame del SwarmRecorder.
        history: OptimizationResult.history con métricas por iteración.
        dimension: dimensión del espacio de búsqueda.
        output_path: ruta de salida (.gif o .mp4).
        objective: callable de la función objetivo (requerido para d=2).
        bounds: BoxBounds del problema (requerido para d=2).
        fps: fotogramas por segundo de la animación.
        resolution: resolución de la cuadrícula de contorno para d=2.

    Returns:
        Path al fichero generado.
    """
    if not frames:
        raise ValueError("frames está vacío; comprueba que SwarmRecorder se pasó a PSO.optimize()")

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    writer_name = "pillow" if suffix == ".gif" else "ffmpeg"

    if dimension == 2:
        if objective is None or bounds is None:
            raise ValueError("objective y bounds son necesarios para animar d=2")
        _animate_2d(frames, history, objective, bounds, output_path, fps, resolution, writer_name)
    elif dimension == 3:
        _animate_3d(frames, history, output_path, fps, writer_name)
    else:
        _animate_convergence_only(frames, history, output_path, fps, writer_name)

    return output_path
