from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from pso.core.types import BoxBounds, IterationMetrics
from pso.viz.recorder import SnapshotFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_contour_grid(objective, bounds: BoxBounds, resolution: int):
    """Builds the contour grid for d=2 (expensive, done once per animation)."""
    x = np.linspace(bounds.lower[0], bounds.upper[0], resolution)
    y = np.linspace(bounds.lower[1], bounds.upper[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: float(objective(np.array([xi, yi]))))(X, Y)
    return X, Y, Z


# ---------------------------------------------------------------------------
# 2D animation
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

    # Left panel: objective function contour
    ax_swarm.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax_swarm.contour(X, Y, Z, levels=10, colors="white", linewidths=0.4, alpha=0.5)
    ax_swarm.set_xlim(bounds.lower[0], bounds.upper[0])
    ax_swarm.set_ylim(bounds.lower[1], bounds.upper[1])
    ax_swarm.set_xlabel("x₀")
    ax_swarm.set_ylabel("x₁")

    scatter_particles = ax_swarm.scatter([], [], c="white", s=20, zorder=3, label="Particles")
    scatter_best = ax_swarm.scatter([], [], c="red", s=120, marker="*", zorder=4, label="Global best")
    title_swarm = ax_swarm.set_title("")
    ax_swarm.legend(fontsize=8, loc="upper right")

    # Right panel: convergence curve
    use_log = min(fitnesses) > 0
    line_conv, = ax_conv.plot([], [], color="steelblue", linewidth=1.5)
    ax_conv.set_xlim(iterations_all[0], iterations_all[-1])
    ymin = min(fitnesses) * 0.5 if min(fitnesses) > 0 else min(fitnesses) - abs(min(fitnesses)) * 0.1
    ymax = max(fitnesses) * 1.5
    ax_conv.set_ylim(ymin, ymax)
    if use_log:
        ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best fitness")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, alpha=0.3)
    # Vertical line tracking the current iteration on the convergence panel
    vline = ax_conv.axvline(x=iterations_all[0], color="red", linewidth=1, alpha=0.6)

    fig.tight_layout()

    # Map iteration number → index in history (used to draw partial convergence)
    iter_to_hist_idx = {m.iteration: i for i, m in enumerate(history)}

    def update(frame_idx: int):
        frame = frames[frame_idx]
        pos = frame.positions

        scatter_particles.set_offsets(pos[:, :2])
        scatter_best.set_offsets(frame.global_best_position[:2].reshape(1, 2))
        title_swarm.set_text(
            f"Iteration {frame.iteration}   |   best = {frame.global_best_value:.4e}"
        )

        # Draw convergence only up to the current iteration
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
# 3D animation
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

    # Compute axis limits from all recorded positions
    all_pos = np.concatenate([f.positions for f in frames], axis=0)
    lims = [(all_pos[:, d].min(), all_pos[:, d].max()) for d in range(3)]

    ax3d.set_xlabel("x₀")
    ax3d.set_ylabel("x₁")
    ax3d.set_zlabel("x₂")
    for d, (lo, hi) in enumerate(lims):
        pad = (hi - lo) * 0.05
        [ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim][d](lo - pad, hi + pad)

    scatter3d = ax3d.scatter([], [], [], c="steelblue", s=20, alpha=0.7, label="Particles")
    best3d = ax3d.scatter([], [], [], c="red", s=120, marker="*", zorder=5, label="Global best")
    title3d = ax3d.set_title("")
    ax3d.legend(fontsize=8)

    # Convergence panel
    use_log = min(fitnesses) > 0
    line_conv, = ax_conv.plot([], [], color="steelblue", linewidth=1.5)
    ax_conv.set_xlim(iterations_all[0], iterations_all[-1])
    ymin = min(fitnesses) * 0.5 if min(fitnesses) > 0 else min(fitnesses) - abs(min(fitnesses)) * 0.1
    ymax = max(fitnesses) * 1.5
    ax_conv.set_ylim(ymin, ymax)
    if use_log:
        ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best fitness")
    ax_conv.set_title("Convergence")
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
            f"Iteration {frame.iteration}   |   best = {frame.global_best_value:.4e}"
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
# Convergence only (d > 3)
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
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title("PSO Convergence")
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
        raise ValueError(f"Unsupported format: '{suffix}'. Use .gif or .mp4")

    anim.save(str(output_path), writer=writer)


# ---------------------------------------------------------------------------
# Public API
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
    """Generates a swarm evolution animation and saves it to disk.

    Args:
        frames: list of SnapshotFrames from SwarmRecorder.
        history: OptimizationResult.history with per-iteration metrics.
        dimension: search space dimensionality.
        output_path: output file path (.gif or .mp4).
        objective: objective function callable (required for d=2).
        bounds: problem BoxBounds (required for d=2).
        fps: frames per second of the animation.
        resolution: contour grid resolution for d=2.

    Returns:
        Path to the generated file.
    """
    if not frames:
        raise ValueError("frames is empty; make sure SwarmRecorder was passed to PSO.optimize()")

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    writer_name = "pillow" if suffix == ".gif" else "ffmpeg"

    if dimension == 2:
        if objective is None or bounds is None:
            raise ValueError("objective and bounds are required to animate d=2")
        _animate_2d(frames, history, objective, bounds, output_path, fps, resolution, writer_name)
    elif dimension == 3:
        _animate_3d(frames, history, output_path, fps, writer_name)
    else:
        # For d > 3 only the convergence curve can be animated meaningfully
        _animate_convergence_only(frames, history, output_path, fps, writer_name)

    return output_path
