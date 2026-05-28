"""color_palette.py — Optimal color palette extraction using PSO.

Finds the N-color palette that minimises the Mean Squared Quantisation
Error (MSQE) in RGB space: for each pixel, its squared distance to the
nearest palette entry is computed, then averaged over all sampled pixels.

Search space
─────────────
  dimension = N × 3   (N colours, each with R, G, B ∈ [0, 255])
  bounds    = [0, 255]^dim   (box constraints, clamped by ClampBoundsPolicy)

Usage
──────
  python color_palette.py photo.jpg
  python color_palette.py photo.jpg --n-colors 16 --swarm-size 60 --seed 0
  python color_palette.py photo.jpg --n-colors 8 --evaluator multiprocessing
  python color_palette.py photo.jpg --n-colors 8 --evaluator asyncio
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pso.core.pso import PSO
from pso.core.types import BoxBounds, PSOConfig, StopCriteria
from pso.parallel import get_evaluator


# ── Objective function ────────────────────────────────────────────────────────

def make_objective(pixels: np.ndarray, n_colors: int):
    """Returns the MSQE objective closure.

    Args:
        pixels:   (M, 3) float64 array of sampled RGB values in [0, 255].
        n_colors: number of palette colours N.

    Returns:
        Callable (x: ndarray[N*3,]) -> float

    The objective clips x to [0, 255] before computing distances, so
    ClampBoundsPolicy and this function are always consistent.

    Complexity: O(M × N) per call — fully vectorised with NumPy so it
    releases the GIL and benefits from BLAS-level parallelism.
    """
    def objective(x: np.ndarray) -> float:
        palette = np.clip(x, 0.0, 255.0).reshape(n_colors, 3)   # (N, 3)
        # diff[m, n, c] = pixel_m_c − palette_n_c
        diff = pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]  # (M, N, 3)
        # squared Euclidean distance from each pixel to each palette colour
        sq_dist = np.einsum("mni,mni->mn", diff, diff)               # (M, N)
        # each pixel contributes the distance to its nearest colour
        return float(np.mean(np.min(sq_dist, axis=1)))

    return objective


# ── Image utilities ───────────────────────────────────────────────────────────

def load_pixels(
    path: Path,
    max_pixels: int = 8_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the image and returns (all_pixels, sampled_pixels).

    all_pixels:    (H*W, 3) uint8 — used for full-image quantisation.
    sampled_pixels:(M, 3) float64 — used as the PSO objective sample.

    Sampling keeps the objective cheap: for a 12 MP image (12M pixels)
    computing distances every iteration would be prohibitive.  8 000
    pixels capture the colour distribution well in practice.
    """
    img = Image.open(path).convert("RGB")
    all_pixels = np.array(img, dtype=np.uint8).reshape(-1, 3)

    rng = np.random.default_rng(seed)
    if len(all_pixels) > max_pixels:
        idx = rng.choice(len(all_pixels), max_pixels, replace=False)
        sampled = all_pixels[idx].astype(np.float64)
    else:
        sampled = all_pixels.astype(np.float64)

    return all_pixels, sampled


def quantize_image(image_path: Path, palette: np.ndarray) -> Image.Image:
    """Replaces every pixel in the original image with its nearest palette colour."""
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.float64)
    H, W, _ = pixels.shape
    flat = pixels.reshape(-1, 3)                                      # (H*W, 3)

    pal = np.clip(palette, 0.0, 255.0)                                # (N, 3)
    diff = flat[:, np.newaxis, :] - pal[np.newaxis, :, :]            # (H*W, N, 3)
    sq_dist = np.einsum("mni,mni->mn", diff, diff)                   # (H*W, N)
    nearest = np.argmin(sq_dist, axis=1)                              # (H*W,)

    quantized = pal[nearest].astype(np.uint8).reshape(H, W, 3)
    return Image.fromarray(quantized)


def save_results_figure(
    palette: np.ndarray,
    history,
    output_path: Path,
    title: str,
) -> None:
    """Saves a figure with colour swatches (+ hex codes) and convergence curve."""
    n = len(palette)
    colors = np.clip(palette, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 1.2 + 4), 3.5))

    # ── Left: palette swatches ──────────────────────────────────────────────
    swatch_w = 60
    swatch = np.zeros((swatch_w, n * swatch_w, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        swatch[:, i * swatch_w:(i + 1) * swatch_w] = c

    axes[0].imshow(swatch)
    axes[0].set_title(f"Palette — {n} colours", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i, c in enumerate(colors):
        brightness = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        text_color = "black" if brightness > 140 else "white"
        axes[0].text(
            i * swatch_w + swatch_w / 2,
            swatch_w / 2,
            f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}",
            ha="center", va="center",
            fontsize=max(5, 8 - n // 6),
            color=text_color,
            fontfamily="monospace",
        )

    # ── Right: convergence curve ────────────────────────────────────────────
    iters = [m.iteration for m in history]
    fitness = [m.best_fitness for m in history]
    axes[1].plot(iters, fitness, linewidth=1.5, color="#2563eb")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSQE")
    axes[1].set_title("Convergence", fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSO-based optimal colour palette extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path, help="Input image (JPEG, PNG, …).")

    g_pal = p.add_argument_group("palette")
    g_pal.add_argument("--n-colors", type=int, default=8, metavar="N",
                       help="Number of palette colours.")

    g_pso = p.add_argument_group("PSO")
    g_pso.add_argument("--swarm-size", type=int, default=40)
    g_pso.add_argument("--iterations", type=int, default=300,
                       help="Maximum PSO iterations.")
    g_pso.add_argument("--stagnation", type=int, default=60,
                       help="Stop after N iterations without improvement.")
    g_pso.add_argument("--min-delta", type=float, default=0.1, dest="min_delta",
                       help="Minimum MSQE improvement to reset the stagnation counter.")
    g_pso.add_argument("--inertia", type=float, default=0.7)
    g_pso.add_argument("--cognitive", type=float, default=1.5)
    g_pso.add_argument("--social", type=float, default=1.5)
    g_pso.add_argument("--seed", type=int, default=42)
    g_pso.add_argument("--max-pixels", type=int, default=8_000,
                       help="Pixels sampled for the objective (speed vs accuracy).")

    g_ev = p.add_argument_group("evaluator (parallelism strategy)")
    g_ev.add_argument("--evaluator", default="sequential",
                      choices=["sequential", "threading", "multiprocessing", "asyncio"])
    g_ev.add_argument("--max-workers", type=int, default=None,
                      help="Worker count for threading / multiprocessing.")
    g_ev.add_argument("--latency", type=float, default=0.0, dest="asyncio_latency",
                      help="Simulated I/O latency per particle in seconds (asyncio only).")

    g_out = p.add_argument_group("output")
    g_out.add_argument("--output-dir", type=Path, default=Path("results/color_palette"))

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image & sample pixels ──────────────────────────────────────────
    print(f"Loading {args.image} …")
    all_pixels, sampled_pixels = load_pixels(args.image, args.max_pixels, args.seed)
    H_W = len(all_pixels)
    img_size = Image.open(args.image).size
    print(f"  Resolution : {img_size[0]}×{img_size[1]}  ({H_W:,} pixels)")
    print(f"  Sampled    : {len(sampled_pixels):,} pixels for objective")

    # ── Build objective & bounds ────────────────────────────────────────────
    dim = args.n_colors * 3
    objective = make_objective(sampled_pixels, args.n_colors)
    bounds = BoxBounds(
        lower=np.zeros(dim),
        upper=np.full(dim, 255.0),
    )

    # ── Build evaluator ─────────────────────────────────────────────────────
    if args.evaluator == "asyncio":
        evaluator = get_evaluator(
            "asyncio",
            latency_s=args.asyncio_latency,
            jitter=0.0,
            seed=args.seed,
        )
    else:
        evaluator = get_evaluator(args.evaluator, max_workers=args.max_workers)

    # ── PSO config & run ────────────────────────────────────────────────────
    config = PSOConfig(
        dimension=dim,
        swarm_size=args.swarm_size,
        inertia_weight=args.inertia,
        cognitive_weight=args.cognitive,
        social_weight=args.social,
        stop=StopCriteria(
            max_iterations=args.iterations,
            stagnation_iterations=args.stagnation,
            min_delta=args.min_delta,
        ),
        seed=args.seed,
    )

    print(f"\nRunning PSO …")
    print(
        f"  n_colors={args.n_colors}  dim={dim}  swarm={args.swarm_size}  "
        f"max_iter={args.iterations}  evaluator={args.evaluator}"
    )

    pso = PSO(
        objective=objective,
        bounds=bounds,
        config=config,
        evaluator=evaluator,
    )

    t0 = time.perf_counter()
    result = pso.optimize()
    elapsed = time.perf_counter() - t0
    evaluator.close()

    palette = np.clip(result.best_position.reshape(args.n_colors, 3), 0.0, 255.0)

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  MSQE (sampled pixels) : {result.best_value:,.2f}")
    print(f"  RMSE per channel      : {result.best_value ** 0.5:.2f} / 255")
    print(f"  Iterations run        : {len(result.history)}")
    if result.converged_iteration:
        print(f"  Converged at iter     : {result.converged_iteration}")
    print(f"  Total time            : {elapsed:.2f}s")
    print(f"  Eval time             : {result.total_eval_time_s:.2f}s  "
          f"({result.total_eval_time_s/elapsed*100:.0f}% of total)")
    print(f"\n  Palette:")
    for i, c in enumerate(palette.astype(int)):
        print(f"    {i+1:2d}.  R={c[0]:3d}  G={c[1]:3d}  B={c[2]:3d}  "
              f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}")
    print(f"{'─'*52}\n")

    # ── Save outputs ────────────────────────────────────────────────────────
    stem = args.image.stem
    n = args.n_colors

    quant_path = args.output_dir / f"{stem}_quantized_{n}colors.png"
    print("Quantizing full image …")
    quantize_image(args.image, palette).save(quant_path)
    print(f"  Saved: {quant_path}")

    fig_path = args.output_dir / f"{stem}_palette_{n}colors.png"
    save_results_figure(
        palette=palette,
        history=result.history,
        output_path=fig_path,
        title=(
            f"PSO Color Quantisation · {args.image.name} · "
            f"{n} colours · MSQE={result.best_value:.1f}"
        ),
    )
    print(f"  Saved: {fig_path}\n")


if __name__ == "__main__":
    main()
