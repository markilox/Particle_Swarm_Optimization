"""make_viz.py — Generates swarm animations and convergence plots.

For d=2: animation with function contour + swarm positions + convergence curve.
For d=3: 3D animation with swarm scatter + convergence curve.
For d>3: animated convergence curve only.

Also always saves a static PNG with the convergence curve.

Usage:
  python make_viz.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pso.core.types import PSOConfig, StopCriteria, BoxBounds
from pso.core.pso import PSO
from pso.objectives.benchmarks import get_objective
from pso.viz import SwarmRecorder, animate_swarm, plot_convergence


CONFIG_FILE = Path(__file__).parent / "configs" / "viz.yaml"


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    p = load_config()

    objective = get_objective(p["objective"])
    bounds = BoxBounds(
        lower=np.full(p["dimension"], p.get("lower_bound", objective.lower_bound)),
        upper=np.full(p["dimension"], p.get("upper_bound", objective.upper_bound)),
    )

    config = PSOConfig(
        dimension=p["dimension"],
        swarm_size=p["swarm_size"],
        inertia_weight=p["inertia"],
        cognitive_weight=p["cognitive"],
        social_weight=p["social"],
        stop=StopCriteria(max_iterations=p["iterations"]),
        seed=p["seed"],
    )

    pso = PSO(objective=objective, bounds=bounds, config=config)
    recorder = SwarmRecorder(every_n=p.get("every_n", 1))

    print(f"\nRunning PSO: {p['objective']} d={p['dimension']}, {p['iterations']} iterations...")
    result = pso.optimize(on_iteration=recorder)
    print(f"  Best value: {result.best_value:.6e}  |  frames recorded: {len(recorder)}")

    output_dir = Path(p["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{p['objective']}_d{p['dimension']}_seed{p['seed']}"

    # --- animation ---
    fmt = p.get("format", "gif")
    anim_path = output_dir / f"{base}.{fmt}"
    print(f"Generating animation → {anim_path}")
    try:
        animate_swarm(
            frames=recorder.frames,
            history=result.history,
            dimension=p["dimension"],
            output_path=anim_path,
            objective=objective,
            bounds=bounds,
            fps=p.get("fps", 10),
            resolution=p.get("resolution", 150),
        )
        print(f"  Animation saved: {anim_path}")
    except Exception as exc:
        print(f"  ERROR generating animation: {exc}", file=sys.stderr)
        print("  Make sure 'pillow' is installed for GIF or 'ffmpeg' for MP4.", file=sys.stderr)

    # --- static convergence plot ---
    conv_path = output_dir / f"{base}_convergence.png"
    print(f"Generating convergence plot → {conv_path}")
    fig = plot_convergence(
        history=result.history,
        title=f"Convergence — {p['objective']}  d={p['dimension']}  seed={p['seed']}",
    )
    fig.savefig(conv_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {conv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
