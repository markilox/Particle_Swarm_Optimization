"""run_pso.py — Runs a single PSO instance from a YAML config file.

Usage:
  python run_pso.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pso.core.types import BoxBounds, PSOConfig, StopCriteria
from pso.experiments.runner import ExperimentConfig, run_experiment
from pso.objectives.benchmarks import get_objective
from pso.parallel import get_evaluator
from pso.viz import SwarmRecorder, animate_swarm, plot_convergence


CONFIG_FILE = Path(__file__).parent / "configs" / "pso.yaml"


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = load_config()

    objective = get_objective(params["objective"])
    ev_name = params["evaluator"]
    if ev_name == "asyncio":
        evaluator = get_evaluator(
            "asyncio",
            strategy=params.get("asyncio_strategy", "gather"),
            latency_s=params.get("asyncio_latency_s", 0.0),
            jitter=params.get("asyncio_jitter", 0.0),
            n_workers=params.get("asyncio_n_workers", 4),
            seed=params.get("seed"),
        )
    else:
        evaluator = get_evaluator(ev_name, max_workers=params.get("max_workers"))

    bounds = BoxBounds(
        lower=np.full(params["dimension"], params.get("lower_bound", objective.lower_bound)),
        upper=np.full(params["dimension"], params.get("upper_bound", objective.upper_bound)),
    )

    config = PSOConfig(
        dimension=params["dimension"],
        swarm_size=params["swarm_size"],
        inertia_weight=params["inertia"],
        cognitive_weight=params["cognitive"],
        social_weight=params["social"],
        stop=StopCriteria(
            max_iterations=params["iterations"],
            tolerance=params.get("tolerance"),
            stagnation_iterations=params.get("stagnation"),
        ),
        seed=params.get("seed"),
    )

    recorder = SwarmRecorder(every_n=1)

    exp = ExperimentConfig(
        objective=objective,
        pso_config=config,
        evaluator=evaluator,
        evaluator_name=params["evaluator"],
        output_dir=params.get("output_dir", "results/runs"),
        bounds=bounds,
    )

    result = run_experiment(exp, on_iteration=recorder)

    print(f"\n{'='*50}")
    print(f"Objective:       {params['objective']}  d={params['dimension']}")
    print(f"Evaluator:       {params['evaluator']}")
    print(f"Best value:      {result.best_value:.8e}")
    print(f"Best position:   {result.best_position}")
    print(f"Iterations:      {len(result.history)}")
    if result.converged_iteration:
        print(f"Converged at:    iteration {result.converged_iteration}")
    print(f"Total time:      {result.total_time_s:.4f}s")
    print(f"  - evaluation:  {result.total_eval_time_s:.4f}s")
    print(f"  - update:      {result.total_update_time_s:.4f}s")
    print(f"{'='*50}\n")

    ans = input("¿Generar visualización? [s/N]: ").strip().lower()
    if ans == "s":
        viz_cfg_path = Path(__file__).parent / "configs" / "viz.yaml"
        with open(viz_cfg_path) as f:
            import yaml as _yaml
            viz = _yaml.safe_load(f)

        output_dir = Path(viz.get("output_dir", "results/viz"))
        output_dir.mkdir(parents=True, exist_ok=True)
        base = f"{params['objective']}_d{params['dimension']}_seed{params.get('seed', 0)}"

        fmt = viz.get("format", "gif")
        anim_path = output_dir / f"{base}.{fmt}"
        print(f"Generating animation → {anim_path}")
        try:
            animate_swarm(
                frames=recorder.frames,
                history=result.history,
                dimension=params["dimension"],
                output_path=anim_path,
                objective=objective,
                bounds=bounds,
                fps=viz.get("fps", 10),
                resolution=viz.get("resolution", 150),
            )
            print(f"  Animation saved: {anim_path}")
        except Exception as exc:
            print(f"  ERROR generating animation: {exc}", file=sys.stderr)

        conv_path = output_dir / f"{base}_convergence.png"
        print(f"Generating convergence plot → {conv_path}")
        fig = plot_convergence(
            history=result.history,
            title=f"Convergence — {params['objective']}  d={params['dimension']}  seed={params.get('seed')}",
        )
        fig.savefig(conv_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {conv_path}\n")


if __name__ == "__main__":
    main()
