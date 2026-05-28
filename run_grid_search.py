"""run_grid_search.py — PSO hyperparameter grid search from a YAML config file.

Usage:
  python run_grid_search.py
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from prettytable import PrettyTable

from pso.core.types import BoxBounds
from pso.experiments.grid_search import (
    GridSearchConfig,
    run_grid_search,
    save_grid_search_results,
    save_grid_search_summary,
)
from pso.objectives.benchmarks import get_objective
from pso.parallel import get_evaluator


CONFIG_FILE = Path(__file__).parent / "configs" / "grid_search.yaml"


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _make_evaluator(params: dict):
    ev_name = params["evaluator"]
    if ev_name == "asyncio":
        return get_evaluator(
            "asyncio",
            strategy=params.get("asyncio_strategy", "gather"),
            latency_s=params.get("asyncio_latency_s", 0.0),
            jitter=params.get("asyncio_jitter", 0.0),
            n_workers=params.get("asyncio_n_workers", 4),
            seed=params.get("seed"),
        )
    return get_evaluator(ev_name, max_workers=params.get("max_workers"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("grid_search")

    params = load_config()

    objective = get_objective(params["objective"])
    grid = GridSearchConfig(
        inertia_weights=params["inertia"],
        cognitive_weights=params["cognitive"],
        social_weights=params["social"],
        swarm_sizes=params["swarm_sizes"],
        seeds=params["seeds"],
        max_iterations=params["iterations"],
    )

    bounds = BoxBounds(
        lower=np.full(params["dimension"], params.get("lower_bound", objective.lower_bound)),
        upper=np.full(params["dimension"], params.get("upper_bound", objective.upper_bound)),
    )

    entries = run_grid_search(
        objective=objective,
        dimension=params["dimension"],
        grid=grid,
        evaluator_factory=lambda: _make_evaluator(params),
        evaluator_name=params["evaluator"],
        output_dir=None,
        logger=logger,
        bounds=bounds,
    )

    output_dir = Path(params.get("output_dir", "results/grid_search"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"grid_{params['objective']}_{params['evaluator']}_d{params['dimension']}_{ts}.csv"
    save_grid_search_results(entries, csv_path)
    summary_path = csv_path.with_name(csv_path.stem + "_summary.csv")
    save_grid_search_summary(entries, summary_path)

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for e in entries:
        key = (e.inertia_weight, e.cognitive_weight, e.social_weight, e.swarm_size)
        grouped[key].append(e.best_value)

    ranking = sorted(
        grouped.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
    )

    table = PrettyTable()
    table.field_names = ["#", "w", "c1", "c2", "n", "Mean fitness", "Min fitness", "Max fitness"]
    table.align["Mean fitness"] = "r"
    table.align["Min fitness"] = "r"
    table.align["Max fitness"] = "r"
    for rank, (key, values) in enumerate(ranking[:10], 1):
        w, c1, c2, n = key
        table.add_row([
            rank, w, c1, c2, n,
            f"{sum(values)/len(values):.4e}",
            f"{min(values):.4e}",
            f"{max(values):.4e}",
        ])

    print(f"\nGrid search completed: {len(entries)} runs")
    print(f"Top {min(10, len(ranking))} configurations (mean over {len(params['seeds'])} seeds):\n")
    print(table)
    print(f"\nResults saved to:  {csv_path}")
    print(f"Summary saved to:  {summary_path}\n")


if __name__ == "__main__":
    main()
