
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml
from prettytable import PrettyTable

from pso.core.types import PSOConfig, StopCriteria
from pso.experiments.runner import ExperimentConfig, run_experiment
from pso.objectives.benchmarks import get_objective
from pso.parallel import get_evaluator


CONFIG_FILE = Path(__file__).parent / "configs" / "benchmarks.yaml"

SUMMARY_FIELDS = [
    "run_id", "objective", "dimension", "evaluator", "seed",
    "inertia_weight", "cognitive_weight", "social_weight", "swarm_size",
    "best_value", "total_time_s", "total_eval_time_s", "total_update_time_s",
    "converged_iteration", "total_iterations",
]

HISTORY_FIELDS = ["run_id", "iteration", "best_fitness", "eval_time_s", "update_time_s"]


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _evaluator_kwargs(ev_name: str, params: dict, seed: int | None = None) -> dict:
    if ev_name == "asyncio":
        return {
            "strategy": params.get("asyncio_strategy", "gather"),
            "latency_s": params.get("asyncio_latency_s", 0.0),
            "jitter":    params.get("asyncio_jitter", 0.0),
            "n_workers": params.get("asyncio_n_workers", 4),
            "seed":      seed,
        }
    return {"max_workers": params.get("max_workers")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("benchmarks")

    params = load_config()

    combos = [
        (obj_name, dim, ev_name, seed)
        for obj_name in params["objectives"]
        for dim in params["dimensions"]
        for ev_name in params["evaluators"]
        for seed in params["seeds"]
    ]
    total = len(combos)

    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{ts}.csv"
    history_path = output_dir / f"history_{ts}.csv"

    print(f"\nLaunching {total} experiments → {output_dir}\n")

    suite_start = time.perf_counter()
    table_rows = []

    with (
        open(summary_path, "w", newline="") as sf,
        open(history_path, "w", newline="") as hf,
    ):
        summary_writer = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        history_writer = csv.DictWriter(hf, fieldnames=HISTORY_FIELDS)
        summary_writer.writeheader()
        history_writer.writeheader()

        for idx, (obj_name, dim, ev_name, seed) in enumerate(combos, 1):
            objective = get_objective(obj_name)
            evaluator = get_evaluator(ev_name, **_evaluator_kwargs(ev_name, params, seed))

            config = PSOConfig(
                dimension=dim,
                swarm_size=params["swarm_size"],
                inertia_weight=params["inertia"],
                cognitive_weight=params["cognitive"],
                social_weight=params["social"],
                stop=StopCriteria(max_iterations=params["iterations"]),
                seed=seed,
            )
            silent = logging.getLogger("pso.benchmark.run")
            silent.setLevel(logging.WARNING)

            exp = ExperimentConfig(
                objective=objective,
                pso_config=config,
                evaluator=evaluator,
                evaluator_name=ev_name,
                output_dir=None,
                logger=silent,
            )
            result = run_experiment(exp)

            run_id = f"{obj_name}_{ev_name}_d{dim}_seed{seed}"

            summary_writer.writerow({
                "run_id": run_id,
                "objective": obj_name,
                "dimension": dim,
                "evaluator": ev_name,
                "seed": seed,
                "inertia_weight": params["inertia"],
                "cognitive_weight": params["cognitive"],
                "social_weight": params["social"],
                "swarm_size": params["swarm_size"],
                "best_value": result.best_value,
                "total_time_s": result.total_time_s,
                "total_eval_time_s": result.total_eval_time_s,
                "total_update_time_s": result.total_update_time_s,
                "converged_iteration": result.converged_iteration,
                "total_iterations": len(result.history),
            })

            for m in result.history:
                history_writer.writerow({
                    "run_id": run_id,
                    "iteration": m.iteration,
                    "best_fitness": m.best_fitness,
                    "eval_time_s": m.eval_time_s,
                    "update_time_s": m.update_time_s,
                })

            table_rows.append({
                "objective": obj_name,
                "dimension": dim,
                "evaluator": ev_name,
                "seed": seed,
                "best_value": result.best_value,
                "total_time_s": result.total_time_s,
                "total_eval_time_s": result.total_eval_time_s,
                "total_iterations": len(result.history),
            })
            logger.info(
                f"[{idx}/{total}] {obj_name} d={dim} {ev_name} seed={seed} "
                f"best={result.best_value:.4e} t={result.total_time_s:.3f}s"
            )

    suite_time = time.perf_counter() - suite_start

    table = PrettyTable()
    table.field_names = ["Objective", "d", "Evaluator", "Seed", "Best fitness", "Time (s)", "Eval (s)", "Iters"]
    table.align["Objective"] = "l"
    table.align["Evaluator"] = "l"
    table.align["Best fitness"] = "r"
    table.align["Eval (s)"] = "r"
    for r in table_rows:
        table.add_row([
            r["objective"],
            r["dimension"],
            r["evaluator"],
            r["seed"],
            f"{r['best_value']:.4e}",
            f"{r['total_time_s']:.3f}",
            f"{r['total_eval_time_s']:.3f}",
            r["total_iterations"],
        ])
    print()
    print(table)
    print(f"\nSuite completed in {suite_time:.2f}s")
    print(f"  {summary_path}  ({total} rows)")
    print(f"  {history_path}  (~{total * params['iterations']} rows)\n")


if __name__ == "__main__":
    main()
