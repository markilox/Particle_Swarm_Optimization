"""run_benchmarks.py — Suite de benchmarks: todas las funciones × dimensiones × evaluadores.

Pregunta los parámetros de forma interactiva y guarda resultados en dos CSVs:
  summary_<ts>.csv  — una fila por run con métricas finales y configuración completa.
  history_<ts>.csv  — una fila por iteración, con run_id para cruzar con summary.

Uso:
  python run_benchmarks.py
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

from prettytable import PrettyTable

from pso.core.types import PSOConfig, StopCriteria
from pso.experiments.runner import ExperimentConfig, run_experiment
from pso.objectives.benchmarks import list_objectives, get_objective
from pso.parallel import EVALUATOR_NAMES, get_evaluator


SUMMARY_FIELDS = [
    "run_id", "objective", "dimension", "evaluator", "seed",
    "inertia_weight", "cognitive_weight", "social_weight", "swarm_size",
    "best_value", "total_time_s", "total_eval_time_s", "total_update_time_s",
    "converged_iteration", "total_iterations",
]

HISTORY_FIELDS = ["run_id", "iteration", "best_fitness", "eval_time_s", "update_time_s"]


def ask_list(prompt: str, options: list[str], default: list[str]) -> list[str]:
    """Muestra opciones disponibles y pide una selección separada por espacios."""
    print(f"  Opciones: {', '.join(options)}")
    raw = input(f"  {prompt} [{' '.join(default)}]: ").strip()
    if not raw:
        return default
    chosen = raw.split()
    invalid = [x for x in chosen if x not in options]
    if invalid:
        print(f"  Valores no válidos: {', '.join(invalid)}. Se usarán los valores por defecto.")
        return default
    return chosen


def ask_int_list(prompt: str, default: list[int]) -> list[int]:
    """Pide una lista de enteros separados por espacios."""
    raw = input(f"  {prompt} [{' '.join(map(str, default))}]: ").strip()
    if not raw:
        return default
    try:
        return [int(x) for x in raw.split()]
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


def ask_str(prompt: str, default: str) -> str:
    raw = input(f"  {prompt} [{default}]: ").strip()
    return raw if raw else default


def prompt_params() -> dict:
    """Pregunta todos los parámetros de forma interactiva y devuelve un dict."""
    print("\n=== Configuración de la suite de benchmarks ===\n")

    print("Funciones objetivo:")
    objectives = ask_list("Selecciona (separadas por espacio)", list_objectives(), list_objectives())

    print("\nDimensiones:")
    dimensions = ask_int_list("Dimensiones (separadas por espacio)", [2, 10, 30])

    print("\nEvaluadores:")
    evaluators = ask_list("Selecciona (separadas por espacio)", EVALUATOR_NAMES, EVALUATOR_NAMES)

    print("\nSemillas:")
    seeds = ask_int_list("Seeds (separadas por espacio)", [42, 43, 44])

    print("\nHiperparámetros del PSO:")
    swarm_size = ask_int("Tamaño del enjambre", 30)
    iterations = ask_int("Número de iteraciones", 200)
    inertia   = ask_float("Inercia w", 0.7)
    cognitive = ask_float("Cognitivo c1", 1.5)
    social    = ask_float("Social c2", 1.5)

    print("\nSalida:")
    output_dir = ask_str("Directorio de resultados", "results/benchmarks")

    return {
        "objectives": objectives,
        "dimensions": dimensions,
        "evaluators": evaluators,
        "seeds": seeds,
        "swarm_size": swarm_size,
        "iterations": iterations,
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
        "output_dir": output_dir,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("benchmarks")

    params = prompt_params()

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

    print(f"\nLanzando {total} experimentos → {output_dir}\n")

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
            evaluator = get_evaluator(ev_name)

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
    table.field_names = ["Objetivo", "d", "Evaluador", "Seed", "Best fitness", "Tiempo (s)", "Eval (s)", "Iters"]
    table.align["Objetivo"] = "l"
    table.align["Evaluador"] = "l"
    table.align["Best fitness"] = "r"
    table.align["Tiempo (s)"] = "r"
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
    print(f"\nSuite completada en {suite_time:.2f}s")
    print(f"  {summary_path}  ({total} filas)")
    print(f"  {history_path}  ({total * params['iterations']} filas aprox.)\n")


if __name__ == "__main__":
    main()
