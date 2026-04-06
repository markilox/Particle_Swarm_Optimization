"""run_grid_search.py — Grid search de hiperparámetros PSO.

Itera sobre el producto cartesiano de (w, c1, c2, swarm_size, seed)
y guarda los resultados en un CSV para análisis posterior.

Uso:
  python run_grid_search.py
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from prettytable import PrettyTable

from pso.experiments.grid_search import (
    GridSearchConfig,
    run_grid_search,
    save_grid_search_results,
)
from pso.objectives.benchmarks import get_objective, list_objectives
from pso.parallel import EVALUATOR_NAMES, get_evaluator


def ask_one(prompt: str, options: list[str], default: str) -> str:
    """Pide elegir una opción de la lista."""
    print(f"  Opciones: {', '.join(options)}")
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    if raw not in options:
        print(f"  Opción no válida. Se usará '{default}'.")
        return default
    return raw


def ask_float_list(prompt: str, default: list[float]) -> list[float]:
    """Pide una lista de floats separados por espacios."""
    raw = input(f"  {prompt} [{' '.join(map(str, default))}]: ").strip()
    if not raw:
        return default
    try:
        return [float(x) for x in raw.split()]
    except ValueError:
        print("  Valor no válido. Se usará el valor por defecto.")
        return default


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


def ask_str(prompt: str, default: str) -> str:
    raw = input(f"  {prompt} [{default}]: ").strip()
    return raw if raw else default


def prompt_params() -> dict:
    """Pregunta todos los parámetros de forma interactiva."""
    print("\n=== Configuración del Grid Search ===\n")

    print("Función objetivo:")
    objective = ask_one("Selecciona", list_objectives(), "sphere")

    print("\nDimensión:")
    dimension = ask_int("Dimensión del espacio", 10)

    print("\nEvaluador:")
    evaluator = ask_one("Selecciona", EVALUATOR_NAMES, "sequential")

    print("\nRejilla de hiperparámetros (valores separados por espacio):")
    inertia   = ask_float_list("Valores de inercia w", [0.4, 0.7, 0.9])
    cognitive = ask_float_list("Valores de cognitivo c1", [1.0, 1.5, 2.0])
    social    = ask_float_list("Valores de social c2", [1.0, 1.5, 2.0])
    swarm_sizes = ask_int_list("Tamaños de enjambre", [30])

    print("\nSemillas y criterio de parada:")
    seeds      = ask_int_list("Seeds", [42, 43, 44, 45, 46])
    iterations = ask_int("Máximo de iteraciones", 200)

    print("\nSalida:")
    output_dir = ask_str("Directorio de resultados", "results/grid_search")

    return {
        "objective": objective,
        "dimension": dimension,
        "evaluator": evaluator,
        "inertia": inertia,
        "cognitive": cognitive,
        "social": social,
        "swarm_sizes": swarm_sizes,
        "seeds": seeds,
        "iterations": iterations,
        "output_dir": output_dir,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("grid_search")

    params = prompt_params()

    objective = get_objective(params["objective"])
    grid = GridSearchConfig(
        inertia_weights=params["inertia"],
        cognitive_weights=params["cognitive"],
        social_weights=params["social"],
        swarm_sizes=params["swarm_sizes"],
        seeds=params["seeds"],
        max_iterations=params["iterations"],
    )

    entries = run_grid_search(
        objective=objective,
        dimension=params["dimension"],
        grid=grid,
        evaluator_factory=lambda: get_evaluator(params["evaluator"]),
        evaluator_name=params["evaluator"],
        output_dir=None,
        logger=logger,
    )

    output_dir = Path(params["output_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"grid_{params['objective']}_{params['evaluator']}_d{params['dimension']}_{ts}.csv"
    save_grid_search_results(entries, csv_path)

    # Ranking de configuraciones (promedio de best_value sobre seeds)
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for e in entries:
        key = (e.inertia_weight, e.cognitive_weight, e.social_weight, e.swarm_size)
        grouped[key].append(e.best_value)

    ranking = sorted(
        grouped.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
    )

    table = PrettyTable()
    table.field_names = ["#", "w", "c1", "c2", "n", "Fitness medio", "Fitness min", "Fitness max"]
    table.align["Fitness medio"] = "r"
    table.align["Fitness min"] = "r"
    table.align["Fitness max"] = "r"
    for rank, (key, values) in enumerate(ranking[:10], 1):
        w, c1, c2, n = key
        table.add_row([
            rank, w, c1, c2, n,
            f"{sum(values)/len(values):.4e}",
            f"{min(values):.4e}",
            f"{max(values):.4e}",
        ])

    print(f"\nGrid search completado: {len(entries)} ejecuciones")
    print(f"Top {min(10, len(ranking))} configuraciones (media sobre {len(params['seeds'])} seeds):\n")
    print(table)
    print(f"\nResultados guardados en: {csv_path}\n")


if __name__ == "__main__":
    main()
