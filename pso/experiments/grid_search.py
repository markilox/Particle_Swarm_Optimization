
import csv
import itertools
import logging
from dataclasses import dataclass, fields
from pathlib import Path

from pso.core.types import PSOConfig, StopCriteria
from pso.experiments.runner import ExperimentConfig, run_experiment
from pso.objectives.benchmarks import ObjectiveSpec
from pso.parallel.base import FitnessEvaluator


@dataclass
class GridSearchConfig:
    inertia_weights: list[float]
    cognitive_weights: list[float]
    social_weights: list[float]
    swarm_sizes: list[int]
    seeds: list[int]
    max_iterations: int = 200
    tolerance: float | None = None
    stagnation_iterations: int | None = None


@dataclass
class GridSearchEntry:
    objective: str
    evaluator: str
    dimension: int
    inertia_weight: float
    cognitive_weight: float
    social_weight: float
    swarm_size: int
    seed: int
    best_value: float
    total_time_s: float
    total_eval_time_s: float
    converged_iteration: int | None
    total_iterations: int


def run_grid_search(
    objective: ObjectiveSpec,
    dimension: int,
    grid: GridSearchConfig,
    evaluator_factory,
    evaluator_name: str,
    output_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> list[GridSearchEntry]:
    """Ejecuta el grid search y devuelve la lista de entradas.

    Args:
        evaluator_factory: callable sin argumentos que devuelve un FitnessEvaluator
            nuevo en cada llamada (necesario para evaluadores con estado interno).
        output_dir: si se especifica, cada run individual persiste su metadata.json
            y history.csv en un subdirectorio propio.
    """
    logger = logger or logging.getLogger("pso.grid_search")
    entries: list[GridSearchEntry] = []

    combos = list(
        itertools.product(
            grid.inertia_weights,
            grid.cognitive_weights,
            grid.social_weights,
            grid.swarm_sizes,
            grid.seeds,
        )
    )
    total = len(combos)
    logger.info(
        f"Grid search: {total} combinaciones — "
        f"objetivo={objective.name} d={dimension} evaluador={evaluator_name}"
    )

    for idx, (w, c1, c2, n, seed) in enumerate(combos, 1):
        config = PSOConfig(
            dimension=dimension,
            swarm_size=n,
            inertia_weight=w,
            cognitive_weight=c1,
            social_weight=c2,
            stop=StopCriteria(
                max_iterations=grid.max_iterations,
                tolerance=grid.tolerance,
                stagnation_iterations=grid.stagnation_iterations,
            ),
            seed=seed,
        )
        # Logger silencioso para las sub-ejecuciones del grid (evita spam de logs)
        silent_logger = logging.getLogger("pso.grid_search.run")
        silent_logger.setLevel(logging.WARNING)

        exp = ExperimentConfig(
            objective=objective,
            pso_config=config,
            evaluator=evaluator_factory(),
            evaluator_name=evaluator_name,
            output_dir=output_dir,
            logger=silent_logger,
        )
        result = run_experiment(exp)

        entry = GridSearchEntry(
            objective=objective.name,
            evaluator=evaluator_name,
            dimension=dimension,
            inertia_weight=w,
            cognitive_weight=c1,
            social_weight=c2,
            swarm_size=n,
            seed=seed,
            best_value=result.best_value,
            total_time_s=result.total_time_s,
            total_eval_time_s=result.total_eval_time_s,
            converged_iteration=result.converged_iteration,
            total_iterations=len(result.history),
        )
        entries.append(entry)
        logger.info(
            f"[{idx}/{total}] w={w} c1={c1} c2={c2} n={n} seed={seed} "
            f"best={result.best_value:.6e} time={result.total_time_s:.3f}s"
        )

    return entries


def save_grid_search_results(entries: list[GridSearchEntry], path: str | Path) -> None:
    """Guarda entradas de grid search en CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        return
    field_names = [f.name for f in fields(GridSearchEntry)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {f.name: getattr(entry, f.name) for f in fields(GridSearchEntry)}
            )


def load_grid_search_results(path: str | Path) -> list[GridSearchEntry]:
    """Carga CSV de grid search y devuelve lista de GridSearchEntry."""
    entries = []
    with open(Path(path)) as f:
        for row in csv.DictReader(f):
            ci = row["converged_iteration"]
            entries.append(
                GridSearchEntry(
                    objective=row["objective"],
                    evaluator=row["evaluator"],
                    dimension=int(row["dimension"]),
                    inertia_weight=float(row["inertia_weight"]),
                    cognitive_weight=float(row["cognitive_weight"]),
                    social_weight=float(row["social_weight"]),
                    swarm_size=int(row["swarm_size"]),
                    seed=int(row["seed"]),
                    best_value=float(row["best_value"]),
                    total_time_s=float(row["total_time_s"]),
                    total_eval_time_s=float(row["total_eval_time_s"]),
                    converged_iteration=int(ci) if ci not in ("None", "") else None,
                    total_iterations=int(row["total_iterations"]),
                )
            )
    return entries
