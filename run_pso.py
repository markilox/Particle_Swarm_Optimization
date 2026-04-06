"""run_pso.py — Ejecuta una sola instancia del PSO por línea de comandos.

Ejemplos:
  python run_pso.py --objective sphere --dimension 10
  python run_pso.py --objective rastrigin --dimension 30 --swarm-size 50 --seed 42 --output-dir results/
  python run_pso.py --objective ackley --dimension 2 --evaluator threading --max-workers 4
"""

import argparse
import logging
import sys

from pso.core.types import PSOConfig, StopCriteria
from pso.experiments.runner import ExperimentConfig, run_experiment
from pso.objectives.benchmarks import get_objective, list_objectives
from pso.parallel import EVALUATOR_NAMES, get_evaluator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ejecuta una instancia del PSO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--objective", default="sphere", choices=list_objectives(), help="Función objetivo")
    p.add_argument("--dimension", type=int, default=10, help="Dimensión del espacio de búsqueda")
    p.add_argument("--swarm-size", type=int, default=30, help="Número de partículas")
    p.add_argument("--inertia", type=float, default=0.7, help="Peso de inercia w")
    p.add_argument("--cognitive", type=float, default=1.5, help="Coeficiente cognitivo c1")
    p.add_argument("--social", type=float, default=1.5, help="Coeficiente social c2")
    p.add_argument("--iterations", type=int, default=200, help="Máximo de iteraciones")
    p.add_argument("--tolerance", type=float, default=None, help="Tolerancia de convergencia")
    p.add_argument("--stagnation", type=int, default=None, help="Iteraciones de estancamiento")
    p.add_argument("--seed", type=int, default=None, help="Semilla aleatoria (None = no reproducible)")
    p.add_argument("--evaluator", default="sequential", choices=EVALUATOR_NAMES, help="Estrategia de evaluación")
    p.add_argument("--max-workers", type=int, default=None, help="Workers para threading/multiprocessing")
    p.add_argument("--latency", type=float, default=0.0, help="Latencia simulada para asyncio (segundos)")
    p.add_argument("--output-dir", default=None, help="Directorio de salida para persistir resultados")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    objective = get_objective(args.objective)
    evaluator = get_evaluator(
        args.evaluator,
        max_workers=args.max_workers,
        latency=args.latency,
    )

    config = PSOConfig(
        dimension=args.dimension,
        swarm_size=args.swarm_size,
        inertia_weight=args.inertia,
        cognitive_weight=args.cognitive,
        social_weight=args.social,
        stop=StopCriteria(
            max_iterations=args.iterations,
            tolerance=args.tolerance,
            stagnation_iterations=args.stagnation,
        ),
        seed=args.seed,
    )

    exp = ExperimentConfig(
        objective=objective,
        pso_config=config,
        evaluator=evaluator,
        evaluator_name=args.evaluator,
        output_dir=args.output_dir,
    )

    result = run_experiment(exp)

    print(f"\n{'='*50}")
    print(f"Objetivo:        {args.objective}  d={args.dimension}")
    print(f"Evaluador:       {args.evaluator}")
    print(f"Mejor valor:     {result.best_value:.8e}")
    print(f"Mejor posición:  {result.best_position}")
    print(f"Iteraciones:     {len(result.history)}")
    if result.converged_iteration:
        print(f"Convergió en:    iteración {result.converged_iteration}")
    print(f"Tiempo total:    {result.total_time_s:.4f}s")
    print(f"  - evaluación:  {result.total_eval_time_s:.4f}s")
    print(f"  - actualiz.:   {result.total_update_time_s:.4f}s")
    if args.output_dir:
        print(f"Resultados en:   {args.output_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
