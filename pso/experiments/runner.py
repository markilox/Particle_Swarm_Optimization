"""Runner de experimentos individuales.

Encapsula la construcción del PSO, la ejecución y el guardado opcional
de resultados. Mantiene la lógica de orquestación fuera del core.
"""

import logging
from pathlib import Path

from pso.core.pso import PSO
from pso.core.types import BoxBounds, OptimizationResult, PSOConfig
from pso.io.persistence import save_result
from pso.objectives.benchmarks import ObjectiveSpec
from pso.parallel.base import FitnessEvaluator


class ExperimentConfig:
    """Agrupa todo lo necesario para una ejecución reproducible."""

    def __init__(
        self,
        objective: ObjectiveSpec,
        pso_config: PSOConfig,
        evaluator: FitnessEvaluator,
        evaluator_name: str,
        output_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.objective = objective
        self.pso_config = pso_config
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logger


def run_experiment(exp: ExperimentConfig) -> OptimizationResult:
    """Ejecuta un experimento y, si output_dir está definido, persiste el resultado."""
    lower, upper = exp.objective.bounds(exp.pso_config.dimension)
    pso = PSO(
        objective=exp.objective,
        bounds=BoxBounds(lower=lower, upper=upper),
        config=exp.pso_config,
        evaluator=exp.evaluator,
        logger=exp.logger,
    )
    result = pso.optimize()

    if exp.output_dir is not None:
        save_result(
            result=result,
            config=exp.pso_config,
            objective_name=exp.objective.name,
            evaluator_name=exp.evaluator_name,
            output_dir=exp.output_dir,
        )

    return result
