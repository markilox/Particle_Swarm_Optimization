"""Individual experiment runner.

Encapsulates PSO construction, execution, and optional result persistence.
"""

import logging
from pathlib import Path

from pso.core.pso import PSO
from pso.core.types import BoxBounds, OptimizationResult, PSOConfig
from pso.io.persistence import save_result
from pso.objectives.benchmarks import ObjectiveSpec
from pso.parallel.base import FitnessEvaluator


class ExperimentConfig:
    """Groups everything needed for a reproducible run."""

    def __init__(
        self,
        objective: ObjectiveSpec,
        pso_config: PSOConfig,
        evaluator: FitnessEvaluator,
        evaluator_name: str,
        output_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
        bounds: BoxBounds | None = None,
    ) -> None:
        self.objective = objective
        self.pso_config = pso_config
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logger
        self.bounds = bounds  # if None, defaults to objective.bounds(dimension)


def run_experiment(exp: ExperimentConfig, on_iteration=None) -> OptimizationResult:
    """Runs an experiment and, if output_dir is set, persists the result.

    Always calls exp.evaluator.close() in a finally block so worker pools
    (threading, multiprocessing) are released even if an exception occurs.
    """
    if exp.bounds is not None:
        lower, upper = exp.bounds.lower, exp.bounds.upper
    else:
        lower, upper = exp.objective.bounds(exp.pso_config.dimension)
    pso = PSO(
        objective=exp.objective,
        bounds=BoxBounds(lower=lower, upper=upper),
        config=exp.pso_config,
        evaluator=exp.evaluator,
        logger=exp.logger,
    )
    try:
        result = pso.optimize(on_iteration=on_iteration)
    finally:
        exp.evaluator.close()

    if exp.output_dir is not None:
        save_result(
            result=result,
            config=exp.pso_config,
            objective_name=exp.objective.name,
            evaluator_name=exp.evaluator_name,
            output_dir=exp.output_dir,
        )

    return result
