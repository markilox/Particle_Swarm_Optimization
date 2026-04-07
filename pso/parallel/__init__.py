from pso.parallel.sequential import SequentialEvaluator
from pso.parallel.threading_evaluator import ThreadingEvaluator
from pso.parallel.multiprocessing_evaluator import MultiprocessingEvaluator
from pso.parallel.base import FitnessEvaluator

__all__ = [
    "SequentialEvaluator",
    "ThreadingEvaluator",
    "MultiprocessingEvaluator",
    "FitnessEvaluator",
    "get_evaluator",
    "EVALUATOR_NAMES",
]

EVALUATOR_NAMES = ["sequential", "threading", "multiprocessing"]


def get_evaluator(name: str, **kwargs) -> FitnessEvaluator:
    """Returns an evaluator instance by name.

    kwargs are forwarded to the chosen evaluator's constructor:
      - threading/multiprocessing: max_workers (int)
      - multiprocessing:           chunksize (int)
    """
    name = name.strip().lower()
    if name == "sequential":
        return SequentialEvaluator()
    if name == "threading":
        return ThreadingEvaluator(max_workers=kwargs.get("max_workers"))
    if name == "multiprocessing":
        return MultiprocessingEvaluator(
            max_workers=kwargs.get("max_workers"),
            chunksize=kwargs.get("chunksize", 1),
        )
    available = ", ".join(EVALUATOR_NAMES)
    raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
