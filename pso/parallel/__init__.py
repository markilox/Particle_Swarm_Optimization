from pso.parallel.sequential import SequentialEvaluator
from pso.parallel.threading_evaluator import ThreadingEvaluator
from pso.parallel.multiprocessing_evaluator import MultiprocessingEvaluator
from pso.parallel.asyncio_evaluator import AsyncioEvaluator
from pso.parallel.base import FitnessEvaluator

__all__ = [
    "SequentialEvaluator",
    "ThreadingEvaluator",
    "MultiprocessingEvaluator",
    "AsyncioEvaluator",
    "FitnessEvaluator",
    "get_evaluator",
    "EVALUATOR_NAMES",
]

EVALUATOR_NAMES = ["sequential", "threading", "multiprocessing", "asyncio"]


def get_evaluator(name: str, **kwargs) -> FitnessEvaluator:
    """Returns an evaluator instance by name.

    kwargs are forwarded to the chosen evaluator's constructor:
      - threading/multiprocessing: max_workers (int)
      - multiprocessing:           chunksize (int)
      - asyncio:                   latency_s (float), jitter (float),
                                   strategy (str), n_workers (int), seed (int)
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
    if name == "asyncio":
        return AsyncioEvaluator(
            latency_s=kwargs.get("latency_s", 0.0),
            jitter=kwargs.get("jitter", 0.0),
            strategy=kwargs.get("strategy", "gather"),
            n_workers=kwargs.get("n_workers", 4),
            seed=kwargs.get("seed"),
        )
    available = ", ".join(EVALUATOR_NAMES)
    raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
