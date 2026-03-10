import logging

from parallel.sequential import SequentialEvaluator
from pso.core.pso import BoxBounds, PSO, PSOConfig, StopCriteria
from pso.objectives.benchmarks import get_objective


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    objective = get_objective("rastrigin")
    dimension = 10
    lower, upper = objective.bounds(dimension)

    pso = PSO(
        objective=objective,
        bounds=BoxBounds(lower=lower, upper=upper),
        config=PSOConfig(
            dimension=dimension,
            swarm_size=40,
            inertia_weight=0.7,
            cognitive_weight=1.5,
            social_weight=1.5,
            stop=StopCriteria(max_iterations=150, tolerance=1e-8, stagnation_iterations=30, min_delta=1e-12),
            seed=42,
        ),
        evaluator=SequentialEvaluator(),
    )
    result = pso.optimize()
    print(f"Best value: {result.best_value:.8f}")
    print(f"Best position: {result.best_position}")
    print(f"Iterations executed: {len(result.history)}")
    print(f"Total time (s): {result.total_time_s:.6f}")


if __name__ == "__main__":
    main()
