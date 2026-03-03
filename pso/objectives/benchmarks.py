import numpy as np


class ObjectiveSpec:
    def __init__(self, name: str, fn, lower_bound: float, upper_bound: float) -> None:
        self.name = name
        self.fn = fn
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x: np.ndarray) -> float:
        return float(self.fn(x))

    def bounds(self, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(dimension, self.lower_bound, dtype=float)
        upper = np.full(dimension, self.upper_bound, dtype=float)
        return lower, upper


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def ackley(x: np.ndarray) -> float:
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    d = x.size
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e)


OBJECTIVES: dict[str, ObjectiveSpec] = {
    "sphere": ObjectiveSpec(name="sphere", fn=sphere, lower_bound=-5.12, upper_bound=5.12),
    "rastrigin": ObjectiveSpec(name="rastrigin", fn=rastrigin, lower_bound=-5.12, upper_bound=5.12),
    "rosenbrock": ObjectiveSpec(name="rosenbrock", fn=rosenbrock, lower_bound=-2.048, upper_bound=2.048),
    "ackley": ObjectiveSpec(name="ackley", fn=ackley, lower_bound=-32.768, upper_bound=32.768),
}


def get_objective(name: str) -> ObjectiveSpec:
    key = name.strip().lower()
    if key not in OBJECTIVES:
        available = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVES[key]


def list_objectives() -> list[str]:
    return sorted(OBJECTIVES.keys())
