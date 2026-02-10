import numpy as np

class Objective:
    name: str



# The __call__ method allows instances of the Objective classes to be called like functions.

class Sphere(Objective):

    def __init__(self):
        self.name = "Sphere"

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x**2)

class Rastrigin(Objective):

    def __init__(self):
        self.name = "Rastrigin"

    def __call__(self, x: np.ndarray) -> float:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

class Rosenbrock(Objective):

    def __init__(self):
        self.name = "Rosenbrock"

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

class Ackley(Objective):

    def __init__(self):
        self.name = "Ackley"

    def __call__(self, x: np.ndarray) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

#Function which recieves the name of the objective and returns the corresponding objective function
def get_objective(name: str) -> Objective:
    if name == "Sphere":
        return Sphere()
    elif name == "Rastrigin":
        return Rastrigin()
    elif name == "Rosenbrock":
        return Rosenbrock()
    elif name == "Ackley":
        return Ackley()
    else:
        raise ValueError(f"Objective function '{name}' not recognized.")