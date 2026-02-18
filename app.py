# Quiero que llames a las clases core y benchmarks para probar que todo funciona correctamente
import numpy as np

# importa el módulo de funciones objetivo y la clase PSO real
from pso.objectives import benchmarks
from pso.core.pso import PSO

if __name__ == "__main__":
    # Prueba de las funciones objetivo usando el factory actual
    for name in ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"]:
        obj = benchmarks.get_objective(name)
        x = np.array([0.0, 0.0])
        print(f"{name} at {x}: {obj(x):.6f}")   # invoca __call__

    # Prueba del PSO (la clase PSO todavía es esqueleto, pero al menos se instancia)
    pso = PSO(objective_name="Rastrigin", num_particles=30, num_iterations=100)
    pso.optimize()