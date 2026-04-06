"""Tests unitarios del núcleo PSO.

Cubre los cuatro requisitos explícitos del enunciado:
  1. Reproducibilidad por seed.
  2. Manejo de límites (clamp).
  3. Evolución monótona del mejor global (no debe empeorar).
  4. Correctitud básica en Sphere (debe converger a ~0).

Además incluye tests de contrato de las interfaces principales.
"""

import numpy as np

from pso.core.bounds import ClampBoundsPolicy
from pso.core.pso import PSO
from pso.core.types import BoxBounds, PSOConfig, StopCriteria
from pso.objectives.benchmarks import get_objective
from pso.parallel import get_evaluator


# ---------------------------------------------------------------------------
# Fixtures comunes
# ---------------------------------------------------------------------------

def make_pso(objective_name="sphere", dimension=2, swarm_size=20, iterations=100, seed=42, evaluator_name="sequential"):
    objective = get_objective(objective_name)
    lower, upper = objective.bounds(dimension)
    config = PSOConfig(
        dimension=dimension,
        swarm_size=swarm_size,
        inertia_weight=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        stop=StopCriteria(max_iterations=iterations),
        seed=seed,
    )
    return PSO(
        objective=objective,
        bounds=BoxBounds(lower=lower, upper=upper),
        config=config,
        evaluator=get_evaluator(evaluator_name),
    )


# ---------------------------------------------------------------------------
# 1. Reproducibilidad por seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        """Dos ejecuciones con la misma seed deben producir exactamente el mismo resultado."""
        result_a = make_pso(seed=42).optimize()
        result_b = make_pso(seed=42).optimize()
        assert result_a.best_value == result_b.best_value
        np.testing.assert_array_equal(result_a.best_position, result_b.best_position)

    def test_same_seed_same_history(self):
        """El historial de fitness por iteración debe ser idéntico con la misma seed."""
        result_a = make_pso(seed=7).optimize()
        result_b = make_pso(seed=7).optimize()
        fitnesses_a = [m.best_fitness for m in result_a.history]
        fitnesses_b = [m.best_fitness for m in result_b.history]
        assert fitnesses_a == fitnesses_b

    def test_different_seeds_different_results(self):
        """Seeds distintas deben producir resultados distintos (con alta probabilidad)."""
        result_a = make_pso(seed=1).optimize()
        result_b = make_pso(seed=2).optimize()
        assert result_a.best_value != result_b.best_value


# ---------------------------------------------------------------------------
# 2. Manejo de límites
# ---------------------------------------------------------------------------

class TestBounds:
    def test_clamp_position_inside_bounds(self):
        """ClampBoundsPolicy debe devolver posición dentro de [lower, upper]."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([7.0, -8.0])   # fuera de bounds
        velocity = np.array([1.0, -1.0])
        clipped, _ = ClampBoundsPolicy.apply(position, velocity, bounds)
        assert np.all(clipped >= bounds.lower)
        assert np.all(clipped <= bounds.upper)

    def test_clamp_zeroes_velocity_on_hit(self):
        """La velocidad en las dimensiones que chocan con el límite debe ser 0."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([7.0, 3.0])    # solo x sale del límite
        velocity = np.array([2.0, 1.0])
        _, new_vel = ClampBoundsPolicy.apply(position, velocity, bounds)
        assert new_vel[0] == 0.0           # x chocó → velocidad a 0
        assert new_vel[1] == 1.0           # y no chocó → velocidad intacta

    def test_position_inside_bounds_unchanged(self):
        """Posición ya dentro de los límites no debe modificarse."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([1.0, -2.0])
        velocity = np.array([0.5, 0.5])
        clipped, new_vel = ClampBoundsPolicy.apply(position, velocity, bounds)
        np.testing.assert_array_equal(clipped, position)
        np.testing.assert_array_equal(new_vel, velocity)

    def test_particles_stay_in_bounds_after_optimize(self):
        """Todas las partículas deben estar dentro de los límites al terminar."""
        objective = get_objective("sphere")
        lower, upper = objective.bounds(2)
        bounds = BoxBounds(lower=lower, upper=upper)
        config = PSOConfig(dimension=2, swarm_size=30, stop=StopCriteria(max_iterations=50), seed=0)
        pso = PSO(objective=objective, bounds=bounds, config=config)

        snapshots = []
        result = pso.optimize(on_iteration=lambda it, pos, gbp, gbv: snapshots.append(pos))

        for positions in snapshots:
            assert np.all(positions >= lower)
            assert np.all(positions <= upper)


# ---------------------------------------------------------------------------
# 3. Monotonicidad del mejor global
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_global_best_never_worsens(self):
        """El mejor fitness global no debe aumentar en ninguna iteración."""
        result = make_pso(objective_name="sphere", dimension=5, iterations=200, seed=42).optimize()
        fitnesses = [m.best_fitness for m in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-15, (
                f"El mejor fitness empeoró en iteración {i + 1}: "
                f"{fitnesses[i - 1]:.6e} → {fitnesses[i]:.6e}"
            )

    def test_global_best_never_worsens_rastrigin(self):
        """Monotonicidad también en Rastrigin (multimodal)."""
        result = make_pso(objective_name="rastrigin", dimension=5, iterations=150, seed=99).optimize()
        fitnesses = [m.best_fitness for m in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-15


# ---------------------------------------------------------------------------
# 4. Correctitud en Sphere
# ---------------------------------------------------------------------------

class TestSphereConvergence:
    def test_sphere_converges_d2(self):
        """PSO debe converger cerca de 0 en Sphere d=2 con parámetros razonables."""
        result = make_pso(objective_name="sphere", dimension=2, swarm_size=30, iterations=300, seed=42).optimize()
        assert result.best_value < 1e-6, f"Sphere d=2 no convergió: {result.best_value:.4e}"

    def test_sphere_converges_d10(self):
        """PSO debe converger cerca de 0 en Sphere d=10."""
        result = make_pso(objective_name="sphere", dimension=10, swarm_size=50, iterations=500, seed=42).optimize()
        assert result.best_value < 1e-4, f"Sphere d=10 no convergió: {result.best_value:.4e}"

    def test_sphere_best_position_near_zero(self):
        """La mejor posición encontrada debe estar cerca del óptimo (0, 0)."""
        result = make_pso(objective_name="sphere", dimension=2, swarm_size=30, iterations=300, seed=42).optimize()
        assert np.linalg.norm(result.best_position) < 1e-3


