import numpy as np

from pso.core.bounds import ClampBoundsPolicy
from pso.core.pso import PSO
from pso.core.types import BoxBounds, PSOConfig, StopCriteria
from pso.objectives.benchmarks import get_objective
from pso.parallel import get_evaluator


# ---------------------------------------------------------------------------
# Common fixtures
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
# 1. Reproducibility by seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        """Two runs with the same seed must produce exactly the same result."""
        result_a = make_pso(seed=42).optimize()
        result_b = make_pso(seed=42).optimize()
        assert result_a.best_value == result_b.best_value
        np.testing.assert_array_equal(result_a.best_position, result_b.best_position)

    def test_same_seed_same_history(self):
        """The per-iteration fitness history must be identical with the same seed."""
        result_a = make_pso(seed=7).optimize()
        result_b = make_pso(seed=7).optimize()
        fitnesses_a = [m.best_fitness for m in result_a.history]
        fitnesses_b = [m.best_fitness for m in result_b.history]
        assert fitnesses_a == fitnesses_b

    def test_different_seeds_different_results(self):
        """Different seeds must produce different results (with high probability)."""
        result_a = make_pso(seed=1).optimize()
        result_b = make_pso(seed=2).optimize()
        assert result_a.best_value != result_b.best_value


# ---------------------------------------------------------------------------
# 2. Bounds handling
# ---------------------------------------------------------------------------

class TestBounds:
    def test_clamp_position_inside_bounds(self):
        """ClampBoundsPolicy must return a position within [lower, upper]."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([7.0, -8.0])   # outside bounds
        velocity = np.array([1.0, -1.0])
        clipped, _ = ClampBoundsPolicy.apply(position, velocity, bounds)
        assert np.all(clipped >= bounds.lower)
        assert np.all(clipped <= bounds.upper)

    def test_clamp_zeroes_velocity_on_hit(self):
        """Velocity must be zeroed in dimensions where the particle hit a boundary."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([7.0, 3.0])    # only x is out of bounds
        velocity = np.array([2.0, 1.0])
        _, new_vel = ClampBoundsPolicy.apply(position, velocity, bounds)
        assert new_vel[0] == 0.0           # x hit boundary → velocity zeroed
        assert new_vel[1] == 1.0           # y did not hit → velocity unchanged

    def test_position_inside_bounds_unchanged(self):
        """A position already within bounds must not be modified."""
        bounds = BoxBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))
        position = np.array([1.0, -2.0])
        velocity = np.array([0.5, 0.5])
        clipped, new_vel = ClampBoundsPolicy.apply(position, velocity, bounds)
        np.testing.assert_array_equal(clipped, position)
        np.testing.assert_array_equal(new_vel, velocity)

    def test_particles_stay_in_bounds_after_optimize(self):
        """All particles must remain within bounds throughout the optimization."""
        objective = get_objective("sphere")
        lower, upper = objective.bounds(2)
        bounds = BoxBounds(lower=lower, upper=upper)
        config = PSOConfig(dimension=2, swarm_size=30, stop=StopCriteria(max_iterations=50), seed=0)
        pso = PSO(objective=objective, bounds=bounds, config=config)

        snapshots = []
        pso.optimize(on_iteration=lambda it, pos, gbp, gbv: snapshots.append(pos))

        for positions in snapshots:
            assert np.all(positions >= lower)
            assert np.all(positions <= upper)


# ---------------------------------------------------------------------------
# 3. Monotonicity of the global best
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_global_best_never_worsens(self):
        """The global best fitness must never increase between iterations."""
        result = make_pso(objective_name="sphere", dimension=5, iterations=200, seed=42).optimize()
        fitnesses = [m.best_fitness for m in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-15, (
                f"Global best worsened at iteration {i + 1}: "
                f"{fitnesses[i - 1]:.6e} → {fitnesses[i]:.6e}"
            )

    def test_global_best_never_worsens_rastrigin(self):
        """Monotonicity must also hold on Rastrigin (multimodal)."""
        result = make_pso(objective_name="rastrigin", dimension=5, iterations=150, seed=99).optimize()
        fitnesses = [m.best_fitness for m in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-15


# ---------------------------------------------------------------------------
# 4. Correctness on Sphere
# ---------------------------------------------------------------------------

class TestSphereConvergence:
    def test_sphere_converges_d2(self):
        """PSO must converge near 0 on Sphere d=2 with reasonable parameters."""
        result = make_pso(objective_name="sphere", dimension=2, swarm_size=30, iterations=300, seed=42).optimize()
        assert result.best_value < 1e-6, f"Sphere d=2 did not converge: {result.best_value:.4e}"

    def test_sphere_converges_d10(self):
        """PSO must converge near 0 on Sphere d=10."""
        result = make_pso(objective_name="sphere", dimension=10, swarm_size=50, iterations=500, seed=42).optimize()
        assert result.best_value < 1e-4, f"Sphere d=10 did not converge: {result.best_value:.4e}"

    def test_sphere_best_position_near_zero(self):
        """The best position found must be close to the optimum (0, 0)."""
        result = make_pso(objective_name="sphere", dimension=2, swarm_size=30, iterations=300, seed=42).optimize()
        assert np.linalg.norm(result.best_position) < 1e-3
