"""
Public Test Cases for GA Tutorial 5 - Specialized Algorithms
Tests for CMA-ES, Differential Evolution, and PSO implementations.
"""

import numpy as np
from ga_utils_specialized import *


def test_cma_es_initialization():
    """Test CMA-ES initialization."""
    print("Testing CMA-ES initialization...")

    cma = CMA_ES(dim=5, sigma=0.5)

    assert cma.dim == 5, "Dimension should be 5"
    assert cma.sigma == 0.5, "Initial sigma should be 0.5"
    assert cma.mean.shape == (5,), "Mean should be 5D vector"
    assert cma.C.shape == (5, 5), "Covariance should be 5x5"

    print("✓ CMA-ES initialization test passed!")


def test_cma_es_ask_tell():
    """Test CMA-ES ask-tell interface."""
    print("Testing CMA-ES ask-tell interface...")

    cma = CMA_ES(dim=3, sigma=0.5, pop_size=10)

    # Ask for solutions
    solutions = cma.ask()
    assert solutions.shape == (10, 3), f"Expected (10, 3), got {solutions.shape}"

    # Evaluate (simple sphere function)
    fitness = np.array([np.sum(x**2) for x in solutions])

    # Tell results
    cma.tell(solutions, fitness)

    # Mean should have moved
    assert not np.allclose(cma.mean, np.zeros(3)), "Mean should have changed"

    print("✓ CMA-ES ask-tell test passed!")


def test_differential_evolution():
    """Test Differential Evolution on simple problem."""
    print("Testing Differential Evolution...")

    # Simple 2D sphere
    def sphere_2d(x):
        return np.sum(x**2)

    bounds = [(-5, 5), (-5, 5)]
    de = DifferentialEvolution(bounds=bounds, pop_size=20, F=0.8, CR=0.9)

    # Run a few generations
    best_x, best_f, history = de.optimize(sphere_2d, max_generations=50)

    # Should improve from initial random
    assert best_f < 10.0, f"Should find good solution, got {best_f}"
    assert len(history['best_fitness']) == 50, "Should have 50 generations of history"
    assert history['best_fitness'][-1] <= history['best_fitness'][0], "Fitness should improve or stay same"

    print(f"  Final best fitness: {best_f:.6f}")
    print("✓ Differential Evolution test passed!")


def test_pso():
    """Test Particle Swarm Optimization."""
    print("Testing PSO...")

    # Simple 2D sphere
    def sphere_2d(x):
        return np.sum(x**2)

    bounds = [(-5, 5), (-5, 5)]
    pso = ParticleSwarmOptimizer(n_particles=20, dim=2, bounds=bounds)

    # Run optimization
    best_x, best_f, history = pso.optimize(sphere_2d, max_iterations=50)

    # Should improve
    assert best_f < 10.0, f"Should find good solution, got {best_f}"
    assert len(history['best_fitness']) == 50, "Should have 50 iterations of history"
    assert history['best_fitness'][-1] <= history['best_fitness'][0], "Fitness should improve"

    print(f"  Final best fitness: {best_f:.6f}")
    print(f"  Best position: {best_x}")
    print("✓ PSO test passed!")


def test_de_mutation():
    """Test DE mutation operator."""
    print("Testing DE mutation...")

    bounds = [(-5, 5)] * 3
    de = DifferentialEvolution(bounds=bounds, pop_size=10)

    # Test mutation
    mutant = de.mutate(idx=0)

    assert mutant.shape == (3,), "Mutant should be 3D"
    assert np.all(mutant >= -5) and np.all(mutant <= 5), "Mutant should be within bounds"

    print("✓ DE mutation test passed!")


def test_de_crossover():
    """Test DE crossover operator."""
    print("Testing DE crossover...")

    bounds = [(-5, 5)] * 3
    de = DifferentialEvolution(bounds=bounds, pop_size=10)

    target = np.array([1.0, 2.0, 3.0])
    mutant = np.array([-1.0, -2.0, -3.0])

    trial = de.crossover(target, mutant)

    assert trial.shape == (3,), "Trial should be 3D"
    # Trial should have elements from both target and mutant
    assert not (np.array_equal(trial, target) or np.array_equal(trial, mutant)), \
        "Trial should mix target and mutant"

    print("✓ DE crossover test passed!")


def test_benchmark_functions():
    """Test benchmark function implementations."""
    print("Testing benchmark functions...")

    # Test at known optima
    x_zero = np.zeros(5)

    # Sphere at origin should be 0
    assert sphere(x_zero) == 0.0, "Sphere at origin should be 0"

    # Rastrigin at origin should be 0
    assert np.isclose(rastrigin(x_zero), 0.0), "Rastrigin at origin should be ~0"

    # Ackley at origin should be 0
    assert np.isclose(ackley(x_zero), 0.0, atol=1e-10), "Ackley at origin should be ~0"

    # Rosenbrock at (1,1,1,1,1) should be 0
    x_ones = np.ones(5)
    assert np.isclose(rosenbrock(x_ones), 0.0), "Rosenbrock at (1,1,...) should be ~0"

    print("✓ Benchmark functions test passed!")


def test_pso_velocity_update():
    """Test PSO velocity boundaries."""
    print("Testing PSO velocity updates...")

    bounds = [(-5, 5)] * 2
    pso = ParticleSwarmOptimizer(n_particles=10, dim=2, bounds=bounds)

    # Evaluate initial fitness
    initial_fitness = np.array([sphere(p) for p in pso.positions])

    # Update
    pso.update(initial_fitness)

    # Velocities should be bounded
    v_max = 0.2 * (pso.bounds[:, 1] - pso.bounds[:, 0])
    assert np.all(np.abs(pso.velocities) <= v_max), "Velocities should be clamped"

    # Positions should be within bounds
    assert np.all(pso.positions >= pso.bounds[:, 0]), "Positions should be >= lower bound"
    assert np.all(pso.positions <= pso.bounds[:, 1]), "Positions should be <= upper bound"

    print("✓ PSO velocity update test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running GA Tutorial 5 - Specialized Algorithms Tests")
    print("="*70 + "\n")

    try:
        test_cma_es_initialization()
        test_cma_es_ask_tell()
        test_differential_evolution()
        test_pso()
        test_de_mutation()
        test_de_crossover()
        test_benchmark_functions()
        test_pso_velocity_update()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
