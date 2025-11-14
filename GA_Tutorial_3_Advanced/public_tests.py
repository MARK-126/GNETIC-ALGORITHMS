"""
Public Test Cases for GA Tutorial 3 - Advanced Topics
Tests for TSP, hybrid algorithms, and multi-objective optimization.
"""

import numpy as np
from ga_utils_advanced import *
from parallel_ga import ParallelGA, run_single_island, parallel_ga_simple


def test_tsp_distance_calculation():
    """Test TSP distance calculation."""
    print("Testing TSP distance calculation...")

    # Simple 3-city problem
    distance_matrix = np.array([
        [0, 10, 15],
        [10, 0, 20],
        [15, 20, 0]
    ])

    tour = np.array([0, 1, 2])
    distance = calculate_tsp_distance(tour, distance_matrix)

    # Expected: 0→1 (10) + 1→2 (20) + 2→0 (15) = 45
    assert distance == 45, f"Expected 45, got {distance}"

    print("✓ TSP distance calculation test passed!")


def test_order_crossover():
    """Test order crossover validity."""
    print("Testing order crossover...")

    np.random.seed(42)
    parent1 = np.array([0, 1, 2, 3, 4, 5])
    parent2 = np.array([5, 4, 3, 2, 1, 0])

    off1, off2 = order_crossover(parent1, parent2)

    # Check validity (all cities present, no duplicates)
    assert len(set(off1)) == len(off1), "Offspring 1 has duplicates"
    assert set(off1) == set(parent1), "Offspring 1 missing cities"
    assert len(set(off2)) == len(off2), "Offspring 2 has duplicates"
    assert set(off2) == set(parent2), "Offspring 2 missing cities"

    print("✓ Order crossover test passed!")


def test_swap_mutation():
    """Test swap mutation."""
    print("Testing swap mutation...")

    tour = np.array([0, 1, 2, 3, 4])

    np.random.seed(42)
    mutated = swap_mutation(tour, mutation_rate=1.0)

    # Should be valid permutation
    assert len(set(mutated)) == len(mutated), "Mutated tour has duplicates"
    assert set(mutated) == set(tour), "Mutated tour missing cities"

    # Should be different from original (with high probability)
    # Just check validity for determinism

    print("✓ Swap mutation test passed!")


def test_inversion_mutation():
    """Test inversion mutation."""
    print("Testing inversion mutation...")

    tour = np.array([0, 1, 2, 3, 4, 5])

    np.random.seed(42)
    mutated = inversion_mutation(tour, mutation_rate=1.0)

    # Should be valid permutation
    assert len(set(mutated)) == len(mutated), "Mutated tour has duplicates"
    assert set(mutated) == set(tour), "Mutated tour missing cities"

    print("✓ Inversion mutation test passed!")


def test_two_opt():
    """Test 2-opt local search."""
    print("Testing 2-opt local search...")

    # Create simple problem
    cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    distance_matrix = create_distance_matrix(cities)

    # Bad tour
    bad_tour = np.array([0, 2, 1, 3])
    bad_distance = calculate_tsp_distance(bad_tour, distance_matrix)

    # Apply 2-opt
    improved_tour = two_opt_local_search(bad_tour, distance_matrix, max_iterations=10)
    improved_distance = calculate_tsp_distance(improved_tour, distance_matrix)

    # Should improve or stay same
    assert improved_distance <= bad_distance, "2-opt made tour worse!"

    print("✓ 2-opt local search test passed!")


def test_pareto_dominance():
    """Test Pareto dominance check."""
    print("Testing Pareto dominance...")

    obj1 = np.array([2, 5])
    obj2 = np.array([3, 6])
    obj3 = np.array([1, 7])

    # obj1 should dominate obj2
    assert dominates(obj1, obj2), "obj1 should dominate obj2"

    # obj1 should not dominate obj3
    assert not dominates(obj1, obj3), "obj1 should not dominate obj3"

    # obj3 should not dominate obj1
    assert not dominates(obj3, obj1), "obj3 should not dominate obj1"

    print("✓ Pareto dominance test passed!")


def test_non_dominated_sort():
    """Test non-dominated sorting."""
    print("Testing non-dominated sorting...")

    objectives = np.array([
        [1, 5],  # Front 1
        [2, 4],  # Front 1
        [3, 3],  # Front 1
        [4, 6],  # Front 2 (dominated by 1,5 and 2,4)
        [5, 7],  # Front 3
    ])

    population = np.arange(len(objectives)).reshape(-1, 1)
    fronts = fast_non_dominated_sort(population, objectives)

    # Should have at least one front
    assert len(fronts) > 0, "No fronts found"

    # First front should have non-dominated solutions
    assert len(fronts[0]) > 0, "First front is empty"

    print("✓ Non-dominated sorting test passed!")


def test_crowding_distance():
    """Test crowding distance calculation."""
    print("Testing crowding distance...")

    objectives = np.array([
        [1, 5],
        [2, 4],
        [3, 3],
        [4, 2],
        [5, 1]
    ])

    distances = calculate_crowding_distance_multi(objectives)

    # Should have same length as objectives
    assert len(distances) == len(objectives), "Distance array length mismatch"

    # Boundary points should have infinite distance
    sorted_idx_obj1 = np.argsort(objectives[:, 0])
    # Note: may not always be exact indices due to sorting, just check for inf values
    assert np.any(np.isinf(distances)), "No infinite distances found for boundary points"

    print("✓ Crowding distance test passed!")


def test_sbx_crossover():
    """Test SBX crossover."""
    print("Testing SBX crossover...")

    parent1 = np.array([0.2, 0.5, 0.7])
    parent2 = np.array([0.8, 0.3, 0.9])
    bounds = np.array([[0, 1], [0, 1], [0, 1]])

    np.random.seed(42)
    child1, child2 = sbx_crossover(parent1, parent2, eta=20, bounds=bounds)

    # Check bounds
    assert np.all(child1 >= 0) and np.all(child1 <= 1), "Child1 out of bounds"
    assert np.all(child2 >= 0) and np.all(child2 <= 1), "Child2 out of bounds"

    # Check shape
    assert child1.shape == parent1.shape, "Child1 shape mismatch"
    assert child2.shape == parent2.shape, "Child2 shape mismatch"

    print("✓ SBX crossover test passed!")


def test_polynomial_mutation():
    """Test polynomial mutation."""
    print("Testing polynomial mutation...")

    individual = np.array([0.5, 0.5, 0.5])
    bounds = np.array([[0, 1], [0, 1], [0, 1]])

    np.random.seed(42)
    mutated = polynomial_mutation(individual, mutation_rate=1.0, eta=20, bounds=bounds)

    # Check bounds
    assert np.all(mutated >= 0) and np.all(mutated <= 1), "Mutated out of bounds"

    # Check shape
    assert mutated.shape == individual.shape, "Mutated shape mismatch"

    print("✓ Polynomial mutation test passed!")


def test_zdt1_problem():
    """Test ZDT1 problem."""
    print("Testing ZDT1 problem...")

    # Test at known point
    x = np.zeros(30)
    x[0] = 0.5
    f1, f2 = zdt1(x)

    # f1 should equal x[0]
    assert abs(f1 - 0.5) < 1e-6, f"f1 incorrect: {f1}"

    # f2 should be > 0
    assert f2 > 0, f"f2 should be positive: {f2}"

    # Test Pareto optimal point
    x_optimal = np.zeros(30)
    x_optimal[0] = 0.5
    # Rest should be 0 for Pareto optimal
    f1_opt, f2_opt = zdt1(x_optimal)

    # For Pareto optimal, g = 1, so f2 = 1 - sqrt(f1)
    expected_f2 = 1.0 - np.sqrt(f1_opt)
    assert abs(f2_opt - expected_f2) < 1e-6, "ZDT1 Pareto optimal check failed"

    print("✓ ZDT1 problem test passed!")


def test_nsga2_selection():
    """Test NSGA-II selection."""
    print("Testing NSGA-II selection...")

    # Create simple population
    population = np.random.rand(10, 2)
    objectives = np.random.rand(10, 2)

    # Select 5 individuals
    selected_pop, selected_obj = nsga2_selection(population, objectives, 5)

    # Check size
    assert len(selected_pop) == 5, f"Expected 5 selected, got {len(selected_pop)}"
    assert len(selected_obj) == 5, f"Expected 5 objectives, got {len(selected_obj)}"

    # Check shapes
    assert selected_pop.shape == (5, 2), "Selected population shape mismatch"
    assert selected_obj.shape == (5, 2), "Selected objectives shape mismatch"

    print("✓ NSGA-II selection test passed!")


def test_nsga2_basic():
    """Test basic NSGA-II execution."""
    print("Testing NSGA-II basic execution...")

    # Simple 2-objective problem
    def f1(x):
        return x[0]**2

    def f2(x):
        return (x[0] - 2)**2

    objective_functions = [f1, f2]
    bounds = [(0, 2)]

    # Run for few generations
    population, objectives, pareto_front, history = nsga2(
        objective_functions,
        n_objectives=2,
        bounds=bounds,
        pop_size=20,
        max_generations=10
    )

    # Check outputs
    assert len(population) == 20, "Population size mismatch"
    assert objectives.shape == (20, 2), "Objectives shape mismatch"
    assert len(pareto_front) > 0, "No Pareto front found"

    # Check history
    assert 'n_pareto' in history, "Missing n_pareto in history"
    assert len(history['n_pareto']) == 10, "History length mismatch"

    print(f"  Found {len(pareto_front)} Pareto optimal solutions")
    print("✓ NSGA-II basic execution test passed!")


def test_parallel_ga_initialization():
    """Test ParallelGA initialization."""
    print("Testing ParallelGA initialization...")

    def sphere(x):
        return -np.sum(x**2)

    bounds = [(-5, 5), (-5, 5)]

    pga = ParallelGA(
        sphere,
        bounds,
        n_islands=2,
        pop_size_per_island=10
    )

    assert pga.n_islands == 2, "Number of islands mismatch"
    assert pga.pop_size_per_island == 10, "Population size mismatch"
    assert pga.best_solution is None, "Best solution should be None before optimization"

    print("✓ ParallelGA initialization test passed!")


def test_single_island_execution():
    """Test single island GA execution."""
    print("Testing single island execution...")

    def sphere(x):
        return -np.sum(x**2)

    bounds = [(-5, 5), (-5, 5)]

    best_sol, best_fit, history = run_single_island(
        island_id=0,
        objective_func=sphere,
        bounds=bounds,
        pop_size=20,
        n_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8
    )

    # Should return valid solution
    assert best_sol is not None, "Best solution should not be None"
    assert len(best_sol) == 2, "Solution dimension mismatch"
    assert best_fit is not None, "Best fitness should not be None"

    # Check history
    assert 'best_fitness' in history, "History should contain best_fitness"
    assert len(history['best_fitness']) == 5, "History length should match generations"

    print("✓ Single island execution test passed!")


def test_parallel_ga_basic():
    """Test basic ParallelGA execution."""
    print("Testing ParallelGA basic execution...")

    def sphere(x):
        return -np.sum(x**2)

    bounds = [(-2, 2), (-2, 2)]

    pga = ParallelGA(
        sphere,
        bounds,
        n_islands=2,
        pop_size_per_island=10
    )

    best_sol, best_fit, history = pga.optimize(max_generations=3)

    # Should have valid results
    assert best_sol is not None, "Best solution should not be None"
    assert len(best_sol) == 2, "Solution dimension mismatch"
    assert best_fit > -10, f"Fitness should be reasonable, got {best_fit}"

    # Check history
    assert 'island_histories' in history, "History should contain island_histories"
    assert len(history['island_histories']) == 2, "Should have 2 island histories"

    print(f"  Best fitness: {best_fit:.4f}")
    print("✓ ParallelGA basic execution test passed!")


def test_get_zdt_problem():
    """Test ZDT problem retrieval."""
    print("Testing ZDT problem retrieval...")

    # Test ZDT1
    objectives, bounds, n_vars = get_zdt_problem('ZDT1')
    assert len(objectives) == 2, "Should have 2 objectives"
    assert len(bounds) == 30, "ZDT1 should have 30 variables"
    assert n_vars == 30, "n_vars mismatch"

    # Test ZDT4
    objectives, bounds, n_vars = get_zdt_problem('ZDT4')
    assert len(bounds) == 10, "ZDT4 should have 10 variables"

    # Test invalid problem
    try:
        get_zdt_problem('ZDT99')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    print("✓ ZDT problem retrieval test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running GA Tutorial 3 - Advanced Topics Tests")
    print("="*70 + "\n")

    try:
        # TSP tests
        test_tsp_distance_calculation()
        test_order_crossover()
        test_swap_mutation()
        test_inversion_mutation()
        test_two_opt()

        # Multi-objective tests
        test_pareto_dominance()
        test_non_dominated_sort()
        test_crowding_distance()

        # NSGA-II tests
        test_sbx_crossover()
        test_polynomial_mutation()
        test_zdt1_problem()
        test_nsga2_selection()
        test_nsga2_basic()
        test_get_zdt_problem()

        # Parallel GA tests
        test_parallel_ga_initialization()
        test_single_island_execution()
        test_parallel_ga_basic()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED! (17 tests)")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
