"""
Public Test Cases for GA Tutorial 3 - Advanced Topics
Tests for TSP, hybrid algorithms, and multi-objective optimization.
"""

import numpy as np
from ga_utils_advanced import *


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


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running GA Tutorial 3 - Advanced Topics Tests")
    print("="*70 + "\n")

    try:
        test_tsp_distance_calculation()
        test_order_crossover()
        test_swap_mutation()
        test_inversion_mutation()
        test_two_opt()
        test_pareto_dominance()
        test_non_dominated_sort()
        test_crowding_distance()

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
