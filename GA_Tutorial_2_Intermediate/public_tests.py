"""
Public Test Cases for GA Tutorial 2 - Advanced Operators
These tests validate advanced genetic algorithm implementations.
"""

import numpy as np
import sys
sys.path.append('../GA_Tutorial_1_Basics')
from ga_utils_basics import initialize_population_real
from ga_utils_intermediate import *


def test_rank_based_selection():
    """Test rank-based selection."""
    print("Testing rank_based_selection...")

    np.random.seed(42)
    population = np.eye(5)
    fitness = np.array([1.0, 1000.0, 2.0, 3.0, 4.0])  # Extreme differences

    selected = rank_based_selection(population, fitness, num_parents=10, selection_pressure=1.5)

    assert selected.shape == (10, 5), f"Expected shape (10, 5), got {selected.shape}"

    # Check all selected are from original population
    for sel in selected:
        found = False
        for ind in population:
            if np.array_equal(sel, ind):
                found = True
                break
        assert found, "Selected individual not from original population"

    print("✓ Rank-based selection test passed!")


def test_stochastic_universal_sampling():
    """Test stochastic universal sampling."""
    print("Testing stochastic_universal_sampling...")

    np.random.seed(42)
    population = np.eye(4)
    fitness = np.array([10.0, 20.0, 30.0, 40.0])

    selected = stochastic_universal_sampling(population, fitness, num_parents=8)

    assert selected.shape == (8, 4), f"Expected shape (8, 4), got {selected.shape}"

    # Check all selected are from original population
    for sel in selected:
        found = False
        for ind in population:
            if np.array_equal(sel, ind):
                found = True
                break
        assert found, "Selected individual not from original population"

    print("✓ Stochastic universal sampling test passed!")


def test_boltzmann_selection():
    """Test Boltzmann selection."""
    print("Testing boltzmann_selection...")

    np.random.seed(42)
    population = np.eye(5)
    fitness = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    selected = boltzmann_selection(population, fitness, num_parents=5, temperature=1.0)

    assert selected.shape == (5, 5), f"Expected shape (5, 5), got {selected.shape}"

    print("✓ Boltzmann selection test passed!")


def test_arithmetic_crossover():
    """Test arithmetic crossover."""
    print("Testing arithmetic_crossover...")

    parent1 = np.array([1.0, 2.0, 3.0])
    parent2 = np.array([4.0, 5.0, 6.0])

    off1, off2 = arithmetic_crossover(parent1, parent2, alpha=0.5)

    # With alpha=0.5, offspring should be averages
    expected_avg = (parent1 + parent2) / 2
    assert np.allclose(off1, expected_avg), f"Expected {expected_avg}, got {off1}"
    assert np.allclose(off2, expected_avg), f"Expected {expected_avg}, got {off2}"

    # Test different alpha
    off1, off2 = arithmetic_crossover(parent1, parent2, alpha=0.3)
    expected1 = 0.3 * parent1 + 0.7 * parent2
    expected2 = 0.7 * parent1 + 0.3 * parent2
    assert np.allclose(off1, expected1), "Alpha=0.3 offspring1 incorrect"
    assert np.allclose(off2, expected2), "Alpha=0.3 offspring2 incorrect"

    print("✓ Arithmetic crossover test passed!")


def test_blx_alpha_crossover():
    """Test BLX-alpha crossover."""
    print("Testing blx_alpha_crossover...")

    parent1 = np.array([2.0, 2.0])
    parent2 = np.array([4.0, 4.0])
    bounds = (-10.0, 10.0)

    np.random.seed(42)
    off1, off2 = blx_alpha_crossover(parent1, parent2, alpha=0.5, bounds=bounds)

    # Check bounds
    assert np.all(off1 >= bounds[0]) and np.all(off1 <= bounds[1]), "Offspring1 out of bounds"
    assert np.all(off2 >= bounds[0]) and np.all(off2 <= bounds[1]), "Offspring2 out of bounds"

    # With alpha=0.5, range is [min-0.5*interval, max+0.5*interval]
    # For [2,4]: interval=2, so range should be [1.0, 5.0]
    # Check that offspring are in reasonable range (not exact due to randomness)
    assert np.all(off1 >= 0.5) and np.all(off1 <= 5.5), "Offspring1 outside expected range"

    print("✓ BLX-alpha crossover test passed!")


def test_simulated_binary_crossover():
    """Test simulated binary crossover."""
    print("Testing simulated_binary_crossover...")

    parent1 = np.array([2.0, 2.0])
    parent2 = np.array([4.0, 4.0])
    bounds = (-10.0, 10.0)

    np.random.seed(42)
    off1, off2 = simulated_binary_crossover(parent1, parent2, eta=20, bounds=bounds)

    # Check length
    assert len(off1) == len(parent1), "Offspring length mismatch"
    assert len(off2) == len(parent2), "Offspring length mismatch"

    # Check bounds
    assert np.all(off1 >= bounds[0]) and np.all(off1 <= bounds[1]), "Offspring1 out of bounds"
    assert np.all(off2 >= bounds[0]) and np.all(off2 <= bounds[1]), "Offspring2 out of bounds"

    print("✓ Simulated binary crossover test passed!")


def test_polynomial_mutation():
    """Test polynomial mutation."""
    print("Testing polynomial_mutation...")

    chromosome = np.array([0.0, 0.0, 0.0])
    bounds = (-5.0, 5.0)
    mutation_rate = 0.5

    np.random.seed(42)
    mutated = polynomial_mutation(chromosome, mutation_rate, bounds, eta=20)

    # Check length
    assert len(mutated) == len(chromosome), "Mutated length mismatch"

    # Check bounds
    assert np.all(mutated >= bounds[0]) and np.all(mutated <= bounds[1]), "Mutated out of bounds"

    # Original should not be modified
    assert np.array_equal(chromosome, np.array([0.0, 0.0, 0.0])), "Original modified"

    print("✓ Polynomial mutation test passed!")


def test_adaptive_mutation():
    """Test adaptive mutation."""
    print("Testing adaptive_mutation...")

    chromosome = np.array([0.0, 0.0])
    bounds = (-5.0, 5.0)
    max_gen = 100

    # Early generation - should have larger mutations
    np.random.seed(42)
    mutated_early = adaptive_mutation(chromosome, generation=10, max_generations=max_gen, bounds=bounds)

    # Late generation - should have smaller mutations
    np.random.seed(42)
    mutated_late = adaptive_mutation(chromosome, generation=90, max_generations=max_gen, bounds=bounds)

    # Check bounds
    assert np.all(mutated_early >= bounds[0]) and np.all(mutated_early <= bounds[1]), "Early mutation out of bounds"
    assert np.all(mutated_late >= bounds[0]) and np.all(mutated_late <= bounds[1]), "Late mutation out of bounds"

    print("✓ Adaptive mutation test passed!")


def test_self_adaptive_mutation():
    """Test self-adaptive mutation."""
    print("Testing self_adaptive_mutation...")

    chromosome = np.array([0.0, 0.0, 0.0])
    strategy = np.array([0.5, 0.5, 0.5])
    bounds = (-5.0, 5.0)

    np.random.seed(42)
    mutated_chrom, mutated_strategy = self_adaptive_mutation(chromosome, strategy, bounds)

    # Check lengths
    assert len(mutated_chrom) == len(chromosome), "Mutated chromosome length mismatch"
    assert len(mutated_strategy) == len(strategy), "Mutated strategy length mismatch"

    # Check bounds
    assert np.all(mutated_chrom >= bounds[0]) and np.all(mutated_chrom <= bounds[1]), "Mutated chromosome out of bounds"

    # Strategy should be positive
    assert np.all(mutated_strategy > 0), "Strategy parameters should be positive"

    print("✓ Self-adaptive mutation test passed!")


def test_elitist_replacement():
    """Test elitist replacement."""
    print("Testing elitist_replacement...")

    old_pop = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    old_fitness = np.array([50.0, 40.0, 100.0, 30.0, 90.0])  # Best: 100, 90

    new_pop = np.array([[6.0], [7.0], [8.0], [9.0], [10.0]])
    new_fitness = np.array([60.0, 50.0, 70.0, 40.0, 55.0])

    elite_size = 2

    result_pop, result_fitness = elitist_replacement(old_pop, old_fitness, new_pop.copy(), new_fitness.copy(), elite_size)

    # Best from old generation should be preserved
    assert np.max(result_fitness) >= np.max(old_fitness), "Best fitness not preserved"

    # Should still have same population size
    assert len(result_pop) == len(old_pop), "Population size changed"

    print("✓ Elitist replacement test passed!")


def test_steady_state_replacement():
    """Test steady-state replacement."""
    print("Testing steady_state_replacement...")

    population = np.array([[1.0], [2.0], [3.0]])
    fitness = np.array([10.0, 20.0, 30.0])

    # Better offspring - should replace worst
    offspring = np.array([4.0])
    offspring_fitness = 25.0

    new_pop, new_fitness = steady_state_replacement(population.copy(), fitness.copy(), offspring, offspring_fitness)

    # Check that worst was replaced
    assert np.min(new_fitness) > np.min(fitness), "Worst not replaced by better offspring"

    # Worse offspring - should not replace
    weak_offspring = np.array([0.0])
    weak_fitness = 5.0

    new_pop2, new_fitness2 = steady_state_replacement(new_pop.copy(), new_fitness.copy(), weak_offspring, weak_fitness)

    # Population should be unchanged
    assert np.array_equal(new_fitness2, new_fitness), "Population changed with worse offspring"

    print("✓ Steady-state replacement test passed!")


def test_penalty_function():
    """Test penalty function."""
    print("Testing penalty_function...")

    objective = 10.0

    # No violation
    constraints = [-1.0, -2.0]  # Negative = satisfied
    penalized = penalty_function(objective, constraints, penalty_coefficient=1000)
    assert penalized == objective, "Penalty applied when no violation"

    # With violation
    constraints_violated = [2.0, -1.0]  # First constraint violated
    penalized = penalty_function(objective, constraints_violated, penalty_coefficient=1000)
    assert penalized > objective, "No penalty applied for violation"
    assert penalized == objective + 1000 * 2.0, "Incorrect penalty calculation"

    print("✓ Penalty function test passed!")


def test_repair_bounds():
    """Test repair bounds."""
    print("Testing repair_bounds...")

    infeasible = np.array([-10.0, 3.0, 15.0])
    bounds = (-5.0, 5.0)

    repaired = repair_bounds(infeasible, bounds)

    # Check all values within bounds
    assert np.all(repaired >= bounds[0]) and np.all(repaired <= bounds[1]), "Repaired values out of bounds"

    # Check specific values
    assert repaired[0] == bounds[0], "Lower bound not clipped"
    assert repaired[1] == 3.0, "In-bounds value changed"
    assert repaired[2] == bounds[1], "Upper bound not clipped"

    print("✓ Repair bounds test passed!")


def test_crowding_distance():
    """Test crowding distance calculation."""
    print("Testing crowding_distance...")

    population = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    fitness = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    distances = crowding_distance(population, fitness)

    # Check length
    assert len(distances) == len(population), "Distance array length mismatch"

    # Boundary points should have infinite distance
    sorted_indices = np.argsort(fitness)
    assert np.isinf(distances[sorted_indices[0]]), "First point should have infinite distance"
    assert np.isinf(distances[sorted_indices[-1]]), "Last point should have infinite distance"

    # All distances should be non-negative
    assert np.all(distances >= 0), "Negative distances found"

    print("✓ Crowding distance test passed!")


def test_niching_selection():
    """Test niching selection."""
    print("Testing niching_selection...")

    # Two clusters
    cluster1 = np.array([[0.0, 0.0], [0.1, 0.1]])
    cluster2 = np.array([[5.0, 5.0], [5.1, 5.1]])
    population = np.vstack([cluster1, cluster2])

    fitness = np.array([100.0, 100.0, 100.0, 100.0])  # All equal

    np.random.seed(42)
    selected = niching_selection(population, fitness, num_parents=4, sigma_share=1.0)

    assert selected.shape == (4, 2), f"Expected shape (4, 2), got {selected.shape}"

    print("✓ Niching selection test passed!")


def test_adaptive_crossover_rate():
    """Test adaptive crossover rate."""
    print("Testing adaptive_crossover_rate...")

    max_gen = 100
    initial = 0.9
    final = 0.6

    # At generation 0, should be initial
    rate_0 = adaptive_crossover_rate(0, max_gen, initial, final)
    assert np.isclose(rate_0, initial), f"Expected {initial}, got {rate_0}"

    # At last generation, should be final
    rate_end = adaptive_crossover_rate(99, max_gen, initial, final)
    assert np.isclose(rate_end, final), f"Expected {final}, got {rate_end}"

    # At middle, should be between
    rate_mid = adaptive_crossover_rate(50, max_gen, initial, final)
    assert initial > rate_mid > final, "Middle rate should be between initial and final"

    print("✓ Adaptive crossover rate test passed!")


def test_adaptive_mutation_rate_fitness():
    """Test fitness-based adaptive mutation rate."""
    print("Testing adaptive_mutation_rate_fitness...")

    max_fit = 100.0
    min_fit = 10.0

    # Best individual should have low mutation
    rate_best = adaptive_mutation_rate_fitness(max_fit, max_fit, min_fit, k=2)
    rate_worst = adaptive_mutation_rate_fitness(min_fit, max_fit, min_fit, k=2)

    assert rate_best < rate_worst, "Best individual should have lower mutation rate than worst"
    assert 0 <= rate_best <= 1, "Mutation rate should be in [0, 1]"
    assert 0 <= rate_worst <= 1, "Mutation rate should be in [0, 1]"

    print("✓ Adaptive mutation rate (fitness-based) test passed!")


def run_all_tests():
    """Run all public tests."""
    print("\n" + "="*70)
    print("Running GA Tutorial 2 - Advanced Operators Tests")
    print("="*70 + "\n")

    try:
        test_rank_based_selection()
        test_stochastic_universal_sampling()
        test_boltzmann_selection()
        test_arithmetic_crossover()
        test_blx_alpha_crossover()
        test_simulated_binary_crossover()
        test_polynomial_mutation()
        test_adaptive_mutation()
        test_self_adaptive_mutation()
        test_elitist_replacement()
        test_steady_state_replacement()
        test_penalty_function()
        test_repair_bounds()
        test_crowding_distance()
        test_niching_selection()
        test_adaptive_crossover_rate()
        test_adaptive_mutation_rate_fitness()

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
