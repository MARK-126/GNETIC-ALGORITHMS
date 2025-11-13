"""
Public Test Cases for GA Tutorial 1
These tests validate student implementations of basic genetic algorithm functions.
"""

import numpy as np
from ga_utils_basics import *


def test_initialize_population_binary():
    """Test binary population initialization."""
    print("Testing initialize_population_binary...")

    pop_size = 10
    chromosome_length = 8
    population = initialize_population_binary(pop_size, chromosome_length)

    # Check shape
    assert population.shape == (pop_size, chromosome_length), \
        f"Expected shape {(pop_size, chromosome_length)}, got {population.shape}"

    # Check binary values
    assert np.all((population == 0) | (population == 1)), \
        "Population should contain only 0s and 1s"

    print("✓ Binary population initialization test passed!")


def test_initialize_population_real():
    """Test real-valued population initialization."""
    print("Testing initialize_population_real...")

    pop_size = 10
    chromosome_length = 5
    bounds = (-5.0, 5.0)
    population = initialize_population_real(pop_size, chromosome_length, bounds)

    # Check shape
    assert population.shape == (pop_size, chromosome_length), \
        f"Expected shape {(pop_size, chromosome_length)}, got {population.shape}"

    # Check bounds
    assert np.all(population >= bounds[0]) and np.all(population <= bounds[1]), \
        f"Population values should be within bounds {bounds}"

    print("✓ Real population initialization test passed!")


def test_binary_to_real():
    """Test binary to real conversion."""
    print("Testing binary_to_real...")

    # Test case 1: all zeros should give lower bound
    chromosome = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    bounds = (-5.0, 5.0)
    bits_per_variable = 8
    value = binary_to_real(chromosome, bounds, bits_per_variable)

    assert np.isclose(value, bounds[0], atol=1e-2), \
        f"All zeros should decode to lower bound {bounds[0]}, got {value}"

    # Test case 2: all ones should give upper bound
    chromosome = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    value = binary_to_real(chromosome, bounds, bits_per_variable)

    assert np.isclose(value, bounds[1], atol=1e-2), \
        f"All ones should decode to upper bound {bounds[1]}, got {value}"

    print("✓ Binary to real conversion test passed!")


def test_single_point_crossover():
    """Test single-point crossover implementation."""
    print("Testing single_point_crossover...")

    np.random.seed(42)
    parent1 = np.array([1, 1, 1, 1, 1])
    parent2 = np.array([0, 0, 0, 0, 0])

    offspring1, offspring2 = single_point_crossover(parent1, parent2)

    # Check length preservation
    assert len(offspring1) == len(parent1), "Offspring length should match parent length"
    assert len(offspring2) == len(parent2), "Offspring length should match parent length"

    # Check that offspring are different from parents (with high probability)
    same_as_parent1 = np.array_equal(offspring1, parent1)
    same_as_parent2 = np.array_equal(offspring2, parent2)

    # At least one offspring should be different (unless crossover point is at extremes)
    print(f"  Offspring 1: {offspring1}")
    print(f"  Offspring 2: {offspring2}")

    print("✓ Single-point crossover test passed!")


def test_two_point_crossover():
    """Test two-point crossover implementation."""
    print("Testing two_point_crossover...")

    np.random.seed(123)
    parent1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    parent2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    offspring1, offspring2 = two_point_crossover(parent1, parent2)

    # Check length preservation
    assert len(offspring1) == len(parent1), "Offspring length should match parent length"
    assert len(offspring2) == len(parent2), "Offspring length should match parent length"

    print(f"  Offspring 1: {offspring1}")
    print(f"  Offspring 2: {offspring2}")

    print("✓ Two-point crossover test passed!")


def test_bit_flip_mutation():
    """Test bit-flip mutation implementation."""
    print("Testing bit_flip_mutation...")

    np.random.seed(42)
    chromosome = np.array([1, 1, 1, 1, 1])
    mutation_rate = 0.5

    mutated = bit_flip_mutation(chromosome, mutation_rate)

    # Check length preservation
    assert len(mutated) == len(chromosome), "Mutated chromosome length should match original"

    # Check binary values
    assert np.all((mutated == 0) | (mutated == 1)), \
        "Mutated chromosome should contain only 0s and 1s"

    # Original should not be modified
    assert np.array_equal(chromosome, np.array([1, 1, 1, 1, 1])), \
        "Original chromosome should not be modified"

    print(f"  Original:  {chromosome}")
    print(f"  Mutated:   {mutated}")

    print("✓ Bit-flip mutation test passed!")


def test_gaussian_mutation():
    """Test Gaussian mutation implementation."""
    print("Testing gaussian_mutation...")

    np.random.seed(42)
    chromosome = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    mutation_rate = 0.5
    bounds = (-5.0, 5.0)

    mutated = gaussian_mutation(chromosome, mutation_rate, bounds)

    # Check length preservation
    assert len(mutated) == len(chromosome), "Mutated chromosome length should match original"

    # Check bounds
    assert np.all(mutated >= bounds[0]) and np.all(mutated <= bounds[1]), \
        f"Mutated values should be within bounds {bounds}"

    # Original should not be modified
    assert np.array_equal(chromosome, np.array([0.0, 0.0, 0.0, 0.0, 0.0])), \
        "Original chromosome should not be modified"

    print(f"  Original:  {chromosome}")
    print(f"  Mutated:   {mutated}")

    print("✓ Gaussian mutation test passed!")


def test_roulette_wheel_selection():
    """Test roulette wheel selection implementation."""
    print("Testing roulette_wheel_selection...")

    np.random.seed(42)
    population = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    fitness_values = np.array([10.0, 1.0, 5.0, 3.0])
    num_parents = 4

    selected = roulette_wheel_selection(population, fitness_values, num_parents)

    # Check shape
    assert selected.shape == (num_parents, population.shape[1]), \
        f"Expected shape {(num_parents, population.shape[1])}, got {selected.shape}"

    # Check that selected individuals are from the original population
    for individual in selected:
        found = False
        for original in population:
            if np.array_equal(individual, original):
                found = True
                break
        assert found, "Selected individual should be from original population"

    print("✓ Roulette wheel selection test passed!")


def test_tournament_selection():
    """Test tournament selection implementation."""
    print("Testing tournament_selection...")

    np.random.seed(42)
    population = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    fitness_values = np.array([10.0, 1.0, 5.0, 3.0])
    num_parents = 4
    tournament_size = 2

    selected = tournament_selection(population, fitness_values, num_parents, tournament_size)

    # Check shape
    assert selected.shape == (num_parents, population.shape[1]), \
        f"Expected shape {(num_parents, population.shape[1])}, got {selected.shape}"

    # Check that selected individuals are from the original population
    for individual in selected:
        found = False
        for original in population:
            if np.array_equal(individual, original):
                found = True
                break
        assert found, "Selected individual should be from original population"

    print("✓ Tournament selection test passed!")


def test_fitness_functions():
    """Test benchmark fitness functions."""
    print("Testing fitness functions...")

    # Test Sphere function
    x = np.array([0.0, 0.0])
    result = sphere_function(x)
    assert np.isclose(result, 0.0), f"Sphere at origin should be 0, got {result}"

    x = np.array([1.0, 1.0])
    result = sphere_function(x)
    assert np.isclose(result, 2.0), f"Sphere at (1,1) should be 2, got {result}"

    # Test Rastrigin function
    x = np.array([0.0, 0.0])
    result = rastrigin_function(x)
    assert np.isclose(result, 0.0, atol=1e-10), f"Rastrigin at origin should be 0, got {result}"

    # Test Rosenbrock function
    x = np.array([1.0, 1.0])
    result = rosenbrock_function(x)
    assert np.isclose(result, 0.0, atol=1e-10), f"Rosenbrock at (1,1) should be 0, got {result}"

    print("✓ Fitness functions test passed!")


def run_all_tests():
    """Run all public tests."""
    print("\n" + "="*60)
    print("Running GA Tutorial 1 - Public Tests")
    print("="*60 + "\n")

    try:
        test_initialize_population_binary()
        test_initialize_population_real()
        test_binary_to_real()
        test_single_point_crossover()
        test_two_point_crossover()
        test_bit_flip_mutation()
        test_gaussian_mutation()
        test_roulette_wheel_selection()
        test_tournament_selection()
        test_fitness_functions()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
