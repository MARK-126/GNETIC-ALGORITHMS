"""
Test Case Generators for GA Tutorial 1
Provides functions to generate random test data for validating student implementations.
"""

import numpy as np


def generate_random_population_binary_testcase():
    """Generate test case for binary population initialization."""
    pop_size = np.random.randint(5, 20)
    chromosome_length = np.random.randint(8, 32)

    return {
        'pop_size': pop_size,
        'chromosome_length': chromosome_length
    }


def generate_random_population_real_testcase():
    """Generate test case for real-valued population initialization."""
    pop_size = np.random.randint(5, 20)
    chromosome_length = np.random.randint(2, 10)
    lower_bound = np.random.uniform(-10, -1)
    upper_bound = np.random.uniform(1, 10)

    return {
        'pop_size': pop_size,
        'chromosome_length': chromosome_length,
        'bounds': (lower_bound, upper_bound)
    }


def generate_binary_chromosome_testcase():
    """Generate random binary chromosome for testing."""
    length = np.random.randint(8, 32)
    chromosome = np.random.randint(0, 2, size=length)

    return {
        'chromosome': chromosome,
        'bits_per_variable': np.random.choice([8, 10, 16]),
        'bounds': (-5.0, 5.0)
    }


def generate_crossover_testcase():
    """Generate test case for crossover operators."""
    chromosome_length = np.random.randint(8, 32)
    parent1 = np.random.randint(0, 2, size=chromosome_length)
    parent2 = np.random.randint(0, 2, size=chromosome_length)

    return {
        'parent1': parent1,
        'parent2': parent2
    }


def generate_mutation_testcase():
    """Generate test case for mutation operators."""
    chromosome_length = np.random.randint(8, 32)
    chromosome = np.random.randint(0, 2, size=chromosome_length)
    mutation_rate = np.random.uniform(0.01, 0.2)

    return {
        'chromosome': chromosome,
        'mutation_rate': mutation_rate
    }


def generate_selection_testcase():
    """Generate test case for selection operators."""
    pop_size = np.random.randint(10, 50)
    chromosome_length = np.random.randint(5, 20)

    population = np.random.randint(0, 2, size=(pop_size, chromosome_length))
    fitness_values = np.random.uniform(0, 100, size=pop_size)
    num_parents = np.random.randint(2, pop_size // 2)

    return {
        'population': population,
        'fitness_values': fitness_values,
        'num_parents': num_parents
    }


def generate_fitness_evaluation_testcase():
    """Generate test case for fitness evaluation."""
    pop_size = np.random.randint(5, 15)
    num_variables = np.random.choice([2, 3, 5])
    bits_per_variable = np.random.choice([8, 10])
    chromosome_length = num_variables * bits_per_variable

    population = np.random.randint(0, 2, size=(pop_size, chromosome_length))

    return {
        'population': population,
        'num_variables': num_variables,
        'bits_per_variable': bits_per_variable,
        'bounds': (-5.12, 5.12)
    }


def get_expected_crossover_result(parent1, parent2, crossover_point):
    """
    Get expected result for single-point crossover at specific point.
    Used for deterministic testing.
    """
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return offspring1, offspring2


def get_expected_two_point_crossover_result(parent1, parent2, point1, point2):
    """
    Get expected result for two-point crossover at specific points.
    Used for deterministic testing.
    """
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])

    return offspring1, offspring2


def validate_population_shape(population, expected_pop_size, expected_chromosome_length):
    """Validate population array shape."""
    assert population.shape == (expected_pop_size, expected_chromosome_length), \
        f"Expected shape {(expected_pop_size, expected_chromosome_length)}, got {population.shape}"
    return True


def validate_binary_values(array):
    """Validate that array contains only binary values."""
    assert np.all((array == 0) | (array == 1)), \
        "Array should contain only 0s and 1s"
    return True


def validate_bounds(array, lower_bound, upper_bound):
    """Validate that array values are within specified bounds."""
    assert np.all(array >= lower_bound) and np.all(array <= upper_bound), \
        f"Values should be within bounds [{lower_bound}, {upper_bound}]"
    return True


def validate_chromosome_length(chromosome, expected_length):
    """Validate chromosome length."""
    assert len(chromosome) == expected_length, \
        f"Expected length {expected_length}, got {len(chromosome)}"
    return True


def validate_fitness_values(fitness_values, population_size):
    """Validate fitness values array."""
    assert len(fitness_values) == population_size, \
        f"Expected {population_size} fitness values, got {len(fitness_values)}"
    assert np.all(np.isfinite(fitness_values)), \
        "All fitness values should be finite"
    return True


def validate_selected_parents(selected_parents, original_population, num_parents):
    """Validate selected parents from selection operators."""
    assert len(selected_parents) == num_parents, \
        f"Expected {num_parents} selected parents, got {len(selected_parents)}"

    # Check that each selected parent exists in original population
    for parent in selected_parents:
        found = False
        for individual in original_population:
            if np.array_equal(parent, individual):
                found = True
                break
        assert found, "Selected parent should exist in original population"

    return True


# Benchmark problem test data
BENCHMARK_PROBLEMS = {
    'sphere': {
        'name': 'Sphere Function',
        'optimum': np.array([0.0, 0.0]),
        'optimum_value': 0.0,
        'bounds': (-5.12, 5.12),
        'test_points': [
            (np.array([0.0, 0.0]), 0.0),
            (np.array([1.0, 1.0]), 2.0),
            (np.array([2.0, 3.0]), 13.0),
        ]
    },
    'rastrigin': {
        'name': 'Rastrigin Function',
        'optimum': np.array([0.0, 0.0]),
        'optimum_value': 0.0,
        'bounds': (-5.12, 5.12),
        'test_points': [
            (np.array([0.0, 0.0]), 0.0),
        ]
    },
    'rosenbrock': {
        'name': 'Rosenbrock Function',
        'optimum': np.array([1.0, 1.0]),
        'optimum_value': 0.0,
        'bounds': (-5.0, 10.0),
        'test_points': [
            (np.array([1.0, 1.0]), 0.0),
        ]
    }
}


def get_benchmark_problem(problem_name):
    """Get benchmark problem information."""
    if problem_name not in BENCHMARK_PROBLEMS:
        raise ValueError(f"Unknown problem: {problem_name}. Available: {list(BENCHMARK_PROBLEMS.keys())}")

    return BENCHMARK_PROBLEMS[problem_name]


def generate_test_population_with_known_best(pop_size, chromosome_length, encoding='binary'):
    """
    Generate a test population with a known best individual.

    Returns:
    population, best_individual, best_index
    """
    if encoding == 'binary':
        population = np.random.randint(0, 2, size=(pop_size, chromosome_length))
        # Make the first individual all ones (known best for certain problems)
        population[0] = np.ones(chromosome_length, dtype=int)
        best_index = 0
        best_individual = population[0]
    else:
        population = np.random.uniform(-5, 5, size=(pop_size, chromosome_length))
        # Make the first individual close to optimum
        population[0] = np.zeros(chromosome_length)
        best_index = 0
        best_individual = population[0]

    return population, best_individual, best_index
