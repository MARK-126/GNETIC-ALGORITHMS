"""
Test Case Generators for GA Tutorial 2
Provides test data for advanced genetic algorithm operators.
"""

import numpy as np


def generate_selection_testcase():
    """Generate test case for advanced selection methods."""
    pop_size = np.random.randint(10, 30)
    num_variables = np.random.randint(2, 5)

    population = np.random.uniform(-5, 5, size=(pop_size, num_variables))
    fitness = np.random.uniform(0, 100, size=pop_size)
    num_parents = np.random.randint(pop_size // 2, pop_size)

    return {
        'population': population,
        'fitness': fitness,
        'num_parents': num_parents
    }


def generate_crossover_testcase():
    """Generate test case for advanced crossover operators."""
    num_variables = np.random.randint(2, 10)

    parent1 = np.random.uniform(-5, 5, size=num_variables)
    parent2 = np.random.uniform(-5, 5, size=num_variables)
    bounds = (-5.0, 5.0)

    return {
        'parent1': parent1,
        'parent2': parent2,
        'bounds': bounds
    }


def generate_mutation_testcase():
    """Generate test case for advanced mutation operators."""
    num_variables = np.random.randint(2, 10)

    chromosome = np.random.uniform(-5, 5, size=num_variables)
    bounds = (-5.0, 5.0)
    mutation_rate = np.random.uniform(0.01, 0.2)

    return {
        'chromosome': chromosome,
        'bounds': bounds,
        'mutation_rate': mutation_rate
    }


def generate_constraint_testcase():
    """Generate test case for constraint handling."""
    num_variables = np.random.randint(2, 5)

    solution = np.random.uniform(-10, 10, size=num_variables)
    bounds = (-5.0, 5.0)

    # Generate random constraint violations
    num_constraints = np.random.randint(1, 4)
    constraints = [np.random.uniform(-2, 3) for _ in range(num_constraints)]

    return {
        'solution': solution,
        'bounds': bounds,
        'constraints': constraints
    }


# Expected behavior patterns
SELECTION_PATTERNS = {
    'rank_based': {
        'description': 'Selection based on rank, not raw fitness',
        'robust_to_scaling': True,
        'selection_pressure': 'Controlled by parameter'
    },
    'sus': {
        'description': 'Stochastic Universal Sampling - low variance',
        'robust_to_scaling': True,
        'selection_pressure': 'Proportional to fitness'
    },
    'boltzmann': {
        'description': 'Temperature-based selection',
        'robust_to_scaling': False,
        'selection_pressure': 'Controlled by temperature'
    }
}


CROSSOVER_PATTERNS = {
    'arithmetic': {
        'description': 'Convex combination of parents',
        'exploration': 'Low',
        'offspring_range': 'Within parent bounds'
    },
    'blx_alpha': {
        'description': 'Blend with extension beyond parents',
        'exploration': 'High',
        'offspring_range': 'Extended beyond parents'
    },
    'sbx': {
        'description': 'Simulated Binary Crossover',
        'exploration': 'Controlled by eta',
        'offspring_range': 'Adaptive'
    }
}


MUTATION_PATTERNS = {
    'polynomial': {
        'description': 'Self-adaptive perturbation',
        'distribution': 'Polynomial',
        'parameter': 'eta (distribution index)'
    },
    'adaptive': {
        'description': 'Decreases over time',
        'distribution': 'Gaussian',
        'parameter': 'Generation-based'
    },
    'self_adaptive': {
        'description': 'Strategy parameters evolve',
        'distribution': 'Gaussian',
        'parameter': 'Evolved with solution'
    }
}


def validate_advanced_selection(selected, original_population, num_parents):
    """Validate advanced selection output."""
    assert len(selected) == num_parents, \
        f"Expected {num_parents} parents, got {len(selected)}"

    for parent in selected:
        found = False
        for individual in original_population:
            if np.array_equal(parent, individual):
                found = True
                break
        assert found, "Selected parent not in original population"

    return True


def validate_advanced_crossover(offspring1, offspring2, parent1, parent2, bounds):
    """Validate advanced crossover output."""
    # Check dimensions
    assert len(offspring1) == len(parent1), "Offspring dimension mismatch"
    assert len(offspring2) == len(parent2), "Offspring dimension mismatch"

    # Check bounds if provided
    if bounds is not None:
        lower, upper = bounds
        assert np.all(offspring1 >= lower) and np.all(offspring1 <= upper), \
            "Offspring 1 out of bounds"
        assert np.all(offspring2 >= lower) and np.all(offspring2 <= upper), \
            "Offspring 2 out of bounds"

    return True


def validate_advanced_mutation(mutated, original, bounds):
    """Validate advanced mutation output."""
    # Check dimension
    assert len(mutated) == len(original), "Mutated dimension mismatch"

    # Check bounds
    if bounds is not None:
        lower, upper = bounds
        assert np.all(mutated >= lower) and np.all(mutated <= upper), \
            "Mutated values out of bounds"

    # Original should not be modified
    # (This check should be done in the calling code)

    return True


def get_selection_info(method_name):
    """Get information about selection method."""
    if method_name in SELECTION_PATTERNS:
        return SELECTION_PATTERNS[method_name]
    else:
        raise ValueError(f"Unknown selection method: {method_name}")


def get_crossover_info(method_name):
    """Get information about crossover method."""
    if method_name in CROSSOVER_PATTERNS:
        return CROSSOVER_PATTERNS[method_name]
    else:
        raise ValueError(f"Unknown crossover method: {method_name}")


def get_mutation_info(method_name):
    """Get information about mutation method."""
    if method_name in MUTATION_PATTERNS:
        return MUTATION_PATTERNS[method_name]
    else:
        raise ValueError(f"Unknown mutation method: {method_name}")
