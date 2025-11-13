"""
Genetic Algorithm Utilities - Basic Functions
Tutorial 1: Introduction to Genetic Algorithms

This module provides fundamental functions for implementing basic genetic algorithms:
- Population initialization
- Fitness evaluation
- Basic selection, crossover, and mutation operators
- Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# ==================== Population Initialization ====================

def initialize_population_binary(pop_size, chromosome_length):
    """
    Initialize a random binary population.

    Arguments:
    pop_size -- number of individuals in the population
    chromosome_length -- length of each chromosome (number of genes)

    Returns:
    population -- numpy array of shape (pop_size, chromosome_length) with binary values {0, 1}
    """
    population = np.random.randint(0, 2, size=(pop_size, chromosome_length))
    return population


def initialize_population_real(pop_size, chromosome_length, bounds):
    """
    Initialize a random real-valued population.

    Arguments:
    pop_size -- number of individuals in the population
    chromosome_length -- length of each chromosome (number of genes)
    bounds -- tuple (lower_bound, upper_bound) for gene values

    Returns:
    population -- numpy array of shape (pop_size, chromosome_length) with real values
    """
    lower, upper = bounds
    population = np.random.uniform(lower, upper, size=(pop_size, chromosome_length))
    return population


# ==================== Decoding Functions ====================

def binary_to_real(chromosome, bounds, bits_per_variable):
    """
    Convert binary chromosome to real-valued representation.

    Arguments:
    chromosome -- binary array representing the individual
    bounds -- tuple (lower_bound, upper_bound) for the variable
    bits_per_variable -- number of bits used to encode the variable

    Returns:
    value -- decoded real value
    """
    # Convert binary to integer
    integer_value = 0
    for i in range(bits_per_variable):
        integer_value += chromosome[i] * (2 ** (bits_per_variable - 1 - i))

    # Scale to real range
    lower, upper = bounds
    max_integer = 2 ** bits_per_variable - 1
    value = lower + (integer_value / max_integer) * (upper - lower)

    return value


def decode_chromosome(chromosome, num_variables, bounds, bits_per_variable):
    """
    Decode a binary chromosome into multiple real variables.

    Arguments:
    chromosome -- binary array
    num_variables -- number of variables to decode
    bounds -- tuple (lower_bound, upper_bound) for all variables
    bits_per_variable -- number of bits per variable

    Returns:
    variables -- list of decoded real values
    """
    variables = []
    for i in range(num_variables):
        start_idx = i * bits_per_variable
        end_idx = start_idx + bits_per_variable
        gene_segment = chromosome[start_idx:end_idx]
        value = binary_to_real(gene_segment, bounds, bits_per_variable)
        variables.append(value)

    return variables


# ==================== Fitness Functions ====================

def sphere_function(x):
    """
    Sphere function: f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0

    Arguments:
    x -- array of variables

    Returns:
    value -- function value (to be minimized)
    """
    return np.sum(x ** 2)


def rastrigin_function(x):
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0

    Arguments:
    x -- array of variables

    Returns:
    value -- function value (to be minimized)
    """
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1,...,1) = 0

    Arguments:
    x -- array of variables

    Returns:
    value -- function value (to be minimized)
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def evaluate_fitness(population, fitness_function, num_variables, bounds, bits_per_variable, encoding='binary'):
    """
    Evaluate fitness for all individuals in the population.

    Arguments:
    population -- array of individuals
    fitness_function -- function to evaluate fitness
    num_variables -- number of variables (for binary encoding)
    bounds -- bounds for variables (for binary encoding)
    bits_per_variable -- bits per variable (for binary encoding)
    encoding -- 'binary' or 'real'

    Returns:
    fitness_values -- array of fitness values (higher is better)
    """
    fitness_values = []

    for individual in population:
        if encoding == 'binary':
            # Decode binary chromosome to real values
            variables = decode_chromosome(individual, num_variables, bounds, bits_per_variable)
            variables = np.array(variables)
        else:
            # Already real-valued
            variables = individual

        # Evaluate fitness (negate for minimization problems)
        objective_value = fitness_function(variables)
        # Convert to fitness (higher is better): fitness = -objective_value
        fitness = -objective_value
        fitness_values.append(fitness)

    return np.array(fitness_values)


# ==================== Selection Operators ====================

def roulette_wheel_selection(population, fitness_values, num_parents):
    """
    Roulette wheel selection (fitness proportionate selection).

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number of parents to select

    Returns:
    selected_parents -- array of selected individuals
    """
    # Handle negative fitness values by shifting
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        adjusted_fitness = fitness_values - min_fitness + 1e-10
    else:
        adjusted_fitness = fitness_values + 1e-10

    # Calculate selection probabilities
    total_fitness = np.sum(adjusted_fitness)
    probabilities = adjusted_fitness / total_fitness

    # Select parents
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities, replace=True)
    selected_parents = population[selected_indices]

    return selected_parents


def tournament_selection(population, fitness_values, num_parents, tournament_size=3):
    """
    Tournament selection.

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number of parents to select
    tournament_size -- number of individuals in each tournament

    Returns:
    selected_parents -- array of selected individuals
    """
    selected_parents = []

    for _ in range(num_parents):
        # Randomly select individuals for tournament
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_indices]

        # Select the best individual from tournament
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_index])

    return np.array(selected_parents)


# ==================== Crossover Operators ====================

def single_point_crossover(parent1, parent2):
    """
    Single-point crossover operator.

    Arguments:
    parent1 -- first parent chromosome
    parent2 -- second parent chromosome

    Returns:
    offspring1, offspring2 -- two offspring chromosomes
    """
    chromosome_length = len(parent1)

    # Select random crossover point
    crossover_point = np.random.randint(1, chromosome_length)

    # Create offspring
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return offspring1, offspring2


def two_point_crossover(parent1, parent2):
    """
    Two-point crossover operator.

    Arguments:
    parent1 -- first parent chromosome
    parent2 -- second parent chromosome

    Returns:
    offspring1, offspring2 -- two offspring chromosomes
    """
    chromosome_length = len(parent1)

    # Select two random crossover points
    points = sorted(np.random.choice(range(1, chromosome_length), size=2, replace=False))
    point1, point2 = points

    # Create offspring
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])

    return offspring1, offspring2


def uniform_crossover(parent1, parent2, crossover_rate=0.5):
    """
    Uniform crossover operator.

    Arguments:
    parent1 -- first parent chromosome
    parent2 -- second parent chromosome
    crossover_rate -- probability of swapping each gene

    Returns:
    offspring1, offspring2 -- two offspring chromosomes
    """
    chromosome_length = len(parent1)

    # Create offspring as copies of parents
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Swap genes with probability crossover_rate
    for i in range(chromosome_length):
        if np.random.rand() < crossover_rate:
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

    return offspring1, offspring2


# ==================== Mutation Operators ====================

def bit_flip_mutation(chromosome, mutation_rate):
    """
    Bit-flip mutation for binary chromosomes.

    Arguments:
    chromosome -- binary chromosome
    mutation_rate -- probability of flipping each bit

    Returns:
    mutated_chromosome -- chromosome after mutation
    """
    mutated_chromosome = chromosome.copy()

    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]

    return mutated_chromosome


def gaussian_mutation(chromosome, mutation_rate, bounds, sigma=0.1):
    """
    Gaussian mutation for real-valued chromosomes.

    Arguments:
    chromosome -- real-valued chromosome
    mutation_rate -- probability of mutating each gene
    bounds -- tuple (lower_bound, upper_bound) for gene values
    sigma -- standard deviation for Gaussian noise (relative to range)

    Returns:
    mutated_chromosome -- chromosome after mutation
    """
    mutated_chromosome = chromosome.copy()
    lower, upper = bounds
    mutation_range = upper - lower

    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            # Add Gaussian noise
            noise = np.random.normal(0, sigma * mutation_range)
            mutated_chromosome[i] += noise
            # Clip to bounds
            mutated_chromosome[i] = np.clip(mutated_chromosome[i], lower, upper)

    return mutated_chromosome


# ==================== Visualization Functions ====================

def plot_fitness_evolution(fitness_history, title="Fitness Evolution"):
    """
    Plot the evolution of fitness over generations.

    Arguments:
    fitness_history -- dictionary with 'best', 'average', 'worst' fitness per generation
    title -- plot title
    """
    plt.figure(figsize=(10, 6))

    generations = range(len(fitness_history['best']))

    plt.plot(generations, fitness_history['best'], 'g-', linewidth=2, label='Best Fitness')
    plt.plot(generations, fitness_history['average'], 'b--', linewidth=1.5, label='Average Fitness')
    plt.plot(generations, fitness_history['worst'], 'r:', linewidth=1.5, label='Worst Fitness')

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_population_diversity(diversity_history, title="Population Diversity"):
    """
    Plot the diversity of population over generations.

    Arguments:
    diversity_history -- list of diversity values per generation
    title -- plot title
    """
    plt.figure(figsize=(10, 6))

    generations = range(len(diversity_history))
    plt.plot(generations, diversity_history, 'purple', linewidth=2)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Diversity (Standard Deviation)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_2d_function(fitness_function, bounds, best_solution=None, title="Objective Function"):
    """
    Plot a 2D objective function with optional best solution marker.

    Arguments:
    fitness_function -- function to plot
    bounds -- tuple (lower_bound, upper_bound)
    best_solution -- optional array [x, y] of best solution found
    title -- plot title
    """
    fig = plt.figure(figsize=(12, 5))

    # Create grid
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_function(np.array([X[i, j], Y[i, j]]))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)

    if best_solution is not None:
        z_best = fitness_function(best_solution)
        ax1.scatter([best_solution[0]], [best_solution[1]], [z_best],
                   color='red', s=100, marker='*', label='Best Solution')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax1.set_title(title + ' (3D View)')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis)

    if best_solution is not None:
        ax2.plot(best_solution[0], best_solution[1], 'r*', markersize=20, label='Best Solution')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(title + ' (Contour View)')
    fig.colorbar(contour, ax=ax2)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def calculate_diversity(population):
    """
    Calculate population diversity as the average standard deviation across all genes.

    Arguments:
    population -- array of individuals

    Returns:
    diversity -- diversity measure
    """
    # Calculate standard deviation for each gene position
    std_per_gene = np.std(population, axis=0)
    # Average across all genes
    diversity = np.mean(std_per_gene)
    return diversity


def print_generation_stats(generation, fitness_values, best_individual, num_variables=None,
                          bounds=None, bits_per_variable=None, encoding='binary'):
    """
    Print statistics for current generation.

    Arguments:
    generation -- current generation number
    fitness_values -- array of fitness values
    best_individual -- best individual chromosome
    num_variables -- number of variables (for binary encoding)
    bounds -- bounds (for binary encoding)
    bits_per_variable -- bits per variable (for binary encoding)
    encoding -- 'binary' or 'real'
    """
    print(f"\n{'='*60}")
    print(f"Generation {generation}")
    print(f"{'='*60}")
    print(f"Best Fitness:    {np.max(fitness_values):.6f}")
    print(f"Average Fitness: {np.mean(fitness_values):.6f}")
    print(f"Worst Fitness:   {np.min(fitness_values):.6f}")

    if encoding == 'binary' and num_variables is not None:
        best_decoded = decode_chromosome(best_individual, num_variables, bounds, bits_per_variable)
        print(f"Best Solution:   {best_decoded}")
    else:
        print(f"Best Solution:   {best_individual}")

    print(f"{'='*60}")
