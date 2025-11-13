"""
Genetic Algorithm Utilities - Intermediate/Advanced Functions
Tutorial 2: Advanced Genetic Operators

This module provides advanced GA operators and techniques:
- Advanced selection methods (rank-based, stochastic universal sampling)
- Elitism and replacement strategies
- Adaptive parameters (mutation rate, crossover rate)
- Advanced crossover operators (arithmetic, BLX-alpha, SBX)
- Island models and migration
- Constraint handling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ==================== Advanced Selection Operators ====================

def rank_based_selection(population, fitness_values, num_parents, selection_pressure=2.0):
    """
    Rank-based selection - selection probability based on rank, not raw fitness.

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number of parents to select
    selection_pressure -- pressure parameter (1.0-2.0), higher = more selective

    Returns:
    selected_parents -- array of selected individuals
    """
    pop_size = len(population)

    # Rank individuals (0 = worst, pop_size-1 = best)
    ranks = np.argsort(np.argsort(fitness_values))

    # Calculate selection probabilities using linear ranking
    # P(i) = (2 - SP + 2*(SP-1)*(rank-1)/(N-1)) / N
    # where SP = selection pressure, N = population size
    probabilities = np.zeros(pop_size)
    for i in range(pop_size):
        probabilities[i] = (2 - selection_pressure +
                          2 * (selection_pressure - 1) * ranks[i] / (pop_size - 1)) / pop_size

    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities)

    # Select parents
    selected_indices = np.random.choice(pop_size, size=num_parents, p=probabilities, replace=True)
    selected_parents = population[selected_indices]

    return selected_parents


def stochastic_universal_sampling(population, fitness_values, num_parents):
    """
    Stochastic Universal Sampling (SUS) - ensures better spread than roulette wheel.

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number of parents to select

    Returns:
    selected_parents -- array of selected individuals
    """
    # Handle negative fitness values
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        adjusted_fitness = fitness_values - min_fitness + 1e-10
    else:
        adjusted_fitness = fitness_values + 1e-10

    # Calculate cumulative fitness
    total_fitness = np.sum(adjusted_fitness)
    cumulative_fitness = np.cumsum(adjusted_fitness)

    # Calculate pointer distance
    pointer_distance = total_fitness / num_parents

    # Random start point
    start = np.random.uniform(0, pointer_distance)

    # Generate pointers
    pointers = [start + i * pointer_distance for i in range(num_parents)]

    # Select individuals
    selected_parents = []
    for pointer in pointers:
        for i, cum_fit in enumerate(cumulative_fitness):
            if pointer <= cum_fit:
                selected_parents.append(population[i].copy())
                break

    return np.array(selected_parents)


def boltzmann_selection(population, fitness_values, num_parents, temperature=1.0):
    """
    Boltzmann selection - temperature-based selection inspired by simulated annealing.

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number of parents to select
    temperature -- temperature parameter (higher = less selective)

    Returns:
    selected_parents -- array of selected individuals
    """
    # Calculate Boltzmann probabilities
    # P(i) ∝ exp(fitness_i / temperature)

    # Normalize fitness to avoid overflow
    normalized_fitness = fitness_values - np.max(fitness_values)

    # Calculate probabilities
    boltzmann_values = np.exp(normalized_fitness / temperature)
    probabilities = boltzmann_values / np.sum(boltzmann_values)

    # Select parents
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities, replace=True)
    selected_parents = population[selected_indices]

    return selected_parents


# ==================== Advanced Crossover Operators ====================

def arithmetic_crossover(parent1, parent2, alpha=0.5):
    """
    Arithmetic crossover for real-valued chromosomes.
    offspring1 = alpha * parent1 + (1-alpha) * parent2
    offspring2 = (1-alpha) * parent1 + alpha * parent2

    Arguments:
    parent1 -- first parent (real-valued)
    parent2 -- second parent (real-valued)
    alpha -- blending parameter (0-1)

    Returns:
    offspring1, offspring2 -- two offspring
    """
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2

    return offspring1, offspring2


def blx_alpha_crossover(parent1, parent2, alpha=0.5, bounds=None):
    """
    BLX-α (Blend Crossover) for real-valued chromosomes.
    Creates offspring in extended range around parents.

    Arguments:
    parent1 -- first parent
    parent2 -- second parent
    alpha -- extension parameter (typically 0.5)
    bounds -- optional tuple (lower, upper) to clip values

    Returns:
    offspring1, offspring2 -- two offspring
    """
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        min_val = min(parent1[i], parent2[i])
        max_val = max(parent1[i], parent2[i])
        interval = max_val - min_val

        # Extended range
        lower = min_val - alpha * interval
        upper = max_val + alpha * interval

        offspring1[i] = np.random.uniform(lower, upper)
        offspring2[i] = np.random.uniform(lower, upper)

    # Clip to bounds if provided
    if bounds is not None:
        offspring1 = np.clip(offspring1, bounds[0], bounds[1])
        offspring2 = np.clip(offspring2, bounds[0], bounds[1])

    return offspring1, offspring2


def simulated_binary_crossover(parent1, parent2, eta=20, bounds=None):
    """
    Simulated Binary Crossover (SBX) - mimics binary crossover for real values.
    Commonly used in NSGA-II and other modern GAs.

    Arguments:
    parent1 -- first parent
    parent2 -- second parent
    eta -- distribution index (larger = more similar to parents)
    bounds -- optional tuple (lower, upper)

    Returns:
    offspring1, offspring2 -- two offspring
    """
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                # Calculate beta
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

                # Create offspring
                offspring1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                offspring2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]

    # Clip to bounds if provided
    if bounds is not None:
        offspring1 = np.clip(offspring1, bounds[0], bounds[1])
        offspring2 = np.clip(offspring2, bounds[0], bounds[1])

    return offspring1, offspring2


# ==================== Advanced Mutation Operators ====================

def polynomial_mutation(chromosome, mutation_rate, bounds, eta=20):
    """
    Polynomial mutation - used in NSGA-II and other modern GAs.

    Arguments:
    chromosome -- individual to mutate
    mutation_rate -- probability of mutating each gene
    bounds -- tuple (lower, upper)
    eta -- distribution index

    Returns:
    mutated_chromosome -- mutated individual
    """
    mutated = chromosome.copy()
    lower, upper = bounds

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            u = np.random.rand()
            delta_l = (mutated[i] - lower) / (upper - lower)
            delta_u = (upper - mutated[i]) / (upper - lower)

            if u < 0.5:
                delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1.0
            else:
                delta_q = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))

            mutated[i] = mutated[i] + delta_q * (upper - lower)
            mutated[i] = np.clip(mutated[i], lower, upper)

    return mutated


def adaptive_mutation(chromosome, generation, max_generations, bounds,
                     initial_rate=0.1, final_rate=0.01):
    """
    Adaptive mutation that decreases mutation strength over generations.

    Arguments:
    chromosome -- individual to mutate
    generation -- current generation
    max_generations -- total number of generations
    bounds -- tuple (lower, upper)
    initial_rate -- starting mutation rate
    final_rate -- ending mutation rate

    Returns:
    mutated_chromosome -- mutated individual
    """
    # Linear decrease in mutation rate
    current_rate = initial_rate - (initial_rate - final_rate) * generation / max_generations

    # Apply Gaussian mutation with adaptive rate
    mutated = chromosome.copy()
    lower, upper = bounds
    mutation_range = upper - lower

    for i in range(len(mutated)):
        if np.random.rand() < current_rate:
            # Decrease sigma over time as well
            sigma = (1 - generation / max_generations) * 0.2
            noise = np.random.normal(0, sigma * mutation_range)
            mutated[i] += noise
            mutated[i] = np.clip(mutated[i], lower, upper)

    return mutated


def self_adaptive_mutation(chromosome, strategy_params, bounds, tau=0.1, tau_prime=0.01):
    """
    Self-adaptive mutation where mutation parameters evolve with the solution.
    Each individual carries its own mutation strategy.

    Arguments:
    chromosome -- individual to mutate
    strategy_params -- array of strategy parameters (mutation strengths)
    bounds -- tuple (lower, upper)
    tau -- learning rate for global adaptation
    tau_prime -- learning rate for individual adaptation

    Returns:
    mutated_chromosome, mutated_strategy -- mutated individual and strategy
    """
    n = len(chromosome)

    # Evolve strategy parameters
    global_noise = np.random.normal(0, 1)
    individual_noise = np.random.normal(0, 1, size=n)

    mutated_strategy = strategy_params * np.exp(tau_prime * global_noise + tau * individual_noise)
    mutated_strategy = np.clip(mutated_strategy, 1e-10, 1.0)

    # Apply mutation using evolved strategy
    mutated = chromosome.copy()
    lower, upper = bounds

    for i in range(n):
        noise = np.random.normal(0, mutated_strategy[i])
        mutated[i] += noise
        mutated[i] = np.clip(mutated[i], lower, upper)

    return mutated, mutated_strategy


# ==================== Elitism and Replacement Strategies ====================

def elitist_replacement(old_population, old_fitness, new_population, new_fitness, elite_size):
    """
    Elitist replacement - keep best individuals from previous generation.

    Arguments:
    old_population -- previous generation
    old_fitness -- fitness of previous generation
    new_population -- new generation
    new_fitness -- fitness of new generation
    elite_size -- number of elite individuals to preserve

    Returns:
    combined_population, combined_fitness -- merged population with elites
    """
    # Sort old population by fitness (descending)
    elite_indices = np.argsort(old_fitness)[-elite_size:]
    elites = old_population[elite_indices]
    elite_fitness = old_fitness[elite_indices]

    # Replace worst individuals in new population with elites
    worst_indices = np.argsort(new_fitness)[:elite_size]

    new_population[worst_indices] = elites
    new_fitness[worst_indices] = elite_fitness

    return new_population, new_fitness


def steady_state_replacement(population, fitness_values, offspring, offspring_fitness):
    """
    Steady-state replacement - replace worst with offspring if better.

    Arguments:
    population -- current population
    fitness_values -- current fitness
    offspring -- new offspring
    offspring_fitness -- offspring fitness

    Returns:
    updated_population, updated_fitness
    """
    # Find worst individual
    worst_idx = np.argmin(fitness_values)

    # Replace if offspring is better
    if offspring_fitness > fitness_values[worst_idx]:
        population[worst_idx] = offspring
        fitness_values[worst_idx] = offspring_fitness

    return population, fitness_values


# ==================== Constraint Handling ====================

def penalty_function(objective_value, constraints, penalty_coefficient=1000):
    """
    Simple penalty function for constraint handling.

    Arguments:
    objective_value -- original objective value
    constraints -- list of constraint violations (positive = violated)
    penalty_coefficient -- penalty multiplier

    Returns:
    penalized_value -- objective value with penalty
    """
    total_violation = sum(max(0, c) for c in constraints)
    penalized = objective_value + penalty_coefficient * total_violation
    return penalized


def death_penalty(individual, constraints):
    """
    Death penalty - assign very bad fitness if constraints violated.

    Arguments:
    individual -- solution to check
    constraints -- list of constraint values (positive = violated)

    Returns:
    is_feasible -- True if all constraints satisfied
    """
    return all(c <= 0 for c in constraints)


def repair_bounds(chromosome, bounds):
    """
    Repair chromosome by clipping to feasible bounds.

    Arguments:
    chromosome -- potentially infeasible solution
    bounds -- tuple (lower, upper)

    Returns:
    repaired_chromosome -- feasible solution
    """
    lower, upper = bounds
    return np.clip(chromosome, lower, upper)


# ==================== Diversity Maintenance ====================

def crowding_distance(population, fitness_values):
    """
    Calculate crowding distance for diversity preservation (used in NSGA-II).

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values

    Returns:
    distances -- crowding distance for each individual
    """
    pop_size = len(population)
    distances = np.zeros(pop_size)

    # Sort by fitness
    sorted_indices = np.argsort(fitness_values)

    # Boundary points have infinite distance
    distances[sorted_indices[0]] = np.inf
    distances[sorted_indices[-1]] = np.inf

    # Calculate distances for middle points
    fitness_range = fitness_values[sorted_indices[-1]] - fitness_values[sorted_indices[0]]

    if fitness_range > 0:
        for i in range(1, pop_size - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]

            distances[idx] = (fitness_values[next_idx] - fitness_values[prev_idx]) / fitness_range

    return distances


def niching_selection(population, fitness_values, num_parents, sigma_share=0.1):
    """
    Niching selection to maintain population diversity.

    Arguments:
    population -- array of individuals
    fitness_values -- array of fitness values
    num_parents -- number to select
    sigma_share -- sharing radius

    Returns:
    selected_parents
    """
    pop_size = len(population)
    shared_fitness = fitness_values.copy()

    # Calculate sharing
    for i in range(pop_size):
        niche_count = 0
        for j in range(pop_size):
            distance = np.linalg.norm(population[i] - population[j])
            if distance < sigma_share:
                sharing = 1 - (distance / sigma_share)
                niche_count += sharing

        if niche_count > 0:
            shared_fitness[i] = fitness_values[i] / niche_count

    # Tournament selection with shared fitness
    selected = []
    for _ in range(num_parents):
        tournament_indices = np.random.choice(pop_size, size=3, replace=False)
        winner_idx = tournament_indices[np.argmax(shared_fitness[tournament_indices])]
        selected.append(population[winner_idx].copy())

    return np.array(selected)


# ==================== Adaptive Parameters ====================

def adaptive_crossover_rate(generation, max_generations, initial_rate=0.9, final_rate=0.6):
    """
    Adaptive crossover rate that changes over generations.

    Arguments:
    generation -- current generation
    max_generations -- total generations
    initial_rate -- starting rate
    final_rate -- ending rate

    Returns:
    current_rate
    """
    return initial_rate - (initial_rate - final_rate) * generation / max_generations


def adaptive_mutation_rate_fitness(fitness, max_fitness, min_fitness, k=2):
    """
    Adapt mutation rate based on individual fitness.
    Better individuals get lower mutation rates.

    Arguments:
    fitness -- individual's fitness
    max_fitness -- best fitness in population
    min_fitness -- worst fitness in population
    k -- scaling factor

    Returns:
    mutation_rate
    """
    if max_fitness - min_fitness > 0:
        normalized_fitness = (fitness - min_fitness) / (max_fitness - min_fitness)
        # Higher fitness -> lower mutation
        mutation_rate = 0.5 * (1 - normalized_fitness) ** k
    else:
        mutation_rate = 0.1

    return mutation_rate


# ==================== Visualization ====================

def plot_pareto_front(objectives1, objectives2, title="Pareto Front"):
    """
    Plot Pareto front for multi-objective optimization.

    Arguments:
    objectives1 -- first objective values
    objectives2 -- second objective values
    title -- plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives1, objectives2, c='blue', alpha=0.6, s=50)
    plt.xlabel('Objective 1', fontsize=12)
    plt.ylabel('Objective 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_selection_pressure(fitness_values, selection_method, num_samples=1000):
    """
    Visualize selection pressure of different selection methods.

    Arguments:
    fitness_values -- population fitness
    selection_method -- selection function
    num_samples -- number of selection trials
    """
    population = np.arange(len(fitness_values)).reshape(-1, 1)

    selection_counts = np.zeros(len(population))

    for _ in range(num_samples):
        selected = selection_method(population, fitness_values, 1)
        selection_counts[int(selected[0, 0])] += 1

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(fitness_values)), fitness_values, alpha=0.7)
    plt.xlabel('Individual')
    plt.ylabel('Fitness')
    plt.title('Population Fitness')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(range(len(selection_counts)), selection_counts / num_samples, alpha=0.7, color='green')
    plt.xlabel('Individual')
    plt.ylabel('Selection Probability')
    plt.title(f'Selection Pressure ({num_samples} trials)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
