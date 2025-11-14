"""
Genetic Algorithm Utilities - Advanced Applications
Tutorial 3: TSP, Multi-Objective Optimization, and Hybrid Algorithms

This module provides:
- TSP (Traveling Salesman Problem) specific operators
- Multi-objective optimization basics
- Hybrid GA with local search
- Advanced visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import permutations


# ==================== TSP Utilities ====================

def calculate_tsp_distance(tour, distance_matrix):
    """
    Calculate total distance of a TSP tour.

    Arguments:
    tour -- array of city indices representing the tour
    distance_matrix -- matrix of distances between cities

    Returns:
    total_distance -- total tour distance
    """
    total_distance = 0
    n_cities = len(tour)

    for i in range(n_cities):
        from_city = tour[i]
        to_city = tour[(i + 1) % n_cities]  # Wrap around to start
        total_distance += distance_matrix[from_city, to_city]

    return total_distance


def generate_random_cities(n_cities, bounds=(0, 100)):
    """
    Generate random city coordinates.

    Arguments:
    n_cities -- number of cities
    bounds -- tuple (min, max) for coordinates

    Returns:
    cities -- array of shape (n_cities, 2) with (x, y) coordinates
    """
    return np.random.uniform(bounds[0], bounds[1], size=(n_cities, 2))


def create_distance_matrix(cities):
    """
    Create distance matrix from city coordinates.

    Arguments:
    cities -- array of city coordinates (n_cities, 2)

    Returns:
    distance_matrix -- symmetric matrix of Euclidean distances
    """
    n_cities = len(cities)
    distance_matrix = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])

    return distance_matrix


# ==================== TSP-Specific Operators ====================

def initialize_tsp_population(pop_size, n_cities):
    """
    Initialize population of TSP tours (permutations).

    Arguments:
    pop_size -- population size
    n_cities -- number of cities

    Returns:
    population -- array of tours
    """
    population = []
    for _ in range(pop_size):
        tour = np.random.permutation(n_cities)
        population.append(tour)
    return np.array(population)


def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) for TSP - preserves relative order.

    Arguments:
    parent1, parent2 -- parent tours

    Returns:
    offspring1, offspring2 -- offspring tours
    """
    size = len(parent1)

    # Choose two random crossover points
    cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False))

    # Create offspring 1
    offspring1 = np.full(size, -1)
    offspring1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    # Fill remaining positions from parent2
    current_pos = cx_point2
    for city in np.concatenate([parent2[cx_point2:], parent2[:cx_point2]]):
        if city not in offspring1:
            if current_pos >= size:
                current_pos = 0
            offspring1[current_pos] = city
            current_pos += 1

    # Create offspring 2 (symmetric process)
    offspring2 = np.full(size, -1)
    offspring2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

    current_pos = cx_point2
    for city in np.concatenate([parent1[cx_point2:], parent1[:cx_point2]]):
        if city not in offspring2:
            if current_pos >= size:
                current_pos = 0
            offspring2[current_pos] = city
            current_pos += 1

    return offspring1.astype(int), offspring2.astype(int)


def partially_mapped_crossover(parent1, parent2):
    """
    Partially Mapped Crossover (PMX) for TSP.

    Arguments:
    parent1, parent2 -- parent tours

    Returns:
    offspring1, offspring2 -- offspring tours
    """
    size = len(parent1)

    # Choose crossover points
    cx_point1, cx_point2 = sorted(np.random.choice(range(size), 2, replace=False))

    # Initialize offspring as copies
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Swap segments
    offspring1[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
    offspring2[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]

    # Fix conflicts for offspring1
    for i in range(cx_point1, cx_point2):
        if parent1[i] not in offspring1[cx_point1:cx_point2]:
            # Find where to put parent1[i]
            current = parent2[i]
            while cx_point1 <= np.where(parent2 == current)[0][0] < cx_point2:
                current = parent2[np.where(parent1 == current)[0][0]]
            offspring1[np.where(parent2 == current)[0][0]] = parent1[i]

    # Fix conflicts for offspring2
    for i in range(cx_point1, cx_point2):
        if parent2[i] not in offspring2[cx_point1:cx_point2]:
            current = parent1[i]
            while cx_point1 <= np.where(parent1 == current)[0][0] < cx_point2:
                current = parent1[np.where(parent2 == current)[0][0]]
            offspring2[np.where(parent1 == current)[0][0]] = parent2[i]

    return offspring1, offspring2


def swap_mutation(tour, mutation_rate=0.1):
    """
    Swap mutation for TSP - swaps two random cities.

    Arguments:
    tour -- tour to mutate
    mutation_rate -- probability of mutation

    Returns:
    mutated_tour
    """
    mutated = tour.copy()

    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

    return mutated


def inversion_mutation(tour, mutation_rate=0.1):
    """
    Inversion mutation for TSP - reverses a segment.

    Arguments:
    tour -- tour to mutate
    mutation_rate -- probability of mutation

    Returns:
    mutated_tour
    """
    mutated = tour.copy()

    if np.random.rand() < mutation_rate:
        idx1, idx2 = sorted(np.random.choice(len(tour), 2, replace=False))
        mutated[idx1:idx2+1] = mutated[idx1:idx2+1][::-1]

    return mutated


def scramble_mutation(tour, mutation_rate=0.1):
    """
    Scramble mutation - shuffles a segment.

    Arguments:
    tour -- tour to mutate
    mutation_rate -- probability of mutation

    Returns:
    mutated_tour
    """
    mutated = tour.copy()

    if np.random.rand() < mutation_rate:
        idx1, idx2 = sorted(np.random.choice(len(tour), 2, replace=False))
        segment = mutated[idx1:idx2+1].copy()
        np.random.shuffle(segment)
        mutated[idx1:idx2+1] = segment

    return mutated


# ==================== Local Search for Hybrid GA ====================

def two_opt_local_search(tour, distance_matrix, max_iterations=100):
    """
    2-opt local search for TSP improvement.

    Arguments:
    tour -- initial tour
    distance_matrix -- distance matrix
    max_iterations -- max iterations for improvement

    Returns:
    improved_tour -- locally optimized tour
    """
    n = len(tour)
    improved_tour = tour.copy()
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 1):
            for j in range(i + 2, n):
                # Calculate current distance
                current_dist = (distance_matrix[improved_tour[i], improved_tour[i+1]] +
                              distance_matrix[improved_tour[j], improved_tour[(j+1) % n]])

                # Calculate new distance after swap
                new_dist = (distance_matrix[improved_tour[i], improved_tour[j]] +
                          distance_matrix[improved_tour[i+1], improved_tour[(j+1) % n]])

                # If improvement, make the swap
                if new_dist < current_dist:
                    improved_tour[i+1:j+1] = improved_tour[i+1:j+1][::-1]
                    improved = True

    return improved_tour


# ==================== Multi-Objective Optimization ====================

def dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2 (for minimization).

    Arguments:
    obj1, obj2 -- arrays of objective values

    Returns:
    True if obj1 dominates obj2
    """
    # obj1 dominates obj2 if it's no worse in all objectives and better in at least one
    return np.all(obj1 <= obj2) and np.any(obj1 < obj2)


def fast_non_dominated_sort(population, objectives):
    """
    Fast non-dominated sorting (NSGA-II).

    Arguments:
    population -- array of individuals
    objectives -- array of objective values (pop_size, n_objectives)

    Returns:
    fronts -- list of arrays, each containing indices of individuals in that front
    """
    pop_size = len(population)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]
    fronts = [[]]

    # For each solution
    for i in range(pop_size):
        for j in range(pop_size):
            if i != j:
                if dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif dominates(objectives[j], objectives[i]):
                    domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    # Find subsequent fronts
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove empty last front


def calculate_crowding_distance_multi(objectives):
    """
    Calculate crowding distance for multi-objective optimization.

    Arguments:
    objectives -- array of objective values (n_solutions, n_objectives)

    Returns:
    distances -- crowding distances
    """
    n_solutions = objectives.shape[0]
    n_objectives = objectives.shape[1]

    distances = np.zeros(n_solutions)

    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_indices = np.argsort(objectives[:, obj_idx])

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Calculate distances for other points
        obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]

        if obj_range > 0:
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                distances[idx] += (objectives[sorted_indices[i+1], obj_idx] -
                                 objectives[sorted_indices[i-1], obj_idx]) / obj_range

    return distances


# ==================== Visualization ====================

def plot_tsp_tour(cities, tour, title="TSP Tour"):
    """
    Plot TSP tour on 2D plane.

    Arguments:
    cities -- city coordinates
    tour -- tour (sequence of city indices)
    title -- plot title
    """
    plt.figure(figsize=(10, 8))

    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=3, label='Cities')

    # Plot tour
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        plt.plot([cities[from_city, 0], cities[to_city, 0]],
                [cities[from_city, 1], cities[to_city, 1]],
                'b-', alpha=0.6, linewidth=2)

    # Label cities
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_pareto_front_2d(objectives, title="Pareto Front"):
    """
    Plot 2D Pareto front.

    Arguments:
    objectives -- array of shape (n_solutions, 2)
    title -- plot title
    """
    # Perform non-dominated sorting
    dummy_pop = np.arange(len(objectives)).reshape(-1, 1)
    fronts = fast_non_dominated_sort(dummy_pop, objectives)

    plt.figure(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, front in enumerate(fronts[:5]):  # Plot first 5 fronts
        front_objectives = objectives[front]
        color = colors[i % len(colors)]
        label = f'Front {i+1}' if i == 0 else f'Front {i+1}'
        marker = 'o' if i == 0 else 's'
        size = 100 if i == 0 else 50

        plt.scatter(front_objectives[:, 0], front_objectives[:, 1],
                   c=color, marker=marker, s=size, alpha=0.7, label=label, edgecolors='black')

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence_multi_objective(history, title="Multi-Objective Convergence"):
    """
    Plot convergence for multi-objective optimization.

    Arguments:
    history -- dictionary with 'hypervolume' or other metrics
    title -- plot title
    """
    plt.figure(figsize=(10, 5))

    if 'hypervolume' in history:
        plt.plot(history['hypervolume'], linewidth=2, label='Hypervolume')
        plt.ylabel('Hypervolume')
    elif 'spread' in history:
        plt.plot(history['spread'], linewidth=2, label='Spread')
        plt.ylabel('Spread')

    plt.xlabel('Generation')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== Benchmark TSP Instances ====================

def create_symmetric_tsp(n_cities):
    """Create a symmetric TSP instance."""
    cities = generate_random_cities(n_cities)
    distance_matrix = create_distance_matrix(cities)
    return cities, distance_matrix


def create_circle_tsp(n_cities, radius=50):
    """
    Create TSP with cities arranged in a circle (known optimal).

    Arguments:
    n_cities -- number of cities
    radius -- circle radius

    Returns:
    cities, distance_matrix, optimal_tour, optimal_distance
    """
    angles = np.linspace(0, 2*np.pi, n_cities, endpoint=False)
    cities = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    distance_matrix = create_distance_matrix(cities)

    # Optimal tour is simply 0,1,2,...,n-1
    optimal_tour = np.arange(n_cities)
    optimal_distance = calculate_tsp_distance(optimal_tour, distance_matrix)

    return cities, distance_matrix, optimal_tour, optimal_distance


# ==================== Hybrid GA Helpers ====================

def lamarckian_evolution(population, distance_matrix, local_search_prob=0.2):
    """
    Apply local search to some individuals (Lamarckian evolution).

    Arguments:
    population -- current population
    distance_matrix -- TSP distance matrix
    local_search_prob -- probability of applying local search

    Returns:
    improved_population
    """
    improved_pop = []

    for individual in population:
        if np.random.rand() < local_search_prob:
            improved = two_opt_local_search(individual, distance_matrix, max_iterations=50)
            improved_pop.append(improved)
        else:
            improved_pop.append(individual)

    return np.array(improved_pop)


def baldwin_evolution(population, fitness_values, distance_matrix, local_search_prob=0.2):
    """
    Baldwin effect - use local search for fitness evaluation only, don't modify genotype.

    Arguments:
    population -- current population
    fitness_values -- current fitness
    distance_matrix -- TSP distance matrix
    local_search_prob -- probability of local search evaluation

    Returns:
    adjusted_fitness
    """
    adjusted_fitness = fitness_values.copy()

    for i, individual in enumerate(population):
        if np.random.rand() < local_search_prob:
            improved = two_opt_local_search(individual, distance_matrix, max_iterations=50)
            improved_distance = calculate_tsp_distance(improved, distance_matrix)
            adjusted_fitness[i] = -improved_distance  # Fitness = negative distance

    return adjusted_fitness


# ==================== NSGA-II Complete Implementation ====================

def nsga2_selection(population, objectives, n_select):
    """
    NSGA-II selection based on non-domination rank and crowding distance.

    Arguments:
    population -- current population
    objectives -- objective values (pop_size, n_objectives)
    n_select -- number of individuals to select

    Returns:
    selected_population -- selected individuals
    selected_objectives -- their objectives
    """
    # Fast non-dominated sort
    fronts = fast_non_dominated_sort(population, objectives)

    # Calculate crowding distance for each front
    crowding_distances = np.zeros(len(population))
    for front in fronts:
        if len(front) > 0:
            front_objectives = objectives[front]
            distances = calculate_crowding_distance_multi(front_objectives)
            crowding_distances[front] = distances

    # Select individuals
    selected_indices = []

    for front in fronts:
        if len(selected_indices) + len(front) <= n_select:
            # Add entire front
            selected_indices.extend(front)
        else:
            # Add part of front based on crowding distance
            remaining = n_select - len(selected_indices)
            front_distances = crowding_distances[front]
            # Sort by crowding distance (descending)
            sorted_front = [front[i] for i in np.argsort(-front_distances)]
            selected_indices.extend(sorted_front[:remaining])
            break

    selected_population = population[selected_indices]
    selected_objectives = objectives[selected_indices]

    return selected_population, selected_objectives


def nsga2(objective_functions, n_objectives, bounds, pop_size=100, max_generations=100,
          crossover_rate=0.9, mutation_rate=None, eta_c=20, eta_m=20):
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    Complete implementation for multi-objective optimization.

    Arguments:
    objective_functions -- list of objective functions to minimize
    n_objectives -- number of objectives
    bounds -- list of (min, max) tuples for each variable
    pop_size -- population size (should be even)
    max_generations -- maximum generations
    crossover_rate -- crossover probability
    mutation_rate -- mutation probability (default: 1/n_vars)
    eta_c -- distribution index for SBX crossover
    eta_m -- distribution index for polynomial mutation

    Returns:
    final_population -- final population
    final_objectives -- final objective values
    pareto_front -- indices of Pareto optimal solutions
    history -- evolution history
    """
    n_vars = len(bounds)
    bounds = np.array(bounds)

    if mutation_rate is None:
        mutation_rate = 1.0 / n_vars

    # Ensure pop_size is even
    if pop_size % 2 != 0:
        pop_size += 1

    # Initialize population
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], (pop_size, n_vars)
    )

    # Evaluate objectives
    objectives = np.array([[f(ind) for f in objective_functions]
                          for ind in population])

    history = {
        'hypervolume': [],
        'n_pareto': [],
        'spread': []
    }

    for generation in range(max_generations):
        # Create offspring through crossover and mutation
        offspring = []

        for i in range(0, pop_size, 2):
            # Select parents (binary tournament)
            parent1_idx = binary_tournament_selection_nsga2(
                population, objectives, 2
            )
            parent2_idx = binary_tournament_selection_nsga2(
                population, objectives, 2
            )

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover (SBX)
            if np.random.rand() < crossover_rate:
                child1, child2 = sbx_crossover(parent1, parent2, eta_c, bounds)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation (Polynomial)
            child1 = polynomial_mutation(child1, mutation_rate, eta_m, bounds)
            child2 = polynomial_mutation(child2, mutation_rate, eta_m, bounds)

            offspring.extend([child1, child2])

        offspring = np.array(offspring)

        # Evaluate offspring
        offspring_objectives = np.array([[f(ind) for f in objective_functions]
                                        for ind in offspring])

        # Combine parent and offspring populations
        combined_population = np.vstack([population, offspring])
        combined_objectives = np.vstack([objectives, offspring_objectives])

        # Select next generation using NSGA-II selection
        population, objectives = nsga2_selection(
            combined_population, combined_objectives, pop_size
        )

        # Track metrics
        fronts = fast_non_dominated_sort(population, objectives)
        history['n_pareto'].append(len(fronts[0]) if len(fronts) > 0 else 0)

        # Spread metric (diversity)
        if len(fronts) > 0 and len(fronts[0]) > 1:
            pareto_objectives = objectives[fronts[0]]
            distances = calculate_crowding_distance_multi(pareto_objectives)
            history['spread'].append(np.std(distances[np.isfinite(distances)]))
        else:
            history['spread'].append(0.0)

    # Get final Pareto front
    fronts = fast_non_dominated_sort(population, objectives)
    pareto_front = fronts[0] if len(fronts) > 0 else []

    return population, objectives, pareto_front, history


def binary_tournament_selection_nsga2(population, objectives, n_tournaments=1):
    """
    Binary tournament selection for NSGA-II.

    Compares based on:
    1. Non-domination rank
    2. Crowding distance (if same rank)

    Arguments:
    population -- current population
    objectives -- objective values
    n_tournaments -- number of tournaments

    Returns:
    selected_index -- index of winner
    """
    pop_size = len(population)

    # Get ranks
    fronts = fast_non_dominated_sort(population, objectives)
    ranks = np.zeros(pop_size, dtype=int)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank

    # Get crowding distances
    crowding_distances = np.zeros(pop_size)
    for front in fronts:
        if len(front) > 1:
            front_objectives = objectives[front]
            distances = calculate_crowding_distance_multi(front_objectives)
            crowding_distances[front] = distances

    # Tournament
    idx1, idx2 = np.random.choice(pop_size, 2, replace=False)

    # Compare
    if ranks[idx1] < ranks[idx2]:  # Lower rank is better
        return idx1
    elif ranks[idx1] > ranks[idx2]:
        return idx2
    else:
        # Same rank, compare crowding distance
        if crowding_distances[idx1] > crowding_distances[idx2]:
            return idx1
        else:
            return idx2


def sbx_crossover(parent1, parent2, eta, bounds):
    """
    Simulated Binary Crossover (SBX) for real-coded GAs.

    Arguments:
    parent1, parent2 -- parent solutions
    eta -- distribution index (larger eta -> more similar to parents)
    bounds -- variable bounds

    Returns:
    child1, child2 -- offspring
    """
    n_vars = len(parent1)
    child1 = np.zeros(n_vars)
    child2 = np.zeros(n_vars)

    for i in range(n_vars):
        if np.random.rand() < 0.5:
            if np.abs(parent1[i] - parent2[i]) > 1e-6:
                # Calculate beta
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])

                lower_bound = bounds[i, 0]
                upper_bound = bounds[i, 1]

                beta_l = 1.0 + 2.0 * (y1 - lower_bound) / (y2 - y1)
                beta_u = 1.0 + 2.0 * (upper_bound - y2) / (y2 - y1)

                alpha = 2.0 - beta_l ** -(eta + 1.0)
                rand = np.random.rand()

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                child1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                alpha = 2.0 - beta_u ** -(eta + 1.0)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                child2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                # Clamp to bounds
                child1[i] = np.clip(child1[i], lower_bound, upper_bound)
                child2[i] = np.clip(child2[i], lower_bound, upper_bound)
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2


def polynomial_mutation(individual, mutation_rate, eta, bounds):
    """
    Polynomial mutation for real-coded GAs.

    Arguments:
    individual -- solution to mutate
    mutation_rate -- probability of mutating each variable
    eta -- distribution index
    bounds -- variable bounds

    Returns:
    mutated -- mutated solution
    """
    mutated = individual.copy()
    n_vars = len(individual)

    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            y = mutated[i]
            lower_bound = bounds[i, 0]
            upper_bound = bounds[i, 1]

            delta_1 = (y - lower_bound) / (upper_bound - lower_bound)
            delta_2 = (upper_bound - y) / (upper_bound - lower_bound)

            rand = np.random.rand()

            if rand < 0.5:
                delta_q = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta_1) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta_2) ** (eta + 1.0)) ** (1.0 / (eta + 1.0))

            mutated[i] = y + delta_q * (upper_bound - lower_bound)
            mutated[i] = np.clip(mutated[i], lower_bound, upper_bound)

    return mutated


# ==================== ZDT Test Problems ====================

def zdt1(x):
    """
    ZDT1 test problem.

    Convex Pareto front.
    n_vars = 30

    Returns:
    (f1, f2) -- tuple of two objectives
    """
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


def zdt2(x):
    """
    ZDT2 test problem.

    Non-convex Pareto front.
    n_vars = 30

    Returns:
    (f1, f2) -- tuple of two objectives
    """
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return f1, f2


def zdt3(x):
    """
    ZDT3 test problem.

    Disconnected Pareto front.
    n_vars = 30

    Returns:
    (f1, f2) -- tuple of two objectives
    """
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return f1, f2


def zdt4(x):
    """
    ZDT4 test problem.

    Many local Pareto fronts.
    n_vars = 10
    x[0] in [0,1], x[1:] in [-5,5]

    Returns:
    (f1, f2) -- tuple of two objectives
    """
    n = len(x)
    f1 = x[0]
    g = 1.0 + 10.0 * (n - 1) + np.sum(x[1:]**2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


def zdt6(x):
    """
    ZDT6 test problem.

    Non-uniform search space, low density near Pareto front.
    n_vars = 10

    Returns:
    (f1, f2) -- tuple of two objectives
    """
    n = len(x)
    f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
    g = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return f1, f2


def get_zdt_problem(problem_name):
    """
    Get ZDT problem configuration.

    Arguments:
    problem_name -- 'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6'

    Returns:
    objective_functions, bounds, n_vars
    """
    if problem_name.upper() == 'ZDT1':
        return [lambda x: zdt1(x)[0], lambda x: zdt1(x)[1]], [(0.0, 1.0)] * 30, 30
    elif problem_name.upper() == 'ZDT2':
        return [lambda x: zdt2(x)[0], lambda x: zdt2(x)[1]], [(0.0, 1.0)] * 30, 30
    elif problem_name.upper() == 'ZDT3':
        return [lambda x: zdt3(x)[0], lambda x: zdt3(x)[1]], [(0.0, 1.0)] * 30, 30
    elif problem_name.upper() == 'ZDT4':
        bounds = [(0.0, 1.0)] + [(-5.0, 5.0)] * 9
        return [lambda x: zdt4(x)[0], lambda x: zdt4(x)[1]], bounds, 10
    elif problem_name.upper() == 'ZDT6':
        return [lambda x: zdt6(x)[0], lambda x: zdt6(x)[1]], [(0.0, 1.0)] * 10, 10
    else:
        raise ValueError(f"Unknown ZDT problem: {problem_name}")
