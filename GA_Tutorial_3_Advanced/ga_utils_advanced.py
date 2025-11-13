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
