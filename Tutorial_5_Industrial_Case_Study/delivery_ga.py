"""
Genetic Algorithm for Vehicle Routing Problem with Time Windows (VRPTW)

Specialized GA with route-aware operators and multi-objective fitness.
"""

import numpy as np
from typing import List, Tuple, Dict
from delivery_problem import DeliveryProblem, Route, Customer
import copy


# ==================== Chromosome Encoding/Decoding ====================

def encode_solution(routes: List[List[int]], n_customers: int) -> np.ndarray:
    """
    Encode routes into chromosome.

    Chromosome format: customer sequence with -1 separators for vehicles
    Example: [2, 5, -1, 0, 3, 4, -1, 1] means:
      Vehicle 0: customers [2, 5]
      Vehicle 1: customers [0, 3, 4]
      Vehicle 2: customer [1]

    Arguments:
    routes -- list of routes, each route is list of customer IDs
    n_customers -- total number of customers

    Returns:
    chromosome -- 1D array
    """
    chromosome = []
    for route in routes:
        chromosome.extend(route)
        chromosome.append(-1)  # Separator

    # Remove last separator
    if chromosome and chromosome[-1] == -1:
        chromosome.pop()

    return np.array(chromosome, dtype=int)


def decode_solution(chromosome: np.ndarray, problem: DeliveryProblem) -> List[Route]:
    """
    Decode chromosome into routes.

    Arguments:
    chromosome -- encoded solution
    problem -- DeliveryProblem instance

    Returns:
    routes -- list of Route objects
    """
    routes = []
    current_route = []
    vehicle_id = 0

    for gene in chromosome:
        if gene == -1:
            # End of route
            if current_route:
                routes.append(Route(vehicle_id, current_route, problem))
                current_route = []
                vehicle_id += 1
        elif gene >= 0:
            current_route.append(int(gene))

    # Add last route if exists
    if current_route and vehicle_id < len(problem.vehicles):
        routes.append(Route(vehicle_id, current_route, problem))

    return routes


def create_random_solution(problem: DeliveryProblem) -> np.ndarray:
    """
    Create random valid solution.

    Arguments:
    problem -- DeliveryProblem instance

    Returns:
    chromosome -- random encoded solution
    """
    n_customers = len(problem.customers)
    customer_ids = list(range(n_customers))
    np.random.shuffle(customer_ids)

    # Randomly split into routes
    n_vehicles = min(len(problem.vehicles), n_customers)
    routes = []

    # Random split points
    split_points = sorted(np.random.choice(range(1, n_customers), n_vehicles - 1, replace=False))
    split_points = [0] + list(split_points) + [n_customers]

    for i in range(len(split_points) - 1):
        route = customer_ids[split_points[i]:split_points[i+1]]
        if route:
            routes.append(route)

    return encode_solution(routes, n_customers)


# ==================== Fitness Evaluation ====================

def evaluate_solution(chromosome: np.ndarray, problem: DeliveryProblem,
                     weights: Dict[str, float] = None) -> Tuple[float, Dict]:
    """
    Evaluate solution quality (multi-objective).

    Fitness components:
    - Total distance (minimize)
    - Number of vehicles used (minimize)
    - Time window violations (minimize, high penalty)
    - Capacity violations (minimize, high penalty)
    - VIP customer satisfaction (maximize)

    Arguments:
    chromosome -- encoded solution
    problem -- DeliveryProblem instance
    weights -- optional weight dictionary for objectives

    Returns:
    fitness -- scalar fitness value (higher is better)
    metrics -- dictionary with detailed metrics
    """
    if weights is None:
        weights = {
            'distance': 0.3,
            'vehicles': 0.1,
            'time_violations': 0.3,
            'capacity_violations': 0.2,
            'vip_bonus': 0.1
        }

    # Decode solution
    routes = decode_solution(chromosome, problem)

    # Initialize metrics
    total_distance = 0.0
    total_cost = 0.0
    total_time_violations = 0.0
    total_capacity_violations = 0.0
    n_vehicles_used = len(routes)
    vip_score = 0.0

    delivered_customers = set()

    for route in routes:
        total_distance += route.total_distance
        total_cost += route.get_cost()
        total_time_violations += route.time_violations
        total_capacity_violations += route.capacity_violations

        # Track delivered customers
        for cust_id in route.customer_sequence:
            delivered_customers.add(cust_id)

            # VIP bonus (if delivered on time)
            customer = problem.customers[cust_id]
            if customer.priority == 1 and route.time_violations == 0:
                vip_score += 10.0

    # Penalty for undelivered customers (critical)
    n_undelivered = len(problem.customers) - len(delivered_customers)
    undelivered_penalty = n_undelivered * 1000.0

    # Calculate fitness (higher is better)
    # Normalize components
    distance_component = -total_distance / 100.0  # Normalize by typical scale
    vehicle_component = -n_vehicles_used * 10.0
    time_violation_component = -total_time_violations * 100.0  # High penalty
    capacity_violation_component = -total_capacity_violations * 50.0

    fitness = (
        weights['distance'] * distance_component +
        weights['vehicles'] * vehicle_component +
        weights['time_violations'] * time_violation_component +
        weights['capacity_violations'] * capacity_violation_component +
        weights['vip_bonus'] * vip_score -
        undelivered_penalty
    )

    metrics = {
        'total_distance': total_distance,
        'total_cost': total_cost,
        'n_vehicles': n_vehicles_used,
        'time_violations': total_time_violations,
        'capacity_violations': total_capacity_violations,
        'vip_score': vip_score,
        'n_delivered': len(delivered_customers),
        'n_undelivered': n_undelivered,
        'is_feasible': (total_time_violations == 0 and
                       total_capacity_violations == 0 and
                       n_undelivered == 0)
    }

    return fitness, metrics


# ==================== Genetic Operators ====================

def route_aware_crossover(parent1: np.ndarray, parent2: np.ndarray,
                         problem: DeliveryProblem) -> Tuple[np.ndarray, np.ndarray]:
    """
    Route-aware crossover operator.

    Preserves valid routes from both parents and repairs duplicates.

    Arguments:
    parent1, parent2 -- parent chromosomes
    problem -- DeliveryProblem instance

    Returns:
    child1, child2 -- offspring chromosomes
    """
    # Decode parents
    routes1 = decode_solution(parent1, problem)
    routes2 = decode_solution(parent2, problem)

    # Select random routes from each parent
    n_routes = min(len(routes1), len(routes2))
    if n_routes == 0:
        return parent1.copy(), parent2.copy()

    crossover_point = np.random.randint(1, n_routes) if n_routes > 1 else 1

    # Child 1: first part from parent1, second from parent2
    child1_routes = [r.customer_sequence for r in routes1[:crossover_point]]
    child2_routes = [r.customer_sequence for r in routes2[:crossover_point]]

    # Add remaining routes and repair
    delivered1 = set(c for r in child1_routes for c in r)
    delivered2 = set(c for r in child2_routes for c in r)

    # Add missing customers from parent2
    for route in routes2[crossover_point:]:
        filtered_route = [c for c in route.customer_sequence if c not in delivered1]
        if filtered_route:
            child1_routes.append(filtered_route)
            delivered1.update(filtered_route)

    # Add missing customers from parent1
    for route in routes1[crossover_point:]:
        filtered_route = [c for c in route.customer_sequence if c not in delivered2]
        if filtered_route:
            child2_routes.append(filtered_route)
            delivered2.update(filtered_route)

    # Add any remaining undelivered customers
    all_customers = set(range(len(problem.customers)))
    undelivered1 = all_customers - delivered1
    undelivered2 = all_customers - delivered2

    if undelivered1:
        child1_routes.append(list(undelivered1))
    if undelivered2:
        child2_routes.append(list(undelivered2))

    child1 = encode_solution(child1_routes, len(problem.customers))
    child2 = encode_solution(child2_routes, len(problem.customers))

    return child1, child2


def insertion_mutation(chromosome: np.ndarray, problem: DeliveryProblem,
                      mutation_rate: float = 0.1) -> np.ndarray:
    """
    Insertion mutation: remove customer and reinsert at best position.

    Arguments:
    chromosome -- chromosome to mutate
    problem -- DeliveryProblem instance
    mutation_rate -- probability of mutation

    Returns:
    mutated_chromosome
    """
    if np.random.rand() > mutation_rate:
        return chromosome.copy()

    routes = decode_solution(chromosome, problem)
    if not routes:
        return chromosome.copy()

    # Select random route
    route_idx = np.random.randint(len(routes))
    route = routes[route_idx].customer_sequence

    if len(route) < 2:
        return chromosome.copy()

    # Remove random customer
    remove_idx = np.random.randint(len(route))
    customer = route.pop(remove_idx)

    # Try to insert in best position (greedy)
    best_position = 0
    best_cost = float('inf')

    for route_idx2 in range(len(routes)):
        test_route = routes[route_idx2].customer_sequence.copy()
        for insert_idx in range(len(test_route) + 1):
            test_route_with_insert = test_route[:insert_idx] + [customer] + test_route[insert_idx:]
            test_route_obj = Route(routes[route_idx2].vehicle_id, test_route_with_insert, problem)

            if test_route_obj.total_distance < best_cost:
                best_cost = test_route_obj.total_distance
                best_position = (route_idx2, insert_idx)

    # Insert at best position
    target_route_idx, insert_idx = best_position
    routes[target_route_idx].customer_sequence.insert(insert_idx, customer)

    # Re-encode
    route_lists = [r.customer_sequence for r in routes]
    return encode_solution(route_lists, len(problem.customers))


def swap_mutation(chromosome: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Swap mutation: swap two random customers.

    Arguments:
    chromosome -- chromosome to mutate
    mutation_rate -- probability of mutation

    Returns:
    mutated_chromosome
    """
    if np.random.rand() > mutation_rate:
        return chromosome.copy()

    mutated = chromosome.copy()

    # Get indices of customers (not separators)
    customer_indices = np.where(mutated >= 0)[0]

    if len(customer_indices) < 2:
        return mutated

    # Swap two random customers
    idx1, idx2 = np.random.choice(customer_indices, 2, replace=False)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

    return mutated


def inversion_mutation(chromosome: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Inversion mutation: reverse a segment of route.

    Arguments:
    chromosome -- chromosome to mutate
    mutation_rate -- probability of mutation

    Returns:
    mutated_chromosome
    """
    if np.random.rand() > mutation_rate:
        return chromosome.copy()

    mutated = chromosome.copy()

    # Get indices of customers
    customer_indices = np.where(mutated >= 0)[0]

    if len(customer_indices) < 2:
        return mutated

    # Select segment to reverse
    idx1, idx2 = sorted(np.random.choice(len(customer_indices), 2, replace=False))
    actual_idx1 = customer_indices[idx1]
    actual_idx2 = customer_indices[idx2]

    # Reverse segment
    mutated[actual_idx1:actual_idx2+1] = mutated[actual_idx1:actual_idx2+1][::-1]

    return mutated


# ==================== Main GA ====================

def delivery_genetic_algorithm(problem: DeliveryProblem, pop_size: int = 100,
                               max_generations: int = 200, mutation_rate: float = 0.15,
                               crossover_rate: float = 0.8, elite_size: int = 5,
                               verbose: bool = True) -> Tuple[np.ndarray, float, Dict, List]:
    """
    Genetic Algorithm for Vehicle Routing Problem.

    Arguments:
    problem -- DeliveryProblem instance
    pop_size -- population size
    max_generations -- maximum generations
    mutation_rate -- mutation probability
    crossover_rate -- crossover probability
    elite_size -- number of elite individuals
    verbose -- print progress

    Returns:
    best_solution -- best chromosome found
    best_fitness -- fitness of best solution
    best_metrics -- detailed metrics of best solution
    history -- evolution history
    """
    if verbose:
        print("\n" + "="*70)
        print("GENETIC ALGORITHM FOR VEHICLE ROUTING")
        print("="*70)
        print(f"Problem: {problem.name}")
        print(f"Customers: {len(problem.customers)}")
        print(f"Vehicles: {len(problem.vehicles)}")
        print(f"Population size: {pop_size}")
        print(f"Max generations: {max_generations}")
        print("="*70 + "\n")

    # Initialize population
    population = [create_random_solution(problem) for _ in range(pop_size)]

    # Track history
    history = {
        'best_fitness': [],
        'mean_fitness': [],
        'best_distance': [],
        'best_vehicles': [],
        'feasibility_rate': []
    }

    best_overall_fitness = -float('inf')
    best_overall_solution = None
    best_overall_metrics = None

    for generation in range(max_generations):
        # Evaluate population
        fitness_values = []
        metrics_list = []

        for individual in population:
            fitness, metrics = evaluate_solution(individual, problem)
            fitness_values.append(fitness)
            metrics_list.append(metrics)

        fitness_values = np.array(fitness_values)

        # Track best
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > best_overall_fitness:
            best_overall_fitness = fitness_values[best_idx]
            best_overall_solution = population[best_idx].copy()
            best_overall_metrics = metrics_list[best_idx]

        # Update history
        history['best_fitness'].append(fitness_values[best_idx])
        history['mean_fitness'].append(fitness_values.mean())
        history['best_distance'].append(metrics_list[best_idx]['total_distance'])
        history['best_vehicles'].append(metrics_list[best_idx]['n_vehicles'])

        feasible_count = sum(1 for m in metrics_list if m['is_feasible'])
        history['feasibility_rate'].append(feasible_count / pop_size)

        # Progress report
        if verbose and (generation % 20 == 0 or generation == max_generations - 1):
            best_metrics = metrics_list[best_idx]
            print(f"Gen {generation:3d} | Fit: {fitness_values[best_idx]:7.2f} | "
                  f"Dist: {best_metrics['total_distance']:6.1f}km | "
                  f"Veh: {best_metrics['n_vehicles']} | "
                  f"Feasible: {feasible_count}/{pop_size} | "
                  f"TW Viol: {best_metrics['time_violations']:.1f}")

        # Selection (Tournament)
        selected = []
        for _ in range(pop_size - elite_size):
            tournament_idx = np.random.choice(pop_size, 3, replace=False)
            winner = tournament_idx[np.argmax(fitness_values[tournament_idx])]
            selected.append(population[winner].copy())

        # Elitism
        elite_indices = np.argsort(fitness_values)[-elite_size:]
        elite = [population[i].copy() for i in elite_indices]

        # Crossover
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if np.random.rand() < crossover_rate:
                child1, child2 = route_aware_crossover(selected[i], selected[i+1], problem)
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i].copy(), selected[i+1].copy()])

        # Mutation (mix of strategies)
        for i in range(len(offspring)):
            mutation_type = np.random.choice(['swap', 'insertion', 'inversion'], p=[0.4, 0.4, 0.2])

            if mutation_type == 'swap':
                offspring[i] = swap_mutation(offspring[i], mutation_rate)
            elif mutation_type == 'insertion':
                offspring[i] = insertion_mutation(offspring[i], problem, mutation_rate)
            else:
                offspring[i] = inversion_mutation(offspring[i], mutation_rate)

        # Form new population
        population = elite + offspring[:pop_size - elite_size]

    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best fitness: {best_overall_fitness:.2f}")
        print(f"Total distance: {best_overall_metrics['total_distance']:.2f} km")
        print(f"Total cost: ${best_overall_metrics['total_cost']:.2f}")
        print(f"Vehicles used: {best_overall_metrics['n_vehicles']}/{len(problem.vehicles)}")
        print(f"Time violations: {best_overall_metrics['time_violations']:.2f} hours")
        print(f"Capacity violations: {best_overall_metrics['capacity_violations']:.2f} kg")
        print(f"Feasible solution: {best_overall_metrics['is_feasible']}")
        print("="*70 + "\n")

    return best_overall_solution, best_overall_fitness, best_overall_metrics, history


if __name__ == "__main__":
    from delivery_problem import DeliveryProblem, Customer, Depot, Vehicle

    # Create test problem
    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Customer {i}", 4.6097 + np.random.randn()*0.05,
                -74.0817 + np.random.randn()*0.05, np.random.uniform(5, 25),
                (8 + np.random.uniform(0, 2), 12 + np.random.uniform(0, 4)),
                np.random.uniform(5, 15), np.random.choice([1, 2, 3]))
        for i in range(20)
    ]

    vehicles = [Vehicle(i, 100.0, 150.0, 35.0, 50.0, 0.5) for i in range(5)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test Problem")

    # Run GA
    best_solution, best_fitness, metrics, history = delivery_genetic_algorithm(
        problem, pop_size=50, max_generations=100
    )

    print("\nBest solution routes:")
    routes = decode_solution(best_solution, problem)
    for i, route in enumerate(routes):
        print(f"  {route}")
