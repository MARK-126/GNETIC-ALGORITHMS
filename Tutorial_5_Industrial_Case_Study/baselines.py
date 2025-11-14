"""
Baseline Heuristics for Vehicle Routing Problem

Classic algorithms for comparison with GA:
- Nearest Neighbor (Greedy)
- Clarke-Wright Savings Algorithm
- Sweep Algorithm
"""

import numpy as np
from typing import List, Tuple, Dict
from delivery_problem import DeliveryProblem, Route, Customer
from delivery_ga import evaluate_solution, encode_solution


# ==================== Nearest Neighbor ====================

def nearest_neighbor(problem: DeliveryProblem, start_customer: int = 0) -> Tuple[np.ndarray, float, Dict]:
    """
    Nearest Neighbor heuristic (greedy approach).

    Builds routes by always selecting the nearest unvisited customer.

    Arguments:
    problem -- DeliveryProblem instance
    start_customer -- optional starting customer

    Returns:
    solution -- encoded solution
    fitness -- solution fitness
    metrics -- detailed metrics
    """
    n_customers = len(problem.customers)
    unvisited = set(range(n_customers))
    routes = []
    vehicle_id = 0

    while unvisited and vehicle_id < len(problem.vehicles):
        route = []
        current_location = 0  # Start at depot
        current_load = 0.0
        vehicle = problem.vehicles[vehicle_id]

        # Start with specified customer if first route
        if vehicle_id == 0 and start_customer in unvisited:
            route.append(start_customer)
            unvisited.remove(start_customer)
            current_location = start_customer + 1
            current_load += problem.customers[start_customer].demand

        # Build route greedily
        while unvisited:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')

            for customer_id in unvisited:
                customer = problem.customers[customer_id]
                customer_location = customer_id + 1

                # Check capacity
                if current_load + customer.demand > vehicle.capacity:
                    continue

                distance = problem.get_distance(current_location, customer_location)

                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer_id

            if best_customer is None:
                # No feasible customer, start new route
                break

            route.append(best_customer)
            unvisited.remove(best_customer)
            current_location = best_customer + 1
            current_load += problem.customers[best_customer].demand

        if route:
            routes.append(route)
            vehicle_id += 1

    # Encode solution
    solution = encode_solution(routes, n_customers)
    fitness, metrics = evaluate_solution(solution, problem)

    return solution, fitness, metrics


# ==================== Clarke-Wright Savings ====================

def clarke_wright(problem: DeliveryProblem) -> Tuple[np.ndarray, float, Dict]:
    """
    Clarke-Wright Savings Algorithm.

    Classic VRP heuristic that merges routes based on savings.

    Savings s(i,j) = distance(depot,i) + distance(depot,j) - distance(i,j)

    Arguments:
    problem -- DeliveryProblem instance

    Returns:
    solution -- encoded solution
    fitness -- solution fitness
    metrics -- detailed metrics
    """
    n_customers = len(problem.customers)

    # Calculate all savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            # Savings from merging routes i and j
            saving = (problem.get_distance(0, i + 1) +
                     problem.get_distance(0, j + 1) -
                     problem.get_distance(i + 1, j + 1))
            savings.append((saving, i, j))

    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])

    # Initialize: each customer in own route
    routes = [[i] for i in range(n_customers)]
    route_assignment = {i: i for i in range(n_customers)}  # customer -> route_id

    # Merge routes based on savings
    for saving, i, j in savings:
        route_i = route_assignment[i]
        route_j = route_assignment[j]

        if route_i == route_j:
            continue  # Already in same route

        # Check if merging is feasible
        merged_route = routes[route_i] + routes[route_j]

        # Check capacity
        total_demand = sum(problem.customers[c].demand for c in merged_route)
        if total_demand > problem.vehicles[0].capacity:
            continue

        # Check if customers are at route ends
        if i == routes[route_i][-1] and j == routes[route_j][0]:
            # Merge: route_i + route_j
            routes[route_i].extend(routes[route_j])
            # Update assignments
            for customer in routes[route_j]:
                route_assignment[customer] = route_i
            routes[route_j] = []

        elif j == routes[route_j][-1] and i == routes[route_i][0]:
            # Merge: route_j + route_i
            routes[route_j].extend(routes[route_i])
            for customer in routes[route_i]:
                route_assignment[customer] = route_j
            routes[route_i] = []

    # Remove empty routes
    routes = [r for r in routes if r]

    # Limit to available vehicles
    if len(routes) > len(problem.vehicles):
        routes = routes[:len(problem.vehicles)]

    # Encode solution
    solution = encode_solution(routes, n_customers)
    fitness, metrics = evaluate_solution(solution, problem)

    return solution, fitness, metrics


# ==================== Sweep Algorithm ====================

def sweep_algorithm(problem: DeliveryProblem) -> Tuple[np.ndarray, float, Dict]:
    """
    Sweep Algorithm for VRP.

    Sorts customers by polar angle from depot and builds routes
    by sweeping around.

    Arguments:
    problem -- DeliveryProblem instance

    Returns:
    solution -- encoded solution
    fitness -- solution fitness
    metrics -- detailed metrics
    """
    n_customers = len(problem.customers)

    # Calculate polar angles relative to depot
    depot_lat = problem.depot.lat
    depot_lon = problem.depot.lon

    angles = []
    for i, customer in enumerate(problem.customers):
        delta_lat = customer.lat - depot_lat
        delta_lon = customer.lon - depot_lon
        angle = np.arctan2(delta_lat, delta_lon)
        angles.append((angle, i))

    # Sort by angle
    angles.sort()
    sorted_customers = [customer_id for _, customer_id in angles]

    # Build routes by sweeping
    routes = []
    current_route = []
    current_load = 0.0
    vehicle_id = 0

    for customer_id in sorted_customers:
        customer = problem.customers[customer_id]

        # Check if adding customer exceeds capacity
        if (current_load + customer.demand > problem.vehicles[vehicle_id].capacity and
            current_route):
            # Start new route
            routes.append(current_route)
            current_route = []
            current_load = 0.0
            vehicle_id += 1

            if vehicle_id >= len(problem.vehicles):
                # No more vehicles, force add remaining
                break

        current_route.append(customer_id)
        current_load += customer.demand

    # Add last route
    if current_route and vehicle_id < len(problem.vehicles):
        routes.append(current_route)

    # Encode solution
    solution = encode_solution(routes, n_customers)
    fitness, metrics = evaluate_solution(solution, problem)

    return solution, fitness, metrics


# ==================== Random Solution ====================

def random_solution(problem: DeliveryProblem) -> Tuple[np.ndarray, float, Dict]:
    """
    Random solution for baseline comparison.

    Arguments:
    problem -- DeliveryProblem instance

    Returns:
    solution -- encoded random solution
    fitness -- solution fitness
    metrics -- detailed metrics
    """
    from delivery_ga import create_random_solution

    solution = create_random_solution(problem)
    fitness, metrics = evaluate_solution(solution, problem)

    return solution, fitness, metrics


# ==================== Comparison Function ====================

def compare_algorithms(problem: DeliveryProblem, verbose: bool = True) -> Dict:
    """
    Compare all baseline algorithms.

    Arguments:
    problem -- DeliveryProblem instance
    verbose -- print results

    Returns:
    results -- dictionary with results for each algorithm
    """
    if verbose:
        print("\n" + "="*70)
        print("BASELINE ALGORITHMS COMPARISON")
        print("="*70)
        print(f"Problem: {problem.name}")
        print(f"Customers: {len(problem.customers)}")
        print(f"Vehicles: {len(problem.vehicles)}")
        print("="*70 + "\n")

    results = {}

    # Random
    if verbose:
        print("Running Random Solution...")
    random_sol, random_fit, random_metrics = random_solution(problem)
    results['random'] = {
        'solution': random_sol,
        'fitness': random_fit,
        'metrics': random_metrics
    }

    # Nearest Neighbor
    if verbose:
        print("Running Nearest Neighbor...")
    nn_sol, nn_fit, nn_metrics = nearest_neighbor(problem)
    results['nearest_neighbor'] = {
        'solution': nn_sol,
        'fitness': nn_fit,
        'metrics': nn_metrics
    }

    # Clarke-Wright
    if verbose:
        print("Running Clarke-Wright...")
    cw_sol, cw_fit, cw_metrics = clarke_wright(problem)
    results['clarke_wright'] = {
        'solution': cw_sol,
        'fitness': cw_fit,
        'metrics': cw_metrics
    }

    # Sweep
    if verbose:
        print("Running Sweep Algorithm...")
    sweep_sol, sweep_fit, sweep_metrics = sweep_algorithm(problem)
    results['sweep'] = {
        'solution': sweep_sol,
        'fitness': sweep_fit,
        'metrics': sweep_metrics
    }

    # Print comparison
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"{'Algorithm':<20} {'Distance (km)':<15} {'Vehicles':<10} {'Feasible':<10}")
        print("-"*70)

        for alg_name, alg_results in results.items():
            metrics = alg_results['metrics']
            feasible = "✓" if metrics['is_feasible'] else "✗"
            print(f"{alg_name:<20} {metrics['total_distance']:<15.2f} "
                  f"{metrics['n_vehicles']:<10} {feasible:<10}")

        print("="*70 + "\n")

    return results


if __name__ == "__main__":
    from delivery_problem import DeliveryProblem, Customer, Depot, Vehicle

    # Create test problem
    depot = Depot(0, "Central Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Customer {i}",
                4.6097 + np.random.randn()*0.05,
                -74.0817 + np.random.randn()*0.05,
                np.random.uniform(5, 20),
                (8 + np.random.uniform(0, 2), 14 + np.random.uniform(0, 4)),
                np.random.uniform(5, 15),
                np.random.choice([1, 2, 3], p=[0.1, 0.3, 0.6]))
        for i in range(15)
    ]

    vehicles = [Vehicle(i, 80.0, 120.0, 35.0, 50.0, 0.5) for i in range(4)]

    problem = DeliveryProblem(customers, depot, vehicles, "Baseline Test")

    # Compare algorithms
    results = compare_algorithms(problem, verbose=True)
