"""
Public Test Cases for Tutorial 5 - Industrial Case Study
Tests for delivery optimization problem, GA, and baselines.
"""

import numpy as np
from delivery_problem import Customer, Depot, Vehicle, DeliveryProblem, Route
from delivery_ga import (
    encode_solution, decode_solution, create_random_solution,
    evaluate_solution, route_aware_crossover, swap_mutation
)
from baselines import nearest_neighbor, clarke_wright, sweep_algorithm


def test_customer_creation():
    """Test Customer creation."""
    print("Testing Customer creation...")

    customer = Customer(
        id=0,
        name="Test Store",
        lat=4.6097,
        lon=-74.0817,
        demand=15.5,
        time_window=(8.0, 12.0),
        service_time=10.0,
        priority=2
    )

    assert customer.id == 0, "Customer ID mismatch"
    assert customer.demand == 15.5, "Customer demand mismatch"
    assert customer.time_window == (8.0, 12.0), "Time window mismatch"
    assert customer.priority == 2, "Priority mismatch"

    print("✓ Customer creation test passed!")


def test_depot_and_vehicle():
    """Test Depot and Vehicle creation."""
    print("Testing Depot and Vehicle creation...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)
    assert depot.opening_time == 6.0, "Depot opening time mismatch"
    assert depot.closing_time == 22.0, "Depot closing time mismatch"

    vehicle = Vehicle(0, 100.0, 150.0, 35.0, 50.0, 0.5)
    assert vehicle.capacity == 100.0, "Vehicle capacity mismatch"
    assert vehicle.speed == 35.0, "Vehicle speed mismatch"

    print("✓ Depot and Vehicle creation test passed!")


def test_delivery_problem_creation():
    """Test DeliveryProblem creation and distance matrix."""
    print("Testing DeliveryProblem creation...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(0, "Store A", 4.6200, -74.0700, 15.0, (8.0, 12.0), 10, 2),
        Customer(1, "Store B", 4.6050, -74.0900, 20.0, (9.0, 14.0), 15, 3),
    ]

    vehicles = [Vehicle(0, 50.0, 100.0, 35.0, 50.0, 0.5)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Check distance matrix
    assert problem.distance_matrix.shape == (3, 3), "Distance matrix shape mismatch"
    assert problem.distance_matrix[0, 0] == 0.0, "Self-distance should be zero"

    # Distance should be symmetric
    assert np.isclose(problem.distance_matrix[1, 2], problem.distance_matrix[2, 1]), \
        "Distance matrix should be symmetric"

    print("✓ DeliveryProblem creation test passed!")


def test_route_evaluation():
    """Test Route evaluation."""
    print("Testing Route evaluation...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(0, "Store A", 4.6200, -74.0700, 15.0, (8.0, 12.0), 10, 2),
        Customer(1, "Store B", 4.6050, -74.0900, 20.0, (9.0, 14.0), 15, 3),
    ]

    vehicles = [Vehicle(0, 100.0, 150.0, 35.0, 50.0, 0.5)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Create route
    route = Route(0, [0, 1], problem)

    assert route.total_distance > 0, "Route distance should be positive"
    assert route.total_demand == 35.0, f"Total demand should be 35.0, got {route.total_demand}"
    assert route.vehicle_id == 0, "Vehicle ID mismatch"

    print("✓ Route evaluation test passed!")


def test_encoding_decoding():
    """Test solution encoding and decoding."""
    print("Testing encoding/decoding...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Store {i}", 4.6097 + i*0.01, -74.0817, 10.0,
                (8.0, 14.0), 10, 3)
        for i in range(5)
    ]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(2)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Create routes
    routes = [[0, 1, 2], [3, 4]]

    # Encode
    chromosome = encode_solution(routes, 5)

    # Should contain customer IDs and separators
    assert len(chromosome) > 0, "Chromosome should not be empty"
    assert -1 in chromosome, "Chromosome should contain separator (-1)"

    # Decode
    decoded_routes = decode_solution(chromosome, problem)

    assert len(decoded_routes) == 2, f"Should have 2 routes, got {len(decoded_routes)}"
    assert len(decoded_routes[0].customer_sequence) == 3, "First route should have 3 customers"
    assert len(decoded_routes[1].customer_sequence) == 2, "Second route should have 2 customers"

    print("✓ Encoding/decoding test passed!")


def test_random_solution_generation():
    """Test random solution generation."""
    print("Testing random solution generation...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Store {i}", 4.6097 + np.random.rand()*0.05,
                -74.0817 + np.random.rand()*0.05, 10.0,
                (8.0, 14.0), 10, 3)
        for i in range(10)
    ]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(3)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Generate random solution
    solution = create_random_solution(problem)

    # Decode and check
    routes = decode_solution(solution, problem)

    # Check all customers are assigned
    assigned_customers = set()
    for route in routes:
        assigned_customers.update(route.customer_sequence)

    assert len(assigned_customers) == 10, "All customers should be assigned"
    assert assigned_customers == set(range(10)), "All customer IDs should be present"

    print("✓ Random solution generation test passed!")


def test_fitness_evaluation():
    """Test fitness evaluation."""
    print("Testing fitness evaluation...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(0, "Store A", 4.6200, -74.0700, 15.0, (8.0, 12.0), 10, 2),
        Customer(1, "Store B", 4.6050, -74.0900, 20.0, (9.0, 14.0), 15, 3),
    ]

    vehicles = [Vehicle(0, 100.0, 150.0, 35.0, 50.0, 0.5)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Create solution
    routes = [[0, 1]]
    chromosome = encode_solution(routes, 2)

    # Evaluate
    fitness, metrics = evaluate_solution(chromosome, problem)

    assert isinstance(fitness, (int, float)), "Fitness should be numeric"
    assert 'total_distance' in metrics, "Metrics should contain total_distance"
    assert 'n_vehicles' in metrics, "Metrics should contain n_vehicles"
    assert 'is_feasible' in metrics, "Metrics should contain is_feasible"
    assert metrics['n_delivered'] == 2, "Should deliver 2 customers"

    print("✓ Fitness evaluation test passed!")


def test_crossover_operator():
    """Test route-aware crossover."""
    print("Testing crossover operator...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [Customer(i, f"Store {i}", 4.61, -74.08, 10.0, (8.0, 14.0), 10, 3)
                for i in range(6)]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(2)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Create two parent solutions
    parent1 = encode_solution([[0, 1, 2], [3, 4, 5]], 6)
    parent2 = encode_solution([[0, 3], [1, 4], [2, 5]], 6)

    # Crossover
    child1, child2 = route_aware_crossover(parent1, parent2, problem)

    # Check children are valid
    routes1 = decode_solution(child1, problem)
    routes2 = decode_solution(child2, problem)

    assigned1 = set(c for r in routes1 for c in r.customer_sequence)
    assigned2 = set(c for r in routes2 for c in r.customer_sequence)

    assert len(assigned1) == 6, "Child 1 should assign all 6 customers"
    assert len(assigned2) == 6, "Child 2 should assign all 6 customers"
    assert assigned1 == set(range(6)), "All customer IDs should be in child 1"
    assert assigned2 == set(range(6)), "All customer IDs should be in child 2"

    print("✓ Crossover operator test passed!")


def test_mutation_operator():
    """Test mutation operators."""
    print("Testing mutation operators...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [Customer(i, f"Store {i}", 4.61, -74.08, 10.0, (8.0, 14.0), 10, 3)
                for i in range(5)]

    vehicles = [Vehicle(0, 100.0, 150.0, 35.0, 50.0, 0.5)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Create solution
    original = encode_solution([[0, 1, 2, 3, 4]], 5)

    # Apply mutation (with rate=1.0 to ensure it happens)
    mutated = swap_mutation(original.copy(), mutation_rate=1.0)

    # Mutation should change chromosome
    # But all customers should still be present
    routes = decode_solution(mutated, problem)
    assigned = set(c for r in routes for c in r.customer_sequence)

    assert len(assigned) == 5, "All customers should still be assigned after mutation"
    assert assigned == set(range(5)), "All customer IDs should be present"

    print("✓ Mutation operator test passed!")


def test_nearest_neighbor():
    """Test Nearest Neighbor heuristic."""
    print("Testing Nearest Neighbor heuristic...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Store {i}", 4.6097 + np.random.rand()*0.02,
                -74.0817 + np.random.rand()*0.02, 10.0,
                (8.0, 14.0), 10, 3)
        for i in range(8)
    ]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(2)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Run nearest neighbor
    solution, fitness, metrics = nearest_neighbor(problem)

    assert isinstance(fitness, (int, float)), "Fitness should be numeric"
    assert metrics['n_delivered'] > 0, "Should deliver some customers"
    assert metrics['total_distance'] > 0, "Total distance should be positive"

    print("✓ Nearest Neighbor test passed!")


def test_clarke_wright():
    """Test Clarke-Wright heuristic."""
    print("Testing Clarke-Wright heuristic...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Store {i}", 4.6097 + np.random.rand()*0.02,
                -74.0817 + np.random.rand()*0.02, 10.0,
                (8.0, 14.0), 10, 3)
        for i in range(8)
    ]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(2)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Run Clarke-Wright
    solution, fitness, metrics = clarke_wright(problem)

    assert isinstance(fitness, (int, float)), "Fitness should be numeric"
    assert metrics['n_delivered'] > 0, "Should deliver some customers"

    print("✓ Clarke-Wright test passed!")


def test_sweep_algorithm():
    """Test Sweep Algorithm."""
    print("Testing Sweep Algorithm...")

    depot = Depot(0, "Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(i, f"Store {i}", 4.6097 + np.random.rand()*0.02,
                -74.0817 + np.random.rand()*0.02, 10.0,
                (8.0, 14.0), 10, 3)
        for i in range(8)
    ]

    vehicles = [Vehicle(i, 50.0, 100.0, 35.0, 50.0, 0.5) for i in range(2)]

    problem = DeliveryProblem(customers, depot, vehicles, "Test")

    # Run Sweep
    solution, fitness, metrics = sweep_algorithm(problem)

    assert isinstance(fitness, (int, float)), "Fitness should be numeric"
    assert metrics['n_delivered'] > 0, "Should deliver some customers"

    print("✓ Sweep Algorithm test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running Tutorial 5 - Industrial Case Study Tests")
    print("="*70 + "\n")

    try:
        test_customer_creation()
        test_depot_and_vehicle()
        test_delivery_problem_creation()
        test_route_evaluation()
        test_encoding_decoding()
        test_random_solution_generation()
        test_fitness_evaluation()
        test_crossover_operator()
        test_mutation_operator()
        test_nearest_neighbor()
        test_clarke_wright()
        test_sweep_algorithm()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED! (12 tests)")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
