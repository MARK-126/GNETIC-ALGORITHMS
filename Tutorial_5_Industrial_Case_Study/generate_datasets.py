"""
Generate realistic synthetic delivery datasets

Creates problems of various sizes for testing and benchmarking.
"""

import numpy as np
import json
from delivery_problem import Customer, Depot, Vehicle, DeliveryProblem


def generate_customers_in_city(n_customers: int, city_center: tuple,
                               city_radius_km: float = 10.0,
                               seed: int = None) -> list:
    """
    Generate customers distributed in a city.

    Arguments:
    n_customers -- number of customers
    city_center -- (lat, lon) of city center
    city_radius_km -- approximate city radius
    seed -- random seed for reproducibility

    Returns:
    customers -- list of Customer objects
    """
    if seed is not None:
        np.random.seed(seed)

    center_lat, center_lon = city_center
    customers = []

    # Convert km to approximate degrees (rough approximation)
    lat_range = city_radius_km / 111.0  # 1 degree lat ≈ 111 km
    lon_range = city_radius_km / (111.0 * np.cos(np.radians(center_lat)))

    for i in range(n_customers):
        # Generate location (roughly uniform in circle)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.sqrt(np.random.uniform(0, 1)) * city_radius_km

        lat = center_lat + (radius / 111.0) * np.cos(angle)
        lon = center_lon + (radius / (111.0 * np.cos(np.radians(center_lat)))) * np.sin(angle)

        # Generate demand (log-normal distribution)
        demand = np.random.lognormal(mean=np.log(15), sigma=0.5)
        demand = np.clip(demand, 5, 50)

        # Generate time window
        # Morning deliveries (8-12), afternoon (12-18), or flexible (8-18)
        window_type = np.random.choice(['morning', 'afternoon', 'flexible'], p=[0.3, 0.4, 0.3])

        if window_type == 'morning':
            earliest = np.random.uniform(7, 9)
            latest = np.random.uniform(11, 13)
        elif window_type == 'afternoon':
            earliest = np.random.uniform(12, 14)
            latest = np.random.uniform(16, 19)
        else:  # flexible
            earliest = np.random.uniform(7, 10)
            latest = np.random.uniform(15, 19)

        # Service time (proportional to demand)
        service_time = np.random.uniform(5, 15) + demand * 0.2

        # Priority (1=VIP, 2=urgent, 3=normal)
        priority = np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7])

        customer = Customer(
            id=i,
            name=f"Customer_{i}",
            lat=lat,
            lon=lon,
            demand=demand,
            time_window=(earliest, latest),
            service_time=service_time,
            priority=priority
        )

        customers.append(customer)

    return customers


def generate_fleet(n_vehicles: int, capacity: float = 100.0,
                  max_distance: float = 150.0) -> list:
    """
    Generate homogeneous fleet.

    Arguments:
    n_vehicles -- number of vehicles
    capacity -- vehicle capacity (kg)
    max_distance -- maximum distance per vehicle (km)

    Returns:
    vehicles -- list of Vehicle objects
    """
    vehicles = []

    for i in range(n_vehicles):
        vehicle = Vehicle(
            id=i,
            capacity=capacity,
            max_distance=max_distance,
            speed=35.0,  # Average 35 km/h in city
            fixed_cost=50.0,  # Fixed cost per vehicle
            variable_cost=0.5  # Cost per km
        )
        vehicles.append(vehicle)

    return vehicles


def generate_problem(n_customers: int, n_vehicles: int, city_name: str,
                    city_center: tuple, problem_name: str = None,
                    seed: int = None) -> DeliveryProblem:
    """
    Generate complete delivery problem.

    Arguments:
    n_customers -- number of customers
    n_vehicles -- number of vehicles
    city_name -- name of city
    city_center -- (lat, lon) of city center
    problem_name -- optional problem name
    seed -- random seed

    Returns:
    problem -- DeliveryProblem instance
    """
    if problem_name is None:
        problem_name = f"{city_name}_{n_customers}customers_{n_vehicles}vehicles"

    # Create depot
    depot = Depot(
        id=0,
        name=f"{city_name} Central Warehouse",
        lat=city_center[0],
        lon=city_center[1],
        opening_time=6.0,
        closing_time=22.0
    )

    # Generate customers
    customers = generate_customers_in_city(n_customers, city_center, seed=seed)

    # Generate fleet
    vehicles = generate_fleet(n_vehicles)

    problem = DeliveryProblem(customers, depot, vehicles, problem_name)

    return problem


# ==================== Pre-defined City Datasets ====================

def generate_bogota_dataset(n_customers: int = 30, n_vehicles: int = 5,
                           seed: int = 42) -> DeliveryProblem:
    """Generate problem for Bogotá, Colombia."""
    return generate_problem(
        n_customers, n_vehicles,
        "Bogota", (4.6097, -74.0817),
        seed=seed
    )


def generate_medellin_dataset(n_customers: int = 30, n_vehicles: int = 5,
                             seed: int = 42) -> DeliveryProblem:
    """Generate problem for Medellín, Colombia."""
    return generate_problem(
        n_customers, n_vehicles,
        "Medellin", (6.2442, -75.5812),
        seed=seed
    )


def generate_cali_dataset(n_customers: int = 30, n_vehicles: int = 5,
                         seed: int = 42) -> DeliveryProblem:
    """Generate problem for Cali, Colombia."""
    return generate_problem(
        n_customers, n_vehicles,
        "Cali", (3.4516, -76.5320),
        seed=seed
    )


# ==================== Benchmark Suite ====================

def generate_benchmark_suite(output_dir: str = "data/"):
    """
    Generate full benchmark suite with various problem sizes.

    Creates problems of sizes:
    - Small: 10-20 customers, 2-3 vehicles
    - Medium: 30-50 customers, 5-8 vehicles
    - Large: 80-100 customers, 10-15 vehicles

    Arguments:
    output_dir -- directory to save datasets
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    problems = []

    # Small problems
    for n_customers in [10, 15, 20]:
        n_vehicles = max(2, n_customers // 7)
        problem = generate_bogota_dataset(n_customers, n_vehicles, seed=42)
        filename = f"{output_dir}small_{n_customers}c_{n_vehicles}v.json"
        problem.save_to_file(filename)
        problems.append((filename, problem))
        print(f"Created: {filename}")

    # Medium problems
    for n_customers in [30, 40, 50]:
        n_vehicles = max(4, n_customers // 8)
        problem = generate_bogota_dataset(n_customers, n_vehicles, seed=42)
        filename = f"{output_dir}medium_{n_customers}c_{n_vehicles}v.json"
        problem.save_to_file(filename)
        problems.append((filename, problem))
        print(f"Created: {filename}")

    # Large problems
    for n_customers in [80, 100]:
        n_vehicles = max(8, n_customers // 10)
        problem = generate_bogota_dataset(n_customers, n_vehicles, seed=42)
        filename = f"{output_dir}large_{n_customers}c_{n_vehicles}v.json"
        problem.save_to_file(filename)
        problems.append((filename, problem))
        print(f"Created: {filename}")

    print(f"\n✓ Generated {len(problems)} benchmark problems")
    return problems


if __name__ == "__main__":
    print("Generating Delivery Problem Datasets")
    print("="*70)

    # Generate example problems
    print("\n1. Small example (15 customers, 3 vehicles):")
    small_problem = generate_bogota_dataset(n_customers=15, n_vehicles=3, seed=42)
    print(f"   {small_problem}")
    print(f"   Total demand: {sum(c.demand for c in small_problem.customers):.2f} kg")
    print(f"   Total capacity: {sum(v.capacity for v in small_problem.vehicles):.2f} kg")

    print("\n2. Medium example (30 customers, 5 vehicles):")
    medium_problem = generate_bogota_dataset(n_customers=30, n_vehicles=5, seed=42)
    print(f"   {medium_problem}")
    print(f"   Total demand: {sum(c.demand for c in medium_problem.customers):.2f} kg")
    print(f"   Total capacity: {sum(v.capacity for v in medium_problem.vehicles):.2f} kg")

    print("\n3. Large example (50 customers, 8 vehicles):")
    large_problem = generate_bogota_dataset(n_customers=50, n_vehicles=8, seed=42)
    print(f"   {large_problem}")
    print(f"   Total demand: {sum(c.demand for c in large_problem.customers):.2f} kg")
    print(f"   Total capacity: {sum(v.capacity for v in large_problem.vehicles):.2f} kg")

    # Generate full benchmark suite
    print("\n" + "="*70)
    print("Generating Benchmark Suite")
    print("="*70)
    generate_benchmark_suite()
