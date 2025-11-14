"""
Vehicle Routing Problem with Time Windows (VRPTW)
Delivery optimization problem modeling
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class Customer:
    """Represents a delivery customer."""
    id: int
    name: str
    lat: float
    lon: float
    demand: float  # Weight/volume in kg or m³
    time_window: Tuple[float, float]  # (earliest, latest) in hours from start
    service_time: float  # Time to complete delivery (minutes)
    priority: int  # 1=VIP, 2=urgent, 3=normal

    def __repr__(self):
        return f"Customer({self.id}, demand={self.demand:.1f}kg, tw={self.time_window})"


@dataclass
class Depot:
    """Represents a distribution center."""
    id: int
    name: str
    lat: float
    lon: float
    opening_time: float  # Hours from midnight
    closing_time: float

    def __repr__(self):
        return f"Depot({self.id}, {self.name})"


@dataclass
class Vehicle:
    """Represents a delivery vehicle."""
    id: int
    capacity: float  # Maximum weight/volume in kg or m³
    max_distance: float  # Maximum distance per day (km)
    speed: float  # Average speed (km/h)
    fixed_cost: float  # Fixed cost per vehicle used
    variable_cost: float  # Cost per km

    def __repr__(self):
        return f"Vehicle({self.id}, capacity={self.capacity}kg)"


class DeliveryProblem:
    """
    Complete delivery optimization problem.

    Combines customers, depots, vehicles, and constraints.
    """

    def __init__(self, customers: List[Customer], depot: Depot,
                 vehicles: List[Vehicle], name: str = "Delivery Problem"):
        """
        Initialize delivery problem.

        Arguments:
        customers -- list of Customer objects
        depot -- Depot object (single depot for now)
        vehicles -- list of Vehicle objects
        name -- problem name
        """
        self.customers = customers
        self.depot = depot
        self.vehicles = vehicles
        self.name = name

        # Pre-calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()

        # Pre-calculate travel time matrix (can vary by time of day)
        self.travel_time_matrix = self._calculate_travel_time_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate Euclidean distances between all locations.

        Returns:
        distance_matrix -- (n_locations, n_locations) matrix
                          Index 0 is depot, 1..n are customers
        """
        n_customers = len(self.customers)
        n_locations = n_customers + 1  # +1 for depot

        distance_matrix = np.zeros((n_locations, n_locations))

        # Get all coordinates
        coords = [(self.depot.lat, self.depot.lon)]
        for customer in self.customers:
            coords.append((customer.lat, customer.lon))

        # Calculate distances
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance_matrix[i, j] = self._haversine_distance(
                        coords[i][0], coords[i][1],
                        coords[j][0], coords[j][1]
                    )

        return distance_matrix

    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points (km).

        Uses Haversine formula for realistic distances.
        """
        # Earth radius in km
        R = 6371.0

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        distance = R * c
        return distance

    def _calculate_travel_time_matrix(self, time_of_day: str = 'midday') -> np.ndarray:
        """
        Calculate travel times considering traffic.

        Arguments:
        time_of_day -- 'morning_rush', 'midday', 'evening_rush'

        Returns:
        travel_time_matrix -- (n_locations, n_locations) in hours
        """
        # Traffic factors
        traffic_factors = {
            'morning_rush': 1.5,  # 50% slower
            'midday': 1.0,
            'evening_rush': 1.6,
            'night': 0.8
        }

        factor = traffic_factors.get(time_of_day, 1.0)

        # Assume average speed from first vehicle (simplified)
        avg_speed = self.vehicles[0].speed if self.vehicles else 30.0

        # Time = Distance / Speed, adjusted by traffic
        travel_time_matrix = (self.distance_matrix / avg_speed) * factor

        return travel_time_matrix

    def get_distance(self, loc1: int, loc2: int) -> float:
        """Get distance between two locations (0=depot, 1..n=customers)."""
        return self.distance_matrix[loc1, loc2]

    def get_travel_time(self, loc1: int, loc2: int, current_time: float = 12.0) -> float:
        """
        Get travel time between locations considering time of day.

        Arguments:
        loc1, loc2 -- location indices
        current_time -- current time in hours (0-24)

        Returns:
        travel_time -- in hours
        """
        # Simple time-of-day logic
        if 7 <= current_time < 10 or 16 <= current_time < 19:
            factor = 1.5  # Rush hour
        elif 22 <= current_time or current_time < 6:
            factor = 0.8  # Night
        else:
            factor = 1.0  # Normal

        base_time = self.distance_matrix[loc1, loc2] / self.vehicles[0].speed
        return base_time * factor

    def get_customer(self, customer_id: int) -> Customer:
        """Get customer by ID."""
        return self.customers[customer_id]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'depot': {
                'id': self.depot.id,
                'name': self.depot.name,
                'lat': self.depot.lat,
                'lon': self.depot.lon,
                'opening_time': self.depot.opening_time,
                'closing_time': self.depot.closing_time
            },
            'customers': [
                {
                    'id': c.id,
                    'name': c.name,
                    'lat': c.lat,
                    'lon': c.lon,
                    'demand': c.demand,
                    'time_window': c.time_window,
                    'service_time': c.service_time,
                    'priority': c.priority
                } for c in self.customers
            ],
            'vehicles': [
                {
                    'id': v.id,
                    'capacity': v.capacity,
                    'max_distance': v.max_distance,
                    'speed': v.speed,
                    'fixed_cost': v.fixed_cost,
                    'variable_cost': v.variable_cost
                } for v in self.vehicles
            ]
        }

    def save_to_file(self, filename: str):
        """Save problem to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str):
        """Load problem from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        depot = Depot(
            id=data['depot']['id'],
            name=data['depot']['name'],
            lat=data['depot']['lat'],
            lon=data['depot']['lon'],
            opening_time=data['depot']['opening_time'],
            closing_time=data['depot']['closing_time']
        )

        customers = [
            Customer(
                id=c['id'],
                name=c['name'],
                lat=c['lat'],
                lon=c['lon'],
                demand=c['demand'],
                time_window=tuple(c['time_window']),
                service_time=c['service_time'],
                priority=c['priority']
            ) for c in data['customers']
        ]

        vehicles = [
            Vehicle(
                id=v['id'],
                capacity=v['capacity'],
                max_distance=v['max_distance'],
                speed=v['speed'],
                fixed_cost=v['fixed_cost'],
                variable_cost=v['variable_cost']
            ) for v in data['vehicles']
        ]

        return cls(customers, depot, vehicles, data['name'])

    def __repr__(self):
        return f"DeliveryProblem({self.name}, {len(self.customers)} customers, {len(self.vehicles)} vehicles)"


class Route:
    """Represents a vehicle route."""

    def __init__(self, vehicle_id: int, customer_sequence: List[int], problem: DeliveryProblem):
        """
        Initialize route.

        Arguments:
        vehicle_id -- ID of vehicle assigned
        customer_sequence -- ordered list of customer IDs
        problem -- DeliveryProblem instance
        """
        self.vehicle_id = vehicle_id
        self.customer_sequence = customer_sequence
        self.problem = problem

        # Calculate route metrics
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_demand = 0.0
        self.time_violations = 0
        self.capacity_violations = 0
        self.is_feasible = False

        self._evaluate()

    def _evaluate(self):
        """Evaluate route and calculate all metrics."""
        if not self.customer_sequence:
            self.is_feasible = True
            return

        vehicle = self.problem.vehicles[self.vehicle_id]
        current_location = 0  # Start at depot
        current_time = self.problem.depot.opening_time
        current_load = 0.0

        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_demand = 0.0
        self.time_violations = 0
        self.capacity_violations = 0

        for customer_id in self.customer_sequence:
            customer = self.problem.customers[customer_id]
            customer_location = customer_id + 1  # +1 because depot is 0

            # Travel to customer
            travel_time = self.problem.get_travel_time(current_location, customer_location, current_time)
            travel_distance = self.problem.get_distance(current_location, customer_location)

            current_time += travel_time
            self.total_distance += travel_distance

            # Check time window
            earliest, latest = customer.time_window
            if current_time < earliest:
                # Wait until time window opens
                current_time = earliest
            elif current_time > latest:
                # Late delivery
                self.time_violations += (current_time - latest)

            # Service customer
            current_time += customer.service_time / 60.0  # Convert minutes to hours
            current_load += customer.demand
            self.total_demand += customer.demand

            # Check capacity
            if current_load > vehicle.capacity:
                self.capacity_violations += (current_load - vehicle.capacity)

            current_location = customer_location

        # Return to depot
        return_distance = self.problem.get_distance(current_location, 0)
        return_time = self.problem.get_travel_time(current_location, 0, current_time)

        self.total_distance += return_distance
        current_time += return_time
        self.total_time = current_time - self.problem.depot.opening_time

        # Check if depot is still open
        if current_time > self.problem.depot.closing_time:
            self.time_violations += (current_time - self.problem.depot.closing_time)

        # Check max distance
        if self.total_distance > vehicle.max_distance:
            # This is a hard constraint violation
            pass

        # Route is feasible if no violations
        self.is_feasible = (self.time_violations == 0 and self.capacity_violations == 0)

    def get_cost(self) -> float:
        """Calculate total cost of route."""
        vehicle = self.problem.vehicles[self.vehicle_id]
        return vehicle.fixed_cost + vehicle.variable_cost * self.total_distance

    def __repr__(self):
        status = "✓" if self.is_feasible else "✗"
        return f"Route({status}, vehicle={self.vehicle_id}, customers={len(self.customer_sequence)}, dist={self.total_distance:.1f}km)"


if __name__ == "__main__":
    # Example usage
    print("Delivery Problem Module")
    print("=" * 70)

    # Create a simple example
    depot = Depot(0, "Central Warehouse", 4.6097, -74.0817, 6.0, 22.0)

    customers = [
        Customer(0, "Store A", 4.6200, -74.0700, 15.0, (8.0, 12.0), 10, 2),
        Customer(1, "Store B", 4.6050, -74.0900, 20.0, (9.0, 14.0), 15, 3),
        Customer(2, "Store C", 4.6150, -74.0650, 10.0, (7.0, 11.0), 10, 1),
    ]

    vehicles = [
        Vehicle(0, 50.0, 100.0, 35.0, 50.0, 0.5),
        Vehicle(1, 50.0, 100.0, 35.0, 50.0, 0.5),
    ]

    problem = DeliveryProblem(customers, depot, vehicles, "Example Problem")

    print(f"\n{problem}")
    print(f"Distance matrix shape: {problem.distance_matrix.shape}")
    print(f"Distance depot to customer 0: {problem.get_distance(0, 1):.2f} km")

    # Create a route
    route = Route(0, [0, 1, 2], problem)
    print(f"\n{route}")
    print(f"Total distance: {route.total_distance:.2f} km")
    print(f"Total time: {route.total_time:.2f} hours")
    print(f"Cost: ${route.get_cost():.2f}")
