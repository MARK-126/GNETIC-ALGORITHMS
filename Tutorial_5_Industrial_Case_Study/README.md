# Tutorial 5: Industrial Case Study - Vehicle Routing with Time Windows

## 📚 Overview

This tutorial presents a **complete real-world industrial case study** solving the Vehicle Routing Problem with Time Windows (VRPTW) using Genetic Algorithms. This is the same type of optimization problem used by companies like Amazon, Uber Eats, Rappi, and FedEx for daily delivery operations.

**Problem:** Optimize routes for multiple vehicles to deliver packages to customers while:
- Minimizing total distance and cost
- Respecting delivery time windows
- Adhering to vehicle capacity constraints
- Minimizing number of vehicles used
- Prioritizing VIP customers

---

## 🎯 Learning Objectives

- Model complex real-world logistics problems
- Design specialized GA operators for routing
- Handle multiple constraints (time windows, capacity, priorities)
- Compare GA with classical heuristics
- Evaluate multi-objective solutions
- Apply GA to production-scale problems

---

## 📋 Contents

### Files

| File | Description |
|------|-------------|
| `C5W1_Assignment_Vehicle_Routing.ipynb` | **NEW:** Interactive Coursera-style assignment for VRPTW |
| `delivery_problem.py` | Core problem modeling (Customer, Depot, Vehicle, Problem) |
| `delivery_ga.py` | Specialized GA with route-aware operators |
| `baselines.py` | Classical heuristics (Nearest Neighbor, Clarke-Wright, Sweep) |
| `generate_datasets.py` | Synthetic dataset generator |
| `public_tests.py` | 12 automated tests |
| `data/` | Benchmark problem instances |
| `README.md` | This documentation |

### 🎓 Interactive Assignment (Coursera-style)

**New!** `C5W1_Assignment_Vehicle_Routing.ipynb` provides a complete hands-on industrial case study:

- **Real-world problem modeling** - Create VRPTW instances with realistic constraints
- **Route evaluation exercise** - Implement distance calculation for delivery routes
- **Specialized GA operators** - Use domain-specific crossover and mutation
- **Algorithm comparison** - Compare GA with Nearest Neighbor, Clarke-Wright, Sweep
- **Geographic visualization** - Plot delivery routes on maps
- **Business impact calculation** - Quantify cost savings from optimization
- **Challenge: 50-customer problem** - Scale up to production-size instances

**Learn how Amazon and Uber Eats optimize their deliveries!**

### Problem Components

**1. Customer**
- Location (lat, lon)
- Demand (weight/volume)
- Time window (earliest, latest delivery time)
- Service time (minutes to complete delivery)
- Priority (1=VIP, 2=urgent, 3=normal)

**2. Depot**
- Distribution center location
- Operating hours

**3. Vehicle**
- Capacity (kg or m³)
- Maximum distance per day
- Speed, costs (fixed + per-km)

**4. Constraints**
- **Time windows**: Deliver within customer's time window
- **Capacity**: Don't exceed vehicle capacity
- **Depot hours**: Return before closing time
- **Max distance**: Don't exceed vehicle's daily limit

---

## 🚀 Quick Start

### Basic Usage

```python
from delivery_problem import Customer, Depot, Vehicle, DeliveryProblem
from delivery_ga import delivery_genetic_algorithm
from baselines import compare_algorithms

# Create problem
depot = Depot(0, "Central Warehouse", 4.6097, -74.0817, 6.0, 22.0)

customers = [
    Customer(0, "Store A", 4.6200, -74.0700, 15.0, (8.0, 12.0), 10, 2),
    Customer(1, "Store B", 4.6050, -74.0900, 20.0, (9.0, 14.0), 15, 3),
    # ... more customers
]

vehicles = [
    Vehicle(0, 100.0, 150.0, 35.0, 50.0, 0.5),  # capacity, max_dist, speed, fixed_cost, var_cost
    Vehicle(1, 100.0, 150.0, 35.0, 50.0, 0.5),
]

problem = DeliveryProblem(customers, depot, vehicles, "Delivery Problem")

# Solve with GA
best_solution, best_fitness, metrics, history = delivery_genetic_algorithm(
    problem,
    pop_size=100,
    max_generations=200,
    mutation_rate=0.15,
    crossover_rate=0.8
)

# View results
from delivery_ga import decode_solution
routes = decode_solution(best_solution, problem)

for i, route in enumerate(routes):
    print(f"Vehicle {i}: {route.customer_sequence}")
    print(f"  Distance: {route.total_distance:.2f} km")
    print(f"  Cost: ${route.get_cost():.2f}")
```

### Generate and Solve Benchmark Problems

```python
from generate_datasets import generate_bogota_dataset
from delivery_ga import delivery_genetic_algorithm
from baselines import compare_algorithms

# Generate problem
problem = generate_bogota_dataset(n_customers=30, n_vehicles=5, seed=42)

# Compare algorithms
baseline_results = compare_algorithms(problem, verbose=True)

# Solve with GA
best_solution, fitness, metrics, history = delivery_genetic_algorithm(
    problem,
    pop_size=100,
    max_generations=200
)

print(f"\nGA Results:")
print(f"  Distance: {metrics['total_distance']:.2f} km")
print(f"  Vehicles: {metrics['n_vehicles']}")
print(f"  Cost: ${metrics['total_cost']:.2f}")
print(f"  Feasible: {metrics['is_feasible']}")
```

### Run Tests

```bash
python public_tests.py
```

---

## 💡 Problem Modeling

### VRPTW Formulation

**Decision Variables:**
- Which customers are served by which vehicle
- Order of customers in each route

**Objectives** (multi-objective):
1. Minimize total distance
2. Minimize number of vehicles
3. Minimize time window violations
4. Minimize capacity violations
5. Maximize VIP customer satisfaction

**Constraints:**
- Each customer visited exactly once
- Vehicle capacity not exceeded
- Time windows respected
- Depot operating hours respected

### Chromosome Encoding

Routes are encoded as integer arrays with separators:

```
[2, 5, 1, -1, 0, 3, 4, -1, 6, 8]
```

Interpretation:
- Vehicle 0: customers [2, 5, 1]
- Vehicle 1: customers [0, 3, 4]
- Vehicle 2: customers [6, 8]

Separator (-1) marks end of route.

---

## 🔧 Genetic Algorithm Components

### 1. Fitness Function

Multi-objective fitness with weighted components:

```python
fitness = (
    0.3 * (-distance/100) +           # Distance (normalized)
    0.1 * (-n_vehicles * 10) +        # Number of vehicles
    0.3 * (-time_violations * 100) +  # Time violations (heavy penalty)
    0.2 * (-capacity_violations * 50) + # Capacity violations
    0.1 * vip_bonus                   # VIP customer bonus
    - undelivered_penalty             # Critical: all must be delivered
)
```

### 2. Specialized Operators

**Route-Aware Crossover:**
- Preserves valid sub-routes from both parents
- Automatically repairs duplicates/omissions
- Maintains route structure

**Insertion Mutation:**
- Removes customer from route
- Reinserts at best position (greedy local search)
- Can move customer between vehicles

**Swap Mutation:**
- Swaps two random customers
- Simple but effective for exploration

**Inversion Mutation:**
- Reverses segment of route
- Similar to 2-opt local search

### 3. Selection and Elitism

- **Tournament selection** (size 3) for diversity
- **Elitism** (top 5) to preserve best solutions
- Balance between exploration and exploitation

---

## 📊 Expected Results

### Performance Comparison (30 customers, 5 vehicles)

| Algorithm | Distance (km) | Vehicles | Time (s) | Feasible | Quality |
|-----------|--------------|----------|----------|----------|---------|
| Random | 450-500 | 5 | <0.1 | No | Baseline |
| Nearest Neighbor | 320-360 | 4-5 | <1 | Sometimes | Good |
| Clarke-Wright | 280-320 | 4 | <2 | Usually | Very Good |
| Sweep | 300-340 | 4-5 | <1 | Usually | Good |
| **GA (100 gen)** | **240-280** | **3-4** | **20-30** | **Yes** | **Excellent** |
| **GA (200 gen)** | **220-260** | **3-4** | **40-60** | **Yes** | **Best** |

### Scalability

| Problem Size | GA Time | Quality vs CW | Best for |
|--------------|---------|---------------|----------|
| Small (10-20) | <10s | Similar | Learning |
| Medium (30-50) | 30-90s | 10-15% better | Testing |
| Large (80-100) | 2-5 min | 15-20% better | Production |
| Very Large (200+) | 10-30 min | 20-25% better | Optimization |

---

## 🎨 Advanced Features

### Multi-Objective Optimization

```python
# Customize fitness weights
weights = {
    'distance': 0.4,        # Prioritize distance
    'vehicles': 0.05,       # Less important
    'time_violations': 0.35, # Critical
    'capacity_violations': 0.15,
    'vip_bonus': 0.05
}

fitness, metrics = evaluate_solution(chromosome, problem, weights=weights)
```

### Hybrid GA with Local Search

```python
# Apply 2-opt to improve routes
def local_optimization(routes, problem):
    improved_routes = []
    for route in routes:
        improved = two_opt(route, problem)
        improved_routes.append(improved)
    return improved_routes

# In main GA loop, after mutation:
if generation % 10 == 0:  # Every 10 generations
    # Apply local search to elite individuals
    for i in elite_indices:
        routes = decode_solution(population[i], problem)
        improved_routes = local_optimization(routes, problem)
        population[i] = encode_solution(improved_routes, n_customers)
```

### Dynamic Re-optimization

```python
# New orders arrive during execution
def reoptimize_with_new_orders(current_solution, new_customers, problem):
    # Add new customers to problem
    problem.customers.extend(new_customers)

    # Use current solution as seed
    population = [current_solution] + [create_random_solution(problem)
                                      for _ in range(pop_size - 1)]

    # Run GA with warm start
    best_solution, _, _, _ = delivery_genetic_algorithm(
        problem, initial_population=population, max_generations=50
    )

    return best_solution
```

---

## 🧪 Exercises

1. **Basic**: Solve small problems (10-15 customers) and verify solutions manually
2. **Tuning**: Experiment with population size and mutation rates
3. **Comparison**: Run all baselines and GA, compare results
4. **Visualization**: Plot routes on a map (lat/lon coordinates)
5. **Constraints**: Add driver break times (lunch, rest periods)
6. **Multi-depot**: Extend to multiple distribution centers
7. **Pickup-Delivery**: Add pickup locations (not just delivery)
8. **Dynamic**: Simulate new orders arriving during route execution
9. **Heterogeneous fleet**: Different vehicle types with different capacities/costs
10. **Traffic**: Incorporate time-dependent travel times (rush hour)

---

## 📚 References

### Vehicle Routing Problem

- **Toth, P., & Vigo, D.** (2014). *Vehicle Routing: Problems, Methods, and Applications*. SIAM.

- **Laporte, G.** (2009). *Fifty years of vehicle routing*. Transportation Science.

### Genetic Algorithms for VRP

- **Baker, B. M., & Ayechew, M. A.** (2003). *A genetic algorithm for the vehicle routing problem*. Computers & Operations Research.

- **Prins, C.** (2004). *A simple and effective evolutionary algorithm for the vehicle routing problem*. Computers & Operations Research.

### Classical Heuristics

- **Clarke, G., & Wright, J. W.** (1964). *Scheduling of vehicles from a central depot to a number of delivery points*. Operations Research.

- **Gillett, B. E., & Miller, L. R.** (1974). *A heuristic algorithm for the vehicle-dispatch problem*. Operations Research.

### Real-World Applications

- **Bräysy, O., & Gendreau, M.** (2005). *Vehicle routing problem with time windows, Part I: Route construction and local search algorithms*. Transportation Science.

---

## 💼 Industrial Applications

This exact problem formulation is used by:

**Logistics & Delivery:**
- Amazon (last-mile delivery)
- FedEx, UPS, DHL (package routing)
- Rappi, Uber Eats, DoorDash (food delivery)

**Services:**
- Waste collection routes
- Home healthcare visits
- Field service technicians

**Retail:**
- Grocery delivery (Instacart, Cornershop)
- Pharmacy deliveries

**Typical Savings:**
- 10-25% reduction in total distance
- 15-30% fewer vehicles needed
- 20-40% improvement in on-time deliveries
- Annual savings: $50K - $500K per depot

---

## ⚡ Quick Reference

### Create Problem

```python
from delivery_problem import DeliveryProblem, Customer, Depot, Vehicle

depot = Depot(id, name, lat, lon, opening, closing)
customer = Customer(id, name, lat, lon, demand, time_window, service_time, priority)
vehicle = Vehicle(id, capacity, max_distance, speed, fixed_cost, variable_cost)

problem = DeliveryProblem(customers, depot, vehicles, name)
```

### Solve with GA

```python
from delivery_ga import delivery_genetic_algorithm

best_solution, fitness, metrics, history = delivery_genetic_algorithm(
    problem,
    pop_size=100,
    max_generations=200,
    mutation_rate=0.15,
    crossover_rate=0.8,
    elite_size=5
)
```

### Run Baselines

```python
from baselines import nearest_neighbor, clarke_wright, sweep_algorithm, compare_algorithms

# Individual algorithms
solution, fitness, metrics = nearest_neighbor(problem)
solution, fitness, metrics = clarke_wright(problem)
solution, fitness, metrics = sweep_algorithm(problem)

# Compare all
results = compare_algorithms(problem, verbose=True)
```

### Generate Datasets

```python
from generate_datasets import generate_bogota_dataset, generate_benchmark_suite

problem = generate_bogota_dataset(n_customers=30, n_vehicles=5, seed=42)
benchmark_problems = generate_benchmark_suite(output_dir="data/")
```

---

## 🤝 Best Practices

1. **Start small**: Test with 10-15 customers before scaling up
2. **Validate constraints**: Always check feasibility of solutions
3. **Tune for your problem**: Different problems need different parameters
4. **Use baselines**: Compare GA against classical heuristics
5. **Monitor convergence**: Plot fitness evolution to detect stagnation
6. **Hybrid approaches**: Combine GA with local search for best results
7. **Real data**: Test on actual delivery data when possible
8. **Parallel evaluation**: Evaluate population in parallel for speed
9. **Warm starts**: Use good initial solutions when available
10. **Domain knowledge**: Incorporate business rules and constraints

### Parameter Guidelines

| Problem Size | Pop Size | Generations | Mutation Rate | Runtime |
|--------------|----------|-------------|---------------|---------|
| Small (10-20) | 50-80 | 100-200 | 0.15-0.20 | <30s |
| Medium (30-50) | 100-150 | 150-300 | 0.12-0.18 | 1-3 min |
| Large (80-100) | 150-250 | 200-400 | 0.10-0.15 | 3-10 min |

---

## 🎓 Key Takeaways

✅ **Realistic Problem**: Identical to problems solved daily by logistics companies
✅ **Multi-Objective**: Balances multiple competing objectives
✅ **Constrained Optimization**: Handles complex real-world constraints
✅ **Specialized Operators**: Route-aware crossover and mutation
✅ **Proven Approach**: GA competitive with or better than classical methods
✅ **Scalable**: Works from 10 to 200+ customers
✅ **Extensible**: Easy to add new constraints and objectives
✅ **Production-Ready**: Can be deployed in real systems

---

**Congratulations! You've completed the comprehensive GA tutorial series!** 🎉

**From fundamentals to production-scale optimization problems, you now have the tools to solve real-world challenges with Genetic Algorithms!** 🚀

**Go forth and optimize!** 🌟
