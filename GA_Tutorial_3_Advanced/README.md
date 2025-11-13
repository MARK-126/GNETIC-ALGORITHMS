# Tutorial 3: Advanced Topics - TSP, Hybrid GAs, and Multi-Objective Optimization

## 📚 Overview

This tutorial covers advanced GA applications including combinatorial optimization, hybrid algorithms, and multi-objective optimization fundamentals.

## 🎯 Learning Objectives

- Solve the Traveling Salesman Problem (TSP) with GAs
- Master permutation-based encoding and operators
- Implement hybrid algorithms (GA + local search)
- Understand multi-objective optimization basics
- Apply NSGA-II concepts (Pareto fronts, crowding distance)

## 📋 Contents

### Files

| File | Description |
|------|-------------|
| `GA_Advanced_Topics.ipynb` | Main tutorial with TSP, hybrid GAs, multi-objective basics |
| `ga_utils_advanced.py` | TSP operators, 2-opt search, multi-objective utilities |
| `public_tests.py` | 8 automated tests |
| `README.md` | This documentation |

### Topics Covered

1. **Traveling Salesman Problem (TSP)**
   - Permutation encoding
   - Order Crossover (OX)
   - Partially Mapped Crossover (PMX)
   - Swap/Inversion/Scramble mutations
   - Complete TSP GA

2. **Hybrid Algorithms**
   - 2-opt local search
   - Lamarckian evolution
   - Baldwin effect
   - Performance comparison

3. **Multi-Objective Optimization**
   - Pareto dominance
   - Non-dominated sorting
   - Crowding distance
   - NSGA-II foundations

## 🚀 Quick Start

```bash
cd GA_Tutorial_3_Advanced
jupyter notebook GA_Advanced_Topics.ipynb
```

### Run Tests
```bash
python public_tests.py
```

## 💡 Key Concepts

### TSP Operators

| Operator | Type | Description |
|----------|------|-------------|
| Order Crossover (OX) | Crossover | Preserves relative order |
| PMX | Crossover | Mapping-based repair |
| Swap Mutation | Mutation | Swaps two cities |
| Inversion Mutation | Mutation | Reverses segment |

**Why special operators?** Standard operators break permutation validity!

### Hybrid GA Performance

Typical improvement over pure GA:
- **Convergence speed**: 40-60% faster
- **Solution quality**: 10-30% better
- **Robustness**: More consistent results

### Multi-Objective Concepts

**Pareto Dominance:** Solution A dominates B if:
- A ≤ B in all objectives
- A < B in at least one objective

**Pareto Front:** Set of non-dominated solutions (optimal tradeoffs)

## 🔧 Usage Examples

### TSP GA

```python
cities, dist_matrix = create_symmetric_tsp(n_cities=20)

best_tour, best_distance, history = tsp_genetic_algorithm(
    distance_matrix=dist_matrix,
    pop_size=100,
    max_generations=500,
    mutation_rate=0.1,
    elite_size=2
)

plot_tsp_tour(cities, best_tour)
```

### Hybrid GA with Local Search

```python
best_tour, best_distance, history = hybrid_tsp_ga(
    distance_matrix=dist_matrix,
    pop_size=100,
    max_generations=300,
    local_search_prob=0.2  # Apply 2-opt to 20% of population
)
```

### Multi-Objective Sorting

```python
# Evaluate multiple objectives
objectives = np.array([[f1_val, f2_val], ...])  # Shape: (n_solutions, 2)

# Non-dominated sorting
fronts = fast_non_dominated_sort(population, objectives)

# Crowding distance
distances = calculate_crowding_distance_multi(objectives)

# Visualize
plot_pareto_front_2d(objectives)
```

## 📊 Expected Results

### TSP Performance (20 cities)

| Method | Typical Distance | Generations |
|--------|-----------------|-------------|
| Random | ~800 | - |
| Greedy | ~500 | - |
| Standard GA | ~350 | 500 |
| **Hybrid GA** | **~300** | **300** |

### Hybrid vs Standard GA

- **Better solutions**: 10-20% improvement
- **Faster convergence**: Reaches good solutions 40% faster
- **Trade-off**: More computation per generation

## 🧪 Exercises

1. Solve TSP with 30+ cities
2. Compare OX vs PMX crossover
3. Experiment with 2-opt iteration limits
4. Implement 3-opt local search
5. Test on asymmetric TSP
6. Explore multi-objective test problems

## 📚 References

### TSP and Permutation GAs

- **Larrañaga, P., et al.** (1999). *Genetic algorithms for the travelling salesman problem: A review of representations and operators*

- **Goldberg, D. E., & Lingle, R.** (1985). *Alleles, loci, and the traveling salesman problem*

### Hybrid Algorithms

- **Hart, W. E., et al.** (2005). *Recent Advances in Memetic Algorithms*

- **Moscato, P.** (1989). *On Evolution, Search, Optimization, Genetic Algorithms and Martial Arts*

### Multi-Objective Optimization

- **Deb, K., et al.** (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation.

- **Zitzler, E., & Thiele, L.** (1999). *Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach*

## 🔜 Next Steps

**Tutorial 4** covers real-world applications:
- Hyperparameter optimization for ML
- Feature selection
- Neural architecture search
- Production scheduling
- Portfolio optimization

## ⚡ Quick Reference

### TSP Crossover
```python
off1, off2 = order_crossover(parent1, parent2)
off1, off2 = partially_mapped_crossover(parent1, parent2)
```

### TSP Mutation
```python
mutated = swap_mutation(tour, mutation_rate=0.1)
mutated = inversion_mutation(tour, mutation_rate=0.1)
mutated = scramble_mutation(tour, mutation_rate=0.1)
```

### Local Search
```python
improved_tour = two_opt_local_search(tour, distance_matrix, max_iterations=100)
```

### Multi-Objective
```python
# Check dominance
is_dominated = dominates(obj1, obj2)

# Sort into fronts
fronts = fast_non_dominated_sort(population, objectives)

# Calculate diversity
distances = calculate_crowding_distance_multi(objectives)
```

## 🤝 Tips for Success

1. **TSP encoding**: Always validate permutations after operations
2. **Local search**: Balance computation vs improvement
3. **Hybrid frequency**: 10-30% local search is usually optimal
4. **Multi-objective**: Maintain diversity on Pareto front
5. **Visualization**: Plot tours and fronts to understand behavior

---

**Master advanced GA techniques for complex real-world problems!** 🚀
