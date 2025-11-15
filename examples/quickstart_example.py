"""
Quick Start Example - Complete GA with Visualization
Demonstrates basic GA usage with the visualization toolkit
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GA_Tutorial_1_Basics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ga_toolkit'))

from ga_utils_basics import genetic_algorithm, sphere_function, rastrigin_function
from visualization import plot_convergence, plot_convergence_comparison

print("="*70)
print("QUICK START EXAMPLE - GENETIC ALGORITHM")
print("="*70)
print("\nThis example demonstrates:")
print("1. Running a basic GA")
print("2. Visualizing convergence")
print("3. Comparing multiple runs")
print("="*70 + "\n")

# ==================== Example 1: Basic GA ====================

print("Example 1: Optimizing Sphere Function")
print("-" * 70)

def fitness_sphere(x):
    """Minimize sphere function: sum(x^2)"""
    return -sphere_function(x)  # Negative because GA maximizes

bounds = [(-5, 5)] * 5  # 5 dimensions

best_solution, best_fitness, history = genetic_algorithm(
    fitness_func=fitness_sphere,
    n_variables=5,
    bounds=bounds,
    pop_size=50,
    max_generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    verbose=True
)

print(f"\nResults:")
print(f"  Best solution: {best_solution}")
print(f"  Best fitness: {best_fitness:.6f}")
print(f"  Expected optimum: [0, 0, 0, 0, 0] with fitness 0")
print(f"  Error: {np.linalg.norm(best_solution):.6f}")

# Visualize convergence
print("\n  Plotting convergence...")
plot_convergence(history, title="Sphere Function Optimization",
                save_path="sphere_convergence.png")

print("\n" + "="*70)

# ==================== Example 2: Harder Problem ====================

print("\nExample 2: Optimizing Rastrigin Function (Harder)")
print("-" * 70)

def fitness_rastrigin(x):
    """Minimize Rastrigin function (multimodal, harder)"""
    return -rastrigin_function(x)

best_solution_r, best_fitness_r, history_r = genetic_algorithm(
    fitness_func=fitness_rastrigin,
    n_variables=5,
    bounds=[(-5.12, 5.12)] * 5,
    pop_size=100,  # Larger population for harder problem
    max_generations=200,
    mutation_rate=0.15,
    crossover_rate=0.8,
    verbose=True
)

print(f"\nResults:")
print(f"  Best solution: {best_solution_r}")
print(f"  Best fitness: {best_fitness_r:.6f}")
print(f"  Expected optimum: [0, 0, 0, 0, 0] with fitness 0")
print(f"  Error: {np.linalg.norm(best_solution_r):.6f}")

# Visualize convergence
print("\n  Plotting convergence...")
plot_convergence(history_r, title="Rastrigin Function Optimization",
                save_path="rastrigin_convergence.png")

print("\n" + "="*70)

# ==================== Example 3: Compare Multiple Runs ====================

print("\nExample 3: Comparing Multiple GA Runs")
print("-" * 70)

# Run GA multiple times with different random seeds
print("Running 3 independent GA trials...")

histories = {}

for trial in range(1, 4):
    np.random.seed(trial * 42)
    print(f"  Trial {trial}/3...", end=" ")

    _, _, hist = genetic_algorithm(
        fitness_func=fitness_sphere,
        n_variables=5,
        bounds=[(-5, 5)] * 5,
        pop_size=50,
        max_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        verbose=False
    )

    histories[f'Trial {trial}'] = hist
    print(f"Done! Final fitness: {hist['best_fitness'][-1]:.4f}")

# Compare all trials
print("\n  Plotting comparison...")
plot_convergence_comparison(histories,
                            title="GA Performance Across 3 Trials",
                            save_path="trials_comparison.png")

print("\n" + "="*70)

# ==================== Summary ====================

print("\n✅ QUICKSTART EXAMPLE COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - sphere_convergence.png")
print("  - rastrigin_convergence.png")
print("  - trials_comparison.png")
print("\nKey Takeaways:")
print("  1. GA successfully optimizes both easy and hard functions")
print("  2. Convergence plots help diagnose GA performance")
print("  3. Multiple trials show GA robustness")
print("\nNext Steps:")
print("  - Check out GA_Tutorial_1_Basics for in-depth learning")
print("  - Try GA_Tutorial_3_Advanced for TSP and NSGA-II")
print("  - Explore Tutorial_5_Industrial_Case_Study for real applications")
print("="*70 + "\n")
