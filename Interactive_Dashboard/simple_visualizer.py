"""
Simple GA Visualizer (No Web Server Required)
Uses matplotlib for static/animated plots

Run with: python simple_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


# ==================== Fitness Functions ====================

def sphere_function(x):
    """Sphere function (minimize)."""
    return -np.sum(x**2)


def rastrigin_function(x):
    """Rastrigin function (minimize)."""
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock_function(x):
    """Rosenbrock function (minimize)."""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


PROBLEM_FUNCTIONS = {
    'sphere': sphere_function,
    'rastrigin': rastrigin_function,
    'rosenbrock': rosenbrock_function
}


# ==================== GA Implementation ====================

class GeneticAlgorithm:
    """Simple GA implementation for visualization."""

    def __init__(self, problem='sphere', pop_size=50, dim=5, bounds=(-5.12, 5.12),
                 mutation_rate=0.1, crossover_rate=0.8, elite_size=2):
        self.problem = problem
        self.fitness_func = PROBLEM_FUNCTIONS[problem]
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.mutation_std = 0.5

        # Initialize
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.generation = 0

        # History
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'best_individual': []
        }

        self.best_fitness = -np.inf
        self.best_individual = None

    def evaluate(self):
        """Evaluate fitness of population."""
        fitness = np.array([self.fitness_func(ind) for ind in self.population])
        return fitness

    def diversity(self):
        """Calculate population diversity."""
        return np.mean(np.std(self.population, axis=0))

    def evolve(self):
        """Run one generation of evolution."""
        # Evaluate
        fitness = self.evaluate()

        # Track best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()

        # Record history
        self.history['best_fitness'].append(fitness.max())
        self.history['mean_fitness'].append(fitness.mean())
        self.history['worst_fitness'].append(fitness.min())
        self.history['diversity'].append(self.diversity())
        self.history['best_individual'].append(self.best_individual.copy())

        # Selection (Tournament)
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            tournament_indices = np.random.choice(self.pop_size, 3, replace=False)
            winner_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
            selected.append(self.population[winner_idx].copy())

        # Elitism
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = [self.population[i].copy() for i in elite_indices]

        # Crossover
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if np.random.rand() < self.crossover_rate:
                point = np.random.randint(1, self.dim)
                child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
                child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i].copy(), selected[i+1].copy()])

        # Mutation
        for individual in offspring:
            for j in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    individual[j] += np.random.normal(0, self.mutation_std)
                    individual[j] = np.clip(individual[j], self.bounds[0], self.bounds[1])

        # New population
        self.population = np.array(elite + offspring[:self.pop_size - self.elite_size])
        self.generation += 1


# ==================== Visualization ====================

def create_static_plots(ga, max_generations=100):
    """Create static plots after optimization."""
    print(f"\n{'='*70}")
    print(f"Running GA Optimization: {ga.problem.upper()}")
    print(f"{'='*70}")
    print(f"Population: {ga.pop_size}, Dimensions: {ga.dim}")
    print(f"Mutation: {ga.mutation_rate}, Crossover: {ga.crossover_rate}\n")

    # Run optimization
    for gen in range(max_generations):
        ga.evolve()
        if gen % 10 == 0:
            print(f"Gen {gen:3d}: Best={ga.best_fitness:.6f}, "
                  f"Mean={ga.history['mean_fitness'][-1]:.6f}, "
                  f"Diversity={ga.history['diversity'][-1]:.4f}")

    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"Best Fitness: {ga.best_fitness:.6f}")
    print(f"Best Individual: {ga.best_individual}")
    print(f"{'='*70}\n")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Fitness Evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ga.history['best_fitness'], 'g-', linewidth=2.5, label='Best Fitness')
    ax1.plot(ga.history['mean_fitness'], 'b-', linewidth=2, label='Mean Fitness')
    ax1.plot(ga.history['worst_fitness'], 'r--', linewidth=1.5, label='Worst Fitness')
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness', fontsize=11)
    ax1.set_title(f'Fitness Evolution - {ga.problem.upper()} Function', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Diversity
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ga.history['diversity'], 'purple', linewidth=2.5, marker='o', markersize=3)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Diversity (Std Dev)', fontsize=11)
    ax2.set_title('Population Diversity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Convergence Rate
    ax3 = fig.add_subplot(gs[1, 1])
    improvements = np.diff(ga.history['best_fitness'])
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax3.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Fitness Improvement', fontsize=11)
    ax3.set_title('Convergence Rate', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, axis='y', alpha=0.3)

    # 4. Population Distribution (2D)
    ax4 = fig.add_subplot(gs[2, 0])
    fitness = ga.evaluate()
    scatter = ax4.scatter(ga.population[:, 0], ga.population[:, 1],
                         c=fitness, cmap='viridis', s=60, alpha=0.7, edgecolors='black')
    ax4.scatter(ga.best_individual[0], ga.best_individual[1],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Best', zorder=5)
    ax4.set_xlabel('Dimension 0', fontsize=11)
    ax4.set_ylabel('Dimension 1', fontsize=11)
    ax4.set_title('Final Population Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    plt.colorbar(scatter, ax=ax4, label='Fitness')

    # 5. Best Individual Evolution (first 3 dimensions)
    ax5 = fig.add_subplot(gs[2, 1])
    best_history = np.array(ga.history['best_individual'])
    for dim in range(min(3, ga.dim)):
        ax5.plot(best_history[:, dim], linewidth=2, label=f'Dim {dim}', marker='o', markersize=2)
    ax5.set_xlabel('Generation', fontsize=11)
    ax5.set_ylabel('Gene Value', fontsize=11)
    ax5.set_title('Best Individual Gene Evolution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'GA Optimization Results - {ga.problem.upper()} Function',
                fontsize=15, fontweight='bold', y=0.995)

    plt.show()


def create_animated_plot(ga, max_generations=100, interval=100):
    """Create animated visualization of optimization."""
    print(f"\n{'='*70}")
    print(f"Animated GA Optimization: {ga.problem.upper()}")
    print(f"{'='*70}")
    print("Close the plot window to end animation.\n")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Axes
    ax1 = fig.add_subplot(gs[0, :])  # Fitness
    ax2 = fig.add_subplot(gs[1, 0])  # Diversity
    ax3 = fig.add_subplot(gs[1, 1])  # Population 2D
    ax4 = fig.add_subplot(gs[2, 0])  # Convergence
    ax5 = fig.add_subplot(gs[2, 1])  # Best individual

    # Initialize plots
    line_best, = ax1.plot([], [], 'g-', linewidth=2.5, label='Best')
    line_mean, = ax1.plot([], [], 'b-', linewidth=2, label='Mean')
    line_worst, = ax1.plot([], [], 'r--', linewidth=1.5, label='Worst')
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness', fontsize=11)
    ax1.set_title(f'Fitness Evolution - {ga.problem.upper()}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    line_div, = ax2.plot([], [], 'purple', linewidth=2.5)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Diversity', fontsize=11)
    ax2.set_title('Population Diversity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    scatter = ax3.scatter([], [], c=[], cmap='viridis', s=60, alpha=0.7)
    best_star = ax3.scatter([], [], c='red', s=200, marker='*', zorder=5)
    ax3.set_xlabel('Dimension 0', fontsize=11)
    ax3.set_ylabel('Dimension 1', fontsize=11)
    ax3.set_title('Population Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlim(ga.bounds[0], ga.bounds[1])
    ax3.set_ylim(ga.bounds[0], ga.bounds[1])

    bars = ax4.bar([], [], color='green')
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Improvement', fontsize=11)
    ax4.set_title('Convergence Rate', fontsize=12, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    lines_genes = []
    for dim in range(min(3, ga.dim)):
        line, = ax5.plot([], [], linewidth=2, label=f'Dim {dim}', marker='o', markersize=2)
        lines_genes.append(line)
    ax5.set_xlabel('Generation', fontsize=11)
    ax5.set_ylabel('Gene Value', fontsize=11)
    ax5.set_title('Best Individual Evolution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(ga.bounds[0], ga.bounds[1])

    def init():
        """Initialize animation."""
        return line_best, line_mean, line_worst, line_div, scatter, best_star, *lines_genes

    def update(frame):
        """Update animation frame."""
        if frame < max_generations:
            ga.evolve()

        # Update fitness plot
        gens = range(len(ga.history['best_fitness']))
        line_best.set_data(gens, ga.history['best_fitness'])
        line_mean.set_data(gens, ga.history['mean_fitness'])
        line_worst.set_data(gens, ga.history['worst_fitness'])
        ax1.set_xlim(0, max(10, len(gens)))
        ax1.set_ylim(min(ga.history['worst_fitness']) * 1.1,
                    max(ga.history['best_fitness']) * 1.1 if max(ga.history['best_fitness']) > 0
                    else max(ga.history['best_fitness']) * 0.9)

        # Update diversity
        line_div.set_data(gens, ga.history['diversity'])
        ax2.set_xlim(0, max(10, len(gens)))
        ax2.set_ylim(0, max(ga.history['diversity']) * 1.1)

        # Update population distribution
        fitness = ga.evaluate()
        scatter.set_offsets(ga.population[:, :2])
        scatter.set_array(fitness)
        if ga.best_individual is not None:
            best_star.set_offsets([[ga.best_individual[0], ga.best_individual[1]]])

        # Update convergence rate
        if len(ga.history['best_fitness']) > 1:
            improvements = np.diff(ga.history['best_fitness'])
            ax4.clear()
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax4.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            ax4.set_xlabel('Generation', fontsize=11)
            ax4.set_ylabel('Improvement', fontsize=11)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.grid(True, axis='y', alpha=0.3)

        # Update best individual evolution
        if len(ga.history['best_individual']) > 0:
            best_history = np.array(ga.history['best_individual'])
            for dim, line in enumerate(lines_genes):
                line.set_data(gens, best_history[:, dim])
            ax5.set_xlim(0, max(10, len(gens)))

        # Update title with current stats
        fig.suptitle(
            f'GA Optimization - Gen {ga.generation} - Best: {ga.best_fitness:.6f}',
            fontsize=14, fontweight='bold'
        )

        return line_best, line_mean, line_worst, line_div, scatter, best_star, *lines_genes

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=max_generations, interval=interval,
                                  blit=False, repeat=False)

    plt.show()


# ==================== Main ====================

def main():
    """Main function with user interaction."""
    print("\n" + "="*70)
    print("🧬 GENETIC ALGORITHM SIMPLE VISUALIZER")
    print("="*70)

    # User input
    print("\nAvailable problems:")
    print("1. Sphere (simple, unimodal)")
    print("2. Rastrigin (multimodal, challenging)")
    print("3. Rosenbrock (valley-shaped)")

    problem_choice = input("\nSelect problem (1-3) [default: 1]: ").strip() or "1"
    problem_map = {'1': 'sphere', '2': 'rastrigin', '3': 'rosenbrock'}
    problem = problem_map.get(problem_choice, 'sphere')

    pop_size = int(input("Population size [default: 50]: ").strip() or "50")
    mutation_rate = float(input("Mutation rate (0.0-0.5) [default: 0.1]: ").strip() or "0.1")
    max_gen = int(input("Maximum generations [default: 100]: ").strip() or "100")

    print("\nVisualization mode:")
    print("1. Static (run then plot)")
    print("2. Animated (live visualization)")

    mode = input("\nSelect mode (1-2) [default: 1]: ").strip() or "1"

    # Create GA
    ga = GeneticAlgorithm(
        problem=problem,
        pop_size=pop_size,
        mutation_rate=mutation_rate
    )

    # Run visualization
    if mode == '2':
        create_animated_plot(ga, max_generations=max_gen, interval=50)
    else:
        create_static_plots(ga, max_generations=max_gen)


if __name__ == '__main__':
    main()
