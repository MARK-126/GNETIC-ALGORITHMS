"""
Visualization utilities for Genetic Algorithms
Shared plotting functions for all tutorials
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import warnings


# ==================== Convergence Plots ====================

def plot_convergence(history: Dict, title: str = "GA Convergence",
                    figsize: Tuple[int, int] = (10, 6),
                    save_path: Optional[str] = None):
    """
    Plot fitness evolution over generations.

    Arguments:
    history -- dict with 'best_fitness' and optionally 'mean_fitness', 'worst_fitness'
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    generations = range(len(history['best_fitness']))

    # Plot best fitness
    ax.plot(generations, history['best_fitness'],
            label='Best Fitness', linewidth=2, color='green', marker='o', markersize=3)

    # Plot mean fitness if available
    if 'mean_fitness' in history:
        ax.plot(generations, history['mean_fitness'],
                label='Mean Fitness', linewidth=1.5, color='blue', alpha=0.7)

    # Plot worst fitness if available
    if 'worst_fitness' in history:
        ax.plot(generations, history['worst_fitness'],
                label='Worst Fitness', linewidth=1, color='red', alpha=0.5, linestyle='--')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


def plot_convergence_comparison(histories: Dict[str, Dict],
                                title: str = "Algorithm Comparison",
                                figsize: Tuple[int, int] = (12, 7),
                                save_path: Optional[str] = None):
    """
    Compare convergence of multiple algorithms.

    Arguments:
    histories -- dict mapping algorithm names to history dicts
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']

    for idx, (alg_name, history) in enumerate(histories.items()):
        generations = range(len(history['best_fitness']))
        color = colors[idx % len(colors)]

        ax.plot(generations, history['best_fitness'],
                label=alg_name, linewidth=2, color=color, marker='o', markersize=3)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


# ==================== Multi-Objective Plots ====================

def plot_pareto_front_2d(objectives: np.ndarray, pareto_indices: Optional[np.ndarray] = None,
                        title: str = "Pareto Front",
                        obj_labels: Tuple[str, str] = ("Objective 1", "Objective 2"),
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None):
    """
    Plot 2D Pareto front.

    Arguments:
    objectives -- (n_solutions, 2) array of objective values
    pareto_indices -- optional indices of Pareto-optimal solutions
    title -- plot title
    obj_labels -- labels for objectives
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all solutions
    ax.scatter(objectives[:, 0], objectives[:, 1],
              alpha=0.4, s=30, color='gray', label='All Solutions')

    # Highlight Pareto front if provided
    if pareto_indices is not None:
        pareto_obj = objectives[pareto_indices]

        # Sort by first objective for line plot
        sorted_idx = np.argsort(pareto_obj[:, 0])
        pareto_sorted = pareto_obj[sorted_idx]

        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
               color='red', linewidth=2, marker='o', markersize=8,
               label='Pareto Front', zorder=5)

    ax.set_xlabel(obj_labels[0], fontsize=12)
    ax.set_ylabel(obj_labels[1], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


def plot_pareto_front_evolution(history: Dict,
                                title: str = "Pareto Front Evolution",
                                figsize: Tuple[int, int] = (12, 10),
                                save_path: Optional[str] = None):
    """
    Animate Pareto front evolution (multiple subplots for key generations).

    Arguments:
    history -- dict with 'pareto_fronts' list (one per generation)
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    if 'pareto_fronts' not in history:
        print("Warning: 'pareto_fronts' not in history")
        return None, None

    n_generations = len(history['pareto_fronts'])

    # Select key generations to plot
    plot_gens = [0, n_generations // 4, n_generations // 2,
                 3 * n_generations // 4, n_generations - 1]
    plot_gens = [g for g in plot_gens if g < n_generations]

    n_plots = len(plot_gens)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=figsize)
    axes = axes.flatten()

    for idx, gen in enumerate(plot_gens):
        ax = axes[idx]
        front = history['pareto_fronts'][gen]

        if len(front) > 0:
            ax.scatter(front[:, 0], front[:, 1],
                      color='red', s=50, alpha=0.7)
            ax.plot(front[:, 0], front[:, 1],
                   color='red', linewidth=1, alpha=0.5)

        ax.set_title(f'Generation {gen}', fontsize=10)
        ax.set_xlabel('Objective 1', fontsize=9)
        ax.set_ylabel('Objective 2', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, axes


# ==================== TSP Visualization ====================

def plot_tsp_tour(cities: np.ndarray, tour: List[int],
                 title: str = "TSP Tour",
                 figsize: Tuple[int, int] = (10, 10),
                 save_path: Optional[str] = None):
    """
    Plot TSP tour on 2D plane.

    Arguments:
    cities -- (n_cities, 2) array of city coordinates
    tour -- list of city indices in tour order
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot cities
    ax.scatter(cities[:, 0], cities[:, 1],
              s=200, c='red', marker='o', zorder=3, alpha=0.7,
              edgecolors='black', linewidths=2)

    # Add city labels
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold')

    # Plot tour
    tour_extended = tour + [tour[0]]  # Return to start
    tour_coords = cities[tour_extended]

    ax.plot(tour_coords[:, 0], tour_coords[:, 1],
           'b-', linewidth=2, alpha=0.6, zorder=2)

    # Calculate and display total distance
    total_distance = 0
    for i in range(len(tour)):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % len(tour)]]
        total_distance += np.linalg.norm(city1 - city2)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\nTotal Distance: {total_distance:.2f}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


# ==================== Population Diversity ====================

def plot_population_diversity(history: Dict,
                              title: str = "Population Diversity",
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None):
    """
    Plot population diversity metrics over generations.

    Arguments:
    history -- dict with diversity metrics
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    generations = range(len(history['best_fitness']))

    if 'diversity' in history:
        ax.plot(generations, history['diversity'],
               label='Diversity', linewidth=2, color='purple')
    elif 'mean_fitness' in history and 'best_fitness' in history:
        # Approximate diversity as fitness range
        diversity = np.array(history['best_fitness']) - np.array(history['mean_fitness'])
        ax.plot(generations, diversity,
               label='Fitness Range', linewidth=2, color='purple')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Diversity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


# ==================== Benchmark Comparison ====================

def plot_algorithm_comparison_boxplot(results: Dict[str, List[float]],
                                     title: str = "Algorithm Performance Comparison",
                                     ylabel: str = "Fitness",
                                     figsize: Tuple[int, int] = (10, 6),
                                     save_path: Optional[str] = None):
    """
    Boxplot comparison of multiple algorithms.

    Arguments:
    results -- dict mapping algorithm names to lists of fitness values
    title -- plot title
    ylabel -- y-axis label
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = list(results.values())
    labels = list(results.keys())

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    notch=True, showmeans=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


# ==================== Heatmap for Parameter Sensitivity ====================

def plot_parameter_heatmap(param1_values: np.ndarray, param2_values: np.ndarray,
                          fitness_matrix: np.ndarray,
                          param1_name: str = "Parameter 1",
                          param2_name: str = "Parameter 2",
                          title: str = "Parameter Sensitivity",
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None):
    """
    Heatmap for parameter sensitivity analysis.

    Arguments:
    param1_values -- array of parameter 1 values
    param2_values -- array of parameter 2 values
    fitness_matrix -- 2D array of fitness values
    param1_name -- name of parameter 1
    param2_name -- name of parameter 2
    title -- plot title
    figsize -- figure size
    save_path -- optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(fitness_matrix, cmap='viridis', aspect='auto',
                  extent=[param1_values[0], param1_values[-1],
                         param2_values[0], param2_values[-1]],
                  origin='lower')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fitness', fontsize=12)

    ax.set_xlabel(param1_name, fontsize=12)
    ax.set_ylabel(param2_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig, ax


# ==================== Helper Functions ====================

def save_all_figures(prefix: str = "ga_plot", format: str = "png"):
    """
    Save all currently open matplotlib figures.

    Arguments:
    prefix -- filename prefix
    format -- image format (png, pdf, svg)
    """
    figs = [plt.figure(n) for n in plt.get_fignums()]

    for idx, fig in enumerate(figs):
        filename = f"{prefix}_{idx+1}.{format}"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("GA Visualization Toolkit")
    print("="*70)

    # Example 1: Convergence plot
    print("\n1. Example: Convergence plot")
    history = {
        'best_fitness': [10, 15, 20, 25, 28, 30, 31, 31.5, 31.8, 32],
        'mean_fitness': [5, 8, 12, 15, 18, 20, 22, 23, 24, 25],
        'worst_fitness': [1, 2, 3, 5, 7, 10, 12, 14, 15, 16]
    }

    plot_convergence(history, title="Example GA Convergence")

    # Example 2: Pareto front
    print("\n2. Example: Pareto front")
    np.random.seed(42)
    objectives = np.random.rand(100, 2)
    pareto_indices = np.array([0, 10, 20, 30, 40])  # Fake Pareto indices

    plot_pareto_front_2d(objectives, pareto_indices,
                        title="Example Pareto Front")

    # Example 3: TSP tour
    print("\n3. Example: TSP tour")
    cities = np.random.rand(10, 2) * 100
    tour = list(range(10))

    plot_tsp_tour(cities, tour, title="Example TSP Tour")

    print("\n" + "="*70)
    print("Examples complete!")
