"""
GA Toolkit - Shared utilities for Genetic Algorithms tutorials
"""

from .visualization import (
    plot_convergence,
    plot_convergence_comparison,
    plot_pareto_front_2d,
    plot_pareto_front_evolution,
    plot_tsp_tour,
    plot_population_diversity,
    plot_algorithm_comparison_boxplot,
    plot_parameter_heatmap,
    save_all_figures
)

__version__ = "1.0.0"
__all__ = [
    'plot_convergence',
    'plot_convergence_comparison',
    'plot_pareto_front_2d',
    'plot_pareto_front_evolution',
    'plot_tsp_tour',
    'plot_population_diversity',
    'plot_algorithm_comparison_boxplot',
    'plot_parameter_heatmap',
    'save_all_figures'
]
