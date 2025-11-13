"""
GA Benchmarking Suite
Automated comparison of Genetic Algorithms and variants

Features:
- Compare standard GA, CMA-ES, DE, PSO
- Multiple benchmark functions
- Statistical analysis (mean, std, success rate)
- Visualization and reporting
- Save results to JSON/CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Callable, List, Dict, Tuple
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GA_Tutorial_5_Specialized'))
from ga_utils_specialized import CMA_ES, DifferentialEvolution, ParticleSwarmOptimizer
from ga_utils_specialized import sphere, rosenbrock, rastrigin, ackley


# ==================== Benchmark Functions ====================

BENCHMARK_FUNCTIONS = {
    'Sphere': {
        'func': sphere,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'description': 'Unimodal, separable'
    },
    'Rosenbrock': {
        'func': rosenbrock,
        'bounds': (-2.048, 2.048),
        'optimum': 0.0,
        'description': 'Unimodal, valley-shaped'
    },
    'Rastrigin': {
        'func': rastrigin,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'description': 'Multimodal, many local optima'
    },
    'Ackley': {
        'func': ackley,
        'bounds': (-32.768, 32.768),
        'optimum': 0.0,
        'description': 'Multimodal, deceptive'
    }
}


# ==================== Standard GA Implementation ====================

def standard_ga(objective_func, dim, bounds, max_evals=5000, pop_size=50):
    """Standard Genetic Algorithm for comparison."""
    max_gen = max_evals // pop_size

    # Initialize
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    best_fitness = np.inf
    best_solution = None

    history = {'best_fitness': [], 'mean_fitness': []}

    for gen in range(max_gen):
        # Evaluate
        fitness = np.array([objective_func(ind) for ind in population])

        # Track best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_solution = population[min_idx].copy()

        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(np.mean(fitness))

        # Selection (tournament)
        selected = []
        for _ in range(pop_size - 2):
            tournament_indices = np.random.choice(pop_size, 3, replace=False)
            winner_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected.append(population[winner_idx].copy())

        # Elitism
        elite_indices = np.argsort(fitness)[:2]
        elite = [population[i].copy() for i in elite_indices]

        # Crossover
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if np.random.rand() < 0.8:
                point = np.random.randint(1, dim)
                child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
                child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i].copy(), selected[i+1].copy()])

        # Mutation
        for individual in offspring:
            for j in range(dim):
                if np.random.rand() < 0.1:
                    individual[j] += np.random.normal(0, 0.5)
                    individual[j] = np.clip(individual[j], bounds[0], bounds[1])

        population = np.array(elite + offspring[:pop_size - 2])

    return best_solution, best_fitness, history


# ==================== Benchmark Runner ====================

class BenchmarkRunner:
    """Run comprehensive benchmarks on optimization algorithms."""

    def __init__(self, dimensions: List[int] = [5, 10, 20],
                 max_evals: int = 5000,
                 n_runs: int = 10):
        """
        Initialize benchmark runner.

        Arguments:
        dimensions -- list of problem dimensions to test
        max_evals -- maximum function evaluations per run
        n_runs -- number of independent runs per configuration
        """
        self.dimensions = dimensions
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.results = {}

    def run_single_test(self, algorithm_name: str, algorithm_func: Callable,
                       problem_name: str, problem_func: Callable,
                       bounds: Tuple[float, float], optimum: float,
                       dim: int) -> Dict:
        """
        Run single benchmark test.

        Returns:
        dict with statistics
        """
        best_fitnesses = []
        final_errors = []
        times = []

        for run in range(self.n_runs):
            start_time = time.time()

            # Run algorithm
            if algorithm_name == 'Standard_GA':
                best_sol, best_fit, _ = standard_ga(
                    problem_func, dim, bounds, self.max_evals
                )
            else:
                best_sol, best_fit, _ = algorithm_func(
                    problem_func, dim, bounds, self.max_evals
                )

            elapsed = time.time() - start_time

            best_fitnesses.append(best_fit)
            final_errors.append(abs(best_fit - optimum))
            times.append(elapsed)

        # Compute statistics
        return {
            'algorithm': algorithm_name,
            'problem': problem_name,
            'dim': dim,
            'best': np.min(best_fitnesses),
            'mean': np.mean(best_fitnesses),
            'median': np.median(best_fitnesses),
            'std': np.std(best_fitnesses),
            'worst': np.max(best_fitnesses),
            'mean_error': np.mean(final_errors),
            'success_rate': np.mean(np.array(final_errors) < 1e-4) * 100,  # % reaching optimum
            'avg_time': np.mean(times),
            'std_time': np.std(times)
        }

    def run_benchmark(self):
        """Run full benchmark suite."""
        algorithms = {
            'Standard_GA': lambda f, d, b, me: standard_ga(f, d, b, me),
            'CMA-ES': self._run_cma_es,
            'DE': self._run_de,
            'PSO': self._run_pso
        }

        results = []
        total_tests = len(algorithms) * len(BENCHMARK_FUNCTIONS) * len(self.dimensions)
        current_test = 0

        print("\n" + "="*70)
        print("GENETIC ALGORITHMS BENCHMARK SUITE")
        print("="*70)
        print(f"Algorithms: {list(algorithms.keys())}")
        print(f"Problems: {list(BENCHMARK_FUNCTIONS.keys())}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Runs per config: {self.n_runs}")
        print(f"Max evaluations: {self.max_evals}")
        print(f"Total tests: {total_tests}")
        print("="*70 + "\n")

        for alg_name, alg_func in algorithms.items():
            for prob_name, prob_config in BENCHMARK_FUNCTIONS.items():
                for dim in self.dimensions:
                    current_test += 1

                    print(f"[{current_test}/{total_tests}] {alg_name:15} | "
                          f"{prob_name:15} | Dim {dim:2d}... ", end='', flush=True)

                    result = self.run_single_test(
                        alg_name, alg_func,
                        prob_name, prob_config['func'],
                        prob_config['bounds'], prob_config['optimum'],
                        dim
                    )

                    results.append(result)

                    print(f"Mean: {result['mean']:.4e}, Success: {result['success_rate']:.0f}%")

        self.results = pd.DataFrame(results)

        print("\n" + "="*70)
        print("BENCHMARK COMPLETE!")
        print("="*70)

        return self.results

    def _run_cma_es(self, objective_func, dim, bounds, max_evals):
        """Run CMA-ES."""
        cma = CMA_ES(dim=dim, sigma=0.5)
        max_gen = max_evals // cma.pop_size

        best_solution = None
        best_fitness = np.inf

        history = {'best_fitness': [], 'mean_fitness': []}

        for gen in range(max_gen):
            solutions = cma.ask()
            fitness = np.array([objective_func(x) for x in solutions])
            cma.tell(solutions, fitness)

            if fitness.min() < best_fitness:
                best_fitness = fitness.min()
                best_solution = solutions[np.argmin(fitness)].copy()

            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(fitness.mean())

            if cma.stop():
                break

        return best_solution, best_fitness, history

    def _run_de(self, objective_func, dim, bounds, max_evals):
        """Run Differential Evolution."""
        bounds_list = [(bounds[0], bounds[1])] * dim
        de = DifferentialEvolution(bounds=bounds_list, pop_size=50)
        max_gen = max_evals // 50

        return de.optimize(objective_func, max_generations=max_gen)

    def _run_pso(self, objective_func, dim, bounds, max_evals):
        """Run Particle Swarm Optimization."""
        bounds_list = [(bounds[0], bounds[1])] * dim
        pso = ParticleSwarmOptimizer(n_particles=50, dim=dim, bounds=bounds_list)
        max_iter = max_evals // 50

        return pso.optimize(objective_func, max_iterations=max_iter)

    def generate_report(self, output_file: str = 'benchmark_report.txt'):
        """Generate text report."""
        if self.results is None or len(self.results) == 0:
            print("No results to report. Run benchmark first.")
            return

        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GENETIC ALGORITHMS BENCHMARK REPORT\n")
            f.write("="*70 + "\n\n")

            # Overall statistics by algorithm
            f.write("OVERALL PERFORMANCE BY ALGORITHM\n")
            f.write("-"*70 + "\n")
            summary = self.results.groupby('algorithm').agg({
                'mean': ['mean', 'std'],
                'success_rate': 'mean',
                'avg_time': 'mean'
            }).round(4)
            f.write(summary.to_string() + "\n\n")

            # By problem
            f.write("\nPERFORMANCE BY PROBLEM\n")
            f.write("-"*70 + "\n")
            for prob in BENCHMARK_FUNCTIONS.keys():
                f.write(f"\n{prob}:\n")
                prob_data = self.results[self.results['problem'] == prob]
                prob_summary = prob_data.groupby('algorithm')['mean'].agg(['mean', 'std']).round(6)
                f.write(prob_summary.to_string() + "\n")

            # Rankings
            f.write("\n\nALGORITHM RANKINGS (by mean fitness)\n")
            f.write("-"*70 + "\n")
            rankings = self.results.groupby('algorithm')['mean'].mean().sort_values()
            for rank, (alg, score) in enumerate(rankings.items(), 1):
                f.write(f"{rank}. {alg:15} - Mean fitness: {score:.6e}\n")

        print(f"\nReport saved to: {output_file}")

    def save_results(self, csv_file: str = 'benchmark_results.csv',
                    json_file: str = 'benchmark_results.json'):
        """Save results to CSV and JSON."""
        if self.results is None or len(self.results) == 0:
            print("No results to save.")
            return

        # Save CSV
        self.results.to_csv(csv_file, index=False)
        print(f"Results saved to CSV: {csv_file}")

        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(self.results.to_dict(orient='records'), f, indent=2)
        print(f"Results saved to JSON: {json_file}")

    def plot_results(self, output_file: str = 'benchmark_plots.png'):
        """Generate visualization plots."""
        if self.results is None or len(self.results) == 0:
            print("No results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean fitness by algorithm
        ax = axes[0, 0]
        summary = self.results.groupby('algorithm')['mean'].mean().sort_values()
        summary.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Mean Fitness (lower is better)', fontsize=11)
        ax.set_title('Overall Performance by Algorithm', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # 2. Success rate by algorithm
        ax = axes[0, 1]
        success = self.results.groupby('algorithm')['success_rate'].mean().sort_values(ascending=False)
        success.plot(kind='bar', ax=ax, color='green', edgecolor='black', alpha=0.7)
        ax.set_ylabel('Success Rate (%)', fontsize=11)
        ax.set_title('Success Rate by Algorithm', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        # 3. Performance by problem (heatmap)
        ax = axes[1, 0]
        pivot = self.results.pivot_table(values='mean', index='algorithm', columns='problem', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='YlOrRd_r', ax=ax, cbar_kws={'label': 'Mean Fitness'})
        ax.set_title('Performance by Problem (Mean Fitness)', fontsize=12, fontweight='bold')

        # 4. Scaling with dimension
        ax = axes[1, 1]
        for alg in self.results['algorithm'].unique():
            alg_data = self.results[self.results['algorithm'] == alg]
            dim_perf = alg_data.groupby('dim')['mean'].mean()
            ax.plot(dim_perf.index, dim_perf.values, marker='o', linewidth=2, label=alg, markersize=8)
        ax.set_xlabel('Dimensionality', fontsize=11)
        ax.set_ylabel('Mean Fitness', fontsize=11)
        ax.set_yscale('log')
        ax.set_title('Scalability with Dimension', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {output_file}")
        plt.show()


# ==================== Main ====================

def main():
    """Main benchmark execution."""
    # Create runner
    runner = BenchmarkRunner(
        dimensions=[5, 10, 20],
        max_evals=5000,
        n_runs=10
    )

    # Run benchmark
    results = runner.run_benchmark()

    # Generate outputs
    runner.generate_report('benchmark_report.txt')
    runner.save_results('benchmark_results.csv', 'benchmark_results.json')
    runner.plot_results('benchmark_plots.png')

    print("\n" + "="*70)
    print("BENCHMARK SUITE COMPLETE!")
    print("="*70)
    print("\nOutputs generated:")
    print("  - benchmark_report.txt   : Detailed text report")
    print("  - benchmark_results.csv  : Raw results data")
    print("  - benchmark_results.json : Results in JSON format")
    print("  - benchmark_plots.png    : Visualization plots")
    print("\nCheck these files for detailed analysis.")


if __name__ == '__main__':
    main()
