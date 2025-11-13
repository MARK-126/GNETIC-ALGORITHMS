"""
Specialized Evolutionary Algorithms
Tutorial 5: CMA-ES, Differential Evolution, Particle Swarm Optimization

This module provides implementations of advanced evolutionary and swarm intelligence algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional


# ==================== CMA-ES Implementation ====================

class CMA_ES:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Best for continuous optimization with unknown landscape.
    Self-adapts step size and search direction.

    Reference: Hansen & Ostermeier (2001)
    """

    def __init__(self, dim: int, sigma: float = 0.5, pop_size: Optional[int] = None):
        """
        Initialize CMA-ES.

        Arguments:
        dim -- problem dimensionality
        sigma -- initial step size (std dev)
        pop_size -- population size (None = auto: 4 + 3*ln(dim))
        """
        self.dim = dim
        self.sigma = sigma

        # Population size
        if pop_size is None:
            self.pop_size = 4 + int(3 * np.log(dim))
        else:
            self.pop_size = pop_size

        # Selection
        self.mu = self.pop_size // 2  # Number of parents

        # Recombination weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)

        # Adaptation parameters
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs

        # Initialize dynamic variables
        self.mean = np.zeros(dim)
        self.C = np.eye(dim)  # Covariance matrix
        self.pc = np.zeros(dim)  # Evolution path for C
        self.ps = np.zeros(dim)  # Evolution path for sigma
        self.B = np.eye(dim)  # Eigenvectors
        self.D = np.ones(dim)  # Eigenvalues (sqrt)

        self.eigeneval = 0
        self.counteval = 0
        self.chi_n = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

    def ask(self) -> np.ndarray:
        """Generate new candidate solutions."""
        self.arz = np.random.randn(self.pop_size, self.dim)
        self.arx = self.mean + self.sigma * self.arz @ (self.B * self.D)
        return self.arx

    def tell(self, solutions: np.ndarray, fitness: np.ndarray):
        """
        Update distribution based on fitness.

        Arguments:
        solutions -- population (same as from ask())
        fitness -- fitness values (to be minimized)
        """
        # Sort by fitness
        arindex = np.argsort(fitness)

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * solutions[arindex[:self.mu]], axis=0)

        # Cumulation for C
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) / self.sigma * \
                  (self.B @ (1 / self.D * (self.B.T @ (self.mean - old_mean))))

        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs)**(2 * self.counteval / self.pop_size)) /
                self.chi_n) < (1.4 + 2 / (self.dim + 1))

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) / self.sigma * \
                  (self.mean - old_mean)

        # Update C
        artmp = 1 / self.sigma * (solutions[arindex[:self.mu]] - old_mean)
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :] +
                           (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)

        # Update sigma
        self.sigma *= np.exp((self.cs / self.damps) *
                            (np.linalg.norm(self.ps) / self.chi_n - 1))

        # Update B and D (eigendecomposition)
        if self.counteval - self.eigeneval > self.pop_size / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)

        self.counteval += self.pop_size

    def result(self) -> Tuple[np.ndarray, float]:
        """Return current best solution."""
        return self.mean, None

    def stop(self) -> bool:
        """Check stopping criteria."""
        # Simple stopping: check if sigma is very small
        return self.sigma < 1e-10


# ==================== Differential Evolution ====================

class DifferentialEvolution:
    """
    Differential Evolution (DE/rand/1/bin).

    Simple and effective global optimization algorithm.
    Good for multimodal continuous problems.

    Reference: Storn & Price (1997)
    """

    def __init__(self, bounds: List[Tuple[float, float]], pop_size: int = None,
                 F: float = 0.8, CR: float = 0.9):
        """
        Initialize Differential Evolution.

        Arguments:
        bounds -- list of (min, max) for each dimension
        pop_size -- population size (None = 10*dim)
        F -- differential weight (0.4 - 1.0)
        CR -- crossover probability (0.6 - 1.0)
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.F = F
        self.CR = CR

        if pop_size is None:
            self.pop_size = 10 * self.dim
        else:
            self.pop_size = max(4, pop_size)  # Minimum 4 for mutation

        # Initialize population
        self.population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.pop_size, self.dim)
        )

        self.fitness = None
        self.best_idx = None
        self.best_solution = None
        self.best_fitness = np.inf

    def mutate(self, idx: int) -> np.ndarray:
        """
        DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3).

        Arguments:
        idx -- index of current individual (to exclude from selection)

        Returns:
        mutant -- mutant vector
        """
        # Select 3 random distinct individuals (different from idx)
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

        # Mutation
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

        # Clip to bounds
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

        return mutant

    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover.

        Arguments:
        target -- current individual
        mutant -- mutant vector

        Returns:
        trial -- trial vector
        """
        trial = np.copy(target)

        # Ensure at least one dimension from mutant
        j_rand = np.random.randint(self.dim)

        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def optimize(self, objective_func: Callable, max_generations: int = 200) -> Tuple[np.ndarray, float, dict]:
        """
        Run DE optimization.

        Arguments:
        objective_func -- function to minimize f(x) -> scalar
        max_generations -- maximum iterations

        Returns:
        best_solution -- best found solution
        best_fitness -- best fitness value
        history -- optimization history
        """
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }

        # Initial evaluation
        self.fitness = np.array([objective_func(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_solution = self.population[self.best_idx].copy()

        for generation in range(max_generations):
            # For each individual
            for i in range(self.pop_size):
                # Mutation
                mutant = self.mutate(i)

                # Crossover
                trial = self.crossover(self.population[i], mutant)

                # Selection
                trial_fitness = objective_func(trial)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Update best
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()

            # Record history
            history['best_fitness'].append(self.best_fitness)
            history['mean_fitness'].append(np.mean(self.fitness))
            history['diversity'].append(np.mean(np.std(self.population, axis=0)))

        return self.best_solution, self.best_fitness, history


# ==================== Particle Swarm Optimization ====================

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO).

    Swarm intelligence algorithm inspired by bird flocking.
    Fast convergence, simple, few parameters.

    Reference: Kennedy & Eberhart (1995)
    """

    def __init__(self, n_particles: int, dim: int, bounds: List[Tuple[float, float]],
                 w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445):
        """
        Initialize PSO.

        Arguments:
        n_particles -- number of particles (swarm size)
        dim -- problem dimensionality
        bounds -- list of (min, max) for each dimension
        w -- inertia weight (0.4 - 0.9)
        c1 -- cognitive parameter (personal best attraction)
        c2 -- social parameter (global best attraction)
        """
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = np.array(bounds)
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize positions
        self.positions = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (n_particles, dim)
        )

        # Initialize velocities (small random)
        v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.velocities = np.random.uniform(-v_max, v_max, (n_particles, dim))

        # Personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(n_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def update(self, fitness: np.ndarray):
        """
        Update velocities and positions.

        Arguments:
        fitness -- current fitness values for all particles
        """
        # Update personal bests
        improved = fitness < self.personal_best_fitness
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = fitness[improved]

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = fitness[best_idx]
            self.global_best_position = self.positions[best_idx].copy()

        # Update velocities
        r1 = np.random.rand(self.n_particles, self.dim)
        r2 = np.random.rand(self.n_particles, self.dim)

        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)

        self.velocities = self.w * self.velocities + cognitive + social

        # Clamp velocities
        v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.velocities = np.clip(self.velocities, -v_max, v_max)

        # Update positions
        self.positions = self.positions + self.velocities

        # Clamp positions to bounds
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self, objective_func: Callable, max_iterations: int = 200) -> Tuple[np.ndarray, float, dict]:
        """
        Run PSO optimization.

        Arguments:
        objective_func -- function to minimize f(x) -> scalar
        max_iterations -- maximum iterations

        Returns:
        best_solution -- best found solution
        best_fitness -- best fitness value
        history -- optimization history
        """
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }

        for iteration in range(max_iterations):
            # Evaluate
            fitness = np.array([objective_func(p) for p in self.positions])

            # Update
            self.update(fitness)

            # Record history
            history['best_fitness'].append(self.global_best_fitness)
            history['mean_fitness'].append(np.mean(fitness))
            history['diversity'].append(np.mean(np.std(self.positions, axis=0)))

        return self.global_best_position, self.global_best_fitness, history


# ==================== Benchmark Functions ====================

def sphere(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x^2)."""
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function."""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """Ackley function."""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


# ==================== Comparison Utilities ====================

def compare_algorithms(objective_func: Callable, dim: int = 10, bounds: Tuple[float, float] = (-5.12, 5.12),
                      max_evals: int = 5000, n_runs: int = 10) -> dict:
    """
    Compare all three algorithms on a benchmark function.

    Arguments:
    objective_func -- function to minimize
    dim -- problem dimensionality
    bounds -- search bounds (min, max)
    max_evals -- maximum function evaluations
    n_runs -- number of independent runs

    Returns:
    results -- dictionary with statistics for each algorithm
    """
    results = {
        'CMA-ES': {'best': [], 'mean': [], 'time': []},
        'DE': {'best': [], 'mean': [], 'time': []},
        'PSO': {'best': [], 'mean': [], 'time': []}
    }

    import time

    bounds_list = [(bounds[0], bounds[1])] * dim

    for run in range(n_runs):
        # CMA-ES
        start = time.time()
        cma = CMA_ES(dim=dim, sigma=0.5)
        gen = 0
        max_gen = max_evals // cma.pop_size
        for gen in range(max_gen):
            solutions = cma.ask()
            fitness = np.array([objective_func(x) for x in solutions])
            cma.tell(solutions, fitness)
            if cma.stop():
                break
        best_x, _ = cma.result()
        results['CMA-ES']['best'].append(objective_func(best_x))
        results['CMA-ES']['time'].append(time.time() - start)

        # DE
        start = time.time()
        de = DifferentialEvolution(bounds=bounds_list, pop_size=50)
        max_gen_de = max_evals // 50
        best_x, best_f, _ = de.optimize(objective_func, max_generations=max_gen_de)
        results['DE']['best'].append(best_f)
        results['DE']['time'].append(time.time() - start)

        # PSO
        start = time.time()
        pso = ParticleSwarmOptimizer(n_particles=50, dim=dim, bounds=bounds_list)
        max_iter_pso = max_evals // 50
        best_x, best_f, _ = pso.optimize(objective_func, max_iterations=max_iter_pso)
        results['PSO']['best'].append(best_f)
        results['PSO']['time'].append(time.time() - start)

    # Compute statistics
    for alg in results:
        results[alg]['mean'] = np.mean(results[alg]['best'])
        results[alg]['std'] = np.std(results[alg]['best'])
        results[alg]['median'] = np.median(results[alg]['best'])
        results[alg]['avg_time'] = np.mean(results[alg]['time'])

    return results


def plot_convergence_comparison(histories: dict, title: str = "Algorithm Comparison"):
    """
    Plot convergence curves for multiple algorithms.

    Arguments:
    histories -- dict of {algorithm_name: history_dict}
    title -- plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'CMA-ES': 'blue', 'DE': 'green', 'PSO': 'red', 'GA': 'orange'}

    # Best fitness
    ax = axes[0]
    for alg_name, history in histories.items():
        color = colors.get(alg_name, 'gray')
        ax.semilogy(history['best_fitness'], linewidth=2.5, label=alg_name, color=color)
    ax.set_xlabel('Generation / Iteration', fontsize=11)
    ax.set_ylabel('Best Fitness (log scale)', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Diversity
    ax = axes[1]
    for alg_name, history in histories.items():
        if 'diversity' in history:
            color = colors.get(alg_name, 'gray')
            ax.plot(history['diversity'], linewidth=2.5, label=alg_name, color=color)
    ax.set_xlabel('Generation / Iteration', fontsize=11)
    ax.set_ylabel('Population Diversity', fontsize=11)
    ax.set_title('Diversity Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
