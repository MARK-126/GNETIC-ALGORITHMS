# Tutorial 5: Specialized Evolutionary Algorithms

## 📚 Overview

This tutorial explores advanced evolutionary and swarm intelligence algorithms that go beyond standard Genetic Algorithms. Learn when and how to use specialized techniques for different problem types.

## 🎯 Learning Objectives

- Understand limitations of standard GAs
- Master **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- Implement **Differential Evolution** (DE)
- Apply **Particle Swarm Optimization** (PSO)
- Compare algorithms on benchmark problems
- Select the right algorithm for your problem

## 📋 Contents

### Files

| File | Description |
|------|-------------|
| `GA_Specialized_Algorithms.ipynb` | Main tutorial with CMA-ES, DE, PSO |
| `ga_utils_specialized.py` | Implementation of specialized algorithms |
| `public_tests.py` | Automated tests for algorithms |
| `README.md` | This documentation |

### Algorithms Covered

#### 1. **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- **Best for**: Continuous optimization, ill-conditioned problems
- **Strengths**: Self-adaptive step sizes, handles rotated problems
- **Use when**: Need robust convergence on continuous real-valued problems
- **Typical speedup over GA**: 2-10× on high-dimensional continuous problems

#### 2. **Differential Evolution** (DE)
- **Best for**: Global optimization, multimodal functions
- **Strengths**: Simple, few parameters, effective exploration
- **Use when**: Many local optima, unknown landscape
- **Typical speedup over GA**: 1.5-5× on global optimization

#### 3. **Particle Swarm Optimization** (PSO)
- **Best for**: Fast convergence, swarm intelligence problems
- **Strengths**: Simple, intuitive, few parameters
- **Use when**: Need quick good solutions, social/swarm behavior
- **Typical speedup over GA**: 2-8× on smooth landscapes

## 🚀 Quick Start

```bash
cd GA_Tutorial_5_Specialized
jupyter notebook GA_Specialized_Algorithms.ipynb
```

### Run Tests
```bash
python public_tests.py
```

## 💡 Key Concepts

### When to Use Each Algorithm

| Problem Type | Recommended Algorithm | Why? |
|--------------|----------------------|------|
| **Smooth, unimodal** | CMA-ES | Fastest convergence, adaptive |
| **Multimodal, continuous** | Differential Evolution | Good global search |
| **Mixed (discrete+continuous)** | Standard GA | Most flexible encoding |
| **Fast approximation** | PSO | Quick convergence to good solutions |
| **High-dimensional (>20D)** | CMA-ES | Handles dimensionality well |
| **Black-box, expensive** | CMA-ES or DE | Sample efficient |
| **Real-time constraints** | PSO | Fast, simple |

### Performance Comparison

**Benchmark**: 100D Rosenbrock function

| Algorithm | Evaluations to 1e-6 | Time | Memory |
|-----------|-------------------|------|---------|
| Standard GA | ~50,000 | 100% | Low |
| CMA-ES | ~10,000 | 80% | High |
| DE | ~15,000 | 60% | Low |
| PSO | ~20,000 | 40% | Low |

## 🔧 Usage Examples

### CMA-ES

```python
from ga_utils_specialized import CMA_ES

# Initialize
cma = CMA_ES(
    dim=10,
    sigma=0.5,  # Initial step size
    pop_size=None  # Auto: 4 + 3*ln(dim)
)

# Optimize
for generation in range(100):
    # Ask for new candidates
    solutions = cma.ask()

    # Evaluate
    fitness = [objective_function(x) for x in solutions]

    # Tell CMA-ES the results
    cma.tell(solutions, fitness)

    # Check convergence
    if cma.stop():
        break

best_solution = cma.result()[0]
best_fitness = cma.result()[1]
```

### Differential Evolution

```python
from ga_utils_specialized import DifferentialEvolution

de = DifferentialEvolution(
    bounds=[(-5, 5)] * 10,  # 10D problem
    pop_size=50,
    F=0.8,  # Differential weight
    CR=0.9  # Crossover probability
)

best_solution, best_fitness, history = de.optimize(
    objective_function,
    max_generations=200
)
```

### Particle Swarm Optimization

```python
from ga_utils_specialized import ParticleSwarmOptimizer

pso = ParticleSwarmOptimizer(
    n_particles=30,
    dim=10,
    bounds=[(-5, 5)] * 10,
    w=0.7,  # Inertia weight
    c1=1.5,  # Cognitive parameter
    c2=1.5   # Social parameter
)

best_solution, best_fitness, history = pso.optimize(
    objective_function,
    max_iterations=200
)
```

## 📊 Algorithm Details

### CMA-ES: How It Works

1. **Start** with Gaussian distribution N(m, σ²C)
   - m: mean (current best guess)
   - σ: step size
   - C: covariance matrix (search shape)

2. **Generate** λ offspring by sampling from distribution

3. **Select** μ best individuals

4. **Update**:
   - Mean: Weighted average of selected
   - Covariance: Learn from successful steps
   - Step size: Cumulative path length control

**Key advantage**: Adapts both magnitude AND direction of search

### Differential Evolution: How It Works

For each individual x_i:

1. **Mutation**: Create mutant vector
   ```
   v_i = x_r1 + F * (x_r2 - x_r3)
   ```
   where r1, r2, r3 are random distinct individuals, F ∈ [0, 2]

2. **Crossover**: Mix with current individual
   ```
   u_i[j] = v_i[j] if rand() < CR else x_i[j]
   ```
   CR: crossover probability ∈ [0, 1]

3. **Selection**: Greedy replacement
   ```
   x_i = u_i if f(u_i) < f(x_i) else x_i
   ```

**Key advantage**: Automatic scaling based on population spread

### PSO: How It Works

Each particle i has:
- Position x_i (current solution)
- Velocity v_i (movement direction/speed)
- Best personal position p_i
- Best global position g (from all particles)

**Update equations**:
```python
v_i = w*v_i + c1*rand()*(p_i - x_i) + c2*rand()*(g - x_i)
x_i = x_i + v_i
```

Where:
- w: inertia (balance exploration/exploitation)
- c1: cognitive (trust own experience)
- c2: social (trust swarm)

**Key advantage**: Simple, fast, intuitive social behavior

## 🧪 Benchmark Problems

All tutorials use standard test functions:

### 1. **Sphere** (Unimodal, Separable)
```
f(x) = Σ xi²
Global minimum: f(0,...,0) = 0
```
**Best algorithm**: CMA-ES (10× faster than GA)

### 2. **Rosenbrock** (Unimodal, Non-separable)
```
f(x) = Σ [100(x_{i+1} - xi²)² + (1 - xi)²]
Global minimum: f(1,...,1) = 0
```
**Best algorithm**: CMA-ES (handles valley well)

### 3. **Rastrigin** (Multimodal, Separable)
```
f(x) = 10n + Σ [xi² - 10cos(2πxi)]
Global minimum: f(0,...,0) = 0
```
**Best algorithm**: DE (good global search)

### 4. **Ackley** (Multimodal, Non-separable)
```
f(x) = -20exp(-0.2√(Σxi²/n)) - exp(Σcos(2πxi)/n) + 20 + e
Global minimum: f(0,...,0) = 0
```
**Best algorithm**: DE or PSO (explore multiple regions)

### 5. **Schwefel** (Multimodal, Deceptive)
```
f(x) = 418.9829n - Σ [xi*sin(√|xi|)]
Global minimum: f(420.9687,...) = 0
```
**Best algorithm**: DE (resistant to deception)

## 📈 Expected Results

### Convergence Speed (20D problems, to 1e-4)

| Problem | Standard GA | CMA-ES | DE | PSO |
|---------|------------|--------|----|----|
| Sphere | 500 gen | 50 gen | 100 gen | 150 gen |
| Rosenbrock | 1000 gen | 150 gen | 300 gen | 400 gen |
| Rastrigin | 800 gen | 200 gen | 250 gen | 300 gen |
| Ackley | 900 gen | 180 gen | 200 gen | 250 gen |

### Success Rate (% reaching global optimum)

| Problem | Standard GA | CMA-ES | DE | PSO |
|---------|------------|--------|----|----|
| Sphere | 95% | 100% | 100% | 98% |
| Rosenbrock | 60% | 95% | 85% | 75% |
| Rastrigin | 50% | 80% | 90% | 70% |
| Ackley | 55% | 85% | 90% | 75% |

## 🔧 Parameter Guidelines

### CMA-ES
```python
# Standard settings (usually work well)
sigma = 0.3  # Initial std dev (1/3 of search range)
pop_size = 4 + int(3 * np.log(dim))  # Auto-sizing

# Advanced tuning
sigma = 0.5  # Larger for multimodal
pop_size *= 2  # Increase for difficult problems
```

### Differential Evolution
```python
# Standard settings
F = 0.8  # Differential weight [0.4 - 1.0]
CR = 0.9  # Crossover rate [0.6 - 1.0]
pop_size = 10 * dim  # Rule of thumb

# For multimodal
F = 0.5  # Lower for more exploration
CR = 0.7  # Lower for more diversity

# For unimodal
F = 0.9  # Higher for faster convergence
CR = 0.95  # Higher for exploitation
```

### PSO
```python
# Standard settings
w = 0.729  # Inertia (constriction coefficient)
c1 = 1.49445  # Cognitive
c2 = 1.49445  # Social
n_particles = 20-40  # Typically

# For exploration
w = 0.9  # Higher inertia
c1 = 2.0  # Trust self more
c2 = 1.0  # Trust swarm less

# For exploitation
w = 0.4  # Lower inertia
c1 = 1.0  # Trust self less
c2 = 2.5  # Trust swarm more
```

## 🎓 Exercises

1. **Compare algorithms** on new benchmark (Griewank, Levy)
2. **Hybridize**: Start with PSO, finish with CMA-ES
3. **Adaptive parameters**: Vary F, CR, w during optimization
4. **Parallel**: Implement island-model with different algorithms
5. **Real application**: Apply to ML hyperparameter tuning
6. **Constraint handling**: Add penalty methods or repair

## 📚 References

### CMA-ES
- **Hansen, N., & Ostermeier, A.** (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation.
- **Hansen, N.** (2016). *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772

### Differential Evolution
- **Storn, R., & Price, K.** (1997). *Differential Evolution – A Simple and Efficient Heuristic for Global Optimization*. Journal of Global Optimization.
- **Das, S., & Suganthan, P. N.** (2011). *Differential Evolution: A Survey*. IEEE Transactions on Evolutionary Computation.

### Particle Swarm Optimization
- **Kennedy, J., & Eberhart, R.** (1995). *Particle Swarm Optimization*. Proceedings of ICNN'95.
- **Shi, Y., & Eberhart, R.** (1998). *A Modified Particle Swarm Optimizer*. IEEE CEC.
- **Clerc, M., & Kennedy, J.** (2002). *The Particle Swarm - Explosion, Stability, and Convergence*. IEEE Transactions on Evolutionary Computation.

## 🔜 Next Steps

**Projects** to try:
- Neural architecture search with specialized algorithms
- Hyperparameter tuning comparison
- Multi-objective extensions (NSGA-II vs MOPSO)
- Hybrid algorithms (GA + CMA-ES local search)

## ⚡ Quick Reference

### Algorithm Selection Flowchart

```
Start
  │
  ├─ Continuous optimization? ─ YES ─┐
  │                                   │
  │                            Smooth/Unimodal? ─ YES ─> CMA-ES
  │                                   │
  │                                   NO
  │                                   │
  │                            Many local optima? ─ YES ─> DE
  │                                   │
  │                                   NO
  │                                   │
  │                            Fast approx? ─ YES ─> PSO
  │                                   │
  │                                   NO ─> DE (safe choice)
  │
  NO
  │
Discrete/Mixed? ─ YES ─> Standard GA
  │
  NO ─> Reconsider problem encoding
```

### Import Cheat Sheet

```python
from ga_utils_specialized import (
    CMA_ES,                    # Covariance Matrix Adaptation
    DifferentialEvolution,      # DE/rand/1/bin variant
    ParticleSwarmOptimizer,     # Standard PSO
    compare_algorithms,         # Benchmark runner
    plot_convergence_comparison # Visualization
)
```

---

**Master advanced evolutionary algorithms for superior optimization performance!** 🚀
