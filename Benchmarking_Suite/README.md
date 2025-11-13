# GA Benchmarking Suite

## 📚 Overview

Automated benchmarking system for comparing Genetic Algorithms and variants. Provides statistical analysis, visualization, and detailed reports.

## 🎯 Features

- **Compare 4 algorithms**: Standard GA, CMA-ES, Differential Evolution, PSO
- **4 benchmark functions**: Sphere, Rosenbrock, Rastrigin, Ackley
- **Multiple dimensions**: Test scalability (5D, 10D, 20D)
- **Statistical rigor**: 10 independent runs per configuration
- **Automated reporting**: Text reports, CSV, JSON, plots
- **Publication-ready figures**: High-quality visualizations

## 🚀 Quick Start

### Run Full Benchmark

```bash
cd Benchmarking_Suite
python ga_benchmark.py
```

This will:
1. Run 480 tests (4 algorithms × 4 problems × 3 dimensions × 10 runs)
2. Generate statistical report
3. Save results to CSV and JSON
4. Create visualization plots

**Output files**:
- `benchmark_report.txt` - Detailed analysis
- `benchmark_results.csv` - Raw data
- `benchmark_results.json` - Structured results
- `benchmark_plots.png` - Visualizations

### Custom Benchmark

```python
from ga_benchmark import BenchmarkRunner

# Configure benchmark
runner = BenchmarkRunner(
    dimensions=[10, 20, 30],  # Test dimensions
    max_evals=10000,          # Evaluations per run
    n_runs=20                 # Independent runs
)

# Run
results = runner.run_benchmark()

# Generate outputs
runner.generate_report()
runner.save_results()
runner.plot_results()
```

---

## 📊 What Gets Measured

### Performance Metrics

1. **Best Fitness**: Best solution found across all runs
2. **Mean Fitness**: Average final fitness (consistency)
3. **Median Fitness**: Robust central tendency
4. **Standard Deviation**: Variability across runs
5. **Worst Fitness**: Worst-case performance
6. **Success Rate**: % of runs reaching global optimum (within 1e-4)
7. **Average Time**: Mean execution time per run
8. **Mean Error**: Average distance from known optimum

### Comparison Dimensions

- **By Algorithm**: Overall performance ranking
- **By Problem**: Which algorithm excels at which problem type
- **By Dimension**: Scalability analysis
- **Statistical Significance**: Mean ± std across runs

---

## 🧪 Benchmark Functions

### 1. Sphere Function
```
f(x) = Σ xi²
Optimum: f(0,...,0) = 0
```
**Properties**:
- Unimodal (single optimum)
- Separable (independent variables)
- Easy baseline

**Expected results**:
- CMA-ES: Fastest convergence
- All algorithms: Should reach optimum

### 2. Rosenbrock Function
```
f(x) = Σ [100(x_{i+1} - xi²)² + (1 - xi)²]
Optimum: f(1,...,1) = 0
```
**Properties**:
- Unimodal
- Non-separable (coupled variables)
- Valley-shaped (hard for basic methods)

**Expected results**:
- CMA-ES: Best (handles valleys well)
- Standard GA: Struggles with valley

### 3. Rastrigin Function
```
f(x) = 10n + Σ [xi² - 10cos(2πxi)]
Optimum: f(0,...,0) = 0
```
**Properties**:
- Highly multimodal (many local optima)
- Separable
- Regular structure

**Expected results**:
- DE: Good global search
- PSO: Also effective
- Success rate < 100% for all

### 4. Ackley Function
```
f(x) = -20exp(-0.2√(Σxi²/n)) - exp(Σcos(2πxi)/n) + 20 + e
Optimum: f(0,...,0) = 0
```
**Properties**:
- Multimodal
- Non-separable
- Nearly flat outer region

**Expected results**:
- DE: Best exploration
- CMA-ES: Good if starts near optimum
- Challenging for all algorithms

---

## 📈 Interpreting Results

### Report Sections

#### 1. Overall Performance by Algorithm
Average performance across all problems and dimensions.

**Example**:
```
Algorithm     Mean Fitness    Success Rate    Avg Time
CMA-ES        1.234e-05      78.3%           2.1s
DE            2.456e-04      65.0%           1.8s
PSO           3.789e-04      58.3%           1.2s
Standard_GA   5.123e-03      45.0%           2.5s
```

**Interpretation**:
- CMA-ES: Best overall accuracy
- PSO: Fastest execution
- DE: Good balance

#### 2. Performance by Problem
Algorithm rankings per benchmark function.

**Example - Rosenbrock**:
```
1. CMA-ES:       1.23e-06
2. DE:           4.56e-04
3. PSO:          7.89e-04
4. Standard_GA:  2.34e-02
```

**Interpretation**:
CMA-ES excels at valley-shaped problems.

#### 3. Scalability Analysis
How performance degrades with dimension.

**Look for**:
- Flat lines = scales well
- Steep lines = struggles with dimension
- Cross-over points = dimension where rankings change

---

## 🎨 Visualization Plots

### 1. Overall Performance Bar Chart
- X-axis: Mean fitness
- Y-axis: Algorithms
- **Lower is better**

### 2. Success Rate Bar Chart
- X-axis: Algorithms
- Y-axis: Success rate (%)
- **Higher is better**

### 3. Performance Heatmap
- Rows: Algorithms
- Columns: Problems
- Color: Mean fitness (red = poor, yellow = good)

### 4. Scalability Line Plot
- X-axis: Dimension
- Y-axis: Mean fitness (log scale)
- Lines: Different algorithms
- **Flatter is better**

---

## 🔧 Customization

### Add New Algorithm

```python
# In ga_benchmark.py

def my_custom_algorithm(objective_func, dim, bounds, max_evals):
    """Your algorithm implementation."""
    # ... implementation ...
    return best_solution, best_fitness, history

# Add to algorithms dict in run_benchmark():
algorithms = {
    'Standard_GA': ...,
    'My_Algorithm': lambda f, d, b, me: my_custom_algorithm(f, d, b, me)
}
```

### Add New Benchmark Function

```python
def schwefel(x):
    """Schwefel function."""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Add to BENCHMARK_FUNCTIONS:
BENCHMARK_FUNCTIONS['Schwefel'] = {
    'func': schwefel,
    'bounds': (-500, 500),
    'optimum': 0.0,
    'description': 'Multimodal, deceptive'
}
```

### Adjust Test Parameters

```python
runner = BenchmarkRunner(
    dimensions=[5, 10, 15, 20, 30],  # More dimensions
    max_evals=10000,                  # More evaluations
    n_runs=30                         # More runs for statistics
)
```

---

## 📊 Example Results

### Typical Performance Ranking

**Unimodal Problems** (Sphere, Rosenbrock):
1. CMA-ES (best)
2. Differential Evolution
3. PSO
4. Standard GA

**Multimodal Problems** (Rastrigin, Ackley):
1. Differential Evolution (best)
2. CMA-ES
3. PSO
4. Standard GA

**Speed** (execution time):
1. PSO (fastest)
2. Differential Evolution
3. CMA-ES
4. Standard GA

---

## 🔬 Statistical Analysis

### Understanding Variability

**Low std dev** (< 10% of mean):
- Algorithm is consistent
- Reliable performance

**High std dev** (> 50% of mean):
- Algorithm is inconsistent
- May need parameter tuning

### Success Rate Interpretation

| Success Rate | Meaning |
|--------------|---------|
| > 90% | Excellent, very reliable |
| 70-90% | Good, generally finds optimum |
| 50-70% | Moderate, hit-or-miss |
| < 50% | Poor, rarely finds optimum |

### Comparing Algorithms

Use **mean ± std** for comparison:
- If ranges don't overlap → statistically significant difference
- If ranges overlap significantly → similar performance

---

## 🎯 Best Practices

### 1. Run Enough Repeats

```python
n_runs = 30  # For publication-quality results
n_runs = 10  # For quick testing (default)
```

### 2. Scale Evaluations with Dimension

```python
# Heuristic: 1000 × dimension
max_evals = {
    5: 5000,
    10: 10000,
    20: 20000,
    30: 30000
}
```

### 3. Report Both Mean and Median

- **Mean**: Sensitive to outliers
- **Median**: Robust to outliers
- Report both for complete picture

### 4. Include Error Bars

When plotting, show mean ± std:
```python
plt.errorbar(x, mean, yerr=std, fmt='o-')
```

---

## 📚 Using Results

### For Research Papers

1. Run with `n_runs=30` for statistical rigor
2. Report mean ± std
3. Include all test configurations in appendix
4. Use generated plots (high DPI)
5. Cite this suite and algorithms

### For Algorithm Selection

1. Identify your problem type (unimodal/multimodal)
2. Check benchmark results for similar problems
3. Consider time constraints (PSO fastest)
4. Test top 2-3 algorithms on your actual problem

### For Parameter Tuning

1. Run baseline with default parameters
2. Identify worst-performing dimension
3. Test parameter variations on that dimension
4. Re-run full benchmark with best parameters

---

## 🐛 Troubleshooting

**Issue**: Benchmark takes too long
**Solution**:
```python
runner = BenchmarkRunner(
    dimensions=[10],  # Reduce dimensions
    max_evals=2000,   # Reduce evaluations
    n_runs=5          # Fewer runs
)
```

**Issue**: All algorithms perform similarly
**Solution**: Check if problem is too easy/hard. Adjust bounds or try different problems.

**Issue**: Results seem random (high variance)
**Solution**: Increase `n_runs` or `max_evals`.

---

## 📈 Advanced Analysis

### Convergence Analysis

Add this to track evolution:
```python
all_histories = []
for run in range(n_runs):
    _, _, history = algorithm(...)
    all_histories.append(history['best_fitness'])

# Plot average convergence
avg_convergence = np.mean(all_histories, axis=0)
plt.plot(avg_convergence)
```

### Parameter Sensitivity

Test different parameter values:
```python
for mutation_rate in [0.05, 0.1, 0.15, 0.2]:
    results = run_benchmark_with_mutation(mutation_rate)
    # Compare results
```

---

## 🤝 Contributing

Ideas for enhancements:
- More benchmark functions (CEC suite)
- Multi-objective benchmarks
- Constrained optimization
- Parallel execution
- Interactive dashboard
- Statistical significance tests (t-test, Mann-Whitney)

---

**Benchmark your algorithms with scientific rigor!** 📊🔬
