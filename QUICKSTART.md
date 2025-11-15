# 🚀 Quick Start Guide - Genetic Algorithms in 5 Minutes

Get started with Genetic Algorithms immediately! This guide gets you running your first GA in under 5 minutes.

---

## ⚡ Step 1: Installation (30 seconds)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS.git
cd GNETIC-ALGORITHMS

# Install minimal dependencies
pip install -r requirements-minimal.txt
```

**That's it!** You're ready to run GAs.

---

## 🎯 Step 2: Your First GA (2 minutes)

Create a file `my_first_ga.py`:

```python
import numpy as np
import sys
sys.path.append('GA_Tutorial_1_Basics')
from ga_utils_basics import genetic_algorithm, sphere_function

# Define problem
def fitness_function(x):
    """Minimize sphere function: sum(x^2)"""
    return -sphere_function(x)  # Negative because GA maximizes

# Run GA
bounds = [(-5, 5)] * 5  # 5 dimensions, each in [-5, 5]

best_solution, best_fitness, history = genetic_algorithm(
    fitness_func=fitness_function,
    n_variables=5,
    bounds=bounds,
    pop_size=50,
    max_generations=100,
    mutation_rate=0.1
)

# Results
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
print(f"Should be close to: [0, 0, 0, 0, 0]")
```

**Run it:**
```bash
python my_first_ga.py
```

**Expected output:**
```
Generation 100/100 | Best: -0.0012 | Mean: -2.45
Best solution: [ 0.012 -0.008  0.015 -0.003  0.001]
Best fitness: -0.0011865
Should be close to: [0, 0, 0, 0, 0]
```

✅ **Congratulations!** You just solved your first optimization problem with GA!

---

## 📊 Step 3: Visualize Results (1 minute)

Add visualization to see convergence:

```python
import sys
sys.path.append('ga_toolkit')
from visualization import plot_convergence

# After running GA:
plot_convergence(history, title="My First GA - Sphere Function")
```

You'll see a beautiful plot showing how the GA converged!

---

## 🎓 Step 4: Learn More (1 minute)

### Choose Your Path:

**Beginner?** Start here:
```bash
cd GA_Tutorial_1_Basics
jupyter notebook GA_Introduction.ipynb
```

**Intermediate?** Jump to:
```bash
cd GA_Tutorial_3_Advanced
# Learn TSP, NSGA-II, Parallel GA
```

**Want Real Applications?** Check out:
```bash
cd Tutorial_5_Industrial_Case_Study
python delivery_ga.py
# Solve vehicle routing like Amazon!
```

---

## 🔥 Common Use Cases (Copy-Paste Ready)

### Optimize ML Hyperparameters

```python
import sys
sys.path.append('GA_Tutorial_4_Applications')
from ga_utils_applications import hyperparameter_ga

# Your ML model fitness
def evaluate_model(params):
    # params is a dict like {'learning_rate': 0.01, 'n_estimators': 100}
    # Train model, return validation accuracy
    return val_accuracy

param_ranges = {
    'learning_rate': (0.001, 0.1),
    'n_estimators': (10, 200),
    'max_depth': (2, 20)
}

best_params, history = hyperparameter_ga(
    param_ranges, X_train, y_train, X_val, y_val,
    pop_size=20, max_generations=30
)

print(f"Best hyperparameters: {best_params}")
```

### Solve Traveling Salesman Problem

```python
import sys
sys.path.append('GA_Tutorial_3_Advanced')
from ga_utils_advanced import create_symmetric_tsp, tsp_genetic_algorithm
from ga_toolkit.visualization import plot_tsp_tour

# Create problem
cities, dist_matrix = create_symmetric_tsp(n_cities=20, seed=42)

# Solve
best_tour, best_distance, history = tsp_genetic_algorithm(
    distance_matrix=dist_matrix,
    pop_size=100,
    max_generations=500
)

# Visualize
plot_tsp_tour(cities, best_tour, title=f"TSP Solution ({best_distance:.2f})")
```

### Multi-Objective Optimization (Pareto Front)

```python
import sys
sys.path.append('GA_Tutorial_3_Advanced')
from ga_utils_advanced import nsga2, get_zdt_problem
from ga_toolkit.visualization import plot_pareto_front_2d

# Get benchmark problem
objective_functions, bounds, n_vars = get_zdt_problem('ZDT1')

# Run NSGA-II
population, objectives, pareto_front, history = nsga2(
    objective_functions,
    n_objectives=2,
    bounds=bounds,
    pop_size=100,
    max_generations=100
)

# Visualize Pareto front
plot_pareto_front_2d(
    objectives,
    pareto_front,
    title="ZDT1 Pareto Front"
)
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

```bash
pip install -r requirements-minimal.txt
```

### "ImportError: cannot import name 'genetic_algorithm'"

Make sure you're in the correct directory and using `sys.path.append()`:

```python
import sys
sys.path.append('GA_Tutorial_1_Basics')  # Adjust path as needed
```

### Plots not showing?

If running in a script (not Jupyter), add:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

## 📚 What's Next?

### Complete Tutorials

| Tutorial | Time | What You'll Learn |
|----------|------|------------------|
| Tutorial 1 | 4-6 hours | GA fundamentals, basic operators |
| Tutorial 2 | 6-8 hours | Advanced operators, selection strategies |
| Tutorial 3 | 8-10 hours | TSP, NSGA-II, Parallel GA |
| Tutorial 4 | 6-8 hours | Real-world applications (ML, scheduling) |
| Tutorial 5 | 4-6 hours | Industrial case study (Vehicle routing) |

### Example Projects

1. **Optimize Neural Network Architecture** (Tutorial 4)
   - Automatically design CNN architectures
   - Better than hand-crafted baselines

2. **Solve Delivery Routes** (Tutorial 5)
   - Like Amazon/Uber Eats routing
   - Save 10-25% on distance

3. **Feature Selection for ML** (Tutorial 4)
   - Reduce features by 30-70%
   - Maintain or improve accuracy

---

## 💡 Pro Tips

1. **Start Small**: Test with small population (20-50) and generations (50-100)
2. **Visualize Always**: Use `plot_convergence()` to see if GA is working
3. **Compare Baselines**: Test against random or greedy methods
4. **Tune Parameters**: Population size matters more than mutation rate
5. **Use Tests**: Run `python public_tests.py` in each tutorial

---

## 🆘 Need Help?

- **Documentation**: Each tutorial has detailed README
- **Tests**: Run `python public_tests.py` to verify everything works
- **Examples**: Check `if __name__ == "__main__"` sections in .py files
- **Issues**: Report at [GitHub Issues](https://github.com/YOUR_USERNAME/GNETIC-ALGORITHMS/issues)

---

## ⚡ Summary - Copy This Template

```python
# 1. Import
import sys
sys.path.append('GA_Tutorial_1_Basics')
from ga_utils_basics import genetic_algorithm

# 2. Define fitness
def my_fitness(x):
    return -sum(x**2)  # Minimize sum of squares

# 3. Configure
bounds = [(-10, 10)] * 5  # 5 variables
best_sol, best_fit, history = genetic_algorithm(
    fitness_func=my_fitness,
    n_variables=5,
    bounds=bounds,
    pop_size=50,
    max_generations=100
)

# 4. Results
print(f"Solution: {best_sol}")
print(f"Fitness: {best_fit}")

# 5. Visualize
sys.path.append('ga_toolkit')
from visualization import plot_convergence
plot_convergence(history)
```

**That's it! Now go optimize something!** 🚀

---

**Next:** Open `GA_Tutorial_1_Basics/README.md` for in-depth learning.
