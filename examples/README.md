# 📚 Examples - Ready-to-Run GA Scripts

This directory contains **complete, executable examples** that demonstrate GA usage with visualizations.

---

## 🚀 Quick Examples

### 1. **quickstart_example.py** - Basic GA with Visualization

**What it does:**
- Optimizes sphere and Rastrigin functions
- Generates convergence plots automatically
- Compares multiple GA runs

**Run it:**
```bash
python examples/quickstart_example.py
```

**Output:**
- Console: Progress and final results
- Files: `sphere_convergence.png`, `rastrigin_convergence.png`, `trials_comparison.png`

**Time:** 1-2 minutes

---

## 📖 Example Code Patterns

### Pattern 1: Basic Optimization

```python
import sys
sys.path.append('../GA_Tutorial_1_Basics')
from ga_utils_basics import genetic_algorithm

def my_fitness(x):
    return -sum(x**2)  # Minimize sum of squares

best_sol, best_fit, history = genetic_algorithm(
    fitness_func=my_fitness,
    n_variables=5,
    bounds=[(-10, 10)] * 5,
    pop_size=50,
    max_generations=100
)

print(f"Best solution: {best_sol}")
print(f"Best fitness: {best_fit}")
```

### Pattern 2: With Visualization

```python
import sys
sys.path.append('../ga_toolkit')
from visualization import plot_convergence

# After running GA:
plot_convergence(history,
                title="My Optimization",
                save_path="my_result.png")
```

### Pattern 3: Compare Algorithms

```python
from visualization import plot_convergence_comparison

histories = {}
for trial in range(3):
    _, _, hist = genetic_algorithm(...)
    histories[f'Trial {trial}'] = hist

plot_convergence_comparison(histories,
                            title="Performance Comparison")
```

---

## 🎯 More Examples (Coming Soon)

Future examples will include:

- `tsp_example.py` - Traveling Salesman Problem with route visualization
- `nsga2_example.py` - Multi-objective optimization with Pareto fronts
- `vrptw_example.py` - Vehicle routing case study
- `ml_optimization_example.py` - Hyperparameter optimization for ML

---

## 💡 Creating Your Own Example

Template for new examples:

```python
"""
My Custom GA Example
Description: What this example does
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GA_Tutorial_X'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ga_toolkit'))

from ga_utils_X import genetic_algorithm
from visualization import plot_convergence

# Your code here
def main():
    print("="*70)
    print("MY CUSTOM GA EXAMPLE")
    print("="*70)

    # Define problem
    def fitness(x):
        # Your fitness function
        pass

    # Run GA
    best_sol, best_fit, history = genetic_algorithm(
        fitness_func=fitness,
        # ... parameters
    )

    # Visualize
    plot_convergence(history)

    print("Done!")

if __name__ == "__main__":
    main()
```

---

## 📊 Expected Results

### quickstart_example.py

**Sphere Function:**
- Expected optimum: [0, 0, 0, 0, 0]
- Typical final fitness: -0.001 to -0.01
- Convergence: Smooth, monotonic

**Rastrigin Function:**
- Expected optimum: [0, 0, 0, 0, 0]
- Typical final fitness: -0.5 to -5.0 (harder problem)
- Convergence: More jagged due to local optima

---

## 🐛 Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure you're running from the repository root or using correct relative paths:

```python
# Option 1: Run from repository root
cd GNETIC-ALGORITHMS
python examples/quickstart_example.py

# Option 2: Adjust sys.path in script
sys.path.append(os.path.abspath('../GA_Tutorial_1_Basics'))
```

### Visualization Not Showing

If plots don't appear:

```python
# Add at top of script
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg' for saving only
```

### Performance Issues

For faster execution:
- Reduce `pop_size` (e.g., 20-30)
- Reduce `max_generations` (e.g., 50)
- These are just demonstrations!

---

## 📝 Notes

- All examples use **minimal dependencies** (numpy, matplotlib)
- Examples are **self-contained** and can run independently
- Generated plots are saved to current directory
- See `QUICKSTART.md` for more beginner-friendly guide

---

**Happy Optimizing!** 🚀
