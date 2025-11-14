# Tutorial 4: Real-World Applications

## 📚 Overview

This tutorial demonstrates how to apply genetic algorithms to practical real-world problems across different domains.

## 🎯 Applications Covered

1. **Hyperparameter Optimization** - Tuning machine learning models
2. **Feature Selection** - Finding optimal feature subsets
3. **Job Scheduling** - Resource allocation and load balancing
4. **Portfolio Optimization** - Financial asset allocation
5. **Neural Architecture Search (NAS)** - Automated neural network design

## 📋 Contents

| File | Description |
|------|-------------|
| `GA_Real_World_Applications.ipynb` | Complete tutorial with 4 practical applications |
| `ga_utils_applications.py` | Application-specific utilities and helpers |
| `neural_architecture_search.py` | **NEW:** Complete NAS framework with TensorFlow/Keras integration |
| `public_tests.py` | 10 validation tests (including NAS tests) |
| `README.md` | This documentation |

## 🚀 Quick Start

```bash
cd GA_Tutorial_4_Applications
jupyter notebook GA_Real_World_Applications.ipynb
```

**Optional:** Install scikit-learn for ML examples:
```bash
pip install scikit-learn
```

## 💡 Application Details

### 1. Hyperparameter Optimization

**Problem:** Find optimal hyperparameters for ML models

**GA Encoding:** Real values normalized to [0,1], decoded to parameter ranges

**Example:**
```python
param_ranges = {
    'n_estimators': (10, 200),
    'max_depth': (2, 20),
    'min_samples_split': (2, 20)
}
```

**Advantages over Grid Search:**
- 10-100x faster
- Better exploration
- Finds better parameters

### 2. Feature Selection

**Problem:** Select minimal feature subset with maximum accuracy

**GA Encoding:** Binary (1 = select feature, 0 = exclude)

**Objectives:**
- Maximize accuracy
- Minimize number of features

**Results:** Typically 30-50% feature reduction with same or better accuracy

### 3. Job Scheduling

**Problem:** Assign jobs to machines to minimize makespan

**GA Encoding:** Integer array (job → machine assignment)

**Metrics:**
- Makespan (max completion time)
- Load balance
- Efficiency

**Performance:** Usually 80-95% of theoretical optimal

### 4. Portfolio Optimization

**Problem:** Allocate capital across assets

**GA Encoding:** Real values (portfolio weights, normalized to sum=1)

**Objectives:**
- Maximize expected return
- Minimize risk (variance)

**Based on:** Markowitz Modern Portfolio Theory

### 5. Neural Architecture Search (NAS)

**Problem:** Automatically design optimal neural network architectures

**GA Encoding:** Layered architecture representation
- Each gene encodes: layer type, filters/units, kernel size, dropout rate
- Training hyperparameters: learning rate, batch size, optimizer

**Architecture Components:**
- Conv2D layers with variable filters (16-256)
- MaxPooling layers
- Dropout layers (0.1-0.5)
- Dense layers with variable units (32-512)
- Automatic flatten before dense layers

**Fitness Function:**
- Validation accuracy (primary)
- Complexity penalty (parameter count)
- Early stopping for efficiency

**Advantages:**
- Automated architecture design
- Finds task-specific architectures
- Balances performance vs complexity
- Much faster than manual tuning

**Typical Results:**
- Finds competitive architectures in 20-50 generations
- Often outperforms hand-designed baselines
- Complexity-aware: avoids over-parameterized models

## 📊 Expected Results

### Hyperparameter Optimization
- **Improvement over default**: 5-15% accuracy gain
- **Vs Grid Search**: 10-50x faster

### Feature Selection
- **Features reduced**: 30-70%
- **Accuracy impact**: ±2% (often improved)
- **Training speed**: 2-5x faster

### Job Scheduling
- **Vs Random**: 40-60% better makespan
- **Vs Greedy**: 10-20% better
- **Efficiency**: 80-95% of optimal

### Portfolio Optimization
- **Sharpe Ratio**: Typically improved 20-40%
- **Risk-adjusted return**: Significantly better than equal-weight

### Neural Architecture Search
- **Accuracy**: Often 2-5% better than baseline architectures
- **Search time**: 20-50 generations with 20-40 individuals
- **Efficiency**: Finds lightweight models (50-80% fewer parameters)
- **Generalization**: Architectures transfer well to similar tasks

## 🔧 Usage Examples

### Hyperparameter Optimization

```python
param_ranges = {
    'learning_rate': (0.001, 0.1),
    'n_estimators': (10, 200)
}

best_params, history = hyperparameter_ga(
    param_ranges, X_train, y_train, X_val, y_val,
    model_class=RandomForestClassifier,
    pop_size=20,
    max_generations=30
)
```

### Feature Selection

```python
best_features, history = feature_selection_ga(
    X_train, y_train, X_val, y_val,
    pop_size=30,
    max_generations=50
)

# Use selected features
selected_indices = decode_feature_mask(best_features)
X_train_selected = X_train[:, selected_indices]
```

### Job Scheduling

```python
processing_times = np.array([5, 3, 7, 2, 4, 6, ...])
n_machines = 4

best_schedule, best_makespan, history = scheduling_ga(
    processing_times, n_machines,
    pop_size=50,
    max_generations=100
)
```

### Portfolio Optimization

```python
returns = np.array([0.08, 0.12, 0.10, ...])  # Expected returns
cov_matrix = ...  # Covariance matrix

best_weights, history = portfolio_ga(
    returns, cov_matrix,
    risk_aversion=0.5,  # 0=risk-neutral, 1=risk-averse
    pop_size=50,
    max_generations=100
)
```

### Neural Architecture Search

```python
from neural_architecture_search import (
    nas_genetic_algorithm,
    LayerGene, ArchitectureChromosome
)

# Prepare data (e.g., MNIST, CIFAR-10)
# X_train, y_train, X_val, y_val = load_data()

# Run NAS
best_architecture, best_fitness, history = nas_genetic_algorithm(
    X_train, y_train, X_val, y_val,
    num_classes=10,
    pop_size=20,                # Population size
    max_generations=30,         # Generations to evolve
    max_layers=8,               # Maximum layers per architecture
    max_epochs_per_eval=5,      # Training epochs per evaluation
    mutation_rate=0.15,
    crossover_rate=0.8
)

# Best architecture found
print(f"Best validation accuracy: {best_architecture.validation_accuracy:.4f}")
print(f"Architecture complexity: {best_architecture.count_parameters()}")

# View layers
for i, layer in enumerate(best_architecture.layers):
    print(f"Layer {i}: {layer}")

# Build and train final model
model = best_architecture.to_keras_model(input_shape=(28, 28, 1))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

# Create custom architecture manually
layers = [
    LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}),
    LayerGene('maxpool2d', {'pool_size': 2}),
    LayerGene('conv2d', {'filters': 64, 'kernel_size': 3, 'activation': 'relu'}),
    LayerGene('dropout', {'rate': 0.3}),
    LayerGene('flatten', {}),
    LayerGene('dense', {'units': 128, 'activation': 'relu'}),
    LayerGene('dense', {'units': 10, 'activation': 'softmax'})
]
arch = ArchitectureChromosome(layers, learning_rate=0.001, batch_size=32)
```

## 🧪 Exercises

1. Compare GA hyperparameter optimization to grid search
2. Test feature selection on real datasets
3. Implement job scheduling with machine-specific capabilities
4. Add transaction costs to portfolio optimization
5. Combine multiple applications (e.g., feature selection + hyperparameter tuning)
6. **Run NAS on MNIST or Fashion-MNIST datasets**
7. **Compare NAS-found architectures to hand-designed baselines**
8. **Experiment with different complexity penalties in NAS**
9. **Add BatchNormalization layers to NAS search space**
10. **Try NAS on CIFAR-10 (more challenging dataset)**

## 📚 References

### Hyperparameter Optimization
- **Bergstra, J., & Bengio, Y.** (2012). *Random search for hyper-parameter optimization*
- **Snoek, J., et al.** (2012). *Practical Bayesian optimization of machine learning algorithms*

### Feature Selection
- **Gu, Q., et al.** (2012). *Generalized Fisher score for feature selection*
- **Chandrashekar, G., & Sahin, F.** (2014). *A survey on feature selection methods*

### Scheduling
- **Pinedo, M. L.** (2012). *Scheduling: Theory, Algorithms, and Systems*
- **Brucker, P.** (2007). *Scheduling Algorithms*

### Portfolio Optimization
- **Markowitz, H.** (1952). *Portfolio selection*. The Journal of Finance
- **Sharpe, W. F.** (1994). *The Sharpe ratio*

### Neural Architecture Search
- **Zoph, B., & Le, Q. V.** (2017). *Neural Architecture Search with Reinforcement Learning*. ICLR.
- **Real, E., et al.** (2019). *Regularized Evolution for Image Classifier Architecture Search*. AAAI.
- **Elsken, T., et al.** (2019). *Neural Architecture Search: A Survey*. JMLR.
- **Stanley, K. O., & Miikkulainen, R.** (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation.

## 🔜 Beyond This Tutorial

### Advanced Topics
- **Constrained optimization**: Handle complex constraints
- **Multi-objective**: Pareto-optimal solutions
- **Hybrid methods**: GA + gradient descent
- **Parallel GAs**: Distributed computation
- **Adaptive GAs**: Self-tuning parameters

### Specialized Variants
- **CMA-ES**: Covariance Matrix Adaptation
- **NEAT**: NeuroEvolution of Augmenting Topologies
- **NSGA-III**: Many-objective optimization
- **MOEA/D**: Decomposition-based multi-objective

### Tools and Libraries
- **DEAP** (Python): Distributed Evolutionary Algorithms
- **PyGAD** (Python): Simple GA library
- **jMetal** (Java): Multi-objective optimization
- **ECJ** (Java): Evolutionary computation

## ⚡ Quick Reference

### Common Patterns

**Minimize objective:**
```python
fitness = -objective_value
```

**Maximize with constraints:**
```python
fitness = objective - penalty * constraint_violation
```

**Multi-objective:**
```python
fitness = w1*obj1 + w2*obj2  # Weighted sum
# OR use Pareto fronts
```

### Parameter Guidelines

| Problem Type | Pop Size | Generations | Mutation Rate |
|--------------|----------|-------------|---------------|
| Hyperparameters | 20-50 | 20-50 | 0.05-0.15 |
| Feature Selection | 30-100 | 50-100 | 0.01-0.05 |
| Scheduling | 50-200 | 100-300 | 0.1-0.2 |
| Portfolio | 50-100 | 100-200 | 0.1-0.2 |
| NAS | 20-40 | 20-50 | 0.1-0.2 |

**Note for NAS:** Start with smaller populations and generations due to expensive fitness evaluations (training networks).

## 🤝 Best Practices

1. **Start simple**: Test on small problems first
2. **Validate**: Use separate test set
3. **Compare baselines**: GA vs simple heuristics
4. **Monitor convergence**: Plot fitness evolution
5. **Tune parameters**: Population size and mutation rate matter
6. **Domain knowledge**: Incorporate problem-specific insights
7. **Hybrid approaches**: Combine GA with local search
8. **NAS**: Use early stopping (3-5 epochs) for fitness evaluation to save time
9. **NAS**: Balance accuracy vs complexity with appropriate penalty weight
10. **NAS**: Start with small max_layers (6-8) before scaling up

---

**Congratulations on completing all 4 tutorials!** 🎉

You now have comprehensive knowledge of Genetic Algorithms from fundamentals to real-world applications!

**Apply these techniques to solve your own optimization problems!** 🚀
