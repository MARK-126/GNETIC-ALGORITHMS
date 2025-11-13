# Data Science Case Studies with Genetic Algorithms

## 📚 Overview

This collection provides **real-world Data Science applications** of Genetic Algorithms. Each case study demonstrates how GAs can solve practical machine learning challenges while teaching both GA techniques and Data Science concepts.

## 🎯 Learning Objectives

- Apply GAs to real-world Data Science problems
- Understand AutoML, feature engineering, and model optimization
- Master ensemble methods and time series forecasting
- Learn best practices for ML model development
- Gain hands-on experience with scikit-learn and modern ML workflows

---

## 📋 Case Studies

### Case Study 1: AutoML with Genetic Algorithms
**Location**: `automl/Case_Study_1_AutoML.ipynb`

**Topics Covered**:
- Automated Machine Learning (AutoML)
- Joint optimization: model selection + hyperparameters + features
- Multi-stage chromosome encoding
- Cross-validation strategies
- Baseline comparisons

**What You'll Build**:
- Complete AutoML system that automatically:
  - Selects best ML algorithm (RF, XGBoost, SVM, etc.)
  - Optimizes hyperparameters
  - Performs feature selection
  - All in one integrated GA

**Key Techniques**:
- Hybrid chromosome encoding (binary + real-valued)
- K-fold cross-validation for fitness
- Model performance benchmarking

**Dataset**: Breast Cancer Wisconsin (real medical data)

**Expected Results**:
- 2-5% accuracy improvement over manual tuning
- Automated feature reduction (30-50%)
- Significant time savings

---

### Case Study 2: Feature Engineering with Genetic Algorithms
**Location**: `feature_engineering/Case_Study_2_Feature_Engineering.ipynb`

**Topics Covered**:
- Automatic feature engineering
- Mathematical feature transformations
- Unary operations (square, sqrt, log, exp, abs)
- Binary operations (add, subtract, multiply, divide)
- Feature interaction discovery
- Preventing overfitting in feature creation

**What You'll Build**:
- GA system that automatically creates new features through:
  - Combining existing features
  - Applying mathematical transformations
  - Discovering non-linear relationships
  - Optimizing feature complexity

**Key Techniques**:
- Feature transformation encoding
- Polynomial feature generation
- Feature importance analysis
- Cross-validation to prevent overfitting

**Dataset**: Make Classification (synthetic, controllable)

**Expected Results**:
- 5-15% accuracy improvement
- Discovery of non-obvious feature interactions
- Reduced manual feature engineering effort

---

### Case Study 3: Ensemble Optimization with Genetic Algorithms
**Location**: `ensemble_optimization/Case_Study_3_Ensemble_Optimization.ipynb`

**Topics Covered**:
- Ensemble learning fundamentals
- Weighted voting ensembles
- Stacking with meta-learners
- Model selection and weighting
- Diversity vs accuracy tradeoff

**What You'll Build**:
- GA-optimized ensemble that:
  - Selects best subset of models
  - Optimizes voting weights
  - Configures meta-learner hyperparameters
  - Balances complexity and performance

**Key Techniques**:
- Soft voting with optimized weights
- Stacking ensemble configuration
- Base model diversity analysis
- Multi-model chromosome encoding

**Dataset**: Breast Cancer Wisconsin (classification)

**Expected Results**:
- 2-5% improvement over best single model
- 1-3% improvement over uniform ensemble
- 30-50% model reduction with maintained accuracy

---

### Case Study 4: Time Series Forecasting with Genetic Algorithms
**Location**: `time_series/Case_Study_4_Time_Series_Optimization.ipynb`

**Topics Covered**:
- Time series fundamentals (trend, seasonality, noise)
- Lag feature selection
- Rolling statistics optimization
- Window size tuning
- Temporal train/validation/test splits
- Forecasting evaluation metrics (RMSE, MAE, MAPE)

**What You'll Build**:
- GA system that optimizes:
  - Which lag features to include
  - Rolling window sizes
  - Model hyperparameters
  - All jointly for best forecast accuracy

**Key Techniques**:
- Temporal feature engineering
- Walk-forward validation
- Multi-window rolling statistics
- Time series cross-validation

**Dataset**: Synthetic time series (trend + seasonality + noise)

**Expected Results**:
- 10-25% RMSE reduction vs simple lag-1 model
- 30-60% improvement vs naive persistence
- Automated feature engineering
- Interpretable feature importance

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Optional (for advanced examples):
```bash
pip install xgboost lightgbm
```

### Running a Case Study

```bash
cd DataScience_Case_Studies/automl
jupyter notebook Case_Study_1_AutoML.ipynb
```

Each notebook is **self-contained** with:
- Theory and background
- Complete working code
- Visualizations
- Performance comparisons
- Exercises for practice

---

## 📊 Comparison Table

| Case Study | Problem Type | Optimization Target | Typical Improvement | Difficulty |
|------------|--------------|---------------------|---------------------|------------|
| **AutoML** | Classification | Model + Hyperparams + Features | 2-5% accuracy | ⭐⭐⭐ |
| **Feature Engineering** | Classification/Regression | Feature transformations | 5-15% accuracy | ⭐⭐⭐⭐ |
| **Ensemble Optimization** | Classification | Model selection + weights | 2-5% accuracy | ⭐⭐⭐ |
| **Time Series** | Forecasting | Lag features + windows | 10-25% RMSE reduction | ⭐⭐⭐⭐ |

---

## 💡 Common Patterns Across Case Studies

### 1. Chromosome Encoding
All case studies use **hybrid encoding**:
- Binary genes for selection (features, models, etc.)
- Real-valued genes for continuous parameters
- Structured layout for interpretability

### 2. Fitness Evaluation
Consistent approach:
- **Primary metric**: Accuracy, RMSE, etc.
- **Complexity penalty**: Favor simpler solutions
- **Cross-validation**: Prevent overfitting
- **Normalization**: Ensure comparable scales

### 3. GA Configuration
Recommended settings:
- **Population size**: 30-50 individuals
- **Generations**: 20-30 for most problems
- **Mutation rate**: 0.1-0.15
- **Crossover rate**: 0.7-0.9
- **Elitism**: 2-5 individuals

### 4. Validation Strategy
Always include:
- Proper train/val/test split
- Cross-validation for fitness
- Baseline comparisons
- Multiple metrics
- Visualization of results

---

## 🔧 Advanced Topics

### Multi-Objective Optimization
Extend any case study to optimize multiple objectives:
```python
# Example: Balance accuracy and complexity
fitness = (accuracy, -n_features)  # Pareto optimization
```

### Hyperparameter Sensitivity
Analyze how GA parameters affect results:
- Population size impact
- Mutation rate effects
- Crossover operator comparison
- Selection pressure analysis

### Scalability
Techniques for larger problems:
- Parallel fitness evaluation
- Island models (distributed GAs)
- Adaptive operators
- Hybrid local search

---

## 📈 Performance Tips

### 1. Preventing Overfitting
- **Always use validation set** separate from test
- **k-fold cross-validation** for fitness
- **Complexity penalties** to favor simpler solutions
- **Early stopping** if validation performance plateaus

### 2. Improving Search Efficiency
- **Start with small populations** (20-30) to iterate quickly
- **Use domain knowledge** to set reasonable parameter ranges
- **Monitor diversity** to avoid premature convergence
- **Adaptive mutation** to balance exploration/exploitation

### 3. Computational Optimization
- **Cache fitness evaluations** for identical chromosomes
- **Parallel evaluation** across population
- **Reduce CV folds** in early generations (e.g., 3-fold → 5-fold)
- **Use fast models** during search, refine final solution

---

## 🎓 Learning Path

### Beginner
1. **Start with**: Case Study 1 (AutoML)
   - Most straightforward application
   - Clear fitness metric
   - Immediate practical value

2. **Then try**: Case Study 3 (Ensemble)
   - Builds on classification knowledge
   - Introduces ensemble concepts
   - Moderate complexity

### Intermediate
3. **Continue with**: Case Study 2 (Feature Engineering)
   - More complex encoding
   - Requires understanding of overfitting
   - Creative problem solving

4. **Advanced**: Case Study 4 (Time Series)
   - Specialized domain knowledge
   - Temporal validation complexities
   - Real-world forecasting challenges

---

## 🛠️ Extending the Case Studies

### Ideas for Enhancement

1. **Different Datasets**:
   - Regression problems (house prices, energy demand)
   - Multi-class classification (image recognition)
   - Large-scale datasets (100k+ samples)

2. **Advanced Models**:
   - Deep learning models (Keras/PyTorch)
   - XGBoost/LightGBM/CatBoost
   - Neural Architecture Search (NAS)

3. **Multi-Objective**:
   - Accuracy vs inference time
   - Performance vs model interpretability
   - Accuracy vs fairness metrics

4. **Production Integration**:
   - MLOps pipelines
   - A/B testing frameworks
   - Model monitoring
   - Automated retraining

---

## 📚 References

### AutoML
- **Feurer, M., et al.** (2015). *Efficient and Robust Automated Machine Learning*. NeurIPS.
- **Thornton, C., et al.** (2013). *Auto-WEKA: Combined Selection and Hyperparameter Optimization*. KDD.

### Feature Engineering
- **Kanter, J. M., & Veeramachaneni, K.** (2015). *Deep Feature Synthesis: Towards Automating Data Science*. IEEE DSAA.
- **Nargesian, F., et al.** (2017). *Learning Feature Engineering for Classification*. IJCAI.

### Ensemble Methods
- **Caruana, R., et al.** (2004). *Ensemble Selection from Libraries of Models*. ICML.
- **Wolpert, D. H.** (1992). *Stacked Generalization*. Neural Networks.

### Time Series
- **Hyndman, R. J., & Athanasopoulos, G.** (2018). *Forecasting: Principles and Practice*. OTexts.
- **De Gooijer, J. G., & Hyndman, R. J.** (2006). *25 Years of Time Series Forecasting*. IJF.

### Genetic Algorithms in ML
- **Eiben, A. E., & Smith, J. E.** (2015). *Introduction to Evolutionary Computing*. Springer.
- **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

---

## 🤝 Contributing

Ideas for new case studies:
- Anomaly detection optimization
- Clustering algorithm selection
- Neural architecture search
- Recommendation system tuning
- Natural language processing applications

---

## 🔜 Next Steps

After completing these case studies, explore:

1. **Tutorial 5**: Specialized Algorithms (CMA-ES, Differential Evolution, PSO)
2. **Interactive Dashboard**: Visualize GA optimization in real-time
3. **Benchmarking Suite**: Compare GA against other optimization methods
4. **Production Deployment**: Take GA-optimized models to production

---

## ⚡ Quick Reference

### Running All Notebooks
```bash
# From DataScience_Case_Studies directory
jupyter notebook
```

### Typical Workflow
```python
# 1. Load data
X_train, y_train, X_test, y_test = load_data()

# 2. Define chromosome encoding
chromosome_length = n_features + n_hyperparams

# 3. Define fitness function
def fitness(chromosome):
    config = decode(chromosome)
    score = cross_val_score(model, X, y, cv=5)
    return score.mean()

# 4. Run GA
best_chromosome, best_fitness, history = genetic_algorithm(
    fitness_func=fitness,
    pop_size=40,
    generations=25
)

# 5. Evaluate final model
final_model = build_model(decode(best_chromosome))
test_score = final_model.score(X_test, y_test)
```

---

**Master Data Science with Genetic Algorithms! These case studies provide hands-on experience with cutting-edge ML optimization techniques.** 🚀
