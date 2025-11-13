"""
Genetic Algorithm Utilities - Real-World Applications
Tutorial 4: Hyperparameter Optimization, Feature Selection, Scheduling

This module provides utilities for practical GA applications.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================== Hyperparameter Optimization ====================

def decode_hyperparameters(chromosome, param_ranges):
    """
    Decode binary/real chromosome to hyperparameters.

    Arguments:
    chromosome -- encoded hyperparameters
    param_ranges -- dict of parameter ranges
        Example: {'learning_rate': (0.001, 0.1), 'n_estimators': (10, 200)}

    Returns:
    hyperparams -- dictionary of decoded hyperparameters
    """
    hyperparams = {}
    n_params = len(param_ranges)
    genes_per_param = len(chromosome) // n_params

    for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
        start_idx = i * genes_per_param
        end_idx = start_idx + genes_per_param
        gene_segment = chromosome[start_idx:end_idx]

        # Decode to real value
        if isinstance(chromosome[0], (int, np.integer)):
            # Binary encoding
            int_value = sum(gene_segment[j] * (2 ** (len(gene_segment) - 1 - j))
                          for j in range(len(gene_segment)))
            max_int = 2 ** len(gene_segment) - 1
            normalized = int_value / max_int
        else:
            # Real encoding (already normalized)
            normalized = np.mean(gene_segment)  # Simple averaging

        # Scale to parameter range
        hyperparams[param_name] = min_val + normalized * (max_val - min_val)

    return hyperparams


def evaluate_ml_model(hyperparams, X_train, y_train, X_val, y_val, model_class):
    """
    Evaluate ML model with given hyperparameters.

    Arguments:
    hyperparams -- dictionary of hyperparameters
    X_train, y_train -- training data
    X_val, y_val -- validation data
    model_class -- ML model class (e.g., RandomForestClassifier)

    Returns:
    score -- validation score (higher is better)
    """
    try:
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        return score
    except:
        return 0.0  # Return poor score if model fails


# ==================== Feature Selection ====================

def decode_feature_mask(chromosome):
    """
    Decode binary chromosome to feature selection mask.

    Arguments:
    chromosome -- binary array (1 = select feature, 0 = exclude)

    Returns:
    selected_features -- indices of selected features
    """
    return np.where(chromosome == 1)[0]


def evaluate_feature_subset(feature_mask, X_train, y_train, X_val, y_val, model):
    """
    Evaluate a subset of features.

    Arguments:
    feature_mask -- binary array indicating selected features
    X_train, y_train -- training data
    X_val, y_val -- validation data
    model -- ML model instance

    Returns:
    fitness -- combined score (accuracy and feature count)
    """
    selected_features = decode_feature_mask(feature_mask)

    if len(selected_features) == 0:
        return 0.0  # No features selected

    # Train on selected features
    X_train_subset = X_train[:, selected_features]
    X_val_subset = X_val[:, selected_features]

    try:
        model.fit(X_train_subset, y_train)
        accuracy = model.score(X_val_subset, y_val)

        # Fitness = accuracy - penalty for too many features
        feature_penalty = len(selected_features) / len(feature_mask) * 0.1
        fitness = accuracy - feature_penalty

        return max(fitness, 0.0)
    except:
        return 0.0


# ==================== Job Scheduling ====================

def decode_schedule(chromosome, n_jobs, n_machines):
    """
    Decode chromosome to job scheduling assignment.

    Arguments:
    chromosome -- permutation or assignment array
    n_jobs -- number of jobs
    n_machines -- number of machines

    Returns:
    schedule -- array of shape (n_jobs,) with machine assignments
    """
    if len(chromosome) == n_jobs:
        # Direct assignment encoding
        return chromosome % n_machines
    else:
        # Permutation encoding - use round-robin
        schedule = np.zeros(n_jobs, dtype=int)
        for i, job in enumerate(chromosome):
            schedule[job] = i % n_machines
        return schedule


def evaluate_schedule(schedule, processing_times, n_machines):
    """
    Evaluate job schedule (minimize makespan).

    Arguments:
    schedule -- machine assignment for each job
    processing_times -- processing time for each job
    n_machines -- number of machines

    Returns:
    makespan -- maximum completion time across all machines
    """
    machine_times = np.zeros(n_machines)

    for job, machine in enumerate(schedule):
        machine_times[machine] += processing_times[job]

    makespan = np.max(machine_times)
    return makespan


# ==================== Portfolio Optimization ====================

def decode_portfolio(chromosome, n_assets):
    """
    Decode chromosome to portfolio weights.

    Arguments:
    chromosome -- real-valued array
    n_assets -- number of assets

    Returns:
    weights -- normalized portfolio weights (sum to 1)
    """
    # Ensure non-negative and normalize
    weights = np.abs(chromosome[:n_assets])
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_assets) / n_assets
    return weights


def evaluate_portfolio(weights, returns, cov_matrix, risk_aversion=0.5):
    """
    Evaluate portfolio (maximize return, minimize risk).

    Arguments:
    weights -- portfolio weights
    returns -- expected returns for each asset
    cov_matrix -- covariance matrix of returns
    risk_aversion -- risk aversion parameter (0=risk-neutral, 1=very risk-averse)

    Returns:
    fitness -- portfolio fitness (return - risk_aversion * risk)
    """
    # Expected return
    portfolio_return = np.dot(weights, returns)

    # Portfolio variance (risk)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)

    # Fitness (higher is better)
    fitness = portfolio_return - risk_aversion * portfolio_risk

    return fitness


# ==================== Neural Architecture Search ====================

def decode_neural_architecture(chromosome):
    """
    Decode chromosome to neural network architecture.

    Arguments:
    chromosome -- array encoding architecture
        [n_layers, layer1_size, layer2_size, ..., activation_idx, dropout_rate]

    Returns:
    architecture -- dictionary describing network
    """
    activations = ['relu', 'tanh', 'sigmoid']

    n_layers = int(chromosome[0]) % 5 + 1  # 1-5 layers
    layer_sizes = [int(abs(chromosome[i+1])) % 256 + 16 for i in range(n_layers)]  # 16-272 neurons
    activation_idx = int(abs(chromosome[n_layers+1])) % len(activations)
    dropout_rate = abs(chromosome[n_layers+2]) % 0.5  # 0-0.5

    architecture = {
        'n_layers': n_layers,
        'layer_sizes': layer_sizes,
        'activation': activations[activation_idx],
        'dropout_rate': dropout_rate
    }

    return architecture


# ==================== Visualization ====================

def plot_hyperparameter_evolution(history, param_names):
    """
    Plot evolution of best hyperparameters.

    Arguments:
    history -- dictionary with 'best_params' per generation
    param_names -- list of parameter names to plot
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))

    if n_params == 1:
        axes = [axes]

    for i, param_name in enumerate(param_names):
        values = [gen_params.get(param_name, 0) for gen_params in history['best_params']]
        axes[i].plot(values, linewidth=2)
        axes[i].set_ylabel(param_name)
        axes[i].set_xlabel('Generation')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_selection_evolution(history):
    """
    Plot evolution of feature selection.

    Arguments:
    history -- dictionary with 'n_features' and 'accuracy' per generation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Number of features
    ax1.plot(history['n_features'], linewidth=2, color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Number of Features Selected')
    ax1.set_title('Feature Count Evolution')
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['accuracy'], linewidth=2, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Accuracy Evolution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_schedule_gantt(schedule, processing_times, n_machines):
    """
    Plot Gantt chart for job schedule.

    Arguments:
    schedule -- machine assignment for each job
    processing_times -- processing time for each job
    n_machines -- number of machines
    """
    fig, ax = plt.subplots(figsize=(12, max(4, n_machines)))

    machine_times = np.zeros(n_machines)
    colors = plt.cm.Set3(np.linspace(0, 1, len(processing_times)))

    for job, machine in enumerate(schedule):
        start_time = machine_times[machine]
        duration = processing_times[job]

        ax.barh(machine, duration, left=start_time, height=0.8,
               color=colors[job], edgecolor='black', label=f'Job {job}')

        machine_times[machine] += duration

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
    ax.set_xlabel('Time')
    ax.set_title(f'Job Schedule (Makespan = {np.max(machine_times):.1f})')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_portfolio_weights(weights, asset_names=None):
    """
    Plot portfolio allocation.

    Arguments:
    weights -- portfolio weights
    asset_names -- optional list of asset names
    """
    if asset_names is None:
        asset_names = [f'Asset {i}' for i in range(len(weights))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax1.bar(range(len(weights)), weights, color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(weights)))
    ax1.set_xticklabels(asset_names, rotation=45, ha='right')
    ax1.set_ylabel('Weight')
    ax1.set_title('Portfolio Weights')
    ax1.grid(True, axis='y', alpha=0.3)

    # Pie chart
    ax2.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Portfolio Allocation')

    plt.tight_layout()
    plt.show()


# ==================== Benchmark Problems ====================

def generate_synthetic_ml_data(n_samples=1000, n_features=20, n_informative=10):
    """Generate synthetic classification data for testing."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        random_state=42
    )

    # Split into train/val
    split_idx = int(0.7 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


def generate_scheduling_problem(n_jobs=20, n_machines=4):
    """Generate random job scheduling problem."""
    processing_times = np.random.randint(1, 20, size=n_jobs)
    return processing_times, n_machines


def generate_portfolio_data(n_assets=10):
    """Generate synthetic portfolio optimization data."""
    # Random returns
    returns = np.random.uniform(0.05, 0.15, size=n_assets)

    # Random covariance matrix (positive semi-definite)
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = np.dot(A, A.T) / n_assets

    return returns, cov_matrix
