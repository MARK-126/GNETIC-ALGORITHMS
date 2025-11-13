"""
Scikit-Learn Integration for Genetic Algorithms
Provides sklearn-compatible estimators for GA-based optimization

Key Features:
- GAClassifier: GA for classification hyperparameter tuning
- GARegressor: GA for regression hyperparameter tuning
- GAFeatureSelector: GA-based feature selection
- Compatible with sklearn pipelines and cross-validation
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Dict, Any, Callable, List, Tuple
import warnings


# ==================== GA Hyperparameter Optimizer ====================

class GAHyperparameterOptimizer(BaseEstimator):
    """
    Base class for GA-based hyperparameter optimization.
    Compatible with scikit-learn's estimator interface.
    """

    def __init__(self, estimator, param_space: Dict[str, tuple],
                 pop_size: int = 30, max_generations: int = 20,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 cv: int = 3, scoring: str = 'accuracy', random_state: int = None):
        """
        Initialize GA optimizer.

        Arguments:
        estimator -- sklearn estimator to optimize
        param_space -- dict of {param_name: (min, max, type)} where type is 'int', 'float', or 'choice'
        pop_size -- population size
        max_generations -- maximum generations
        mutation_rate -- mutation probability
        crossover_rate -- crossover probability
        cv -- cross-validation folds
        scoring -- sklearn scoring metric
        random_state -- random seed
        """
        self.estimator = estimator
        self.param_space = param_space
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

        # Will be set during fit
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.history_ = None

    def _decode_chromosome(self, chromosome: np.ndarray) -> Dict[str, Any]:
        """Decode chromosome to hyperparameters."""
        params = {}
        param_names = list(self.param_space.keys())

        for i, param_name in enumerate(param_names):
            param_def = self.param_space[param_name]

            if len(param_def) == 3:
                min_val, max_val, param_type = param_def

                if param_type == 'int':
                    params[param_name] = int(min_val + chromosome[i] * (max_val - min_val))
                elif param_type == 'float':
                    params[param_name] = min_val + chromosome[i] * (max_val - min_val)
                elif param_type == 'choice':
                    # min_val is actually a list of choices
                    idx = int(chromosome[i] * len(min_val)) % len(min_val)
                    params[param_name] = min_val[idx]
            else:
                # Legacy format: (min, max) assumes float
                min_val, max_val = param_def
                params[param_name] = min_val + chromosome[i] * (max_val - min_val)

        return params

    def _evaluate_fitness(self, chromosome: np.ndarray, X, y) -> float:
        """Evaluate fitness of chromosome (hyperparameters)."""
        try:
            params = self._decode_chromosome(chromosome)

            # Create estimator with parameters
            estimator = self.estimator.__class__(**params)

            # Cross-validation score
            score = cross_val_score(estimator, X, y, cv=self.cv,
                                   scoring=self.scoring, n_jobs=1).mean()

            return score

        except Exception as e:
            # Return poor score if parameters invalid
            warnings.warn(f"Invalid parameters: {e}")
            return 0.0

    def fit(self, X, y):
        """
        Optimize hyperparameters using GA.

        Arguments:
        X -- features
        y -- target

        Returns:
        self
        """
        X, y = check_X_y(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_params = len(self.param_space)

        # Initialize population
        population = np.random.rand(self.pop_size, n_params)

        history = {
            'best_score': [],
            'mean_score': []
        }

        best_overall_score = -np.inf
        best_overall_chromosome = None

        # Evolution
        for generation in range(self.max_generations):
            # Evaluate
            fitness = np.array([self._evaluate_fitness(ind, X, y) for ind in population])

            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_overall_score:
                best_overall_score = fitness[best_idx]
                best_overall_chromosome = population[best_idx].copy()

            history['best_score'].append(fitness.max())
            history['mean_score'].append(fitness.mean())

            # Selection (tournament)
            selected = []
            for _ in range(self.pop_size - 2):
                tournament_indices = np.random.choice(self.pop_size, 3, replace=False)
                winner_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
                selected.append(population[winner_idx].copy())

            # Elitism
            elite_indices = np.argsort(fitness)[-2:]
            elite = [population[i].copy() for i in elite_indices]

            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, n_params)
                    child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
                    child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([selected[i].copy(), selected[i+1].copy()])

            # Mutation
            for individual in offspring:
                for j in range(n_params):
                    if np.random.rand() < self.mutation_rate:
                        individual[j] += np.random.normal(0, 0.1)
                        individual[j] = np.clip(individual[j], 0, 1)

            population = np.array(elite + offspring[:self.pop_size - 2])

        # Store results
        self.best_params_ = self._decode_chromosome(best_overall_chromosome)
        self.best_score_ = best_overall_score
        self.best_estimator_ = self.estimator.__class__(**self.best_params_)
        self.best_estimator_.fit(X, y)
        self.history_ = history

        return self

    def predict(self, X):
        """Predict using best estimator."""
        check_is_fitted(self, 'best_estimator_')
        X = check_array(X)
        return self.best_estimator_.predict(X)


class GAClassifier(GAHyperparameterOptimizer, ClassifierMixin):
    """GA-based hyperparameter optimizer for classification."""

    def __init__(self, estimator, param_space: Dict[str, tuple],
                 pop_size: int = 30, max_generations: int = 20,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 cv: int = 3, scoring: str = 'accuracy', random_state: int = None):
        super().__init__(estimator, param_space, pop_size, max_generations,
                        mutation_rate, crossover_rate, cv, scoring, random_state)

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self, 'best_estimator_')
        X = check_array(X)
        if hasattr(self.best_estimator_, 'predict_proba'):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("Estimator does not support predict_proba")


class GARegressor(GAHyperparameterOptimizer, RegressorMixin):
    """GA-based hyperparameter optimizer for regression."""

    def __init__(self, estimator, param_space: Dict[str, tuple],
                 pop_size: int = 30, max_generations: int = 20,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 cv: int = 3, scoring: str = 'r2', random_state: int = None):
        super().__init__(estimator, param_space, pop_size, max_generations,
                        mutation_rate, crossover_rate, cv, scoring, random_state)


# ==================== GA Feature Selector ====================

class GAFeatureSelector(BaseEstimator, TransformerMixin):
    """
    GA-based feature selection.
    Compatible with sklearn pipelines.
    """

    def __init__(self, estimator, pop_size: int = 30, max_generations: int = 20,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.8,
                 cv: int = 3, scoring: str = 'accuracy',
                 min_features: int = 1, max_features: int = None,
                 random_state: int = None):
        """
        Initialize GA feature selector.

        Arguments:
        estimator -- sklearn estimator for evaluation
        pop_size -- population size
        max_generations -- maximum generations
        mutation_rate -- mutation probability
        crossover_rate -- crossover probability
        cv -- cross-validation folds
        scoring -- sklearn scoring metric
        min_features -- minimum features to select
        max_features -- maximum features (None = all)
        random_state -- random seed
        """
        self.estimator = estimator
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.cv = cv
        self.scoring = scoring
        self.min_features = min_features
        self.max_features = max_features
        self.random_state = random_state

        # Will be set during fit
        self.selected_features_ = None
        self.best_score_ = None
        self.n_features_in_ = None

    def _evaluate_fitness(self, chromosome: np.ndarray, X, y) -> float:
        """Evaluate fitness of feature subset."""
        selected = np.where(chromosome > 0.5)[0]

        if len(selected) < self.min_features:
            return 0.0

        if self.max_features is not None and len(selected) > self.max_features:
            return 0.0

        try:
            # Select features
            X_subset = X[:, selected]

            # Cross-validation score
            score = cross_val_score(self.estimator, X_subset, y,
                                   cv=self.cv, scoring=self.scoring).mean()

            # Penalty for too many features
            penalty = 0.01 * len(selected) / X.shape[1]

            return score - penalty

        except:
            return 0.0

    def fit(self, X, y):
        """
        Select features using GA.

        Arguments:
        X -- features
        y -- target

        Returns:
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = X.shape[1]

        # Initialize population (binary chromosomes)
        population = np.random.rand(self.pop_size, n_features)

        best_overall_score = -np.inf
        best_overall_chromosome = None

        # Evolution
        for generation in range(self.max_generations):
            # Evaluate
            fitness = np.array([self._evaluate_fitness(ind, X, y) for ind in population])

            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_overall_score:
                best_overall_score = fitness[best_idx]
                best_overall_chromosome = population[best_idx].copy()

            # Selection
            selected = []
            for _ in range(self.pop_size - 2):
                tournament_indices = np.random.choice(self.pop_size, 3, replace=False)
                winner_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
                selected.append(population[winner_idx].copy())

            elite_indices = np.argsort(fitness)[-2:]
            elite = [population[i].copy() for i in elite_indices]

            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, n_features)
                    child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
                    child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([selected[i].copy(), selected[i+1].copy()])

            # Mutation (bit flip for binary)
            for individual in offspring:
                for j in range(n_features):
                    if np.random.rand() < self.mutation_rate:
                        individual[j] = 1 - individual[j]  # Flip bit

            population = np.array(elite + offspring[:self.pop_size - 2])

        # Store results
        self.selected_features_ = np.where(best_overall_chromosome > 0.5)[0]
        self.best_score_ = best_overall_score

        return self

    def transform(self, X):
        """Transform X by selecting features."""
        check_is_fitted(self, 'selected_features_')
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        """Get selected features mask or indices."""
        check_is_fitted(self, 'selected_features_')

        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
            return mask


# ==================== Convenience Functions ====================

def ga_grid_search(estimator, param_space: Dict, X, y, **ga_kwargs):
    """
    Convenience function for GA-based hyperparameter search.

    Arguments:
    estimator -- sklearn estimator
    param_space -- parameter space dict
    X, y -- data
    **ga_kwargs -- additional arguments for GAHyperparameterOptimizer

    Returns:
    best_params, best_score, best_estimator
    """
    optimizer = GAHyperparameterOptimizer(estimator, param_space, **ga_kwargs)
    optimizer.fit(X, y)

    return optimizer.best_params_, optimizer.best_score_, optimizer.best_estimator_


def ga_feature_selection(estimator, X, y, **ga_kwargs):
    """
    Convenience function for GA-based feature selection.

    Arguments:
    estimator -- sklearn estimator
    X, y -- data
    **ga_kwargs -- additional arguments for GAFeatureSelector

    Returns:
    selected_features, best_score
    """
    selector = GAFeatureSelector(estimator, **ga_kwargs)
    selector.fit(X, y)

    return selector.selected_features_, selector.best_score_
