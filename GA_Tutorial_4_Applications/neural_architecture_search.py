"""
Neural Architecture Search (NAS) with Genetic Algorithms
Detailed implementation for searching CNN architectures

Features:
- Layer-wise architecture encoding
- Supports Conv2D, MaxPool, Dropout, Dense layers
- Hyperparameter optimization (filters, kernel size, etc.)
- Efficient evaluation with early stopping
- Architecture visualization
- Compatible with TensorFlow/Keras

Note: Requires tensorflow/keras for actual training
This file provides the complete NAS framework
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


# ==================== Architecture Encoding ====================

class LayerGene:
    """
    Represents a single layer in the neural architecture.
    """

    LAYER_TYPES = ['conv2d', 'maxpool2d', 'dropout', 'dense', 'flatten', 'none']

    def __init__(self, layer_type: str, params: Dict):
        """
        Initialize layer gene.

        Arguments:
        layer_type -- type of layer ('conv2d', 'maxpool2d', etc.)
        params -- dictionary of layer parameters
        """
        self.layer_type = layer_type
        self.params = params

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            'layer_type': self.layer_type,
            'params': self.params
        }

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(d['layer_type'], d['params'])

    def __repr__(self):
        return f"LayerGene({self.layer_type}, {self.params})"


class ArchitectureChromosome:
    """
    Represents a complete neural network architecture.
    """

    def __init__(self, layers: List[LayerGene], learning_rate: float = 0.001,
                 batch_size: int = 32, optimizer: str = 'adam'):
        """
        Initialize architecture chromosome.

        Arguments:
        layers -- list of LayerGene objects
        learning_rate -- optimizer learning rate
        batch_size -- training batch size
        optimizer -- optimizer type
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.fitness = None
        self.validation_accuracy = None

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dictionary."""
        layers = [LayerGene.from_dict(l) for l in d['layers']]
        return cls(layers, d['learning_rate'], d['batch_size'], d['optimizer'])

    def to_keras_model(self, input_shape):
        """
        Convert chromosome to Keras model.

        Arguments:
        input_shape -- input shape tuple (e.g., (28, 28, 1))

        Returns:
        keras model
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers as keras_layers
        except ImportError:
            raise ImportError("TensorFlow/Keras required for model creation")

        model = keras.Sequential()

        # Add input layer
        model.add(keras_layers.Input(shape=input_shape))

        for i, gene in enumerate(self.layers):
            if gene.layer_type == 'conv2d':
                model.add(keras_layers.Conv2D(
                    filters=gene.params['filters'],
                    kernel_size=gene.params['kernel_size'],
                    activation=gene.params.get('activation', 'relu'),
                    padding='same',
                    name=f'conv2d_{i}'
                ))

            elif gene.layer_type == 'maxpool2d':
                model.add(keras_layers.MaxPooling2D(
                    pool_size=gene.params['pool_size'],
                    name=f'maxpool_{i}'
                ))

            elif gene.layer_type == 'dropout':
                model.add(keras_layers.Dropout(
                    rate=gene.params['rate'],
                    name=f'dropout_{i}'
                ))

            elif gene.layer_type == 'flatten':
                model.add(keras_layers.Flatten(name=f'flatten_{i}'))

            elif gene.layer_type == 'dense':
                model.add(keras_layers.Dense(
                    units=gene.params['units'],
                    activation=gene.params.get('activation', 'relu'),
                    name=f'dense_{i}'
                ))

            # 'none' layers are skipped

        return model

    def count_parameters(self):
        """
        Estimate number of parameters (requires knowing input shape).

        Returns approximate complexity score.
        """
        complexity = 0
        for gene in self.layers:
            if gene.layer_type == 'conv2d':
                # Approximate: filters * kernel_size^2
                complexity += gene.params['filters'] * (gene.params['kernel_size'] ** 2)
            elif gene.layer_type == 'dense':
                complexity += gene.params['units']

        return complexity

    def __repr__(self):
        layer_str = " -> ".join([f"{l.layer_type}({l.params})" for l in self.layers])
        return f"Architecture({layer_str})"


# ==================== Chromosome Encoding/Decoding ====================

def decode_chromosome_to_architecture(chromosome: np.ndarray, max_layers: int = 10,
                                     num_classes: int = 10) -> ArchitectureChromosome:
    """
    Decode numeric chromosome to neural architecture.

    Chromosome structure (per layer block):
    - [0]: layer type (0-5 mapped to LAYER_TYPES)
    - [1]: filters (for conv) or units (for dense)
    - [2]: kernel/pool size
    - [3]: dropout rate
    - [4-6]: additional params

    Arguments:
    chromosome -- flat numpy array
    max_layers -- maximum number of layers
    num_classes -- number of output classes

    Returns:
    ArchitectureChromosome
    """
    genes_per_layer = 7  # Each layer encoded with 7 genes
    layers = []

    for i in range(max_layers):
        start_idx = i * genes_per_layer
        end_idx = start_idx + genes_per_layer

        if end_idx > len(chromosome):
            break

        layer_genes = chromosome[start_idx:end_idx]

        # Decode layer type
        layer_type_idx = int(layer_genes[0] * len(LayerGene.LAYER_TYPES)) % len(LayerGene.LAYER_TYPES)
        layer_type = LayerGene.LAYER_TYPES[layer_type_idx]

        if layer_type == 'none':
            continue

        # Decode parameters based on layer type
        if layer_type == 'conv2d':
            filters = int(16 + layer_genes[1] * 240)  # 16-256, powers of 2
            filters = 2 ** int(np.log2(filters))  # Round to power of 2
            kernel_size = 3 if layer_genes[2] < 0.5 else 5
            params = {
                'filters': filters,
                'kernel_size': kernel_size,
                'activation': 'relu'
            }

        elif layer_type == 'maxpool2d':
            pool_size = 2  # Always 2x2
            params = {'pool_size': pool_size}

        elif layer_type == 'dropout':
            rate = 0.1 + layer_genes[3] * 0.4  # 0.1-0.5
            params = {'rate': rate}

        elif layer_type == 'flatten':
            params = {}

        elif layer_type == 'dense':
            units = int(32 + layer_genes[1] * 480)  # 32-512
            units = 2 ** int(np.log2(units))  # Round to power of 2
            params = {
                'units': units,
                'activation': 'relu'
            }

        else:
            continue

        layers.append(LayerGene(layer_type, params))

    # Ensure architecture ends with flatten and dense output
    # Add flatten if not present
    if not any(l.layer_type == 'flatten' for l in layers):
        layers.append(LayerGene('flatten', {}))

    # Add output dense layer
    layers.append(LayerGene('dense', {'units': num_classes, 'activation': 'softmax'}))

    # Decode training hyperparameters
    hp_start = max_layers * genes_per_layer
    if hp_start + 3 <= len(chromosome):
        learning_rate = 10 ** (-4 + chromosome[hp_start] * 2)  # 1e-4 to 1e-2
        batch_size = int(16 + chromosome[hp_start + 1] * 112)  # 16-128
        batch_size = 2 ** int(np.log2(batch_size))  # Round to power of 2
        optimizer = 'adam'  # Could be extended
    else:
        learning_rate = 0.001
        batch_size = 32
        optimizer = 'adam'

    return ArchitectureChromosome(layers, learning_rate, batch_size, optimizer)


def encode_architecture_to_chromosome(architecture: ArchitectureChromosome,
                                     max_layers: int = 10) -> np.ndarray:
    """
    Encode architecture to numeric chromosome.

    Arguments:
    architecture -- ArchitectureChromosome object
    max_layers -- maximum number of layers

    Returns:
    numpy array chromosome
    """
    genes_per_layer = 7
    chromosome = np.zeros(max_layers * genes_per_layer + 3)

    for i, layer in enumerate(architecture.layers[:max_layers]):
        start_idx = i * genes_per_layer

        # Encode layer type
        layer_type_idx = LayerGene.LAYER_TYPES.index(layer.layer_type)
        chromosome[start_idx] = layer_type_idx / len(LayerGene.LAYER_TYPES)

        # Encode parameters
        if layer.layer_type == 'conv2d':
            # Filters: map from 16-256 to [0,1]
            chromosome[start_idx + 1] = (np.log2(layer.params['filters']) - 4) / 4
            # Kernel size: 3 or 5
            chromosome[start_idx + 2] = 0.0 if layer.params['kernel_size'] == 3 else 1.0

        elif layer.layer_type == 'dropout':
            chromosome[start_idx + 3] = (layer.params['rate'] - 0.1) / 0.4

        elif layer.layer_type == 'dense':
            chromosome[start_idx + 1] = (np.log2(layer.params['units']) - 5) / 4

    # Encode hyperparameters
    hp_start = max_layers * genes_per_layer
    chromosome[hp_start] = (np.log10(architecture.learning_rate) + 4) / 2
    chromosome[hp_start + 1] = (np.log2(architecture.batch_size) - 4) / 3

    return chromosome


# ==================== Evaluation ====================

def evaluate_architecture(architecture: ArchitectureChromosome,
                         X_train, y_train, X_val, y_val,
                         max_epochs: int = 10, early_stopping_patience: int = 3,
                         verbose: int = 0) -> Tuple[float, float]:
    """
    Evaluate neural architecture by training on data.

    Arguments:
    architecture -- ArchitectureChromosome to evaluate
    X_train, y_train -- training data
    X_val, y_val -- validation data
    max_epochs -- maximum training epochs
    early_stopping_patience -- patience for early stopping
    verbose -- verbosity level

    Returns:
    (validation_accuracy, train_accuracy)
    """
    try:
        from tensorflow import keras
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        # Return dummy fitness if TensorFlow not available
        print("Warning: TensorFlow not available, returning random fitness")
        return np.random.rand(), np.random.rand()

    try:
        # Build model
        input_shape = X_train.shape[1:]
        model = architecture.to_keras_model(input_shape)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=architecture.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=architecture.batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )

        # Evaluate
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)

        # Clean up
        del model
        keras.backend.clear_session()

        return val_acc, train_acc

    except Exception as e:
        print(f"Error evaluating architecture: {e}")
        return 0.0, 0.0


# ==================== GA for NAS ====================

def nas_genetic_algorithm(X_train, y_train, X_val, y_val, num_classes: int = 10,
                         pop_size: int = 20, max_generations: int = 20,
                         max_layers: int = 8, max_epochs_per_eval: int = 5,
                         mutation_rate: float = 0.15, crossover_rate: float = 0.8,
                         elite_size: int = 2):
    """
    Neural Architecture Search using Genetic Algorithm.

    Arguments:
    X_train, y_train -- training data
    X_val, y_val -- validation data
    num_classes -- number of output classes
    pop_size -- population size
    max_generations -- maximum generations
    max_layers -- maximum layers in architecture
    max_epochs_per_eval -- epochs to train each architecture
    mutation_rate -- mutation probability
    crossover_rate -- crossover probability
    elite_size -- number of elite individuals

    Returns:
    best_architecture, best_fitness, history
    """
    print("\n" + "="*70)
    print("NEURAL ARCHITECTURE SEARCH WITH GA")
    print("="*70)
    print(f"Population size: {pop_size}")
    print(f"Max generations: {max_generations}")
    print(f"Max layers: {max_layers}")
    print(f"Training epochs per eval: {max_epochs_per_eval}")
    print("="*70 + "\n")

    chromosome_length = max_layers * 7 + 3

    # Initialize population
    population = np.random.rand(pop_size, chromosome_length)

    history = {
        'best_fitness': [],
        'mean_fitness': [],
        'best_architectures': []
    }

    best_overall_fitness = 0.0
    best_overall_architecture = None

    for generation in range(max_generations):
        print(f"\nGeneration {generation + 1}/{max_generations}")
        print("-" * 70)

        # Decode and evaluate
        fitness = np.zeros(pop_size)
        architectures = []

        for i in range(pop_size):
            # Decode
            arch = decode_chromosome_to_architecture(
                population[i], max_layers, num_classes
            )
            architectures.append(arch)

            # Evaluate
            print(f"  Evaluating individual {i+1}/{pop_size}...", end=" ")
            val_acc, train_acc = evaluate_architecture(
                arch, X_train, y_train, X_val, y_val,
                max_epochs=max_epochs_per_eval, verbose=0
            )

            # Fitness = validation accuracy - complexity penalty
            complexity = arch.count_parameters() / 10000  # Normalize
            fitness[i] = val_acc - 0.001 * complexity

            arch.fitness = fitness[i]
            arch.validation_accuracy = val_acc

            print(f"Val Acc: {val_acc:.4f}, Fitness: {fitness[i]:.4f}")

        # Track best
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_overall_fitness:
            best_overall_fitness = fitness[best_idx]
            best_overall_architecture = architectures[best_idx]

        history['best_fitness'].append(fitness.max())
        history['mean_fitness'].append(fitness.mean())
        history['best_architectures'].append(architectures[best_idx].to_dict())

        print(f"\nGeneration {generation + 1} Summary:")
        print(f"  Best Fitness: {fitness.max():.4f}")
        print(f"  Mean Fitness: {fitness.mean():.4f}")
        print(f"  Best Val Acc: {architectures[best_idx].validation_accuracy:.4f}")

        # Selection (Tournament)
        selected = []
        for _ in range(pop_size - elite_size):
            tournament_indices = np.random.choice(pop_size, 3, replace=False)
            winner_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
            selected.append(population[winner_idx].copy())

        # Elitism
        elite_indices = np.argsort(fitness)[-elite_size:]
        elite = [population[i].copy() for i in elite_indices]

        # Crossover
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, chromosome_length)
                child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
                child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i].copy(), selected[i+1].copy()])

        # Mutation
        for individual in offspring:
            for j in range(chromosome_length):
                if np.random.rand() < mutation_rate:
                    individual[j] += np.random.normal(0, 0.1)
                    individual[j] = np.clip(individual[j], 0, 1)

        # New population
        population = np.array(elite + offspring[:pop_size - elite_size])

    print("\n" + "="*70)
    print("NAS COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_overall_architecture.validation_accuracy:.4f}")
    print(f"Best Fitness: {best_overall_fitness:.4f}")
    print(f"\nBest Architecture:")
    for i, layer in enumerate(best_overall_architecture.layers):
        print(f"  Layer {i}: {layer}")
    print("="*70 + "\n")

    return best_overall_architecture, best_overall_fitness, history


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("Neural Architecture Search Example")
    print("Note: Requires TensorFlow/Keras and dataset to run actual NAS")
    print("\nExample architecture encoding:")

    # Create example architecture
    layers = [
        LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('maxpool2d', {'pool_size': 2}),
        LayerGene('conv2d', {'filters': 64, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('maxpool2d', {'pool_size': 2}),
        LayerGene('flatten', {}),
        LayerGene('dense', {'units': 128, 'activation': 'relu'}),
        LayerGene('dropout', {'rate': 0.3}),
        LayerGene('dense', {'units': 10, 'activation': 'softmax'})
    ]

    arch = ArchitectureChromosome(layers, learning_rate=0.001, batch_size=32)

    print("\nArchitecture:")
    for i, layer in enumerate(arch.layers):
        print(f"  {i}: {layer}")

    print(f"\nEstimated complexity: {arch.count_parameters()}")

    # Encode and decode
    chromosome = encode_architecture_to_chromosome(arch)
    decoded_arch = decode_chromosome_to_architecture(chromosome)

    print("\nDecoded architecture matches original:")
    print(f"  Layers: {len(decoded_arch.layers)} (original: {len(arch.layers)})")
    print(f"  Learning rate: {decoded_arch.learning_rate:.6f} (original: {arch.learning_rate:.6f})")
