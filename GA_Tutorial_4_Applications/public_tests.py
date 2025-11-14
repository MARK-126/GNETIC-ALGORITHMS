"""
Public Test Cases for GA Tutorial 4 - Real-World Applications
Simple validation tests for application utilities.
"""

import numpy as np
from ga_utils_applications import *
from neural_architecture_search import (
    LayerGene, ArchitectureChromosome,
    encode_architecture_to_chromosome, decode_chromosome_to_architecture
)


def test_portfolio_decoding():
    """Test portfolio weight decoding."""
    print("Testing portfolio decoding...")

    chromosome = np.array([0.2, 0.3, 0.5, 0.1])
    weights = decode_portfolio(chromosome, n_assets=4)

    # Should sum to 1
    assert np.isclose(np.sum(weights), 1.0), f"Weights should sum to 1, got {np.sum(weights)}"

    # Should be non-negative
    assert np.all(weights >= 0), "Weights should be non-negative"

    print("✓ Portfolio decoding test passed!")


def test_schedule_evaluation():
    """Test schedule evaluation."""
    print("Testing schedule evaluation...")

    processing_times = np.array([5, 3, 7, 2, 4])
    schedule = np.array([0, 1, 0, 1, 2])
    n_machines = 3

    makespan = evaluate_schedule(schedule, processing_times, n_machines)

    # Machine 0: jobs 0,2 = 5+7 = 12
    # Machine 1: jobs 1,3 = 3+2 = 5
    # Machine 2: job 4 = 4
    # Makespan should be 12
    assert makespan == 12, f"Expected makespan 12, got {makespan}"

    print("✓ Schedule evaluation test passed!")


def test_feature_mask():
    """Test feature mask decoding."""
    print("Testing feature mask decoding...")

    mask = np.array([1, 0, 1, 0, 1, 1, 0])
    selected = decode_feature_mask(mask)

    expected = np.array([0, 2, 4, 5])
    assert np.array_equal(selected, expected), f"Expected {expected}, got {selected}"

    print("✓ Feature mask decoding test passed!")


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")

    X_train, y_train, X_val, y_val = generate_synthetic_ml_data(
        n_samples=100, n_features=10, n_informative=5
    )

    assert X_train.shape[1] == 10, "Should have 10 features"
    assert len(X_train) + len(X_val) == 100, "Total samples should be 100"
    assert len(np.unique(y_train)) == 2, "Should be binary classification"

    print("✓ Synthetic data generation test passed!")


def test_portfolio_data_generation():
    """Test portfolio data generation."""
    print("Testing portfolio data generation...")

    returns, cov_matrix = generate_portfolio_data(n_assets=5)

    assert len(returns) == 5, "Should have 5 asset returns"
    assert cov_matrix.shape == (5, 5), "Covariance matrix should be 5x5"

    # Check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T), "Covariance matrix should be symmetric"

    print("✓ Portfolio data generation test passed!")


def test_layer_gene_creation():
    """Test LayerGene creation and serialization."""
    print("Testing LayerGene creation...")

    # Create conv2d layer
    conv_layer = LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'})
    assert conv_layer.layer_type == 'conv2d', "Layer type mismatch"
    assert conv_layer.params['filters'] == 32, "Filters mismatch"

    # Serialize and deserialize
    layer_dict = conv_layer.to_dict()
    restored_layer = LayerGene.from_dict(layer_dict)
    assert restored_layer.layer_type == conv_layer.layer_type, "Deserialization failed"
    assert restored_layer.params == conv_layer.params, "Params deserialization failed"

    print("✓ LayerGene creation test passed!")


def test_architecture_chromosome():
    """Test ArchitectureChromosome creation."""
    print("Testing ArchitectureChromosome...")

    layers = [
        LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('maxpool2d', {'pool_size': 2}),
        LayerGene('flatten', {}),
        LayerGene('dense', {'units': 10, 'activation': 'softmax'})
    ]

    arch = ArchitectureChromosome(layers, learning_rate=0.001, batch_size=32)

    assert len(arch.layers) == 4, f"Expected 4 layers, got {len(arch.layers)}"
    assert arch.learning_rate == 0.001, "Learning rate mismatch"
    assert arch.batch_size == 32, "Batch size mismatch"

    # Test serialization
    arch_dict = arch.to_dict()
    restored_arch = ArchitectureChromosome.from_dict(arch_dict)
    assert len(restored_arch.layers) == len(arch.layers), "Deserialization layer count mismatch"

    print("✓ ArchitectureChromosome test passed!")


def test_architecture_encoding_decoding():
    """Test architecture encoding and decoding."""
    print("Testing architecture encoding/decoding...")

    # Create simple architecture
    layers = [
        LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('maxpool2d', {'pool_size': 2}),
        LayerGene('dropout', {'rate': 0.3}),
        LayerGene('flatten', {}),
        LayerGene('dense', {'units': 64, 'activation': 'relu'}),
        LayerGene('dense', {'units': 10, 'activation': 'softmax'})
    ]

    arch = ArchitectureChromosome(layers, learning_rate=0.001, batch_size=32)

    # Encode to chromosome
    chromosome = encode_architecture_to_chromosome(arch, max_layers=10)

    # Should be a numpy array
    assert isinstance(chromosome, np.ndarray), "Chromosome should be numpy array"
    assert len(chromosome) == 10 * 7 + 3, f"Expected length {10*7+3}, got {len(chromosome)}"

    # All values should be in [0, 1]
    assert np.all(chromosome >= 0) and np.all(chromosome <= 1), "Chromosome values out of bounds"

    print("✓ Architecture encoding/decoding test passed!")


def test_chromosome_decoding():
    """Test decoding random chromosome to architecture."""
    print("Testing chromosome decoding...")

    np.random.seed(42)
    chromosome = np.random.rand(10 * 7 + 3)

    # Decode
    arch = decode_chromosome_to_architecture(chromosome, max_layers=10, num_classes=10)

    # Should have at least a flatten and output dense layer
    assert len(arch.layers) > 0, "Architecture should have layers"

    # Last layer should be dense with 10 units (output layer)
    assert arch.layers[-1].layer_type == 'dense', "Last layer should be dense"
    assert arch.layers[-1].params['units'] == 10, "Output layer should have 10 units"

    # Should have flatten layer
    has_flatten = any(l.layer_type == 'flatten' for l in arch.layers)
    assert has_flatten, "Architecture should have flatten layer"

    # Learning rate should be in reasonable range
    assert 1e-4 <= arch.learning_rate <= 1e-2, f"Learning rate out of range: {arch.learning_rate}"

    # Batch size should be power of 2
    batch_size = arch.batch_size
    assert batch_size > 0 and (batch_size & (batch_size - 1)) == 0, "Batch size should be power of 2"

    print("✓ Chromosome decoding test passed!")


def test_architecture_complexity():
    """Test architecture complexity calculation."""
    print("Testing architecture complexity...")

    layers = [
        LayerGene('conv2d', {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('conv2d', {'filters': 64, 'kernel_size': 3, 'activation': 'relu'}),
        LayerGene('dense', {'units': 128, 'activation': 'relu'}),
        LayerGene('dense', {'units': 10, 'activation': 'softmax'})
    ]

    arch = ArchitectureChromosome(layers)
    complexity = arch.count_parameters()

    # Should be positive
    assert complexity > 0, "Complexity should be positive"

    # Approximate calculation: 32*9 + 64*9 + 128 + 10 = 288 + 576 + 128 + 10 = 1002
    expected_approx = 32*9 + 64*9 + 128 + 10
    assert complexity == expected_approx, f"Expected ~{expected_approx}, got {complexity}"

    print("✓ Architecture complexity test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running GA Tutorial 4 - Real-World Applications Tests")
    print("="*70 + "\n")

    try:
        test_portfolio_decoding()
        test_schedule_evaluation()
        test_feature_mask()
        test_synthetic_data_generation()
        test_portfolio_data_generation()

        # NAS tests
        test_layer_gene_creation()
        test_architecture_chromosome()
        test_architecture_encoding_decoding()
        test_chromosome_decoding()
        test_architecture_complexity()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED! (10 tests)")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
