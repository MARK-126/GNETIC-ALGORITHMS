"""
Public Test Cases for GA Tutorial 4 - Real-World Applications
Simple validation tests for application utilities.
"""

import numpy as np
from ga_utils_applications import *


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

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
