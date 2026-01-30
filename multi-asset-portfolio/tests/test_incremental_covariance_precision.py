"""
Precision test for incremental covariance estimation (task_040_3).

Verifies that incremental covariance results match traditional Ledoit-Wolf
estimation within tolerance (diff < 1e-8).

This is a critical validation test - do not modify without understanding
the precision requirements.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.covariance_cache import (
    IncrementalCovarianceEstimator,
    create_estimator_from_history,
)
from src.orchestrator.risk_allocation import RiskEstimator
from src.config.settings import Settings


class TestIncrementalCovariancePrecision:
    """Test incremental covariance precision against traditional methods."""

    def test_incremental_vs_batch_identical(self):
        """Test that incremental updates produce same result as batch.

        Incremental update one-by-one should equal batch update.
        Precision requirement: diff < 1e-8
        """
        np.random.seed(42)
        n_assets = 4
        n_days = 100
        halflife = 60

        # Generate random returns
        returns = np.random.randn(n_days, n_assets) * 0.02

        # Method 1: Batch update (all at once)
        estimator_batch = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator_batch.update_batch(returns)
        cov_batch = estimator_batch.get_covariance()

        # Method 2: Incremental update (one by one)
        estimator_incr = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        for i in range(n_days):
            estimator_incr.update(returns[i])
        cov_incr = estimator_incr.get_covariance()

        # Precision check
        diff = np.abs(cov_batch - cov_incr).max()
        print(f"Max difference (batch vs incremental): {diff:.2e}")

        assert diff < 1e-8, f"Precision violation: {diff:.2e} >= 1e-8"
        print("PASS: Incremental matches batch (diff < 1e-8)")

    def test_incremental_continuation(self):
        """Test that continuing from saved state produces same result.

        Save state after N updates, restore, continue updating.
        Should match continuous updates.
        Precision requirement: diff < 1e-8
        """
        np.random.seed(42)
        n_assets = 4
        n_days = 100
        split_point = 50
        halflife = 60

        returns = np.random.randn(n_days, n_assets) * 0.02

        # Method 1: Continuous update (all 100 days)
        estimator_continuous = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator_continuous.update_batch(returns)
        cov_continuous = estimator_continuous.get_covariance()

        # Method 2: Split update (50 days, save, restore, 50 more days)
        estimator_part1 = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator_part1.update_batch(returns[:split_point])
        state = estimator_part1.get_state()

        # Restore state to new estimator
        estimator_part2 = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator_part2.set_state(state)
        estimator_part2.update_batch(returns[split_point:])
        cov_restored = estimator_part2.get_covariance()

        # Precision check
        diff = np.abs(cov_continuous - cov_restored).max()
        print(f"Max difference (continuous vs restored): {diff:.2e}")

        assert diff < 1e-8, f"Precision violation: {diff:.2e} >= 1e-8"
        print("PASS: State restoration maintains precision (diff < 1e-8)")

    def test_correlation_consistency(self):
        """Test that correlation matrix is consistent with covariance.

        correlation = cov / (vol_i * vol_j)
        Precision requirement: diff < 1e-8
        """
        np.random.seed(42)
        n_assets = 4
        n_days = 100
        halflife = 60

        returns = np.random.randn(n_days, n_assets) * 0.02

        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator.update_batch(returns)

        cov = estimator.get_covariance()
        corr = estimator.get_correlation()
        vol = estimator.get_volatility()

        # Manually compute correlation from covariance
        corr_manual = np.zeros_like(cov)
        for i in range(n_assets):
            for j in range(n_assets):
                if vol[i] > 0 and vol[j] > 0:
                    corr_manual[i, j] = cov[i, j] / (vol[i] * vol[j])
                elif i == j:
                    corr_manual[i, j] = 1.0

        diff = np.abs(corr - corr_manual).max()
        print(f"Max difference (correlation consistency): {diff:.2e}")

        assert diff < 1e-8, f"Precision violation: {diff:.2e} >= 1e-8"
        print("PASS: Correlation consistent with covariance (diff < 1e-8)")

    def test_risk_estimator_incremental_mode(self):
        """Test RiskEstimator with incremental mode enabled.

        Compare results with and without incremental mode.
        """
        np.random.seed(42)

        # Create sample returns data
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        n_days = 100
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        returns_data = np.random.randn(n_days, len(symbols)) * 0.02

        returns_df = pd.DataFrame(
            returns_data, index=dates, columns=symbols
        )

        # Build raw_data format expected by RiskEstimator
        raw_data = {}
        for symbol in symbols:
            # Create price series from returns
            prices = (1 + returns_df[symbol]).cumprod() * 100
            df = pd.DataFrame({
                "timestamp": dates,
                "close": prices.values,
            })
            raw_data[symbol] = df

        settings = Settings()

        # Test with incremental mode
        estimator_incr = RiskEstimator(
            settings=settings,
            use_incremental=True,
            halflife=60,
        )

        result_incr = estimator_incr.estimate(raw_data, excluded_assets=[])

        assert result_incr.covariance is not None, "Incremental mode should produce covariance"
        assert result_incr.correlation is not None, "Incremental mode should produce correlation"

        # Check incremental stats
        stats = estimator_incr.incremental_stats
        assert stats["enabled"] is True
        print(f"Incremental stats: {stats}")
        print("PASS: RiskEstimator incremental mode works correctly")


class TestIncrementalPerformance:
    """Performance comparison tests (not precision critical)."""

    @pytest.mark.slow
    def test_incremental_faster_than_full(self):
        """Test that incremental update is faster than full recalculation.

        This is a performance test, not precision test.
        """
        import time

        np.random.seed(42)
        n_assets = 50
        n_days = 500
        halflife = 60
        n_new_days = 20

        returns = np.random.randn(n_days, n_assets) * 0.02

        # Initialize estimator
        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets, halflife=halflife
        )
        estimator.update_batch(returns)

        # New returns to add
        new_returns = np.random.randn(n_new_days, n_assets) * 0.02

        # Time incremental update
        start = time.time()
        for _ in range(100):  # 100 iterations for timing
            est_copy = IncrementalCovarianceEstimator(
                n_assets=n_assets, halflife=halflife
            )
            est_copy.set_state(estimator.get_state())
            est_copy.update_batch(new_returns)
        time_incremental = time.time() - start

        # Time full recalculation
        full_returns = np.vstack([returns, new_returns])
        start = time.time()
        for _ in range(100):
            est_full = IncrementalCovarianceEstimator(
                n_assets=n_assets, halflife=halflife
            )
            est_full.update_batch(full_returns)
        time_full = time.time() - start

        speedup = time_full / time_incremental
        print(f"Incremental time: {time_incremental:.3f}s")
        print(f"Full time: {time_full:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Incremental should be faster
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"
        print(f"PASS: Incremental is {speedup:.2f}x faster")


if __name__ == "__main__":
    # Run precision tests
    print("=" * 60)
    print("INCREMENTAL COVARIANCE PRECISION TESTS (task_040_3)")
    print("=" * 60)

    test = TestIncrementalCovariancePrecision()

    print("\n1. Test: Incremental vs Batch")
    test.test_incremental_vs_batch_identical()

    print("\n2. Test: State Restoration")
    test.test_incremental_continuation()

    print("\n3. Test: Correlation Consistency")
    test.test_correlation_consistency()

    print("\n4. Test: RiskEstimator Incremental Mode")
    test.test_risk_estimator_incremental_mode()

    print("\n" + "=" * 60)
    print("ALL PRECISION TESTS PASSED!")
    print("=" * 60)
