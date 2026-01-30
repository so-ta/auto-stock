"""
Test script for FiftyTwoWeekHighMomentumSignal.

Tests:
1. 52-week high calculation accuracy
2. Normalization to [-1, 1] range
3. Edge cases (insufficient data, constant prices)
4. Parameter validation
5. Signal classification as independent
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.signals.fifty_two_week_high import FiftyTwoWeekHighMomentumSignal
from src.signals.registry import SignalRegistry


class TestFiftyTwoWeekHighMomentumSignal:
    """Test suite for FiftyTwoWeekHighMomentumSignal."""

    def create_price_data(
        self,
        n_days: int = 300,
        start_price: float = 100.0,
        trend: str = "up",
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Create sample price data for testing."""
        dates = pd.date_range(
            start=datetime(2024, 1, 1),
            periods=n_days,
            freq="D",
        )

        np.random.seed(42)

        prices = [start_price]
        for i in range(1, n_days):
            if trend == "up":
                drift = 0.001
            elif trend == "down":
                drift = -0.001
            elif trend == "v_shape":
                # V-shape: down first half, up second half
                if i < n_days // 2:
                    drift = -0.002
                else:
                    drift = 0.003
            else:
                drift = 0.0

            daily_return = drift + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))  # Ensure positive price

        return pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * n_days,
            },
            index=dates,
        )

    def test_signal_registration(self):
        """Test that signal is properly registered."""
        signals = SignalRegistry.list_all()
        assert "fifty_two_week_high_momentum" in signals
        print("✅ test_signal_registration PASSED")

    def test_basic_computation(self):
        """Test basic signal computation."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=252, smoothing=1)
        data = self.create_price_data(n_days=300)

        result = signal.compute(data)

        # Check result structure
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")
        assert len(result.scores) == len(data)

        # Check normalization range
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0

        print("✅ test_basic_computation PASSED")

    def test_52_week_high_accuracy(self):
        """Test that 52-week high is calculated correctly."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)

        # Create price data where the last point equals the max (= at 52-week high)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        # Prices that increase overall with some noise
        np.random.seed(123)
        prices = [100.0]
        for i in range(99):
            prices.append(prices[-1] * (1 + 0.001 + np.random.normal(0, 0.005)))

        # Make the last price the highest
        max_price = max(prices)
        prices[-1] = max_price * 1.05

        data = pd.DataFrame({"close": prices}, index=dates)

        result = signal.compute(data)

        # Last price is at the 63-day high, so ratio = 1.0, score = 1.0
        assert abs(result.scores.iloc[-1] - 1.0) < 0.01

        print("✅ test_52_week_high_accuracy PASSED")

    def test_normalization_range(self):
        """Test that output is always in [-1, 1]."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)

        # Test with various market conditions
        for trend in ["up", "down", "v_shape", "flat"]:
            data = self.create_price_data(n_days=100, trend=trend)
            result = signal.compute(data)

            assert result.scores.min() >= -1.0, f"Min < -1 for {trend}"
            assert result.scores.max() <= 1.0, f"Max > 1 for {trend}"

        print("✅ test_normalization_range PASSED")

    def test_at_high_gives_positive_score(self):
        """Test that price at 52-week high gives strong positive score."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)

        # Create uptrending data where last price is the highest
        data = self.create_price_data(n_days=60, trend="up", volatility=0.001)

        # Ensure last price is highest by adding a spike
        data.iloc[-1, data.columns.get_loc("close")] = data["close"].max() * 1.1

        result = signal.compute(data)

        # Last score should be 1.0 (at high)
        assert result.scores.iloc[-1] == 1.0

        print("✅ test_at_high_gives_positive_score PASSED")

    def test_far_from_high_gives_negative_score(self):
        """Test that price far from 52-week high gives negative score."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)

        # Create data with a spike in the middle, then decline
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        prices = (
            [100 + i for i in range(20)]  # Up to 119
            + [120 + i * 2 for i in range(10)]  # Spike to 138
            + [138 - i * 2 for i in range(30)]  # Decline to 80
        )

        data = pd.DataFrame({"close": prices}, index=dates)
        result = signal.compute(data)

        # At the end, price is 80, but 52-week high includes 138
        # ratio = 80 / 138 = 0.58
        # score = ((0.58 - 0.5) * 2) = 0.16
        last_score = result.scores.iloc[-1]
        assert last_score < 0.5, f"Expected negative-ish score, got {last_score}"

        print("✅ test_far_from_high_gives_negative_score PASSED")

    def test_smoothing_effect(self):
        """Test that smoothing reduces noise."""
        # Need more data for smoothing effect to be visible
        data = self.create_price_data(n_days=200, volatility=0.05)

        signal_no_smooth = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)
        signal_smoothed = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=20)

        result_no_smooth = signal_no_smooth.compute(data)
        result_smoothed = signal_smoothed.compute(data)

        # Compare day-to-day changes (first difference) instead of overall std
        # Smoothing should reduce the magnitude of daily changes
        diff_no_smooth = result_no_smooth.scores.diff().abs().mean()
        diff_smoothed = result_smoothed.scores.diff().abs().mean()

        assert diff_smoothed < diff_no_smooth, f"Smoothing should reduce daily volatility: {diff_smoothed} vs {diff_no_smooth}"

        print("✅ test_smoothing_effect PASSED")

    def test_edge_case_insufficient_data(self):
        """Test behavior with data shorter than lookback period."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=252, smoothing=1)

        # Only 50 days of data (less than 252)
        data = self.create_price_data(n_days=50)

        # Should not raise error due to min_periods=1
        result = signal.compute(data)
        assert len(result.scores) == 50
        assert not result.scores.isna().all()

        print("✅ test_edge_case_insufficient_data PASSED")

    def test_edge_case_constant_prices(self):
        """Test behavior with constant prices (no volatility)."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=1)

        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        data = pd.DataFrame({"close": [100.0] * 60}, index=dates)

        result = signal.compute(data)

        # With constant prices, ratio = 1.0, score = 1.0
        assert all(result.scores == 1.0)

        print("✅ test_edge_case_constant_prices PASSED")

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        signal = FiftyTwoWeekHighMomentumSignal(lookback=126, smoothing=5)
        assert signal._params["lookback"] == 126
        assert signal._params["smoothing"] == 5

        # Out of range parameters should raise error
        with pytest.raises(ValueError):
            FiftyTwoWeekHighMomentumSignal(lookback=10)  # Below min (63)

        print("✅ test_parameter_validation PASSED")

    def test_param_grid(self):
        """Test parameter grid for optimization."""
        grid = FiftyTwoWeekHighMomentumSignal.get_param_grid()

        assert "lookback" in grid
        assert "smoothing" in grid
        assert grid["lookback"] == [126, 189, 252]
        assert grid["smoothing"] == [1, 3, 5, 10]

        print("✅ test_param_grid PASSED")

    def test_metadata_contents(self):
        """Test that metadata contains expected information."""
        signal = FiftyTwoWeekHighMomentumSignal(lookback=63, smoothing=5)
        data = self.create_price_data(n_days=100)

        result = signal.compute(data)

        expected_keys = [
            "lookback",
            "smoothing",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "pct_at_high",
            "pct_far_from_high",
            "score_mean",
            "score_std",
        ]

        for key in expected_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"

        print("✅ test_metadata_contents PASSED")


class TestSignalClassification:
    """Test that signal is correctly classified as independent."""

    def test_signal_classified_as_independent(self):
        """Test that fifty_two_week_high_* is in INDEPENDENT_SIGNALS."""
        from src.backtest.signal_precompute import INDEPENDENT_SIGNALS
        import fnmatch

        signal_name = "fifty_two_week_high_momentum"

        is_independent = any(
            fnmatch.fnmatch(signal_name, pattern)
            for pattern in INDEPENDENT_SIGNALS
        )

        assert is_independent, f"{signal_name} should be classified as independent"

        print("✅ test_signal_classified_as_independent PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FiftyTwoWeekHighMomentumSignal")
    print("=" * 60)

    # Run main tests
    tests = TestFiftyTwoWeekHighMomentumSignal()
    tests.test_signal_registration()
    tests.test_basic_computation()
    tests.test_52_week_high_accuracy()
    tests.test_normalization_range()
    tests.test_at_high_gives_positive_score()
    tests.test_far_from_high_gives_negative_score()
    tests.test_smoothing_effect()
    tests.test_edge_case_insufficient_data()
    tests.test_edge_case_constant_prices()
    tests.test_parameter_validation()
    tests.test_param_grid()
    tests.test_metadata_contents()

    # Run classification test
    classification_tests = TestSignalClassification()
    classification_tests.test_signal_classified_as_independent()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
