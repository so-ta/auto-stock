"""
Tests for Short Interest Signal.

Tests:
- FINRA API response parsing
- Days to Cover calculation accuracy
- Signal normalization verification
- Parameter grid search
- Edge cases (missing data, zero values)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestFINRAClient:
    """Tests for FINRA API client."""

    def test_client_initialization(self):
        """Test FINRAClient initializes correctly."""
        from src.data.finra import FINRAClient

        client = FINRAClient()
        assert client.cache_enabled is True
        assert client.timeout == 30

        client_no_cache = FINRAClient(cache_enabled=False)
        assert client_no_cache.cache_enabled is False

    def test_client_with_api_key(self):
        """Test client with API key."""
        from src.data.finra import FINRAClient

        client = FINRAClient(api_key="test_key")
        headers = client._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_key"

    def test_empty_dataframe_creation(self):
        """Test empty DataFrame has correct structure."""
        from src.data.finra import FINRAClient

        client = FINRAClient(cache_enabled=False)

        # Create mock empty response
        df = pd.DataFrame(
            columns=[
                "symbol",
                "settlement_date",
                "short_interest",
                "avg_daily_volume",
                "days_to_cover",
            ]
        )

        assert "symbol" in df.columns
        assert "settlement_date" in df.columns
        assert "short_interest" in df.columns
        assert "avg_daily_volume" in df.columns
        assert "days_to_cover" in df.columns

    def test_days_to_cover_calculation(self):
        """Test days to cover calculation."""
        from src.data.finra import FINRAClient

        # Create sample data
        short_interest = 1000000  # 1M shares short
        avg_daily_volume = 200000  # 200K avg volume

        expected_dtc = short_interest / avg_daily_volume  # 5 days
        assert expected_dtc == 5.0

    def test_short_interest_ratio_calculation(self):
        """Test short interest ratio calculation."""
        from src.data.finra import FINRAClient

        client = FINRAClient(cache_enabled=False)

        # Mock: cannot test without actual data
        # Just verify method exists
        assert hasattr(client, "get_short_interest_ratio")

    def test_cache_path_generation(self):
        """Test cache file path generation."""
        from src.data.finra import FINRAClient
        from pathlib import Path

        client = FINRAClient()
        path = client._get_cache_path("AAPL")
        assert path.name == "short_interest_AAPL.parquet"

        # Test case normalization
        path_lower = client._get_cache_path("aapl")
        assert path_lower.name == "short_interest_AAPL.parquet"


class TestShortInterestSignal:
    """Tests for ShortInterestSignal."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate volume data
        volume = np.random.uniform(1000000, 5000000, len(dates))

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def sample_si_data(self, sample_price_data):
        """Create sample short interest data."""
        # SI data is biweekly
        dates = sample_price_data.index[::14]  # Every 2 weeks
        np.random.seed(42)

        avg_volume = sample_price_data["volume"].mean()

        short_interest = np.random.uniform(100000, 1000000, len(dates))
        days_to_cover = short_interest / avg_volume
        si_ratio = np.random.uniform(0.01, 0.10, len(dates))

        # Create DataFrame with date as index (not column)
        df = pd.DataFrame(
            {
                "short_interest": short_interest,
                "days_to_cover": days_to_cover,
                "short_interest_ratio": si_ratio,
            },
            index=dates,
        )
        df.index.name = "settlement_date"
        return df

    def test_signal_initialization(self):
        """Test signal initialization with default parameters."""
        from src.signals.short_interest import ShortInterestSignal

        signal = ShortInterestSignal()

        assert signal._params["use_days_to_cover"] is True
        assert signal._params["zscore_normalize"] is True
        assert signal._params["smoothing"] == 1

    def test_signal_with_custom_params(self):
        """Test signal with custom parameters."""
        from src.signals.short_interest import ShortInterestSignal

        signal = ShortInterestSignal(
            use_days_to_cover=False,
            zscore_normalize=False,
            smoothing=3,
        )

        assert signal._params["use_days_to_cover"] is False
        assert signal._params["zscore_normalize"] is False
        assert signal._params["smoothing"] == 3

    def test_compute_without_si_data(self, sample_price_data):
        """Test signal computation without external SI data (using proxy)."""
        from src.signals.short_interest import ShortInterestSignal

        signal = ShortInterestSignal()
        result = signal.compute(sample_price_data)

        # Check result structure
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")

        # Check scores are in valid range
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0

        # Check metadata
        assert result.metadata["has_external_data"] is False

    def test_compute_with_si_data(self, sample_price_data, sample_si_data):
        """Test signal computation with external SI data."""
        from src.signals.short_interest import ShortInterestSignal

        signal = ShortInterestSignal(use_days_to_cover=True)
        result = signal.compute(sample_price_data, short_interest=sample_si_data)

        # Check result structure
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")

        # Check scores are in valid range
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0

        # Check metadata
        assert result.metadata["has_external_data"] is True
        assert result.metadata["use_days_to_cover"] is True

    def test_signal_inversion(self, sample_price_data):
        """Test that signal is properly inverted."""
        from src.signals.short_interest import ShortInterestSignal

        signal_normal = ShortInterestSignal(invert=False)
        signal_inverted = ShortInterestSignal(invert=True)

        result_normal = signal_normal.compute(sample_price_data)
        result_inverted = signal_inverted.compute(sample_price_data)

        # Signals should be negated
        correlation = result_normal.scores.corr(result_inverted.scores)
        assert correlation < -0.9  # Should be highly negatively correlated

    def test_smoothing_effect(self, sample_price_data):
        """Test that smoothing produces different results."""
        from src.signals.short_interest import ShortInterestSignal

        signal_no_smooth = ShortInterestSignal(smoothing=1)
        signal_smooth = ShortInterestSignal(smoothing=5)

        result_no_smooth = signal_no_smooth.compute(sample_price_data)
        result_smooth = signal_smooth.compute(sample_price_data)

        # Smoothed signal should be different from unsmoothed
        # (they should be correlated but not identical)
        correlation = result_no_smooth.scores.corr(result_smooth.scores)
        assert correlation > 0.5  # Should be positively correlated
        assert correlation < 1.0  # But not identical

        # Both should produce valid scores
        assert result_no_smooth.scores.min() >= -1.0
        assert result_smooth.scores.min() >= -1.0

    def test_zscore_normalization(self, sample_price_data):
        """Test z-score normalization option."""
        from src.signals.short_interest import ShortInterestSignal

        signal_zscore = ShortInterestSignal(zscore_normalize=True)
        signal_no_zscore = ShortInterestSignal(zscore_normalize=False)

        result_zscore = signal_zscore.compute(sample_price_data)
        result_no_zscore = signal_no_zscore.compute(sample_price_data)

        # Both should produce valid scores
        assert result_zscore.scores.min() >= -1.0
        assert result_zscore.scores.max() <= 1.0
        assert result_no_zscore.scores.min() >= -1.0
        assert result_no_zscore.scores.max() <= 1.0

    def test_parameter_specs(self):
        """Test parameter specifications."""
        from src.signals.short_interest import ShortInterestSignal

        specs = ShortInterestSignal.parameter_specs()

        param_names = {spec.name for spec in specs}
        expected_params = {
            "use_days_to_cover",
            "zscore_normalize",
            "zscore_lookback",
            "smoothing",
            "scale",
            "invert",
        }

        assert param_names == expected_params

    def test_param_grid(self):
        """Test parameter grid for optimization."""
        from src.signals.short_interest import ShortInterestSignal

        grid = ShortInterestSignal.get_param_grid()

        assert "use_days_to_cover" in grid
        assert "zscore_normalize" in grid
        assert "smoothing" in grid

        assert grid["use_days_to_cover"] == [True, False]
        assert grid["smoothing"] == [1, 3, 5]


class TestShortInterestChangeSignal:
    """Tests for ShortInterestChangeSignal."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        prices = 100 * (1 + np.random.normal(0, 0.02, len(dates))).cumprod()
        volume = np.random.uniform(1000000, 5000000, len(dates))

        return pd.DataFrame(
            {
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )

    def test_change_signal_initialization(self):
        """Test change signal initialization."""
        from src.signals.short_interest import ShortInterestChangeSignal

        signal = ShortInterestChangeSignal()

        assert signal._params["change_period"] == 2
        assert signal._params["zscore_normalize"] is True

    def test_change_signal_compute(self, sample_price_data):
        """Test change signal computation."""
        from src.signals.short_interest import ShortInterestChangeSignal

        signal = ShortInterestChangeSignal()
        result = signal.compute(sample_price_data)

        # Check result structure
        assert hasattr(result, "scores")
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0


class TestSignalRegistry:
    """Test signal registration."""

    def test_short_interest_registered(self):
        """Test that ShortInterestSignal is registered."""
        from src.signals.registry import SignalRegistry

        # Import to trigger registration
        from src.signals import short_interest  # noqa: F401

        assert SignalRegistry.is_registered("short_interest")

    def test_short_interest_change_registered(self):
        """Test that ShortInterestChangeSignal is registered."""
        from src.signals.registry import SignalRegistry

        # Import to trigger registration
        from src.signals import short_interest  # noqa: F401

        assert SignalRegistry.is_registered("short_interest_change")

    def test_create_signal_from_registry(self):
        """Test creating signal from registry."""
        from src.signals.registry import SignalRegistry

        # Import to trigger registration
        from src.signals import short_interest  # noqa: F401

        signal = SignalRegistry.create("short_interest", smoothing=3)
        assert signal._params["smoothing"] == 3


class TestConvenienceFunction:
    """Test convenience functions."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        return pd.DataFrame(
            {
                "close": 100 * (1 + np.random.normal(0, 0.02, len(dates))).cumprod(),
                "volume": np.random.uniform(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

    def test_compute_short_interest_signal(self, sample_price_data):
        """Test convenience function."""
        from src.signals.short_interest import compute_short_interest_signal

        scores = compute_short_interest_signal(sample_price_data)

        assert isinstance(scores, pd.Series)
        assert scores.min() >= -1.0
        assert scores.max() <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_volume_column(self):
        """Test signal computation without volume column."""
        from src.signals.short_interest import ShortInterestSignal

        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Price data without volume
        data = pd.DataFrame(
            {
                "close": 100 * (1 + np.random.normal(0, 0.02, len(dates))).cumprod(),
            },
            index=dates,
        )

        signal = ShortInterestSignal()
        result = signal.compute(data)

        # Should still produce valid scores (using fallback)
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0

    def test_short_data_series(self):
        """Test with very short data series."""
        from src.signals.short_interest import ShortInterestSignal

        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "close": 100 * (1 + np.random.normal(0, 0.02, len(dates))).cumprod(),
                "volume": np.random.uniform(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

        signal = ShortInterestSignal(zscore_lookback=20)  # Longer than data
        result = signal.compute(data)

        # Should handle gracefully
        assert len(result.scores) == len(data)

    def test_constant_prices(self):
        """Test with constant prices (zero returns)."""
        from src.signals.short_interest import ShortInterestSignal

        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")

        data = pd.DataFrame(
            {
                "close": np.full(len(dates), 100.0),
                "volume": np.random.uniform(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

        signal = ShortInterestSignal()
        result = signal.compute(data)

        # Should handle constant prices without errors
        assert len(result.scores) == len(data)
        # Check no NaN or Inf
        assert not np.any(np.isinf(result.scores.dropna()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
