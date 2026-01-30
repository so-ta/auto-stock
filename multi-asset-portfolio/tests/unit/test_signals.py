"""Unit tests for signal modules."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


class TestParameterSpec:
    """Tests for ParameterSpec class."""

    def test_parameter_spec_validation(self):
        """Test parameter validation."""
        from src.signals.base import ParameterSpec

        spec = ParameterSpec(
            name="lookback",
            default=20,
            searchable=True,
            min_value=5,
            max_value=100,
        )

        assert spec.validate(20) is True
        assert spec.validate(5) is True
        assert spec.validate(100) is True
        assert spec.validate(4) is False
        assert spec.validate(101) is False

    def test_parameter_spec_search_range(self):
        """Test generating search range."""
        from src.signals.base import ParameterSpec

        spec = ParameterSpec(
            name="window",
            default=10,
            searchable=True,
            min_value=5,
            max_value=25,
            step=5,
        )

        search_range = spec.search_range()
        assert search_range is not None
        assert 5 in search_range
        assert 10 in search_range
        assert 25 in search_range

    def test_non_searchable_param_no_range(self):
        """Test that non-searchable params return None for search range."""
        from src.signals.base import ParameterSpec

        spec = ParameterSpec(
            name="fixed_param",
            default=10,
            searchable=False,
        )

        assert spec.search_range() is None


class TestSignalResult:
    """Tests for SignalResult class."""

    def test_signal_result_is_valid(self):
        """Test SignalResult validity check."""
        from src.signals.base import SignalResult

        # Valid result
        valid_scores = pd.Series([0.5, -0.3, 0.2, 0.0])
        result = SignalResult(scores=valid_scores)
        assert result.is_valid is True

        # Invalid result (all NaN)
        invalid_scores = pd.Series([np.nan, np.nan, np.nan])
        result_invalid = SignalResult(scores=invalid_scores)
        assert result_invalid.is_valid is False

    def test_signal_result_to_dict(self):
        """Test SignalResult serialization."""
        from src.signals.base import SignalResult

        scores = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
        metadata = {"strategy": "momentum"}
        result = SignalResult(scores=scores, metadata=metadata)

        result_dict = result.to_dict()
        assert "scores" in result_dict
        assert "metadata" in result_dict
        assert result_dict["metadata"]["strategy"] == "momentum"


class TestSignalBase:
    """Tests for Signal base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV DataFrame for testing."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))

        return pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)

    def test_normalize_tanh(self):
        """Test tanh normalization."""
        from src.signals.base import Signal

        values = pd.Series([0.5, 1.0, 2.0, -1.0, -2.0])
        normalized = Signal.normalize_tanh(values, scale=1.0)

        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0
        assert normalized[0] > 0  # positive input -> positive output
        assert normalized[3] < 0  # negative input -> negative output

    def test_normalize_zscore_tanh(self):
        """Test z-score + tanh normalization."""
        from src.signals.base import Signal

        np.random.seed(42)
        values = pd.Series(np.random.normal(0, 1, 100))
        normalized = Signal.normalize_zscore_tanh(values, lookback=20, scale=0.5)

        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0

    def test_normalize_minmax_scaled(self):
        """Test min-max normalization."""
        from src.signals.base import Signal

        values = pd.Series([10, 20, 30, 40, 50])
        normalized = Signal.normalize_minmax_scaled(values, lookback=5)

        # After warmup, should be in [-1, 1]
        assert normalized.iloc[-1] == pytest.approx(1.0)

    def test_unknown_parameter_raises(self, sample_data):
        """Test that unknown parameters raise ValueError."""
        from src.signals.momentum import MomentumSignal

        with pytest.raises(ValueError, match="Unknown parameters"):
            MomentumSignal(unknown_param=123)

    def test_parameter_out_of_range_raises(self, sample_data):
        """Test that out-of-range parameters raise ValueError."""
        from src.signals.momentum import MomentumSignal

        with pytest.raises(ValueError, match="out of range"):
            MomentumSignal(lookback=1)  # min is typically > 1


class TestMomentumSignal:
    """Tests for MomentumSignal."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with clear momentum."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        # Create uptrend data
        prices = 100 + np.arange(n) * 0.5 + np.random.normal(0, 1, n)

        return pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)

    def test_momentum_signal_positive_trend(self, sample_data):
        """Test momentum signal on uptrend data."""
        from src.signals.momentum import MomentumSignal

        signal = MomentumSignal(lookback=20)
        result = signal.compute(sample_data)

        assert result.is_valid
        # In uptrend, later scores should be positive
        assert result.scores.iloc[-1] > 0

    def test_momentum_signal_parameters(self):
        """Test MomentumSignal parameter specs."""
        from src.signals.momentum import MomentumSignal

        specs = MomentumSignal.parameter_specs()
        param_names = [s.name for s in specs]

        assert "lookback" in param_names

    def test_momentum_signal_callable(self, sample_data):
        """Test that signal can be called directly."""
        from src.signals.momentum import MomentumSignal

        signal = MomentumSignal()
        result = signal(sample_data)  # Uses __call__

        assert result.is_valid


class TestMeanReversionSignal:
    """Tests for MeanReversionSignal (Bollinger Bands)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)

        # Mean-reverting data
        prices = 100 + np.cumsum(np.random.normal(0, 1, n))
        prices = prices - (prices - 100) * 0.1  # Pull towards mean

        return pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)

    def test_mean_reversion_signal(self, sample_data):
        """Test mean reversion signal computation."""
        from src.signals.mean_reversion import MeanReversionSignal

        signal = MeanReversionSignal(lookback=20, num_std=2.0)
        result = signal.compute(sample_data)

        assert result.is_valid
        assert result.scores.max() <= 1.0
        assert result.scores.min() >= -1.0


class TestVolatilitySignal:
    """Tests for VolatilitySignal."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with varying volatility."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)

        # First half: low volatility, second half: high volatility
        vol = np.concatenate([
            np.random.normal(0, 0.01, n // 2),
            np.random.normal(0, 0.03, n // 2),
        ])
        prices = 100 * np.exp(np.cumsum(vol))

        return pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)

    def test_volatility_signal(self, sample_data):
        """Test volatility signal computation."""
        from src.signals.volatility import VolatilitySignal

        signal = VolatilitySignal(lookback=20)
        result = signal.compute(sample_data)

        assert result.is_valid
        # High volatility period should have different scores than low vol period
        assert result.scores.iloc[30] != result.scores.iloc[80]


class TestSignalRegistry:
    """Tests for SignalRegistry."""

    def test_register_and_get_signal(self):
        """Test registering and retrieving signals."""
        from src.signals.base import Signal
        from src.signals.registry import SignalRegistry

        registry = SignalRegistry()

        # Register a mock signal class
        class MockSignal(Signal):
            @classmethod
            def parameter_specs(cls):
                return []

            def compute(self, data):
                from src.signals.base import SignalResult
                return SignalResult(scores=pd.Series([0.0]))

        registry.register("mock", MockSignal)

        assert "mock" in registry.list_signals()

        signal_cls = registry.get("mock")
        assert signal_cls == MockSignal

    def test_get_unknown_signal_raises(self):
        """Test that getting unknown signal raises KeyError."""
        from src.signals.registry import SignalRegistry

        registry = SignalRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")
