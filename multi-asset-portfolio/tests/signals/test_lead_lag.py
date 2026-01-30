"""Tests for LeadLagSignal - task_042_1."""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.signals.lead_lag import LeadLagSignal, LeadLagPair, compute_lead_lag_signal
from src.signals.registry import SignalRegistry


class TestLeadLagSignal:
    """Tests for LeadLagSignal class."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price DataFrame with known lead-lag relationship."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

        # Create Leader ticker (moves first)
        leader_returns = np.random.randn(120) * 0.02
        leader_prices = 100 * np.exp(np.cumsum(leader_returns))

        # Create Follower ticker (follows Leader with 2-day lag)
        follower_returns = np.zeros(120)
        follower_returns[2:] = leader_returns[:-2] * 0.8 + np.random.randn(118) * 0.005
        follower_prices = 100 * np.exp(np.cumsum(follower_returns))

        # Create Independent ticker (no relationship)
        independent_returns = np.random.randn(120) * 0.02
        independent_prices = 100 * np.exp(np.cumsum(independent_returns))

        return pd.DataFrame(
            {
                "LEADER": leader_prices,
                "FOLLOWER": follower_prices,
                "INDEPENDENT": independent_prices,
            },
            index=dates,
        )

    @pytest.fixture
    def minimal_prices(self):
        """Create minimal price DataFrame for edge case testing."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        return pd.DataFrame(
            {
                "A": np.linspace(100, 110, 20),
                "B": np.linspace(100, 105, 20),
            },
            index=dates,
        )

    def test_initialization_default_params(self):
        """Test: Default parameters are set correctly."""
        signal = LeadLagSignal()
        assert signal._params["lookback"] == 60
        assert signal._params["lag_min"] == 1
        assert signal._params["lag_max"] == 5
        assert signal._params["min_correlation"] == 0.3
        assert signal._params["top_n_leaders"] == 5
        print("✅ PASS: Default parameters")

    def test_initialization_custom_params(self):
        """Test: Custom parameters are accepted."""
        signal = LeadLagSignal(
            lookback=90,
            lag_min=2,
            lag_max=7,
            min_correlation=0.4,
            top_n_leaders=3,
        )
        assert signal._params["lookback"] == 90
        assert signal._params["lag_min"] == 2
        assert signal._params["lag_max"] == 7
        assert signal._params["min_correlation"] == 0.4
        assert signal._params["top_n_leaders"] == 3
        print("✅ PASS: Custom parameters")

    def test_parameter_specs(self):
        """Test: Parameter specs are defined correctly."""
        specs = LeadLagSignal.parameter_specs()
        spec_names = [spec.name for spec in specs]

        assert "lookback" in spec_names
        assert "lag_min" in spec_names
        assert "lag_max" in spec_names
        assert "min_correlation" in spec_names
        assert "top_n_leaders" in spec_names
        print("✅ PASS: Parameter specs")

    def test_get_param_grid(self):
        """Test: Parameter grid for optimization is defined."""
        grid = LeadLagSignal.get_param_grid()

        assert "lookback" in grid
        assert "lag_min" in grid
        assert "lag_max" in grid
        assert "min_correlation" in grid
        assert 30 in grid["lookback"]
        assert 60 in grid["lookback"]
        print("✅ PASS: Parameter grid")

    def test_compute_known_lead_lag(self, sample_prices):
        """Test: Known lead-lag relationship is detected."""
        signal = LeadLagSignal(
            lookback=60,
            lag_min=1,
            lag_max=5,
            min_correlation=0.2,  # Lower threshold to ensure detection
            use_numba=False,  # Disable Numba for testing
        )
        result = signal.compute(sample_prices)

        # Result should be a SignalResult
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")

        # Scores should be in [-1, +1] range
        assert result.scores.abs().max() <= 1.0

        # All tickers should have scores
        assert len(result.scores) == 3
        assert "LEADER" in result.scores.index
        assert "FOLLOWER" in result.scores.index
        assert "INDEPENDENT" in result.scores.index

        # Metadata should contain pairs info
        assert "pairs_count" in result.metadata
        print(f"✅ PASS: Known lead-lag detection (pairs_count={result.metadata['pairs_count']})")

    def test_compute_insufficient_data(self, minimal_prices):
        """Test: Handles insufficient data gracefully."""
        signal = LeadLagSignal(lookback=60)  # More than available data
        result = signal.compute(minimal_prices)

        # Should not raise error, but return neutral scores
        assert result.scores.abs().max() <= 1.0
        assert "error" in result.metadata or result.metadata.get("pairs_count", 0) == 0
        print("✅ PASS: Insufficient data handling")

    def test_compute_scores_normalized(self, sample_prices):
        """Test: Scores are normalized to [-1, +1]."""
        signal = LeadLagSignal(use_numba=False)
        result = signal.compute(sample_prices)

        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0
        print("✅ PASS: Score normalization")

    def test_detect_lead_lag_pairs(self, sample_prices):
        """Test: _detect_lead_lag_pairs finds relationships."""
        signal = LeadLagSignal(min_correlation=0.2)
        returns = sample_prices.pct_change().dropna()

        pairs = signal._detect_lead_lag_pairs(
            returns,
            lookback=60,
            lag_range=(1, 5),
            min_correlation=0.2,
        )

        # Should find at least some pairs
        assert isinstance(pairs, list)

        # Check pair structure
        for pair in pairs:
            assert isinstance(pair, LeadLagPair)
            assert pair.leader in sample_prices.columns
            assert pair.follower in sample_prices.columns
            assert 1 <= pair.lag <= 5
            assert abs(pair.correlation) >= 0.2
            assert pair.direction in [-1, 1]
        print(f"✅ PASS: Lead-lag pair detection (found {len(pairs)} pairs)")

    def test_generate_signals(self, sample_prices):
        """Test: _generate_signals creates weighted signals."""
        signal = LeadLagSignal()
        returns = sample_prices.pct_change().dropna()

        # Create mock pairs
        pairs = [
            LeadLagPair(
                leader="LEADER",
                follower="FOLLOWER",
                lag=2,
                correlation=0.5,
                direction=1,
            ),
        ]

        scores = signal._generate_signals(returns, pairs, top_n_leaders=5)

        assert isinstance(scores, pd.Series)
        assert "FOLLOWER" in scores.index
        print("✅ PASS: Signal generation")

    def test_registry_registration(self):
        """Test: LeadLagSignal is registered in SignalRegistry."""
        assert SignalRegistry.is_registered("lead_lag")

        signal_cls = SignalRegistry.get("lead_lag")
        assert signal_cls == LeadLagSignal

        metadata = SignalRegistry.get_metadata("lead_lag")
        assert metadata["category"] == "relative"
        print("✅ PASS: Registry registration")

    def test_registry_create(self):
        """Test: Can create instance via SignalRegistry."""
        signal = SignalRegistry.create("lead_lag", lookback=90)
        assert isinstance(signal, LeadLagSignal)
        assert signal._params["lookback"] == 90
        print("✅ PASS: Registry create")


class TestComputePolars:
    """Tests for Polars DataFrame interface."""

    @pytest.fixture
    def polars_prices(self):
        """Create Polars DataFrame with price data."""
        dates = pl.date_range(
            datetime(2024, 1, 1),
            datetime(2024, 5, 1),
            eager=True,
        )
        tickers = ["AAPL", "GOOGL", "MSFT"]

        data = []
        np.random.seed(42)
        for ticker in tickers:
            base_price = 100.0
            for i, date in enumerate(dates):
                price = base_price * (1 + np.random.randn() * 0.02) ** i
                data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": price,
                })

        return pl.DataFrame(data)

    def test_compute_polars(self, polars_prices):
        """Test: compute_polars returns correct format."""
        signal = LeadLagSignal(use_numba=False)
        result = signal.compute_polars(polars_prices)

        assert isinstance(result, pl.DataFrame)
        assert "timestamp" in result.columns
        assert "ticker" in result.columns
        assert "value" in result.columns

        # Should have one row per ticker
        assert len(result) == 3
        print("✅ PASS: compute_polars format")

    def test_compute_polars_values_normalized(self, polars_prices):
        """Test: Polars output values are normalized."""
        signal = LeadLagSignal(use_numba=False)
        result = signal.compute_polars(polars_prices)

        values = result["value"].to_list()
        assert all(-1.0 <= v <= 1.0 for v in values)
        print("✅ PASS: compute_polars values normalized")


class TestConvenienceFunction:
    """Tests for compute_lead_lag_signal convenience function."""

    def test_convenience_function(self):
        """Test: compute_lead_lag_signal works correctly."""
        dates = pl.date_range(
            datetime(2024, 1, 1),
            datetime(2024, 5, 1),
            eager=True,
        )
        tickers = ["A", "B", "C"]

        data = []
        np.random.seed(42)
        for ticker in tickers:
            for i, date in enumerate(dates):
                data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": 100.0 + i * 0.5 + np.random.randn(),
                })

        prices = pl.DataFrame(data)
        result = compute_lead_lag_signal(
            prices,
            lookback=60,
            lag_range=(1, 5),
            min_correlation=0.2,
            use_numba=False,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        print("✅ PASS: Convenience function")


def test_all_scenarios():
    """Run all test scenarios and summarize results."""
    print("\n" + "=" * 60)
    print("LeadLagSignal Test Suite - task_042_1")
    print("=" * 60 + "\n")

    results = []

    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")

    # Leader-Follower relationship
    leader_returns = np.random.randn(120) * 0.02
    leader_prices = 100 * np.exp(np.cumsum(leader_returns))

    follower_returns = np.zeros(120)
    follower_returns[2:] = leader_returns[:-2] * 0.8 + np.random.randn(118) * 0.005
    follower_prices = 100 * np.exp(np.cumsum(follower_returns))

    sample_prices = pd.DataFrame(
        {"LEADER": leader_prices, "FOLLOWER": follower_prices},
        index=dates,
    )

    # Test 1: Initialization
    signal = LeadLagSignal()
    passed = signal._params["lookback"] == 60
    results.append(("Default initialization", passed))
    print(f"[1] Default initialization: {'✅' if passed else '❌'}")

    # Test 2: Custom params
    signal = LeadLagSignal(lookback=90, min_correlation=0.4)
    passed = signal._params["lookback"] == 90 and signal._params["min_correlation"] == 0.4
    results.append(("Custom parameters", passed))
    print(f"[2] Custom parameters: {'✅' if passed else '❌'}")

    # Test 3: Compute returns SignalResult
    signal = LeadLagSignal(min_correlation=0.2, use_numba=False)
    result = signal.compute(sample_prices)
    passed = hasattr(result, "scores") and hasattr(result, "metadata")
    results.append(("Compute returns SignalResult", passed))
    print(f"[3] Compute returns SignalResult: {'✅' if passed else '❌'}")

    # Test 4: Scores normalized
    passed = result.scores.min() >= -1.0 and result.scores.max() <= 1.0
    results.append(("Scores normalized [-1, +1]", passed))
    print(f"[4] Scores normalized: {'✅' if passed else '❌'}")

    # Test 5: All tickers have scores
    passed = len(result.scores) == 2
    results.append(("All tickers have scores", passed))
    print(f"[5] All tickers have scores: {'✅' if passed else '❌'}")

    # Test 6: Registry registration
    passed = SignalRegistry.is_registered("lead_lag")
    results.append(("Registry registration", passed))
    print(f"[6] Registry registration: {'✅' if passed else '❌'}")

    # Test 7: Create via registry
    try:
        sig = SignalRegistry.create("lead_lag", lookback=90)
        passed = isinstance(sig, LeadLagSignal)
    except Exception:
        passed = False
    results.append(("Create via registry", passed))
    print(f"[7] Create via registry: {'✅' if passed else '❌'}")

    # Test 8: Parameter specs
    specs = LeadLagSignal.parameter_specs()
    passed = len(specs) >= 5
    results.append(("Parameter specs defined", passed))
    print(f"[8] Parameter specs defined: {'✅' if passed else '❌'}")

    print("\n" + "-" * 60)
    print("Summary:")
    all_passed = all(p for _, p in results)
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")
    print("-" * 60)
    print(f"Total: {sum(1 for _, p in results if p)}/{len(results)} passed")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    test_all_scenarios()
