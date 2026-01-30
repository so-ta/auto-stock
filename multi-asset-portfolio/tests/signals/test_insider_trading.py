"""
Tests for Insider Trading Signal and SEC EDGAR Client.

Test Coverage:
1. SECEdgarClient - API response parsing, rate limiting
2. InsiderTradingSignal - Signal computation, parameter validation
3. SignalRegistry integration
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd


class TestSECEdgarClient:
    """Tests for SEC EDGAR API client."""

    def test_insider_transaction_properties(self):
        """Test InsiderTransaction dataclass properties."""
        from src.data.sec_edgar import InsiderTransaction

        # Test purchase transaction
        purchase = InsiderTransaction(
            filing_date=datetime(2024, 1, 15),
            transaction_date=datetime(2024, 1, 14),
            insider_name="John Smith",
            insider_title="CEO",
            transaction_type="P",
            shares=10000.0,
            price_per_share=150.0,
            total_value=1500000.0,
            shares_owned_after=50000.0,
        )

        assert purchase.is_purchase is True
        assert purchase.is_sale is False
        assert purchase.is_executive is True

        # Test sale transaction
        sale = InsiderTransaction(
            filing_date=datetime(2024, 1, 15),
            transaction_date=datetime(2024, 1, 14),
            insider_name="Jane Doe",
            insider_title="Director",
            transaction_type="S",
            shares=5000.0,
            price_per_share=155.0,
            total_value=775000.0,
            shares_owned_after=45000.0,
        )

        assert sale.is_purchase is False
        assert sale.is_sale is True
        assert sale.is_executive is False

    def test_client_initialization(self):
        """Test SECEdgarClient initialization."""
        from src.data.sec_edgar import SECEdgarClient, SECEdgarClientConfig

        # Default config
        client = SECEdgarClient()
        assert client.config.rate_limit_delay == 0.1
        assert "MultiAssetPortfolio" in client.config.user_agent

        # Custom config
        custom_config = SECEdgarClientConfig(
            user_agent="TestApp/1.0",
            rate_limit_delay=0.2,
            max_retries=5,
        )
        client = SECEdgarClient(custom_config)
        assert client.config.user_agent == "TestApp/1.0"
        assert client.config.max_retries == 5

    @patch("src.data.sec_edgar.SECEdgarClient._request")
    def test_get_transaction_summary(self, mock_request):
        """Test transaction summary computation."""
        from src.data.sec_edgar import SECEdgarClient, InsiderTransaction

        # Mock CIK lookup
        mock_request.side_effect = [
            # First call: company tickers
            {"0": {"ticker": "AAPL", "cik_str": "320193"}},
            # Second call: filings
            {
                "filings": {
                    "recent": {
                        "form": ["4", "10-K", "4"],
                        "filingDate": ["2024-01-15", "2024-01-10", "2024-01-05"],
                        "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-24-000003"],
                    }
                }
            },
        ]

        client = SECEdgarClient()

        # Note: get_transaction_summary returns empty if no transactions parsed
        # This tests the summary calculation logic with mock data
        summary = client.get_transaction_summary("AAPL", days=30)

        assert summary["ticker"] == "AAPL"
        assert summary["period_days"] == 30
        assert "net_sentiment" in summary

    def test_rate_limiting(self):
        """Test rate limiting between requests."""
        from src.data.sec_edgar import SECEdgarClient
        import time

        client = SECEdgarClient()
        client.config.rate_limit_delay = 0.1

        # Simulate two rapid calls
        start = time.time()
        client._rate_limit()
        client._rate_limit()
        elapsed = time.time() - start

        # Should have at least one delay
        assert elapsed >= 0.09  # Allow small timing variance


class TestInsiderTradingSignal:
    """Tests for InsiderTradingSignal class."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate synthetic price data
        returns = np.random.randn(100) * 0.02
        close = 100 * np.exp(np.cumsum(returns))

        # Generate volume with some spikes
        volume = np.random.uniform(1000000, 2000000, 100)
        volume[10] = 5000000  # Volume spike
        volume[50] = 4500000  # Another spike

        return pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    def test_signal_initialization(self):
        """Test signal initialization with parameters."""
        from src.signals.insider_trading import InsiderTradingSignal

        # Default parameters
        signal = InsiderTradingSignal()
        assert signal._params["lookback_days"] == 90
        assert signal._params["min_transactions"] == 2
        assert signal._params["executive_only"] is False

        # Custom parameters
        signal = InsiderTradingSignal(
            lookback_days=60,
            min_transactions=3,
            executive_only=True,
        )
        assert signal._params["lookback_days"] == 60
        assert signal._params["min_transactions"] == 3
        assert signal._params["executive_only"] is True

    def test_parameter_validation(self):
        """Test parameter validation."""
        from src.signals.insider_trading import InsiderTradingSignal

        # Valid parameters
        signal = InsiderTradingSignal(lookback_days=30)
        assert signal._params["lookback_days"] == 30

        # Invalid parameter (out of range)
        with pytest.raises(ValueError):
            InsiderTradingSignal(lookback_days=10)  # Below min_value of 30

        with pytest.raises(ValueError):
            InsiderTradingSignal(lookback_days=200)  # Above max_value of 180

    def test_compute_basic(self, sample_price_data):
        """Test basic signal computation."""
        from src.signals.insider_trading import InsiderTradingSignal

        signal = InsiderTradingSignal(lookback_days=30, min_transactions=1)
        result = signal.compute(sample_price_data)

        # Check result structure
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")
        assert isinstance(result.scores, pd.Series)

        # Check score range
        assert result.scores.min() >= -1.0
        assert result.scores.max() <= 1.0

        # Check metadata
        assert result.metadata["lookback_days"] == 30
        assert "score_mean" in result.metadata
        assert "score_std" in result.metadata

    def test_compute_output_range(self, sample_price_data):
        """Test that output is always in [-1, +1] range."""
        from src.signals.insider_trading import InsiderTradingSignal

        # Test with various parameter combinations
        param_sets = [
            {"lookback_days": 30, "min_transactions": 1},
            {"lookback_days": 90, "min_transactions": 3},
            {"lookback_days": 180, "executive_only": True},
        ]

        for params in param_sets:
            signal = InsiderTradingSignal(**params)
            result = signal.compute(sample_price_data)

            assert result.scores.min() >= -1.0, f"Score below -1 with {params}"
            assert result.scores.max() <= 1.0, f"Score above +1 with {params}"

    def test_compute_no_volume(self, sample_price_data):
        """Test computation when volume data is missing."""
        from src.signals.insider_trading import InsiderTradingSignal

        # Remove volume column
        data_no_volume = sample_price_data.drop(columns=["volume"])

        signal = InsiderTradingSignal()
        result = signal.compute(data_no_volume)

        # Should still work with default volume
        assert len(result.scores) == len(data_no_volume)
        assert not result.scores.isna().all()

    def test_parameter_grid(self):
        """Test parameter grid generation."""
        from src.signals.insider_trading import InsiderTradingSignal

        grid = InsiderTradingSignal.get_param_grid()

        assert "lookback_days" in grid
        assert "min_transactions" in grid
        assert "executive_only" in grid

        assert 30 in grid["lookback_days"]
        assert 90 in grid["lookback_days"]
        assert 180 in grid["lookback_days"]

    def test_parameter_combinations(self):
        """Test parameter combination generation."""
        from src.signals.insider_trading import InsiderTradingSignal

        combinations = InsiderTradingSignal.get_param_combinations()

        assert len(combinations) > 0
        assert all(isinstance(c, dict) for c in combinations)

        # Each combination should have all searchable params
        for combo in combinations:
            assert "lookback_days" in combo or "min_transactions" in combo

    def test_convenience_function(self, sample_price_data):
        """Test the convenience function."""
        from src.signals.insider_trading import compute_insider_signal

        scores = compute_insider_signal(
            sample_price_data,
            lookback_days=60,
            min_transactions=2,
        )

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_price_data)
        assert scores.min() >= -1.0
        assert scores.max() <= 1.0


class TestSignalRegistration:
    """Tests for SignalRegistry integration."""

    def test_signal_registered(self):
        """Test that InsiderTradingSignal is registered."""
        from src.signals.registry import SignalRegistry

        # Import to trigger registration
        from src.signals import insider_trading  # noqa: F401

        assert SignalRegistry.is_registered("insider_trading")

    def test_create_from_registry(self):
        """Test creating signal from registry."""
        from src.signals.registry import SignalRegistry
        from src.signals import insider_trading  # noqa: F401

        signal = SignalRegistry.create(
            "insider_trading",
            lookback_days=60,
            min_transactions=3,
        )

        assert signal._params["lookback_days"] == 60
        assert signal._params["min_transactions"] == 3

    def test_signal_metadata(self):
        """Test signal metadata from registry."""
        from src.signals.registry import SignalRegistry
        from src.signals import insider_trading  # noqa: F401

        metadata = SignalRegistry.get_metadata("insider_trading")

        assert metadata["category"] == "independent"
        assert "sec" in metadata["tags"]
        assert "fundamental" in metadata["tags"]

    def test_list_by_category(self):
        """Test filtering signals by category."""
        from src.signals.registry import SignalRegistry
        from src.signals import insider_trading  # noqa: F401

        independent_signals = SignalRegistry.list_by_category("independent")
        assert "insider_trading" in independent_signals


class TestInsiderTradingProxyLogic:
    """Tests for the insider activity proxy computation."""

    @pytest.fixture
    def price_with_volume_spike(self):
        """Create data with clear volume spike pattern."""
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")

        # Steady prices with upward move after volume spike
        close = np.ones(60) * 100
        volume = np.ones(60) * 1000000

        # Create volume spike at day 20 followed by price increase
        volume[20] = 5000000  # 5x normal
        close[21:] = 105  # Price up 5% after spike

        return pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    def test_volume_spike_detection(self, price_with_volume_spike):
        """Test that volume spikes are detected as potential insider activity."""
        from src.signals.insider_trading import InsiderTradingSignal

        signal = InsiderTradingSignal(lookback_days=30, min_transactions=1)
        result = signal.compute(price_with_volume_spike)

        # Should have some positive scores after the volume spike
        # (since price went up afterward)
        recent_scores = result.scores.iloc[25:]
        assert recent_scores.mean() >= 0

    def test_min_transactions_threshold(self, price_with_volume_spike):
        """Test minimum transactions threshold."""
        from src.signals.insider_trading import InsiderTradingSignal

        # With high threshold, should get zero signal
        signal = InsiderTradingSignal(lookback_days=30, min_transactions=5)
        result = signal.compute(price_with_volume_spike)

        # Most values should be zero due to insufficient "transactions"
        zero_count = (result.scores == 0).sum()
        assert zero_count > len(result.scores) * 0.5

    def test_asymmetric_weighting(self, price_with_volume_spike):
        """Test that purchases are weighted more than sales."""
        from src.signals.insider_trading import InsiderTradingSignal

        # Default: purchase_weight=1.0, sale_weight=0.3
        signal = InsiderTradingSignal(lookback_days=30, min_transactions=1)

        # Signal should be more responsive to buying patterns
        assert signal._params["purchase_weight"] > signal._params["sale_weight"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from src.signals.insider_trading import InsiderTradingSignal

        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )

        signal = InsiderTradingSignal()

        with pytest.raises(ValueError, match="empty"):
            signal.compute(empty_df)

    def test_missing_close_column(self):
        """Test handling of missing required column."""
        from src.signals.insider_trading import InsiderTradingSignal

        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"open": np.ones(10), "high": np.ones(10), "low": np.ones(10)},
            index=dates,
        )

        signal = InsiderTradingSignal()

        with pytest.raises(ValueError, match="close"):
            signal.compute(df)

    def test_non_datetime_index(self):
        """Test handling of non-datetime index."""
        from src.signals.insider_trading import InsiderTradingSignal

        df = pd.DataFrame(
            {
                "open": np.ones(10),
                "high": np.ones(10),
                "low": np.ones(10),
                "close": np.ones(10),
            },
            index=range(10),  # Integer index instead of DatetimeIndex
        )

        signal = InsiderTradingSignal()

        with pytest.raises(ValueError, match="DatetimeIndex"):
            signal.compute(df)

    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        from src.signals.insider_trading import InsiderTradingSignal

        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        close = np.random.randn(50).cumsum() + 100
        close[10:15] = np.nan  # Add some NaNs

        df = pd.DataFrame(
            {
                "close": close,
                "volume": np.random.uniform(1000000, 2000000, 50),
            },
            index=dates,
        )

        signal = InsiderTradingSignal()
        result = signal.compute(df)

        # Output should not have any NaN
        assert not result.scores.isna().any()

    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        from src.signals.insider_trading import InsiderTradingSignal

        dates = pd.date_range(start="2024-01-01", periods=1, freq="D")
        df = pd.DataFrame(
            {"close": [100.0], "volume": [1000000]},
            index=dates,
        )

        signal = InsiderTradingSignal()
        result = signal.compute(df)

        assert len(result.scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
