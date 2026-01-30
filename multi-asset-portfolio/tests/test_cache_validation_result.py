"""
Test CacheValidationResult and ticker difference detection (task_041_7a).

Verifies:
1. CacheValidationResult dataclass structure
2. New ticker detection (new_tickers field)
3. Removed tickers handling (cache still valid)
4. Conversion from old 4-element tuple pattern
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.signal_precompute import (
    CacheValidationResult,
    SignalPrecomputer,
    PRECOMPUTE_VERSION,
)


class TestCacheValidationResult:
    """Test CacheValidationResult dataclass."""

    def test_dataclass_structure(self):
        """Test CacheValidationResult has all required fields."""
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Test reason",
        )
        assert result.can_use_cache is True
        assert result.reason == "Test reason"
        assert result.incremental_start is None
        assert result.missing_signals is None
        assert result.new_tickers is None

    def test_full_cache_hit(self):
        """Test full cache hit result."""
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Full cache hit - dates match exactly",
        )
        assert result.can_use_cache is True
        assert result.incremental_start is None
        assert result.missing_signals is None
        assert result.new_tickers is None

    def test_new_tickers_result(self):
        """Test result with new tickers."""
        result = CacheValidationResult(
            can_use_cache=True,
            reason="New tickers added: 3",
            new_tickers=["NVDA", "AMD", "INTC"],
        )
        assert result.can_use_cache is True
        assert result.new_tickers == ["NVDA", "AMD", "INTC"]
        assert len(result.new_tickers) == 3

    def test_incremental_time_result(self):
        """Test result with incremental time update."""
        start = datetime(2024, 6, 1)
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Incremental update: 2024-05-31 -> 2024-06-30",
            incremental_start=start,
        )
        assert result.can_use_cache is True
        assert result.incremental_start == start
        assert result.new_tickers is None

    def test_combined_result(self):
        """Test result with multiple incremental conditions."""
        start = datetime(2024, 6, 1)
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Incremental update + new signals + new tickers",
            incremental_start=start,
            missing_signals=["momentum_120"],
            new_tickers=["TSLA"],
        )
        assert result.can_use_cache is True
        assert result.incremental_start == start
        assert result.missing_signals == ["momentum_120"]
        assert result.new_tickers == ["TSLA"]

    def test_full_recomputation_result(self):
        """Test result requiring full recomputation."""
        result = CacheValidationResult(
            can_use_cache=False,
            reason="Version changed: 1.0.0 -> 1.2.0",
        )
        assert result.can_use_cache is False
        assert "Version changed" in result.reason


class TestTickerDifferenceDetection:
    """Test ticker difference detection in validate_cache_incremental."""

    @pytest.fixture
    def precomputer(self, tmp_path):
        """Create precomputer with temp directory."""
        return SignalPrecomputer(cache_dir=tmp_path)

    @pytest.fixture
    def base_prices(self):
        """Create base price data."""
        dates = pl.date_range(
            datetime(2024, 1, 1),
            datetime(2024, 3, 31),
            "1d",
            eager=True,
        )
        tickers = ["AAPL", "MSFT", "GOOGL"]

        data = []
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
            for i, date in enumerate(dates):
                data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": float(prices[i]),
                    "high": float(prices[i] * 1.02),
                    "low": float(prices[i] * 0.98),
                    "volume": 1000000,
                })

        return pl.DataFrame(data)

    def test_new_tickers_detection(self, precomputer, base_prices):
        """Test detection of newly added tickers."""
        config = {"momentum_periods": [20], "volatility_periods": [],
                  "rsi_periods": [], "zscore_periods": [], "sharpe_periods": []}

        # First, precompute with base tickers
        precomputer.precompute_all(base_prices, config, force=True)

        # Add new tickers
        dates = base_prices.select("timestamp").unique().to_series()
        new_ticker_data = []
        for ticker in ["NVDA", "AMD"]:
            np.random.seed(hash(ticker) % 2**32)
            prices = 150 + np.cumsum(np.random.randn(len(dates)) * 3)
            for i, date in enumerate(dates):
                new_ticker_data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": float(prices[i]),
                    "high": float(prices[i] * 1.02),
                    "low": float(prices[i] * 0.98),
                    "volume": 500000,
                })

        extended_prices = pl.concat([base_prices, pl.DataFrame(new_ticker_data)])

        # Validate - should detect new tickers
        result = precomputer.validate_cache_incremental(extended_prices, config)

        assert isinstance(result, CacheValidationResult)
        assert result.can_use_cache is True
        assert result.new_tickers is not None
        assert set(result.new_tickers) == {"NVDA", "AMD"}
        print(f"New tickers detected: {result.new_tickers}")
        print(f"Reason: {result.reason}")

    def test_removed_tickers_only(self, precomputer, base_prices):
        """Test that removing tickers keeps cache valid."""
        config = {"momentum_periods": [20], "volatility_periods": [],
                  "rsi_periods": [], "zscore_periods": [], "sharpe_periods": []}

        # First, precompute with all tickers
        precomputer.precompute_all(base_prices, config, force=True)

        # Remove one ticker
        reduced_prices = base_prices.filter(pl.col("ticker") != "GOOGL")

        # Validate - should be valid (cache still usable)
        result = precomputer.validate_cache_incremental(reduced_prices, config)

        assert isinstance(result, CacheValidationResult)
        assert result.can_use_cache is True
        assert result.new_tickers is None  # No new tickers
        print(f"Reason: {result.reason}")

    def test_no_ticker_change(self, precomputer, base_prices):
        """Test validation when tickers haven't changed."""
        config = {"momentum_periods": [20], "volatility_periods": [],
                  "rsi_periods": [], "zscore_periods": [], "sharpe_periods": []}

        # Precompute
        precomputer.precompute_all(base_prices, config, force=True)

        # Validate with same tickers
        result = precomputer.validate_cache_incremental(base_prices, config)

        assert isinstance(result, CacheValidationResult)
        assert result.can_use_cache is True
        assert result.new_tickers is None
        assert "Full cache hit" in result.reason or "Dates match" in result.reason


class TestBackwardCompatibility:
    """Test that CacheValidationResult provides equivalent info to old tuple."""

    def test_tuple_equivalent_full_hit(self):
        """Test equivalent to old (True, reason, None, None) case."""
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Full cache hit",
        )

        # Old code would have: can_use, reason, incr_start, missing = result
        can_use = result.can_use_cache
        reason = result.reason
        incr_start = result.incremental_start
        missing = result.missing_signals

        assert can_use is True
        assert reason == "Full cache hit"
        assert incr_start is None
        assert missing is None

    def test_tuple_equivalent_incremental(self):
        """Test equivalent to old (True, reason, date, signals) case."""
        start = datetime(2024, 6, 1)
        result = CacheValidationResult(
            can_use_cache=True,
            reason="Incremental + signals",
            incremental_start=start,
            missing_signals=["momentum_120"],
        )

        can_use = result.can_use_cache
        reason = result.reason
        incr_start = result.incremental_start
        missing = result.missing_signals

        assert can_use is True
        assert incr_start == start
        assert missing == ["momentum_120"]


if __name__ == "__main__":
    print("=" * 60)
    print("CACHE VALIDATION RESULT TESTS (task_041_7a)")
    print("=" * 60)

    # Run basic structure tests
    print("\n1. Testing CacheValidationResult structure...")
    test = TestCacheValidationResult()
    test.test_dataclass_structure()
    print("   PASS: Dataclass structure")
    test.test_new_tickers_result()
    print("   PASS: New tickers result")
    test.test_combined_result()
    print("   PASS: Combined result")

    print("\n2. Testing backward compatibility...")
    compat_test = TestBackwardCompatibility()
    compat_test.test_tuple_equivalent_full_hit()
    print("   PASS: Tuple equivalent (full hit)")
    compat_test.test_tuple_equivalent_incremental()
    print("   PASS: Tuple equivalent (incremental)")

    print("\n" + "=" * 60)
    print("BASIC TESTS PASSED!")
    print("Run pytest for full integration tests")
    print("=" * 60)
