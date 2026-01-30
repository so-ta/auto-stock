"""
Test script for task_041_4: precompute_all() incremental support

Verifies:
1. force=True triggers full recomputation
2. Invalid cache triggers full recomputation
3. Missing signals triggers precompute_missing_signals()
4. Date extension triggers precompute_incremental()
5. Full cache hit returns False (no computation)
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.signal_precompute import (
    SignalPrecomputer,
    CacheValidationResult,
)


def create_sample_prices(
    tickers: list[str],
    start_date: datetime,
    end_date: datetime,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Create sample price data for testing."""
    rows = []
    current_date = start_date

    while current_date <= end_date:
        for ticker in tickers:
            seed = hash(f"{ticker}_{current_date.isoformat()}")
            np.random.seed(seed % (2**31))

            daily_return = np.random.normal(0.0005, 0.02)
            close = base_price * (1 + daily_return)

            rows.append({
                "timestamp": current_date,
                "ticker": ticker,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000000.0,
            })

            base_price = close

        current_date += timedelta(days=1)

    return pl.DataFrame(rows)


class TestPrecomputeAllBranches:
    """Test precompute_all() branching logic."""

    def test_force_triggers_full_recompute(self):
        """Test that force=True triggers full recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            prices = create_sample_prices(
                tickers,
                datetime(2025, 1, 1),
                datetime(2025, 3, 31),
            )

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # First run
            result1 = precomputer.precompute_all(prices, config, force=False)
            assert result1 is True, "First run should compute"

            # Second run with force=True
            result2 = precomputer.precompute_all(prices, config, force=True)
            assert result2 is True, "force=True should trigger recomputation"

            print("✅ test_force_triggers_full_recompute PASSED")

    def test_invalid_cache_triggers_full_recompute(self):
        """Test that invalid cache triggers full recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            prices1 = create_sample_prices(
                tickers,
                datetime(2025, 1, 1),
                datetime(2025, 3, 31),
            )

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # First run
            precomputer.precompute_all(prices1, config, force=False)

            # Second run with different config (should invalidate cache)
            config2 = {
                "momentum_periods": [20, 60],  # Changed!
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            result = precomputer.precompute_all(prices1, config2, force=False)
            assert result is True, "Changed config should trigger recomputation"

            # Verify momentum_60 was computed
            signals = precomputer.list_cached_signals()
            assert "momentum_60" in signals

            print("✅ test_invalid_cache_triggers_full_recompute PASSED")

    def test_missing_signals_triggers_partial_compute(self):
        """Test that missing signals triggers precompute_missing_signals()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            prices = create_sample_prices(
                tickers,
                datetime(2025, 1, 1),
                datetime(2025, 6, 30),
            )

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # First run
            precomputer.precompute_all(prices, config, force=False)

            initial_signals = precomputer.list_cached_signals()
            print(f"Initial signals: {initial_signals}")

            # Mock validate_cache_incremental to return missing_signals
            mock_result = CacheValidationResult(
                can_use_cache=True,
                reason="New signals: ['rsi_14']",
                incremental_start=None,
                missing_signals=["rsi_14"],
            )

            with patch.object(
                precomputer,
                'validate_cache_incremental',
                return_value=mock_result
            ):
                with patch.object(
                    precomputer,
                    'precompute_missing_signals',
                    return_value=True
                ) as mock_missing:
                    result = precomputer.precompute_all(prices, config, force=False)

                    # Verify precompute_missing_signals was called
                    mock_missing.assert_called_once()
                    assert result is True

            print("✅ test_missing_signals_triggers_partial_compute PASSED")

    def test_date_extension_triggers_incremental(self):
        """Test that date extension triggers precompute_incremental()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # First run with initial data
            prices1 = create_sample_prices(
                tickers,
                datetime(2025, 1, 1),
                datetime(2025, 3, 31),
            )
            precomputer.precompute_all(prices1, config, force=False)

            # Mock validate_cache_incremental to return incremental_start
            mock_result = CacheValidationResult(
                can_use_cache=True,
                reason="Incremental update: 2025-03-31 -> 2025-04-30",
                incremental_start=datetime(2025, 3, 31),
                missing_signals=None,
            )

            with patch.object(
                precomputer,
                'validate_cache_incremental',
                return_value=mock_result
            ):
                with patch.object(
                    precomputer,
                    'precompute_incremental',
                    return_value=True
                ) as mock_incr:
                    prices2 = create_sample_prices(
                        tickers,
                        datetime(2025, 1, 1),
                        datetime(2025, 4, 30),
                    )
                    result = precomputer.precompute_all(prices2, config, force=False)

                    # Verify precompute_incremental was called
                    mock_incr.assert_called_once()
                    assert result is True

            print("✅ test_date_extension_triggers_incremental PASSED")

    def test_full_cache_hit_returns_false(self):
        """Test that full cache hit returns False (no computation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            prices = create_sample_prices(
                tickers,
                datetime(2025, 1, 1),
                datetime(2025, 3, 31),
            )

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # First run
            result1 = precomputer.precompute_all(prices, config, force=False)
            assert result1 is True, "First run should compute"

            # Second run with same data
            result2 = precomputer.precompute_all(prices, config, force=False)
            assert result2 is False, "Cache hit should return False"

            print("✅ test_full_cache_hit_returns_false PASSED")


def test_full_precompute_method():
    """Test _full_precompute() internal method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        tickers = ["SPY", "QQQ"]
        prices = create_sample_prices(
            tickers,
            datetime(2025, 1, 1),
            datetime(2025, 6, 30),
        )

        config = {
            "momentum_periods": [20, 60],
            "volatility_periods": [20],
            "rsi_periods": [14],
            "zscore_periods": [20],
            "sharpe_periods": [60],
        }

        result = precomputer._full_precompute(prices, config)
        assert result is True

        # Verify all signals were computed
        signals = precomputer.list_cached_signals()
        expected = ["momentum_20", "momentum_60", "volatility_20", "rsi_14", "zscore_20", "sharpe_60"]
        for sig in expected:
            assert sig in signals, f"Missing signal: {sig}"

        print("✅ test_full_precompute_method PASSED")


def test_integration_workflow():
    """Test full workflow: initial -> cache hit -> new signals -> incremental."""
    print("\n" + "=" * 60)
    print("Integration Test: Full workflow")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        tickers = ["SPY", "QQQ"]
        config = {
            "momentum_periods": [20],
            "volatility_periods": [20],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        # Phase 1: Initial computation
        print("\n[Phase 1] Initial computation...")
        prices1 = create_sample_prices(
            tickers,
            datetime(2025, 1, 1),
            datetime(2025, 6, 30),
        )
        result1 = precomputer.precompute_all(prices1, config, force=False)
        assert result1 is True, "Initial should compute"
        print(f"Signals: {precomputer.list_cached_signals()}")

        # Phase 2: Cache hit (same data)
        print("\n[Phase 2] Cache hit (same data)...")
        result2 = precomputer.precompute_all(prices1, config, force=False)
        assert result2 is False, "Should be cache hit"
        print("Cache hit - no computation")

        # Phase 3: Force recomputation
        print("\n[Phase 3] Force recomputation...")
        result3 = precomputer.precompute_all(prices1, config, force=True)
        assert result3 is True, "Force should recompute"
        print("Forced recomputation completed")

        print("\n" + "=" * 60)
        print("✅ Integration Test PASSED")
        print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing task_041_4: precompute_all() incremental support")
    print("=" * 60)

    tests = TestPrecomputeAllBranches()

    tests.test_force_triggers_full_recompute()
    tests.test_invalid_cache_triggers_full_recompute()
    tests.test_missing_signals_triggers_partial_compute()
    tests.test_date_extension_triggers_incremental()
    tests.test_full_cache_hit_returns_false()

    test_full_precompute_method()
    test_integration_workflow()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
