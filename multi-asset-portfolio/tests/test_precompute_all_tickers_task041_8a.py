"""
Test script for task_041_8a: precompute_all() new_tickers support

Verifies:
1. New tickers triggers precompute_for_new_tickers()
2. New signals triggers precompute_missing_signals()
3. Date extension triggers precompute_incremental()
4. Full cache hit returns False
5. All branches work correctly
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

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


class TestPrecomputeAllNewTickers:
    """Test precompute_all() new tickers support."""

    def test_new_tickers_triggers_precompute_for_new_tickers(self):
        """Test that new tickers triggers precompute_for_new_tickers()."""
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

            # Initial computation
            precomputer.precompute_all(prices, config, force=False)

            # Mock validate_cache_incremental to return new_tickers
            mock_result = CacheValidationResult(
                can_use_cache=True,
                reason="New tickers: ['QQQ', 'IWM']",
                incremental_start=None,
                missing_signals=None,
                new_tickers=["QQQ", "IWM"],
            )

            with patch.object(
                precomputer,
                'validate_cache_incremental',
                return_value=mock_result
            ):
                with patch.object(
                    precomputer,
                    'precompute_for_new_tickers',
                    return_value=True
                ) as mock_new_tickers:
                    # Extended data with new tickers
                    extended_prices = create_sample_prices(
                        ["SPY", "QQQ", "IWM"],
                        datetime(2025, 1, 1),
                        datetime(2025, 3, 31),
                    )
                    result = precomputer.precompute_all(extended_prices, config, force=False)

                    # Verify precompute_for_new_tickers was called
                    mock_new_tickers.assert_called_once()
                    assert result is True

            print("✅ test_new_tickers_triggers_precompute_for_new_tickers PASSED")

    def test_priority_new_tickers_over_missing_signals(self):
        """Test that new_tickers is checked before missing_signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Mock result with both new_tickers AND missing_signals
            mock_result = CacheValidationResult(
                can_use_cache=True,
                reason="New tickers + new signals",
                incremental_start=None,
                missing_signals=["rsi_14"],
                new_tickers=["QQQ"],
            )

            with patch.object(
                precomputer,
                'validate_cache_incremental',
                return_value=mock_result
            ):
                with patch.object(
                    precomputer,
                    'precompute_for_new_tickers',
                    return_value=True
                ) as mock_new_tickers:
                    with patch.object(
                        precomputer,
                        'precompute_missing_signals',
                        return_value=True
                    ) as mock_missing:
                        prices = create_sample_prices(
                            ["SPY", "QQQ"],
                            datetime(2025, 1, 1),
                            datetime(2025, 3, 31),
                        )
                        config = {"momentum_periods": [20]}

                        result = precomputer.precompute_all(prices, config, force=False)

                        # new_tickers should be called, not missing_signals
                        mock_new_tickers.assert_called_once()
                        mock_missing.assert_not_called()
                        assert result is True

            print("✅ test_priority_new_tickers_over_missing_signals PASSED")

    def test_priority_new_tickers_over_incremental(self):
        """Test that new_tickers is checked before incremental_start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Mock result with both new_tickers AND incremental_start
            mock_result = CacheValidationResult(
                can_use_cache=True,
                reason="New tickers + time extension",
                incremental_start=datetime(2025, 3, 31),
                missing_signals=None,
                new_tickers=["QQQ"],
            )

            with patch.object(
                precomputer,
                'validate_cache_incremental',
                return_value=mock_result
            ):
                with patch.object(
                    precomputer,
                    'precompute_for_new_tickers',
                    return_value=True
                ) as mock_new_tickers:
                    with patch.object(
                        precomputer,
                        'precompute_incremental',
                        return_value=True
                    ) as mock_incr:
                        prices = create_sample_prices(
                            ["SPY", "QQQ"],
                            datetime(2025, 1, 1),
                            datetime(2025, 4, 30),
                        )
                        config = {"momentum_periods": [20]}

                        result = precomputer.precompute_all(prices, config, force=False)

                        # new_tickers should be called, not incremental
                        mock_new_tickers.assert_called_once()
                        mock_incr.assert_not_called()
                        assert result is True

            print("✅ test_priority_new_tickers_over_incremental PASSED")

    def test_full_branch_coverage(self):
        """Test all branches with proper mocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {"momentum_periods": [20]}
            prices = create_sample_prices(
                ["SPY"],
                datetime(2025, 1, 1),
                datetime(2025, 3, 31),
            )

            # Test 1: can_use_cache=False → _full_precompute
            mock_result1 = CacheValidationResult(
                can_use_cache=False,
                reason="Cache invalid",
            )
            with patch.object(precomputer, 'validate_cache_incremental', return_value=mock_result1):
                with patch.object(precomputer, '_full_precompute', return_value=True) as mock_full:
                    precomputer.precompute_all(prices, config, force=False)
                    mock_full.assert_called_once()
            print("  ✅ Branch: can_use_cache=False → _full_precompute")

            # Test 2: new_tickers → precompute_for_new_tickers
            mock_result2 = CacheValidationResult(
                can_use_cache=True,
                reason="New tickers",
                new_tickers=["QQQ"],
            )
            with patch.object(precomputer, 'validate_cache_incremental', return_value=mock_result2):
                with patch.object(precomputer, 'precompute_for_new_tickers', return_value=True) as mock_new:
                    precomputer.precompute_all(prices, config, force=False)
                    mock_new.assert_called_once()
            print("  ✅ Branch: new_tickers → precompute_for_new_tickers")

            # Test 3: missing_signals → precompute_missing_signals
            mock_result3 = CacheValidationResult(
                can_use_cache=True,
                reason="New signals",
                missing_signals=["rsi_14"],
            )
            with patch.object(precomputer, 'validate_cache_incremental', return_value=mock_result3):
                with patch.object(precomputer, 'precompute_missing_signals', return_value=True) as mock_miss:
                    precomputer.precompute_all(prices, config, force=False)
                    mock_miss.assert_called_once()
            print("  ✅ Branch: missing_signals → precompute_missing_signals")

            # Test 4: incremental_start → precompute_incremental
            mock_result4 = CacheValidationResult(
                can_use_cache=True,
                reason="Time extension",
                incremental_start=datetime(2025, 3, 31),
            )
            with patch.object(precomputer, 'validate_cache_incremental', return_value=mock_result4):
                with patch.object(precomputer, 'precompute_incremental', return_value=True) as mock_incr:
                    precomputer.precompute_all(prices, config, force=False)
                    mock_incr.assert_called_once()
            print("  ✅ Branch: incremental_start → precompute_incremental")

            # Test 5: Full cache hit → return False
            mock_result5 = CacheValidationResult(
                can_use_cache=True,
                reason="Full cache hit",
            )
            with patch.object(precomputer, 'validate_cache_incremental', return_value=mock_result5):
                result = precomputer.precompute_all(prices, config, force=False)
                assert result is False
            print("  ✅ Branch: Full cache hit → return False")

            print("✅ test_full_branch_coverage PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing task_041_8a: precompute_all() new_tickers support")
    print("=" * 60)

    tests = TestPrecomputeAllNewTickers()

    tests.test_new_tickers_triggers_precompute_for_new_tickers()
    tests.test_priority_new_tickers_over_missing_signals()
    tests.test_priority_new_tickers_over_incremental()
    tests.test_full_branch_coverage()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
