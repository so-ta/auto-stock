"""
Test script for task_041_3: Incremental Signal Precomputation

Verifies:
1. precompute_incremental() implementation
2. Incremental results match full computation (diff < 1e-10)
3. Correct append to existing cache
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.signal_precompute import SignalPrecomputer, PRECOMPUTE_VERSION


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
            # Generate deterministic but varied prices
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


class TestPrecomputeIncremental:
    """Test precompute_incremental functionality."""

    def test_incremental_basic(self):
        """Test basic incremental computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Create initial data (300 days)
            tickers = ["SPY", "QQQ", "IWM"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 10, 27)  # ~300 days

            prices = create_sample_prices(tickers, start_date, end_date)

            # Full precompute
            config = {
                "momentum_periods": [20, 60],
                "volatility_periods": [20],
                "rsi_periods": [14],
                "zscore_periods": [20],
                "sharpe_periods": [60],
            }
            precomputer.precompute_all(prices, config, force=True)

            print("✅ Initial full precomputation completed")

            # Load initial end date from metadata
            metadata = precomputer._load_precompute_metadata()
            assert metadata is not None
            initial_end_date = metadata.cached_end_date
            print(f"Initial cached_end_date: {initial_end_date}")

            # Extend data by 20 days
            new_end_date = datetime(2025, 11, 16)  # +20 days
            extended_prices = create_sample_prices(tickers, start_date, new_end_date)

            # Incremental update
            incremental_start = datetime.fromisoformat(initial_end_date.split("T")[0])
            success = precomputer.precompute_incremental(
                extended_prices,
                start_from=incremental_start,
                config=config,
            )

            assert success, "Incremental precomputation should succeed"
            print("✅ Incremental precomputation completed")

            # Verify metadata updated
            metadata_after = precomputer._load_precompute_metadata()
            assert metadata_after is not None
            assert metadata_after.cached_end_date > initial_end_date
            print(f"Updated cached_end_date: {metadata_after.cached_end_date}")

            print("✅ test_incremental_basic PASSED")

    def test_incremental_precision(self):
        """Test that incremental results match full computation (diff < 1e-10)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two separate precomputers for comparison
            precomputer_incr = SignalPrecomputer(cache_dir=f"{tmpdir}/incr")
            precomputer_full = SignalPrecomputer(cache_dir=f"{tmpdir}/full")

            tickers = ["SPY", "QQQ"]
            start_date = datetime(2025, 1, 1)
            mid_date = datetime(2025, 9, 1)  # Split point
            end_date = datetime(2025, 10, 15)

            config = {
                "momentum_periods": [20, 60],
                "volatility_periods": [20],
                "rsi_periods": [14],
                "zscore_periods": [20],
                "sharpe_periods": [60],
            }

            # Full dataset
            full_prices = create_sample_prices(tickers, start_date, end_date)

            # Initial dataset (up to mid_date)
            initial_prices = full_prices.filter(pl.col("timestamp") <= mid_date)

            # --- Approach 1: Full computation on full data ---
            precomputer_full.precompute_all(full_prices, config, force=True)

            # --- Approach 2: Initial + Incremental ---
            precomputer_incr.precompute_all(initial_prices, config, force=True)
            precomputer_incr.precompute_incremental(
                full_prices,
                start_from=mid_date,
                config=config,
            )

            # Compare all signals
            signals_to_check = [
                "momentum_20", "momentum_60",
                "volatility_20",
                "rsi_14",
                "zscore_20",
                "sharpe_60",
            ]

            max_diffs = {}
            for signal_name in signals_to_check:
                full_df = precomputer_full.load_signal(signal_name)
                incr_df = precomputer_incr.load_signal(signal_name)

                # Join on timestamp and ticker
                joined = full_df.join(
                    incr_df,
                    on=["timestamp", "ticker"],
                    suffix="_incr",
                )

                if len(joined) == 0:
                    continue

                # Calculate difference
                diffs = (joined["value"] - joined["value_incr"]).abs()
                max_diff = diffs.max()
                max_diffs[signal_name] = float(max_diff) if max_diff is not None else 0.0

                # Check precision
                assert max_diff <= 1e-10, f"{signal_name}: max_diff={max_diff} > 1e-10"
                print(f"✅ {signal_name}: max_diff = {max_diff:.2e} (< 1e-10)")

            print("\n✅ test_incremental_precision PASSED: All signals within tolerance")
            return max_diffs

    def test_append_to_cache(self):
        """Test that incremental data is correctly appended to cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            start_date = datetime(2025, 1, 1)
            mid_date = datetime(2025, 6, 1)
            end_date = datetime(2025, 6, 30)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # Initial data
            initial_prices = create_sample_prices(tickers, start_date, mid_date)
            precomputer.precompute_all(initial_prices, config, force=True)

            # Get initial row count
            initial_signal = precomputer.load_signal("momentum_20")
            initial_count = len(initial_signal)
            print(f"Initial signal rows: {initial_count}")

            # Extended data
            extended_prices = create_sample_prices(tickers, start_date, end_date)

            # Incremental update
            precomputer.precompute_incremental(
                extended_prices,
                start_from=mid_date,
                config=config,
            )

            # Get updated row count
            updated_signal = precomputer.load_signal("momentum_20")
            updated_count = len(updated_signal)
            print(f"Updated signal rows: {updated_count}")

            # Should have more rows after incremental update
            assert updated_count > initial_count, \
                f"Expected more rows after incremental: {updated_count} <= {initial_count}"

            # Check no duplicates in timestamps for each ticker
            dup_check = (
                updated_signal
                .group_by(["timestamp", "ticker"])
                .agg(pl.count().alias("cnt"))
                .filter(pl.col("cnt") > 1)
            )
            assert len(dup_check) == 0, f"Found duplicate entries: {dup_check}"

            print("✅ test_append_to_cache PASSED: Data correctly appended without duplicates")

    def test_get_max_lookback(self):
        """Test _get_max_lookback calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20, 60, 120, 252],
                "volatility_periods": [20, 60],
                "rsi_periods": [14],
                "zscore_periods": [20, 60],
                "sharpe_periods": [60, 252],
            }

            max_lookback = precomputer._get_max_lookback(config)
            assert max_lookback == 252, f"Expected 252, got {max_lookback}"
            print("✅ test_get_max_lookback PASSED")


def test_full_workflow():
    """Test complete workflow: full compute -> incremental -> verify precision."""
    print("\n" + "=" * 60)
    print("Full Workflow Test: precompute_all -> precompute_incremental")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        tickers = ["AAPL", "GOOGL", "MSFT"]
        start_date = datetime(2025, 1, 1)
        phase1_end = datetime(2025, 8, 31)
        phase2_end = datetime(2025, 9, 30)

        config = {
            "momentum_periods": [20, 60, 120],
            "volatility_periods": [20, 60],
            "rsi_periods": [14],
            "zscore_periods": [20],
            "sharpe_periods": [60],
        }

        # Phase 1: Full computation
        print("\n[Phase 1] Full computation...")
        prices_phase1 = create_sample_prices(tickers, start_date, phase1_end)
        precomputer.precompute_all(prices_phase1, config, force=True)

        # Verify cache created
        cached_signals = precomputer.list_cached_signals()
        print(f"Cached signals: {cached_signals}")

        # Phase 2: Incremental update
        print("\n[Phase 2] Incremental update...")
        prices_phase2 = create_sample_prices(tickers, start_date, phase2_end)

        # Validate cache to get incremental start date
        result = precomputer.validate_cache_incremental(prices_phase2, config)
        # Use CacheValidationResult attributes
        can_use = result.can_use_cache
        reason = result.reason
        incr_start = result.incremental_start
        missing_signals = result.missing_signals
        print(f"Cache validation: can_use={can_use}, reason={reason}")
        print(f"Incremental start: {incr_start}")
        if missing_signals:
            print(f"Missing signals: {missing_signals}")

        if can_use and incr_start:
            precomputer.precompute_incremental(prices_phase2, incr_start, config)
        else:
            precomputer.precompute_all(prices_phase2, config, force=True)

        # Phase 3: Verify against fresh full computation
        print("\n[Phase 3] Verify precision against fresh full computation...")
        precomputer_verify = SignalPrecomputer(cache_dir=f"{tmpdir}/verify")
        precomputer_verify.precompute_all(prices_phase2, config, force=True)

        all_passed = True
        for signal_name in cached_signals:
            incr_df = precomputer.load_signal(signal_name)
            full_df = precomputer_verify.load_signal(signal_name)

            joined = full_df.join(
                incr_df,
                on=["timestamp", "ticker"],
                suffix="_incr",
            )

            if len(joined) == 0:
                continue

            diffs = (joined["value"] - joined["value_incr"]).abs()
            max_diff = diffs.max()

            if max_diff > 1e-10:
                print(f"❌ {signal_name}: max_diff = {max_diff:.2e} > 1e-10")
                all_passed = False
            else:
                print(f"✅ {signal_name}: max_diff = {max_diff:.2e}")

        assert all_passed, "Some signals failed precision check"
        print("\n" + "=" * 60)
        print("✅ Full Workflow Test PASSED: diff < 1e-10 for all signals")
        print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing task_041_3: Incremental Signal Precomputation")
    print("=" * 60)

    tests = TestPrecomputeIncremental()

    tests.test_get_max_lookback()
    tests.test_incremental_basic()
    tests.test_append_to_cache()
    tests.test_incremental_precision()

    test_full_workflow()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
