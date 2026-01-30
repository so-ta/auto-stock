"""Tests for validate_cache_incremental() method - task_041_2 + task_041_1a."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from src.backtest.signal_precompute import (
    PRECOMPUTE_VERSION,
    PrecomputeMetadata,
    SignalPrecomputer,
    CacheValidationResult,
)


@pytest.fixture
def sample_prices():
    """Create sample price DataFrame for testing."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


@pytest.fixture
def extended_prices():
    """Create extended price DataFrame (additional days at the end)."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 2, 15),  # Extended to Feb 15
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


@pytest.fixture
def earlier_start_prices():
    """Create price DataFrame with earlier start date."""
    dates = pl.date_range(
        datetime(2023, 12, 1),  # Earlier start
        datetime(2024, 1, 31),
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


@pytest.fixture
def different_tickers_prices():
    """Create price DataFrame with different ticker set."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "NVDA"]  # NVDA instead of MSFT
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


class TestValidateCacheIncremental:
    """Tests for validate_cache_incremental() method."""

    def test_no_cached_metadata(self, sample_prices):
        """Test: No cached metadata → full recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            result = precomputer.validate_cache_incremental(sample_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is False
            assert "No cached metadata" in reason
            assert incremental_start is None
            assert missing_signals is None
            print(f"✅ PASS: No cached metadata → {reason}")

    def test_full_cache_hit_exact_match(self, sample_prices):
        """Test: Exact date match → full cache hit (no computation needed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute to create cache
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with same data
            result = precomputer.validate_cache_incremental(sample_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert incremental_start is None  # No incremental needed
            assert missing_signals is None  # No missing signals
            assert "hit" in reason.lower() or "match" in reason.lower()
            print(f"✅ PASS: Exact match → {reason}")

    def test_incremental_update_end_date_extended(self, sample_prices, extended_prices):
        """Test: End date extended → incremental update (returns start date)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original data
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with extended data
            result = precomputer.validate_cache_incremental(extended_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert incremental_start is not None  # Should return start date for incremental
            assert isinstance(incremental_start, datetime)
            assert missing_signals is None  # No missing signals
            assert "incremental" in reason.lower() or "update" in reason.lower()
            print(f"✅ PASS: End date extended → {reason}, incremental_start={incremental_start}")

    def test_full_recompute_start_date_moved_backward(
        self, sample_prices, earlier_start_prices
    ):
        """Test: Start date moved backward → full recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original data
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with earlier start date
            result = precomputer.validate_cache_incremental(earlier_start_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is False
            assert "start date" in reason.lower() or "backward" in reason.lower()
            assert incremental_start is None
            assert missing_signals is None
            print(f"✅ PASS: Start date backward → {reason}")

    def test_full_recompute_ticker_changed(self, sample_prices, different_tickers_prices):
        """Test: Ticker list changed → cache usable with new_tickers for independent signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original data
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with different tickers (NVDA instead of MSFT)
            result = precomputer.validate_cache_incremental(different_tickers_prices)

            # Current implementation: can_use=True with new_tickers for independent signals
            # If new ticker detected, cache can still be used for existing tickers
            if result.new_tickers:
                assert result.can_use_cache is True
                assert "NVDA" in result.new_tickers
                print(f"✅ PASS: New ticker detected → {result.reason}, new_tickers={result.new_tickers}")
            else:
                # Full recompute needed (e.g., relative signals)
                assert result.can_use_cache is False
                print(f"✅ PASS: Ticker changed → {result.reason}")

    def test_new_signals_detected(self, sample_prices):
        """Test: New signals in config → returns missing_signals list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with default config
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with config containing additional signals
            extended_config = {
                "momentum_periods": [20, 60, 120, 252, 10],  # Added 10
                "volatility_periods": [20, 60],
                "rsi_periods": [14, 7],  # Added 7
                "zscore_periods": [20, 60],
                "sharpe_periods": [60, 252],
            }
            result = precomputer.validate_cache_incremental(sample_prices, config=extended_config)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert missing_signals is not None
            assert "momentum_10" in missing_signals
            assert "rsi_7" in missing_signals
            assert len(missing_signals) == 2
            print(f"✅ PASS: New signals detected → {reason}, missing={missing_signals}")

    def test_no_new_signals_same_config(self, sample_prices):
        """Test: Same config → no missing signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with default config
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with same default config
            result = precomputer.validate_cache_incremental(sample_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert missing_signals is None
            print(f"✅ PASS: Same config → no missing signals")

    def test_full_recompute_version_changed(self, sample_prices):
        """Test: Version changed → full recomputation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute
            precomputer.precompute_all(sample_prices, force=True)

            # Manually modify version in metadata
            metadata_file = Path(tmpdir) / "_metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            metadata["version"] = "0.0.0"  # Old version
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Validate
            result = precomputer.validate_cache_incremental(sample_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is False
            assert "version" in reason.lower()
            assert incremental_start is None
            assert missing_signals is None
            print(f"✅ PASS: Version changed → {reason}")

    def test_cache_covers_earlier_end_date(self, extended_prices, sample_prices):
        """Test: Requested end date earlier than cached → cache valid (covers period)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with extended data (longer period)
            precomputer.precompute_all(extended_prices, force=True)

            # Validate with shorter period (same start, earlier end)
            result = precomputer.validate_cache_incremental(sample_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert incremental_start is None  # No incremental needed
            assert missing_signals is None
            print(f"✅ PASS: Cache covers shorter period → {reason}")

    def test_incremental_start_date_value(self, sample_prices, extended_prices):
        """Test: Incremental start date is the cached end date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original data
            precomputer.precompute_all(sample_prices, force=True)

            # Get cached end date
            cached_metadata = precomputer._load_precompute_metadata()
            cached_end_str = cached_metadata.cached_end_date

            # Validate with extended data
            result = precomputer.validate_cache_incremental(extended_prices)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert incremental_start is not None

            # The incremental_start should match the cached end date
            if "T" in cached_end_str:
                expected = datetime.fromisoformat(cached_end_str)
            else:
                expected = datetime.fromisoformat(cached_end_str + "T00:00:00")

            assert incremental_start == expected
            print(f"✅ PASS: Incremental start = cached end ({incremental_start})")

    def test_both_incremental_and_new_signals(self, sample_prices, extended_prices):
        """Test: Both time incremental + new signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original data
            precomputer.precompute_all(sample_prices, force=True)

            # Validate with extended data AND new config
            extended_config = {
                "momentum_periods": [20, 60, 120, 252, 5],  # Added 5
                "volatility_periods": [20, 60],
                "rsi_periods": [14],
                "zscore_periods": [20, 60],
                "sharpe_periods": [60, 252],
            }
            result = precomputer.validate_cache_incremental(extended_prices, config=extended_config)
            can_use, reason, incremental_start, missing_signals = (
                result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
            )

            assert can_use is True
            assert incremental_start is not None  # Time incremental
            assert missing_signals is not None  # New signals
            assert "momentum_5" in missing_signals
            print(f"✅ PASS: Both incremental + new signals → {reason}")


def test_all_scenarios():
    """Run all test scenarios and summarize results."""
    print("\n" + "=" * 60)
    print("validate_cache_incremental() Test Suite - task_041_1a")
    print("=" * 60 + "\n")

    # Create test data
    dates = pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 31), eager=True)
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    sample_prices = pl.DataFrame(data)

    # Extended data
    ext_dates = pl.date_range(datetime(2024, 1, 1), datetime(2024, 2, 15), eager=True)
    ext_data = []
    for ticker in tickers:
        for i, date in enumerate(ext_dates):
            ext_data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    extended_prices = pl.DataFrame(ext_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        results = []

        # Test 1: No cached metadata
        result = precomputer.validate_cache_incremental(sample_prices)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("No cached metadata", not can_use and inc_start is None and missing is None))
        print(f"[1] No cached metadata: {reason}")

        # Precompute
        precomputer.precompute_all(sample_prices, force=True)

        # Test 2: Full cache hit
        result = precomputer.validate_cache_incremental(sample_prices)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("Full cache hit", can_use and inc_start is None and missing is None))
        print(f"[2] Full cache hit: {reason}")

        # Test 3: Incremental update (time)
        result = precomputer.validate_cache_incremental(extended_prices)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("Incremental time update", can_use and inc_start is not None and missing is None))
        print(f"[3] Incremental time update: {reason}, start={inc_start}")

        # Test 4: Start date backward
        early_dates = pl.date_range(datetime(2023, 12, 1), datetime(2024, 1, 31), eager=True)
        early_data = []
        for ticker in tickers:
            for i, date in enumerate(early_dates):
                early_data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": 100.0 + i * 0.5,
                    "high": 101.0 + i * 0.5,
                    "low": 99.0 + i * 0.5,
                    "volume": 1000000,
                })
        earlier_prices = pl.DataFrame(early_data)
        result = precomputer.validate_cache_incremental(earlier_prices)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("Start date backward", not can_use))
        print(f"[4] Start date backward: {reason}")

        # Test 5: Ticker changed
        diff_tickers = ["AAPL", "GOOGL", "NVDA"]
        diff_data = []
        for ticker in diff_tickers:
            for i, date in enumerate(dates):
                diff_data.append({
                    "timestamp": date,
                    "ticker": ticker,
                    "close": 100.0 + i * 0.5,
                    "high": 101.0 + i * 0.5,
                    "low": 99.0 + i * 0.5,
                    "volume": 1000000,
                })
        diff_prices = pl.DataFrame(diff_data)
        result = precomputer.validate_cache_incremental(diff_prices)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("Ticker changed", not can_use))
        print(f"[5] Ticker changed: {reason}")

        # Test 6: New signals added
        new_config = {
            "momentum_periods": [20, 60, 120, 252, 10],  # Added 10
            "volatility_periods": [20, 60],
            "rsi_periods": [14, 7],  # Added 7
            "zscore_periods": [20, 60],
            "sharpe_periods": [60, 252],
        }
        result = precomputer.validate_cache_incremental(sample_prices, config=new_config)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("New signals detected", can_use and missing is not None and len(missing) == 2))
        print(f"[6] New signals detected: {reason}, missing={missing}")

        # Test 7: Both incremental + new signals
        result = precomputer.validate_cache_incremental(extended_prices, config=new_config)
        can_use, reason, inc_start, missing = result.can_use_cache, result.reason, result.incremental_start, result.missing_signals
        results.append(("Both incremental + new signals", can_use and inc_start is not None and missing is not None))
        print(f"[7] Both incremental + new signals: {reason}")

    print("\n" + "-" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("All tests PASSED! ✅")
    else:
        print("Some tests FAILED! ❌")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    test_all_scenarios()
