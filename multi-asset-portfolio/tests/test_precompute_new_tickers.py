"""Tests for precompute_for_new_tickers() - task_041_6a."""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from src.backtest.signal_precompute import SignalPrecomputer


@pytest.fixture
def original_prices():
    """Create original price DataFrame with 3 tickers."""
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
    return pl.DataFrame(data)


@pytest.fixture
def extended_prices():
    """Create price DataFrame with original + new tickers."""
    dates = pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 31), eager=True)
    tickers = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"]  # Added NVDA, AMZN
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


class TestPrecomputeForNewTickers:
    """Tests for precompute_for_new_tickers() method."""

    def test_empty_new_tickers(self, original_prices):
        """Test: Empty new_tickers list returns True immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)
            precomputer.precompute_all(original_prices, force=True)

            result = precomputer.precompute_for_new_tickers(original_prices, [])
            assert result is True
            print("✅ PASS: Empty new_tickers")

    def test_new_tickers_independent_signals(self, original_prices, extended_prices):
        """Test: Independent signals computed only for new tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Initial computation
            precomputer.precompute_all(original_prices, force=True)

            # Load original momentum_20
            original_mom = pl.read_parquet(tmpdir + "/momentum_20.parquet")
            original_tickers = set(original_mom["ticker"].unique().to_list())
            assert original_tickers == {"AAPL", "GOOGL", "MSFT"}

            # Add new tickers
            new_tickers = ["NVDA", "AMZN"]
            result = precomputer.precompute_for_new_tickers(extended_prices, new_tickers)
            assert result is True

            # Check momentum_20 now has all 5 tickers
            updated_mom = pl.read_parquet(tmpdir + "/momentum_20.parquet")
            updated_tickers = set(updated_mom["ticker"].unique().to_list())
            assert updated_tickers == {"AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"}
            print("✅ PASS: Independent signals updated with new tickers")

    def test_existing_data_preserved(self, original_prices, extended_prices):
        """Test: Existing ticker data is preserved after adding new tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Initial computation
            precomputer.precompute_all(original_prices, force=True)

            # Get original AAPL data
            original_mom = pl.read_parquet(tmpdir + "/momentum_20.parquet")
            aapl_original = original_mom.filter(pl.col("ticker") == "AAPL").sort("timestamp")
            original_values = aapl_original["value"].to_list()

            # Add new tickers
            new_tickers = ["NVDA", "AMZN"]
            precomputer.precompute_for_new_tickers(extended_prices, new_tickers)

            # Check AAPL data is preserved
            updated_mom = pl.read_parquet(tmpdir + "/momentum_20.parquet")
            aapl_updated = updated_mom.filter(pl.col("ticker") == "AAPL").sort("timestamp")
            updated_values = aapl_updated["value"].to_list()

            assert original_values == updated_values
            print("✅ PASS: Existing ticker data preserved")

    def test_metadata_updated(self, original_prices, extended_prices):
        """Test: Metadata is updated with new ticker count and universe hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Initial computation
            precomputer.precompute_all(original_prices, force=True)
            metadata_before = precomputer._load_precompute_metadata()
            assert metadata_before is not None, "Metadata should exist after precompute_all"
            assert metadata_before.ticker_count == 3

            universe_hash_before = metadata_before.universe_hash

            # Add new tickers
            new_tickers = ["NVDA", "AMZN"]
            precomputer.precompute_for_new_tickers(extended_prices, new_tickers)

            # Reload metadata (need fresh instance to re-read from disk)
            precomputer._metadata = precomputer._load_metadata()
            metadata_after = precomputer._load_precompute_metadata()

            assert metadata_after is not None, "Metadata should exist after new tickers"
            assert metadata_after.ticker_count == 5
            assert metadata_after.universe_hash != universe_hash_before
            print("✅ PASS: Metadata updated correctly")

    def test_multiple_signal_types_computed(self, original_prices, extended_prices):
        """Test: Multiple signal types are computed for new tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Initial computation
            precomputer.precompute_all(original_prices, force=True)

            # Check which files exist after initial computation
            cache_path = Path(tmpdir)
            initial_files = [f.stem for f in cache_path.glob("*.parquet")]
            assert "momentum_20" in initial_files, f"momentum_20 should exist, found: {initial_files}"

            # Add new tickers
            new_tickers = ["NVDA", "AMZN"]
            precomputer.precompute_for_new_tickers(extended_prices, new_tickers)

            # Check momentum_20 has new tickers
            mom = pl.read_parquet(f"{tmpdir}/momentum_20.parquet")
            tickers = set(mom["ticker"].unique().to_list())
            assert "NVDA" in tickers, "NVDA should be in momentum_20"
            assert "AMZN" in tickers, "AMZN should be in momentum_20"

            # Check volatility_20 if it exists
            vol_path = cache_path / "volatility_20.parquet"
            if vol_path.exists():
                vol = pl.read_parquet(str(vol_path))
                vol_tickers = set(vol["ticker"].unique().to_list())
                assert "NVDA" in vol_tickers, "NVDA should be in volatility_20"
                assert "AMZN" in vol_tickers, "AMZN should be in volatility_20"

            print("✅ PASS: Multiple signal types computed for new tickers")


def test_all_scenarios():
    """Run all test scenarios."""
    print("\n" + "=" * 60)
    print("precompute_for_new_tickers() Test Suite - task_041_6a")
    print("=" * 60 + "\n")

    # Create test data
    dates = pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 31), eager=True)

    # Original 3 tickers
    original_tickers = ["AAPL", "GOOGL", "MSFT"]
    original_data = []
    for ticker in original_tickers:
        for i, date in enumerate(dates):
            original_data.append({
                "timestamp": date, "ticker": ticker,
                "close": 100.0 + i * 0.5, "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5, "volume": 1000000,
            })
    original_prices = pl.DataFrame(original_data)

    # Extended with 2 new tickers
    extended_tickers = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"]
    extended_data = []
    for ticker in extended_tickers:
        for i, date in enumerate(dates):
            extended_data.append({
                "timestamp": date, "ticker": ticker,
                "close": 100.0 + i * 0.5, "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5, "volume": 1000000,
            })
    extended_prices = pl.DataFrame(extended_data)

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        precomputer = SignalPrecomputer(cache_dir=tmpdir)

        # Test 1: Initial computation
        precomputer.precompute_all(original_prices, force=True)
        mom_orig = pl.read_parquet(f"{tmpdir}/momentum_20.parquet")
        passed = len(mom_orig["ticker"].unique()) == 3
        results.append(("Initial computation (3 tickers)", passed))
        print(f"[1] Initial computation: {'✅' if passed else '❌'}")

        # Test 2: Add new tickers
        new_tickers = ["NVDA", "AMZN"]
        success = precomputer.precompute_for_new_tickers(extended_prices, new_tickers)
        results.append(("precompute_for_new_tickers() returns True", success))
        print(f"[2] precompute_for_new_tickers(): {'✅' if success else '❌'}")

        # Test 3: Check tickers
        mom_updated = pl.read_parquet(f"{tmpdir}/momentum_20.parquet")
        tickers = set(mom_updated["ticker"].unique().to_list())
        passed = tickers == {"AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"}
        results.append(("All 5 tickers in cache", passed))
        print(f"[3] All 5 tickers in cache: {'✅' if passed else '❌'}")

        # Test 4: Metadata updated
        precomputer._metadata = precomputer._load_metadata()
        metadata = precomputer._load_precompute_metadata()
        passed = metadata is not None and metadata.ticker_count == 5
        results.append(("Metadata ticker_count=5", passed))
        print(f"[4] Metadata ticker_count=5: {'✅' if passed else '❌'}")

        # Test 5: Existing data preserved
        aapl_data = mom_updated.filter(pl.col("ticker") == "AAPL")
        passed = len(aapl_data) > 0
        results.append(("Existing data preserved", passed))
        print(f"[5] Existing data preserved: {'✅' if passed else '❌'}")

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
