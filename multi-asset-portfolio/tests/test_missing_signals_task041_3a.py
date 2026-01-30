"""
Test script for task_041_3a: precompute_missing_signals()

Verifies:
1. Only missing signals are computed
2. Existing cache is preserved
3. Metadata is correctly updated
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.signal_precompute import SignalPrecomputer


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


class TestPrecomputeMissingSignals:
    """Test precompute_missing_signals functionality."""

    def test_compute_missing_only(self):
        """Test that only missing signals are computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY", "QQQ"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 6, 30)

            prices = create_sample_prices(tickers, start_date, end_date)

            # Initial computation with limited signals
            initial_config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }
            precomputer.precompute_all(prices, initial_config, force=True)

            # Get initial cached signals
            initial_signals = precomputer.list_cached_signals()
            print(f"Initial signals: {initial_signals}")
            assert "momentum_20" in initial_signals
            assert "volatility_20" in initial_signals
            assert "rsi_14" not in initial_signals

            # Get initial momentum data for verification
            initial_momentum = precomputer.load_signal("momentum_20")
            initial_momentum_hash = hash(initial_momentum["value"].to_list().__str__())

            # Now compute missing signals
            missing = ["rsi_14", "zscore_20"]
            success = precomputer.precompute_missing_signals(prices, missing)

            assert success, "precompute_missing_signals should succeed"

            # Verify new signals exist
            updated_signals = precomputer.list_cached_signals()
            print(f"Updated signals: {updated_signals}")
            assert "rsi_14" in updated_signals
            assert "zscore_20" in updated_signals

            # Verify existing signals unchanged
            updated_momentum = precomputer.load_signal("momentum_20")
            updated_momentum_hash = hash(updated_momentum["value"].to_list().__str__())
            assert initial_momentum_hash == updated_momentum_hash, \
                "Existing signal should be unchanged"

            print("✅ test_compute_missing_only PASSED")

    def test_existing_cache_preserved(self):
        """Test that existing cache is preserved when computing missing signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["AAPL", "GOOGL", "MSFT"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 6, 30)

            prices = create_sample_prices(tickers, start_date, end_date)

            # Initial computation
            initial_config = {
                "momentum_periods": [20, 60],
                "volatility_periods": [20],
                "rsi_periods": [14],
                "zscore_periods": [],
                "sharpe_periods": [],
            }
            precomputer.precompute_all(prices, initial_config, force=True)

            # Store original file modification times
            cache_dir = Path(tmpdir)
            original_mtimes = {}
            for f in cache_dir.glob("*.parquet"):
                original_mtimes[f.name] = f.stat().st_mtime

            print(f"Original files: {list(original_mtimes.keys())}")

            # Add small delay to ensure mtime difference is detectable
            import time
            time.sleep(0.1)

            # Compute missing signals
            missing = ["sharpe_60", "zscore_20"]
            precomputer.precompute_missing_signals(prices, missing)

            # Check file modification times
            for f in cache_dir.glob("*.parquet"):
                if f.name in ["sharpe_60.parquet", "zscore_20.parquet"]:
                    # New files should exist
                    assert f.exists(), f"New signal file {f.name} should exist"
                    print(f"✅ New signal created: {f.name}")
                elif f.name in original_mtimes:
                    # Existing files should NOT be modified
                    current_mtime = f.stat().st_mtime
                    assert current_mtime == original_mtimes[f.name], \
                        f"Existing file {f.name} should not be modified"
                    print(f"✅ Existing signal preserved: {f.name}")

            print("✅ test_existing_cache_preserved PASSED")

    def test_metadata_updated(self):
        """Test that metadata is correctly updated after computing missing signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 6, 30)

            prices = create_sample_prices(tickers, start_date, end_date)

            # Initial computation
            initial_config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }
            precomputer.precompute_all(prices, initial_config, force=True)

            # Get initial metadata
            initial_metadata = precomputer._load_precompute_metadata()
            initial_signals = precomputer._metadata.get("signals", [])
            print(f"Initial signals in metadata: {initial_signals}")

            # Compute missing signals
            missing = ["rsi_14", "volatility_60"]
            precomputer.precompute_missing_signals(prices, missing)

            # Check updated metadata
            updated_signals = precomputer._metadata.get("signals", [])
            print(f"Updated signals in metadata: {updated_signals}")

            assert "rsi_14" in updated_signals
            assert "volatility_60" in updated_signals
            assert "momentum_20" in updated_signals  # Original should still be there

            print("✅ test_metadata_updated PASSED")

    def test_empty_missing_signals(self):
        """Test handling of empty missing signals list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 3, 31)

            prices = create_sample_prices(tickers, start_date, end_date)

            # Initial computation
            initial_config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }
            precomputer.precompute_all(prices, initial_config, force=True)

            # Call with empty list
            success = precomputer.precompute_missing_signals(prices, [])
            assert success, "Should return True for empty list"

            print("✅ test_empty_missing_signals PASSED")

    def test_signal_name_parsing(self):
        """Test signal name parsing in _compute_and_save_signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            tickers = ["SPY"]
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 6, 30)

            prices = create_sample_prices(tickers, start_date, end_date)

            # Test various signal types
            test_signals = [
                "momentum_120",
                "volatility_30",
                "rsi_21",
                "zscore_50",
                "sharpe_90",
            ]

            success = precomputer.precompute_missing_signals(prices, test_signals)
            assert success, "All signals should compute successfully"

            # Verify all signals exist
            cached = precomputer.list_cached_signals()
            for signal in test_signals:
                assert signal in cached, f"Signal {signal} should be cached"

            print("✅ test_signal_name_parsing PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing task_041_3a: precompute_missing_signals()")
    print("=" * 60)

    tests = TestPrecomputeMissingSignals()

    tests.test_empty_missing_signals()
    tests.test_signal_name_parsing()
    tests.test_compute_missing_only()
    tests.test_existing_cache_preserved()
    tests.test_metadata_updated()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
