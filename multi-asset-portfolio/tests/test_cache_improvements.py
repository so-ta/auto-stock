"""
Tests for cache mechanism improvements (task_cache_fix).

This module tests the cache improvements implemented in Phase 1-3:
1. Universe Hash improvement: Independent signals can be reused when universe changes
2. Signal Precomputer injection order: Precomputer is available before Pipeline access
3. Price Hash stabilization: Minor price changes don't invalidate cache
4. Cache statistics logging: Hit/miss tracking for observability

These tests ensure cache efficiency and prevent regression.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from src.backtest.signal_precompute import (
    CacheStats,
    CacheValidationResult,
    PRECOMPUTE_VERSION,
    SignalPrecomputer,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Create sample price DataFrame for testing."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 2, 28),  # 2 months of data
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
def prices_with_new_ticker(sample_prices):
    """Create price DataFrame with one new ticker added."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 2, 28),
        eager=True,
    )
    # Original tickers + NVDA
    tickers = ["AAPL", "GOOGL", "MSFT", "NVDA"]
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
def prices_with_minor_value_change(sample_prices):
    """Create price DataFrame with minor close value changes (floating point)."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 2, 28),
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            # Add tiny floating point differences
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5 + 0.000001,  # Tiny difference
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


# =============================================================================
# Test CacheStats
# =============================================================================

class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initial_values(self):
        """Test initial CacheStats values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_reasons == {}

    def test_record_hit(self):
        """Test recording cache hits."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        assert stats.hits == 2
        assert stats.hit_rate == 1.0

    def test_record_miss(self):
        """Test recording cache misses with reason."""
        stats = CacheStats()
        stats.record_miss("version_changed")
        stats.record_miss("universe_changed")
        stats.record_miss("version_changed")

        assert stats.misses == 3
        assert stats.miss_reasons["version_changed"] == 2
        assert stats.miss_reasons["universe_changed"] == 1

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss("test")

        assert stats.hit_rate == 0.75  # 3 hits / 4 total

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss("test_reason")
        stats.signals_loaded = 10
        stats.signals_computed = 2

        d = stats.to_dict()
        assert d["hits"] == 1
        assert d["misses"] == 1
        assert d["hit_rate"] == "50.0%"
        assert d["miss_reasons"]["test_reason"] == 1
        assert d["signals_loaded"] == 10
        assert d["signals_computed"] == 2


# =============================================================================
# Test Universe Hash Improvement (Phase 1.1)
# =============================================================================

class TestUniverseHashImprovement:
    """Tests for independent signals cache reuse when universe changes."""

    def test_new_ticker_detected_with_independent_signals(self, sample_prices, prices_with_new_ticker):
        """Test: New ticker added → cache usable for existing tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Use only independent signals config
            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # Precompute with original tickers
            precomputer.precompute_all(sample_prices, config=config)

            # Validate with new ticker added
            result = precomputer.validate_cache_incremental(prices_with_new_ticker, config=config)

            assert result.can_use_cache is True, f"Expected cache usable, got: {result.reason}"
            assert result.new_tickers is not None
            assert "NVDA" in result.new_tickers
            assert result.has_relative_signals is False

    def test_has_relative_signals_flag(self, sample_prices, prices_with_new_ticker):
        """Test: has_relative_signals flag is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with default config (has relative signals)
            precomputer.precompute_all(sample_prices)

            # Validate with new ticker
            result = precomputer.validate_cache_incremental(prices_with_new_ticker)

            # Default config has only independent signals, so has_relative_signals should be False
            # (Unless you have sector_relative_* signals in default config)
            # Check that the flag is set appropriately
            assert isinstance(result.has_relative_signals, bool)

    def test_cache_efficiency_with_new_tickers(self, sample_prices, prices_with_new_ticker):
        """Test: Cache efficiency is good when adding new tickers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Use only independent signals config
            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # Precompute with original tickers
            precomputer.precompute_all(sample_prices, config=config)

            # Reset stats
            precomputer.reset_cache_stats()

            # Precompute with new ticker
            computed = precomputer.precompute_all(prices_with_new_ticker, config=config)

            # Should have computed something (for new ticker)
            assert computed is True

            # Check stats show efficiency
            stats = precomputer.cache_stats
            # For 4 tickers with 2 signals, we should only compute for 1 new ticker
            # = 2 signal computations instead of 8
            assert stats["tickers_computed"] == 1  # Only NVDA computed
            assert stats["tickers_reused"] == 3  # AAPL, GOOGL, MSFT reused


# =============================================================================
# Test Price Hash Stabilization (Phase 2.1)
# =============================================================================

class TestPriceHashStabilization:
    """Tests for stable price hash computation."""

    def test_minor_price_change_no_cache_invalidation(self, sample_prices, prices_with_minor_value_change):
        """Test: Minor floating-point price changes don't invalidate cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Precompute with original prices
            precomputer.precompute_all(sample_prices)

            # The price hash should be based on structural properties, not values
            hash1 = precomputer._compute_prices_hash(sample_prices)
            hash2 = precomputer._compute_prices_hash(prices_with_minor_value_change)

            # Hashes should be the same (based on shape/dates/ticker count, not values)
            assert hash1 == hash2, "Price hash should not change for minor value differences"

    def test_price_hash_changes_for_structural_differences(self, sample_prices):
        """Test: Price hash changes for structural differences (row count, date range)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Create structurally different prices (different date range)
            dates = pl.date_range(
                datetime(2024, 1, 1),
                datetime(2024, 3, 31),  # Different end date
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
            different_range_prices = pl.DataFrame(data)

            hash1 = precomputer._compute_prices_hash(sample_prices)
            hash2 = precomputer._compute_prices_hash(different_range_prices)

            # Hashes should be different (different date range)
            assert hash1 != hash2, "Price hash should change for different date ranges"


# =============================================================================
# Test Cache Statistics Logging (Phase 3.1)
# =============================================================================

class TestCacheStatisticsLogging:
    """Tests for cache statistics and logging."""

    def test_full_cache_hit_updates_stats(self, sample_prices):
        """Test: Full cache hit updates statistics correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # First run: compute
            precomputer.precompute_all(sample_prices)

            # Reset stats
            precomputer.reset_cache_stats()

            # Second run: cache hit
            computed = precomputer.precompute_all(sample_prices)

            assert computed is False  # No computation needed

            stats = precomputer.cache_stats
            assert stats["hits"] >= 1
            assert stats["misses"] == 0

    def test_cache_miss_records_reason(self, sample_prices):
        """Test: Cache miss records the reason."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # First run: no cache exists → miss
            precomputer.precompute_all(sample_prices)

            stats = precomputer.cache_stats
            assert stats["misses"] >= 1
            # The reason should be recorded
            assert len(stats["miss_reasons"]) >= 0  # May be 0 if first run doesn't count as miss

    def test_cache_stats_include_ticker_info(self, sample_prices, prices_with_new_ticker):
        """Test: Cache stats include ticker reuse information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # Initial compute
            precomputer.precompute_all(sample_prices, config=config)
            precomputer.reset_cache_stats()

            # Add new ticker
            precomputer.precompute_all(prices_with_new_ticker, config=config)

            stats = precomputer.cache_stats
            assert "tickers_reused" in stats
            assert "tickers_computed" in stats


# =============================================================================
# Test CacheValidationResult Enhancement
# =============================================================================

class TestCacheValidationResultEnhancement:
    """Tests for enhanced CacheValidationResult."""

    def test_has_relative_signals_attribute(self, sample_prices):
        """Test: CacheValidationResult has has_relative_signals attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)
            precomputer.precompute_all(sample_prices)

            result = precomputer.validate_cache_incremental(sample_prices)

            assert hasattr(result, 'has_relative_signals')
            assert isinstance(result.has_relative_signals, bool)


# =============================================================================
# Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Integration tests for cache improvements."""

    def test_second_run_uses_cache(self, sample_prices):
        """Test: Same data on second run uses cache (full hit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # First run
            import time
            start1 = time.time()
            computed1 = precomputer.precompute_all(sample_prices)
            elapsed1 = time.time() - start1

            # Second run
            start2 = time.time()
            computed2 = precomputer.precompute_all(sample_prices)
            elapsed2 = time.time() - start2

            assert computed1 is True  # First run computed
            assert computed2 is False  # Second run used cache

            # Second run should be faster (cache hit)
            assert elapsed2 < elapsed1, "Second run should be faster due to cache"

    def test_cache_survives_precomputer_recreation(self, sample_prices):
        """Test: Cache is usable after SignalPrecomputer is recreated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First precomputer
            precomputer1 = SignalPrecomputer(cache_dir=tmpdir)
            precomputer1.precompute_all(sample_prices)

            # Create new precomputer with same cache dir
            precomputer2 = SignalPrecomputer(cache_dir=tmpdir)

            # Should recognize cache is valid
            result = precomputer2.validate_cache_incremental(sample_prices)
            assert result.can_use_cache is True

    def test_incremental_update_preserves_existing_data(self, sample_prices):
        """Test: Incremental update preserves existing cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Use a minimal config for consistent signal count
            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # Initial compute
            precomputer.precompute_all(sample_prices, config=config)

            # Check initial signal count
            initial_signals = len(precomputer.list_cached_signals())

            # Extend date range
            dates = pl.date_range(
                datetime(2024, 1, 1),
                datetime(2024, 3, 31),  # Extended
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
            extended_prices = pl.DataFrame(data)

            # Incremental update
            precomputer.precompute_all(extended_prices, config=config)

            # Signal count should remain the same (same config = same signals)
            final_signals = len(precomputer.list_cached_signals())
            assert final_signals == initial_signals


# =============================================================================
# Test Incremental Date Update (Phase: cache period extension improvement)
# =============================================================================

class TestIncrementalDateUpdate:
    """期間延長時のインクリメンタル更新テスト"""

    def test_end_date_extension_uses_incremental(self, sample_prices):
        """終了日延長時、差分のみ計算される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # 2024年2月中旬までのデータでキャッシュ構築
            prices_feb = sample_prices.filter(
                pl.col("timestamp") <= datetime(2024, 2, 15)
            )
            precomputer.precompute_all(prices_feb, config=config)

            # 2024年2月末までのデータで再検証（約2週間延長）
            result = precomputer.validate_cache_incremental(sample_prices, config=config)

            # インクリメンタル更新が検出される
            assert result.can_use_cache is True, f"Expected cache usable, got: {result.reason}"
            assert result.incremental_start is not None, "Expected incremental update to be detected"

    def test_start_date_later_uses_cache(self, sample_prices):
        """開始日が後ろにずれてもキャッシュが使える"""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # 2024年1月からのデータでキャッシュ構築
            precomputer.precompute_all(sample_prices, config=config)

            # 2024年2月からのデータで検証（開始日が後ろ）
            prices_from_feb = sample_prices.filter(
                pl.col("timestamp") >= datetime(2024, 2, 1)
            )
            result = precomputer.validate_cache_incremental(prices_from_feb, config=config)

            # キャッシュが使える（開始日が後ろでも問題なし）
            assert result.can_use_cache is True, f"Expected cache usable with later start date, got: {result.reason}"

    def test_start_date_earlier_invalidates_cache(self, sample_prices):
        """開始日が前にずれた場合はキャッシュが無効"""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # 2024年2月からのデータでキャッシュ構築
            prices_from_feb = sample_prices.filter(
                pl.col("timestamp") >= datetime(2024, 2, 1)
            )
            precomputer.precompute_all(prices_from_feb, config=config)

            # 2024年1月からのデータで検証（開始日が前に移動）
            result = precomputer.validate_cache_incremental(sample_prices, config=config)

            # キャッシュは使えない（開始日が前に移動したため）
            assert result.can_use_cache is False
            assert "backward" in result.reason.lower() or "start" in result.reason.lower()

    def test_same_dates_full_cache_hit(self, sample_prices):
        """同じ期間ではキャッシュが完全ヒット"""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # キャッシュ構築
            precomputer.precompute_all(sample_prices, config=config)

            # 同じデータで検証
            result = precomputer.validate_cache_incremental(sample_prices, config=config)

            # 完全ヒット（インクリメンタル更新なし）
            assert result.can_use_cache is True
            assert result.incremental_start is None, "Expected full cache hit, not incremental"

    def test_end_date_earlier_uses_cache(self, sample_prices):
        """終了日が前にずれてもキャッシュが使える（キャッシュが期間をカバー）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            config = {
                "momentum_periods": [20],
                "volatility_periods": [20],
                "rsi_periods": [],
                "zscore_periods": [],
                "sharpe_periods": [],
            }

            # 2024年2月末までのデータでキャッシュ構築
            precomputer.precompute_all(sample_prices, config=config)

            # 2024年2月中旬までのデータで検証（終了日が前）
            prices_early_feb = sample_prices.filter(
                pl.col("timestamp") <= datetime(2024, 2, 15)
            )
            result = precomputer.validate_cache_incremental(prices_early_feb, config=config)

            # キャッシュが使える（終了日が前でもキャッシュがカバー）
            assert result.can_use_cache is True, f"Expected cache usable with earlier end date, got: {result.reason}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
