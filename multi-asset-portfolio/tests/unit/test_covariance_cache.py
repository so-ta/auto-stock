"""
Tests for CovarianceCache module.

Verifies:
1. Basic cache operations (get, put)
2. LRU eviction
3. Key generation
4. Incremental update
5. Cache statistics
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.allocation.covariance import CovarianceResult
from src.allocation.covariance_cache import (
    CovarianceCache,
    get_covariance_cache,
    reset_covariance_cache,
)


@pytest.fixture
def sample_covariance_result():
    """Create a sample CovarianceResult for testing."""
    assets = ["AAPL", "MSFT", "GOOGL"]
    n = len(assets)

    # Create a valid positive definite covariance matrix
    cov_values = np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.05, 0.02],
        [0.01, 0.02, 0.03],
    ])

    corr_values = np.corrcoef(cov_values)

    return CovarianceResult(
        covariance=pd.DataFrame(cov_values, index=assets, columns=assets),
        correlation=pd.DataFrame(corr_values, index=assets, columns=assets),
        volatilities=pd.Series([0.2, 0.22, 0.17], index=assets),
        shrinkage_intensity=0.3,
        effective_samples=252,
        method_used="ledoit_wolf",
    )


@pytest.fixture
def cache():
    """Create a fresh cache instance for each test."""
    return CovarianceCache(max_size=10)


class TestCovarianceCacheBasic:
    """Basic cache operations tests."""

    def test_put_and_get(self, cache, sample_covariance_result):
        """Test basic put and get operations."""
        universe = ["AAPL", "MSFT", "GOOGL"]
        lookback = 252
        method = "ledoit_wolf"
        end_date = datetime(2024, 1, 15)

        # Put into cache
        cache.put(universe, lookback, method, end_date, sample_covariance_result)

        # Get from cache
        result = cache.get(universe, lookback, method, end_date)

        assert result is not None
        assert result.method_used == "ledoit_wolf"
        assert result.shrinkage_intensity == 0.3

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        universe = ["AAPL", "MSFT"]
        result = cache.get(universe, 252, "ledoit_wolf", datetime(2024, 1, 15))
        assert result is None

    def test_key_generation_order_independent(self, cache, sample_covariance_result):
        """Test that universe order doesn't affect cache key."""
        universe1 = ["AAPL", "MSFT", "GOOGL"]
        universe2 = ["GOOGL", "AAPL", "MSFT"]  # Different order
        end_date = datetime(2024, 1, 15)

        cache.put(universe1, 252, "ledoit_wolf", end_date, sample_covariance_result)

        # Should hit cache with different order
        result = cache.get(universe2, 252, "ledoit_wolf", end_date)
        assert result is not None

    def test_different_parameters_different_keys(self, cache, sample_covariance_result):
        """Test that different parameters create different cache entries."""
        universe = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime(2024, 1, 15)

        cache.put(universe, 252, "ledoit_wolf", end_date, sample_covariance_result)
        cache.put(universe, 126, "ledoit_wolf", end_date, sample_covariance_result)

        assert cache.size == 2


class TestCovarianceCacheLRU:
    """LRU eviction tests."""

    def test_lru_eviction(self, sample_covariance_result):
        """Test LRU eviction when cache is full."""
        cache = CovarianceCache(max_size=3)
        universe = ["AAPL", "MSFT"]

        # Fill cache
        for i in range(3):
            end_date = datetime(2024, 1, i + 1)
            cache.put(universe, 252, "ledoit_wolf", end_date, sample_covariance_result)

        assert cache.size == 3

        # Add one more, should evict oldest
        cache.put(universe, 252, "ledoit_wolf", datetime(2024, 1, 10), sample_covariance_result)

        assert cache.size == 3
        assert cache.stats.evictions == 1

        # First entry should be evicted
        result = cache.get(universe, 252, "ledoit_wolf", datetime(2024, 1, 1))
        assert result is None

    def test_access_moves_to_end(self, sample_covariance_result):
        """Test that accessing an entry moves it to end (prevents eviction)."""
        cache = CovarianceCache(max_size=3)
        universe = ["AAPL", "MSFT"]

        # Add 3 entries
        dates = [datetime(2024, 1, i + 1) for i in range(3)]
        for date in dates:
            cache.put(universe, 252, "ledoit_wolf", date, sample_covariance_result)

        # Access first entry (moves it to end)
        cache.get(universe, 252, "ledoit_wolf", dates[0])

        # Add new entry, should evict second entry (not first)
        cache.put(universe, 252, "ledoit_wolf", datetime(2024, 1, 10), sample_covariance_result)

        # First entry should still exist
        result = cache.get(universe, 252, "ledoit_wolf", dates[0])
        assert result is not None

        # Second entry should be evicted
        result = cache.get(universe, 252, "ledoit_wolf", dates[1])
        assert result is None


class TestCovarianceCacheStats:
    """Cache statistics tests."""

    def test_hit_miss_counting(self, cache, sample_covariance_result):
        """Test hit and miss counting."""
        universe = ["AAPL", "MSFT"]
        end_date = datetime(2024, 1, 15)

        # Miss
        cache.get(universe, 252, "ledoit_wolf", end_date)
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

        # Put
        cache.put(universe, 252, "ledoit_wolf", end_date, sample_covariance_result)

        # Hit
        cache.get(universe, 252, "ledoit_wolf", end_date)
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_hit_rate_calculation(self, cache, sample_covariance_result):
        """Test hit rate calculation."""
        universe = ["AAPL", "MSFT"]
        end_date = datetime(2024, 1, 15)

        cache.put(universe, 252, "ledoit_wolf", end_date, sample_covariance_result)

        # 1 miss + 2 hits = 66.7% hit rate
        cache.get(universe, 252, "sample", end_date)  # Miss
        cache.get(universe, 252, "ledoit_wolf", end_date)  # Hit
        cache.get(universe, 252, "ledoit_wolf", end_date)  # Hit

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1
        assert abs(cache.stats.hit_rate - 0.667) < 0.01


class TestCovarianceCacheIncrementalUpdate:
    """Incremental update tests."""

    def test_incremental_update(self, cache, sample_covariance_result):
        """Test incremental covariance update."""
        universe = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime(2024, 1, 15)

        # Create new returns
        new_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            columns=universe,
        )

        new_end_date = datetime(2024, 1, 20)

        # Perform incremental update
        result = cache.incremental_update(
            universe, 252, "ledoit_wolf",
            sample_covariance_result,
            new_returns,
            new_end_date,
        )

        assert result is not None
        assert "incremental" in result.method_used
        assert result.metadata.get("incremental_update") is True
        assert cache.stats.incremental_updates == 1


class TestCovarianceCacheGlobal:
    """Global cache instance tests."""

    def test_get_global_cache(self):
        """Test getting global cache instance."""
        reset_covariance_cache()
        cache1 = get_covariance_cache()
        cache2 = get_covariance_cache()

        assert cache1 is cache2

    def test_reset_global_cache(self, sample_covariance_result):
        """Test resetting global cache."""
        reset_covariance_cache()
        cache = get_covariance_cache()

        universe = ["AAPL", "MSFT"]
        cache.put(universe, 252, "ledoit_wolf", datetime(2024, 1, 15), sample_covariance_result)

        assert cache.size == 1

        reset_covariance_cache()
        cache = get_covariance_cache()

        assert cache.size == 0


class TestCovarianceCacheFullRecalc:
    """Full recalculation detection tests."""

    def test_needs_full_recalc_when_no_cache(self, cache):
        """Test full recalc needed when no cache exists."""
        universe = ["AAPL", "MSFT"]
        result = cache.needs_full_recalculation(
            universe, 252, "ledoit_wolf", datetime(2024, 1, 15)
        )
        assert result is True

    def test_no_full_recalc_when_recent(self, cache, sample_covariance_result):
        """Test no full recalc needed when cache is recent."""
        universe = ["AAPL", "MSFT"]
        end_date = datetime(2024, 1, 15)

        cache.put(universe, 252, "ledoit_wolf", end_date, sample_covariance_result)

        result = cache.needs_full_recalculation(
            universe, 252, "ledoit_wolf", end_date
        )
        assert result is False
