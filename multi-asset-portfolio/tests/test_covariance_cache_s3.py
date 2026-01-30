"""
Tests for CovarianceCache with StorageBackend (S3 integration).

Verifies:
1. Backward compatibility (local mode with cache_dir)
2. StorageBackend mode operations (save_state, load_state)
3. State persistence and restoration via backend
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backtest.covariance_cache import (
    CovarianceCache,
    CovarianceState,
    IncrementalCovarianceEstimator,
    create_estimator_from_history,
)


@pytest.fixture
def sample_estimator():
    """Create a sample estimator with some data."""
    np.random.seed(42)
    n_assets = 4
    n_days = 50
    halflife = 60

    returns = np.random.randn(n_days, n_assets) * 0.02
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    estimator = IncrementalCovarianceEstimator(
        n_assets=n_assets,
        halflife=halflife,
        asset_names=asset_names,
    )
    estimator.update_batch(returns)

    return estimator


@pytest.fixture
def mock_storage_backend():
    """Create a mock StorageBackend for testing."""
    backend = MagicMock()
    backend._storage: Dict[str, Any] = {}

    def mock_exists(path: str) -> bool:
        return path in backend._storage

    def mock_write_pickle(data: Any, path: str) -> None:
        backend._storage[path] = data

    def mock_read_pickle(path: str) -> Any:
        if path not in backend._storage:
            raise FileNotFoundError(f"No file at {path}")
        return backend._storage[path]

    backend.exists = mock_exists
    backend.write_pickle = mock_write_pickle
    backend.read_pickle = mock_read_pickle

    return backend


class TestCovarianceCacheBackwardCompatibility:
    """Test backward compatibility with local mode."""

    def test_local_mode_default(self, tmp_path):
        """Test default local mode with cache_dir."""
        cache_dir = tmp_path / "covariance_cache"
        cache = CovarianceCache(cache_dir=str(cache_dir))

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()
        assert cache._use_backend is False

    def test_local_mode_save_load(self, tmp_path, sample_estimator):
        """Test save and load in local mode."""
        cache_dir = tmp_path / "covariance_cache"
        cache = CovarianceCache(cache_dir=str(cache_dir))

        test_date = datetime(2024, 1, 15)

        # Save state
        saved_path = cache.save_state(test_date, sample_estimator)
        assert Path(saved_path).exists()

        # Load state
        loaded = cache.load_state(test_date)
        assert loaded is not None
        assert loaded.n_assets == sample_estimator.n_assets
        assert loaded.halflife == sample_estimator.halflife
        assert loaded.n_updates == sample_estimator.n_updates

        # Verify covariance matrices match
        original_cov = sample_estimator.get_covariance()
        loaded_cov = loaded.get_covariance()
        np.testing.assert_array_almost_equal(original_cov, loaded_cov)

    def test_local_mode_cache_miss(self, tmp_path):
        """Test cache miss returns None."""
        cache_dir = tmp_path / "covariance_cache"
        cache = CovarianceCache(cache_dir=str(cache_dir))

        test_date = datetime(2024, 1, 15)
        result = cache.load_state(test_date)

        assert result is None


class TestCovarianceCacheStorageBackend:
    """Test StorageBackend mode."""

    def test_backend_mode_initialization(self, mock_storage_backend):
        """Test initialization with storage_backend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        assert cache._use_backend is True
        assert cache._backend is mock_storage_backend
        assert cache._cache_subdir == "covariance"

    def test_backend_mode_save_state(self, mock_storage_backend, sample_estimator):
        """Test save_state with StorageBackend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        test_date = datetime(2024, 1, 15)
        saved_path = cache.save_state(test_date, sample_estimator)

        # Verify path format
        assert saved_path == "covariance/cov_state_20240115.pkl"

        # Verify data was stored
        assert mock_storage_backend.exists(saved_path)
        stored_state = mock_storage_backend._storage[saved_path]
        assert isinstance(stored_state, CovarianceState)
        assert stored_state.n_assets == sample_estimator.n_assets

    def test_backend_mode_load_state(self, mock_storage_backend, sample_estimator):
        """Test load_state with StorageBackend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        test_date = datetime(2024, 1, 15)

        # Save first
        cache.save_state(test_date, sample_estimator)

        # Load
        loaded = cache.load_state(test_date)

        assert loaded is not None
        assert loaded.n_assets == sample_estimator.n_assets
        assert loaded.halflife == sample_estimator.halflife

        # Verify data integrity
        original_cov = sample_estimator.get_covariance()
        loaded_cov = loaded.get_covariance()
        np.testing.assert_array_almost_equal(original_cov, loaded_cov)

    def test_backend_mode_cache_miss(self, mock_storage_backend):
        """Test cache miss with StorageBackend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        test_date = datetime(2024, 1, 15)
        result = cache.load_state(test_date)

        assert result is None

    def test_backend_mode_multiple_dates(self, mock_storage_backend, sample_estimator):
        """Test saving/loading multiple dates with StorageBackend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        dates = [
            datetime(2024, 1, 10),
            datetime(2024, 1, 15),
            datetime(2024, 1, 20),
        ]

        # Save multiple states
        for date in dates:
            cache.save_state(date, sample_estimator)

        # Verify all can be loaded
        for date in dates:
            loaded = cache.load_state(date)
            assert loaded is not None
            assert loaded.n_assets == sample_estimator.n_assets

        # Verify storage contains correct keys
        assert len(mock_storage_backend._storage) == 3


class TestCovarianceCacheFindNearest:
    """Test find_nearest_state functionality."""

    def test_find_nearest_local_mode(self, tmp_path, sample_estimator):
        """Test find_nearest_state in local mode."""
        cache_dir = tmp_path / "covariance_cache"
        cache = CovarianceCache(cache_dir=str(cache_dir))

        # Save state at specific date
        saved_date = datetime(2024, 1, 10)
        cache.save_state(saved_date, sample_estimator)

        # Find nearest from later date
        target_date = datetime(2024, 1, 15)
        found_date, found_estimator = cache.find_nearest_state(target_date, max_days_back=10)

        assert found_date == saved_date
        assert found_estimator is not None

    def test_find_nearest_backend_mode(self, mock_storage_backend, sample_estimator):
        """Test find_nearest_state with StorageBackend."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        # Save state at specific date
        saved_date = datetime(2024, 1, 10)
        cache.save_state(saved_date, sample_estimator)

        # Find nearest from later date
        target_date = datetime(2024, 1, 15)
        found_date, found_estimator = cache.find_nearest_state(target_date, max_days_back=10)

        assert found_date == saved_date
        assert found_estimator is not None

    def test_find_nearest_not_found(self, mock_storage_backend):
        """Test find_nearest_state when no cache exists."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        target_date = datetime(2024, 1, 15)
        found_date, found_estimator = cache.find_nearest_state(target_date, max_days_back=10)

        assert found_date is None
        assert found_estimator is None


class TestCovarianceCacheStatePersistence:
    """Test state persistence and data integrity."""

    def test_state_preservation_across_save_load(self, mock_storage_backend):
        """Test that all state data is preserved across save/load cycle."""
        np.random.seed(123)
        n_assets = 5
        halflife = 30
        asset_names = ["A", "B", "C", "D", "E"]

        # Create estimator with specific data
        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets,
            halflife=halflife,
            asset_names=asset_names,
        )

        # Update with multiple batches
        for _ in range(10):
            returns = np.random.randn(n_assets) * 0.01
            estimator.update(returns)

        # Save state
        cache = CovarianceCache(storage_backend=mock_storage_backend)
        test_date = datetime(2024, 6, 15)
        cache.save_state(test_date, estimator)

        # Load state into new estimator
        loaded = cache.load_state(test_date)

        # Verify all attributes preserved
        assert loaded.n_assets == n_assets
        assert loaded.halflife == halflife
        assert loaded.n_updates == estimator.n_updates
        assert loaded.asset_names == asset_names

        # Verify matrices preserved
        np.testing.assert_array_almost_equal(
            estimator.get_covariance(),
            loaded.get_covariance()
        )
        np.testing.assert_array_almost_equal(
            estimator.get_correlation(),
            loaded.get_correlation()
        )
        np.testing.assert_array_almost_equal(
            estimator.get_volatility(),
            loaded.get_volatility()
        )

    def test_continue_updates_after_load(self, mock_storage_backend):
        """Test that loaded estimator can continue receiving updates."""
        np.random.seed(456)
        n_assets = 3
        halflife = 20

        # Create and update estimator
        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets,
            halflife=halflife,
        )
        initial_returns = np.random.randn(20, n_assets) * 0.01
        estimator.update_batch(initial_returns)

        # Save
        cache = CovarianceCache(storage_backend=mock_storage_backend)
        cache.save_state(datetime(2024, 1, 1), estimator)

        # Load
        loaded = cache.load_state(datetime(2024, 1, 1))

        # Continue updating loaded estimator
        new_returns = np.random.randn(10, n_assets) * 0.01
        loaded.update_batch(new_returns)

        # Verify updates worked
        assert loaded.n_updates == estimator.n_updates + 10


class TestCovarianceCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_backend_priority_over_cache_dir(self, mock_storage_backend, tmp_path):
        """Test that storage_backend takes priority when both are specified."""
        cache_dir = tmp_path / "should_not_use"
        cache = CovarianceCache(
            cache_dir=str(cache_dir),
            storage_backend=mock_storage_backend,
        )

        assert cache._use_backend is True
        assert not cache_dir.exists()  # Should not create local dir

    def test_default_cache_dir(self):
        """Test default cache_dir when neither argument specified."""
        cache = CovarianceCache()

        assert cache._use_backend is False
        assert cache.cache_dir == Path(".cache/covariance")

    def test_load_corrupted_state_returns_none(self, mock_storage_backend):
        """Test that corrupted state returns None gracefully."""
        cache = CovarianceCache(storage_backend=mock_storage_backend)

        # Store invalid data directly
        mock_storage_backend._storage["covariance/cov_state_20240115.pkl"] = "invalid data"

        # Override read_pickle to raise exception for invalid data
        original_read = mock_storage_backend.read_pickle
        def mock_read(path):
            data = mock_storage_backend._storage.get(path)
            if data == "invalid data":
                raise Exception("Invalid pickle data")
            return data
        mock_storage_backend.read_pickle = mock_read

        # Should return None, not raise exception
        result = cache.load_state(datetime(2024, 1, 15))
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
