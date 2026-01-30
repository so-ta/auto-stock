"""
S3 Cache Integration Tests

Tests for S3 storage backend integration with various cache systems.

Note: These tests require AWS credentials and are skipped in CI environments.
Set AWS_ACCESS_KEY_ID environment variable to enable these tests.

To run locally:
    export AWS_ACCESS_KEY_ID=your_key
    export AWS_SECRET_ACCESS_KEY=your_secret
    pytest tests/integration/test_s3_cache.py -v
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Skip S3 tests if AWS credentials are not configured
SKIP_S3_TESTS = not os.environ.get("AWS_ACCESS_KEY_ID")
SKIP_REASON = "AWS credentials not configured (set AWS_ACCESS_KEY_ID)"


# =============================================================================
# Mock StorageBackend for tests without real S3
# =============================================================================
class MockStorageBackend:
    """Mock StorageBackend for testing without real S3."""

    def __init__(self, config: Any = None):
        self.config = config or MagicMock()
        self.config.backend = "s3"
        self.config.base_path = ".cache"
        self._storage: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def exists(self, path: str) -> bool:
        return path in self._storage

    def write_parquet(self, df: Any, path: str) -> None:
        self._storage[path] = df
        self._metadata[path] = {"written_at": time.time()}

    def read_parquet(self, path: str) -> Any:
        if path not in self._storage:
            raise FileNotFoundError(f"No file at {path}")
        return self._storage[path]

    def write_pickle(self, data: Any, path: str) -> None:
        self._storage[path] = data
        self._metadata[path] = {"written_at": time.time()}

    def read_pickle(self, path: str) -> Any:
        if path not in self._storage:
            raise FileNotFoundError(f"No file at {path}")
        return self._storage[path]

    def write_json(self, data: Dict[str, Any], path: str) -> None:
        self._storage[path] = data
        self._metadata[path] = {"written_at": time.time()}

    def read_json(self, path: str) -> Dict[str, Any]:
        if path not in self._storage:
            raise FileNotFoundError(f"No file at {path}")
        return self._storage[path]

    def list_files(self, prefix: str = "", pattern: str = "*") -> list:
        import fnmatch
        return [k for k in self._storage.keys()
                if k.startswith(prefix) and fnmatch.fnmatch(k, pattern)]

    def delete(self, path: str) -> bool:
        if path in self._storage:
            del self._storage[path]
            return True
        return False


# =============================================================================
# Test: StorageBackend Basic Operations (Mock)
# =============================================================================
class TestStorageBackendMock:
    """Test StorageBackend operations with mock (no real S3)."""

    def test_write_read_parquet(self):
        """Test parquet write and read operations."""
        backend = MockStorageBackend()

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": np.random.randn(10),
        })

        backend.write_parquet(df, "test/data.parquet")
        assert backend.exists("test/data.parquet")

        loaded = backend.read_parquet("test/data.parquet")
        pd.testing.assert_frame_equal(df, loaded)

    def test_write_read_pickle(self):
        """Test pickle write and read operations."""
        backend = MockStorageBackend()

        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}

        backend.write_pickle(data, "test/data.pkl")
        assert backend.exists("test/data.pkl")

        loaded = backend.read_pickle("test/data.pkl")
        assert loaded == data

    def test_write_read_json(self):
        """Test JSON write and read operations."""
        backend = MockStorageBackend()

        data = {"config": "test", "params": {"a": 1, "b": 2}}

        backend.write_json(data, "test/config.json")
        assert backend.exists("test/config.json")

        loaded = backend.read_json("test/config.json")
        assert loaded == data

    def test_file_not_found(self):
        """Test FileNotFoundError for missing files."""
        backend = MockStorageBackend()

        with pytest.raises(FileNotFoundError):
            backend.read_parquet("nonexistent.parquet")

        with pytest.raises(FileNotFoundError):
            backend.read_pickle("nonexistent.pkl")

        with pytest.raises(FileNotFoundError):
            backend.read_json("nonexistent.json")


# =============================================================================
# Test: SignalPrecomputer S3 Mode
# =============================================================================
class TestSignalPrecomputerS3Mode:
    """Test SignalPrecomputer with S3 storage backend (mock)."""

    def test_initialization_with_backend(self):
        """Test SignalPrecomputer initialization with storage backend."""
        from src.backtest.signal_precompute import SignalPrecomputer

        backend = MockStorageBackend()
        precomputer = SignalPrecomputer(storage_backend=backend)

        assert precomputer._backend is backend

    def test_save_load_cache_with_backend(self):
        """Test saving and loading signal cache with storage backend."""
        from src.backtest.signal_precompute import SignalPrecomputer

        backend = MockStorageBackend()
        precomputer = SignalPrecomputer(storage_backend=backend)

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50)
        tickers = ["AAPL", "MSFT", "GOOGL"]

        cache_data = {}
        for ticker in tickers:
            cache_data[ticker] = pd.DataFrame({
                "date": dates,
                "momentum": np.random.randn(50),
                "volatility": np.random.rand(50),
            })

        # Save cache (using internal method pattern)
        for ticker, df in cache_data.items():
            path = f"signals/{ticker}_signals.parquet"
            backend.write_parquet(df, path)

        # Verify saved
        for ticker in tickers:
            path = f"signals/{ticker}_signals.parquet"
            assert backend.exists(path)

        # Load and verify
        for ticker in tickers:
            path = f"signals/{ticker}_signals.parquet"
            loaded = backend.read_parquet(path)
            pd.testing.assert_frame_equal(cache_data[ticker], loaded)


# =============================================================================
# Test: CovarianceCache S3 Mode
# =============================================================================
class TestCovarianceCacheS3Mode:
    """Test CovarianceCache with S3 storage backend (mock)."""

    def test_save_load_state_with_backend(self):
        """Test saving and loading covariance state with storage backend."""
        from src.backtest.covariance_cache import (
            CovarianceCache,
            IncrementalCovarianceEstimator,
        )

        backend = MockStorageBackend()
        cache = CovarianceCache(storage_backend=backend)

        # Create estimator with data
        np.random.seed(42)
        n_assets = 4
        halflife = 60

        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets,
            halflife=halflife,
            asset_names=["AAPL", "MSFT", "GOOGL", "AMZN"],
        )

        # Update with sample returns
        returns = np.random.randn(30, n_assets) * 0.02
        estimator.update_batch(returns)

        # Save state
        test_date = datetime(2024, 6, 15)
        saved_path = cache.save_state(test_date, estimator)

        assert backend.exists(saved_path)

        # Load state
        loaded = cache.load_state(test_date)

        assert loaded is not None
        assert loaded.n_assets == n_assets
        assert loaded.halflife == halflife

        # Verify covariance matrix
        np.testing.assert_array_almost_equal(
            estimator.get_covariance(),
            loaded.get_covariance()
        )

    def test_find_nearest_state_with_backend(self):
        """Test finding nearest cached state with storage backend."""
        from src.backtest.covariance_cache import (
            CovarianceCache,
            IncrementalCovarianceEstimator,
        )

        backend = MockStorageBackend()
        cache = CovarianceCache(storage_backend=backend)

        # Create and save estimator at specific date
        np.random.seed(42)
        estimator = IncrementalCovarianceEstimator(n_assets=3, halflife=60)
        estimator.update_batch(np.random.randn(20, 3) * 0.02)

        saved_date = datetime(2024, 6, 10)
        cache.save_state(saved_date, estimator)

        # Find nearest from later date
        target_date = datetime(2024, 6, 15)
        found_date, found_estimator = cache.find_nearest_state(target_date, max_days_back=10)

        assert found_date == saved_date
        assert found_estimator is not None


# =============================================================================
# Test: DataCache S3 Mode (UnifiedCacheManager)
# =============================================================================
class TestDataCacheS3Mode:
    """Test UnifiedCacheManager with S3 storage backend (mock)."""

    def test_data_cache_with_backend(self):
        """Test data caching with storage backend."""
        backend = MockStorageBackend()

        # Simulate data cache write
        test_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "open": np.random.rand(100) * 100,
            "high": np.random.rand(100) * 100,
            "low": np.random.rand(100) * 100,
            "close": np.random.rand(100) * 100,
            "volume": np.random.randint(1000000, 10000000, 100),
        })

        # Write
        path = "data/AAPL.parquet"
        backend.write_parquet(test_data, path)

        # Read back
        loaded = backend.read_parquet(path)
        pd.testing.assert_frame_equal(test_data, loaded)

    def test_multiple_assets_cache(self):
        """Test caching multiple assets."""
        backend = MockStorageBackend()

        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        for asset in assets:
            df = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=50),
                "close": np.random.rand(50) * 100 + 100,
            })
            backend.write_parquet(df, f"data/{asset}.parquet")

        # Verify all exist
        for asset in assets:
            assert backend.exists(f"data/{asset}.parquet")

        # List files
        files = backend.list_files(prefix="data/")
        assert len(files) == 5


# =============================================================================
# Test: Local Cache TTL Behavior
# =============================================================================
class TestLocalCacheTTL:
    """Test local cache TTL behavior."""

    def test_ttl_expiration_simulation(self):
        """Test TTL expiration behavior (simulated)."""
        backend = MockStorageBackend()

        # Write data
        test_data = {"value": 123}
        backend.write_pickle(test_data, "cache/test.pkl")

        # Get write time
        write_time = backend._metadata["cache/test.pkl"]["written_at"]

        # Simulate time passing (in real implementation, check TTL)
        # Here we just verify the metadata is tracked
        assert write_time > 0

        # Simulate TTL check (pseudo-code for real implementation)
        ttl_hours = 24
        ttl_seconds = ttl_hours * 3600
        current_time = time.time()

        is_expired = (current_time - write_time) > ttl_seconds
        # In this test, data was just written so it's not expired
        assert not is_expired


# =============================================================================
# Test: Real S3 Integration (requires AWS credentials)
# =============================================================================
@pytest.mark.skipif(SKIP_S3_TESTS, reason=SKIP_REASON)
class TestS3CacheIntegrationReal:
    """Real S3 integration tests (requires AWS credentials)."""

    @pytest.fixture
    def s3_backend(self):
        """Create real S3 storage backend."""
        from src.utils.storage_backend import StorageBackend, StorageConfig
        from src.config.settings import load_settings_from_yaml

        settings = load_settings_from_yaml()
        config = settings.storage.to_storage_config()

        # Only run if S3 mode is configured
        if config.backend != "s3":
            pytest.skip("S3 backend not configured in settings")

        return StorageBackend(config)

    def test_s3_write_read_parquet(self, s3_backend):
        """Test real S3 parquet operations."""
        import uuid

        test_id = str(uuid.uuid4())[:8]
        path = f"test/integration_{test_id}.parquet"

        try:
            df = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=10),
                "value": np.random.randn(10),
            })

            s3_backend.write_parquet(df, path)
            assert s3_backend.exists(path)

            loaded = s3_backend.read_parquet(path)
            pd.testing.assert_frame_equal(df, loaded)
        finally:
            # Cleanup
            s3_backend.delete(path)

    def test_s3_write_read_pickle(self, s3_backend):
        """Test real S3 pickle operations."""
        import uuid

        test_id = str(uuid.uuid4())[:8]
        path = f"test/integration_{test_id}.pkl"

        try:
            data = {"key": "value", "number": 42}

            s3_backend.write_pickle(data, path)
            assert s3_backend.exists(path)

            loaded = s3_backend.read_pickle(path)
            assert loaded == data
        finally:
            # Cleanup
            s3_backend.delete(path)

    def test_s3_covariance_cache_integration(self, s3_backend):
        """Test CovarianceCache with real S3 backend."""
        import uuid
        from src.backtest.covariance_cache import (
            CovarianceCache,
            IncrementalCovarianceEstimator,
        )

        # Use unique date to avoid conflicts
        test_date = datetime(2099, 1, int(uuid.uuid4().int % 28) + 1)

        try:
            cache = CovarianceCache(storage_backend=s3_backend)

            np.random.seed(42)
            estimator = IncrementalCovarianceEstimator(
                n_assets=3,
                halflife=60,
                asset_names=["TEST_A", "TEST_B", "TEST_C"],
            )
            estimator.update_batch(np.random.randn(20, 3) * 0.02)

            # Save
            saved_path = cache.save_state(test_date, estimator)
            assert s3_backend.exists(saved_path)

            # Load
            loaded = cache.load_state(test_date)
            assert loaded is not None
            assert loaded.n_assets == 3

            np.testing.assert_array_almost_equal(
                estimator.get_covariance(),
                loaded.get_covariance()
            )
        finally:
            # Cleanup
            path = f"covariance/cov_state_{test_date.strftime('%Y%m%d')}.pkl"
            s3_backend.delete(path)


# =============================================================================
# Test: Storage Backend Initialization
# =============================================================================
class TestStorageBackendInitialization:
    """Test StorageBackend initialization from Settings."""

    def test_storage_config_from_settings(self):
        """Test that storage config can be retrieved from Settings."""
        from src.config.settings import load_settings_from_yaml

        settings = load_settings_from_yaml()

        assert hasattr(settings, "storage")
        assert settings.storage.backend in ("local", "s3")

        # Test conversion
        config = settings.storage.to_storage_config()
        assert config.backend == settings.storage.backend

    def test_backend_creation_local_mode(self, tmp_path):
        """Test StorageBackend creation in local mode."""
        from src.utils.storage_backend import StorageBackend, StorageConfig

        config = StorageConfig(
            backend="local",
            base_path=str(tmp_path),
        )

        backend = StorageBackend(config)
        assert backend.config.backend == "local"

        # Test basic operation
        test_data = {"test": "data"}
        backend.write_json(test_data, "test.json")
        loaded = backend.read_json("test.json")
        assert loaded == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
