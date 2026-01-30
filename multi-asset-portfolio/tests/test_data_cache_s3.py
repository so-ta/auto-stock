"""
Test DataCache with StorageBackend integration (task_045_3).

Tests:
1. DataCache with storage_backend parameter
2. Backward compatibility (cache_dir only)
3. DuckDB + storage_backend validation
4. S3 mode read/write operations (mocked)
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data.cache import CacheBackend, DataCache
from src.data.adapters import AssetType, DataFrequency, OHLCVData


class TestDataCacheStorageBackendInit:
    """Test DataCache initialization with storage_backend."""

    def test_init_without_storage_backend(self, tmp_path):
        """Test DataCache with cache_dir only (backward compatibility)."""
        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            max_age_days=1,
        )

        assert cache.cache_dir == tmp_path / "cache"
        assert cache.backend == CacheBackend.PARQUET
        assert cache._storage_backend is None
        assert cache.cache_dir.exists()

    def test_init_with_storage_backend(self, tmp_path):
        """Test DataCache with storage_backend."""
        mock_backend = MagicMock()
        mock_backend.exists.return_value = False

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            max_age_days=1,
            storage_backend=mock_backend,
        )

        assert cache._storage_backend is mock_backend

    def test_duckdb_with_storage_backend_raises(self, tmp_path):
        """Test that DuckDB + storage_backend raises ValueError."""
        mock_backend = MagicMock()

        with pytest.raises(ValueError, match="DuckDB backend is not supported"):
            DataCache(
                cache_dir=tmp_path / "cache",
                backend=CacheBackend.DUCKDB,
                storage_backend=mock_backend,
            )


class TestDataCacheStorageBackendOperations:
    """Test DataCache operations with storage_backend."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCVData."""
        df = pl.DataFrame({
            "date": pl.date_range(
                datetime(2024, 1, 1),
                datetime(2024, 1, 10),
                eager=True
            ),
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000000 + i * 100000 for i in range(10)],
        })

        return OHLCVData(
            symbol="SPY",
            asset_type=AssetType.STOCK,
            frequency=DataFrequency.DAILY,
            data=df,
            source="test",
            fetched_at=datetime.now(timezone.utc),
            adjusted=True,
        )

    def test_put_with_storage_backend(self, tmp_path, sample_ohlcv_data):
        """Test put() uses storage_backend for write."""
        mock_backend = MagicMock()

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            storage_backend=mock_backend,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        cache.put(sample_ohlcv_data, start, end)

        # Verify storage_backend methods were called
        assert mock_backend.write_parquet.called
        assert mock_backend.write_json.called

        # Verify parquet was written with correct path pattern
        parquet_call = mock_backend.write_parquet.call_args
        assert parquet_call[0][1].endswith(".parquet")

        # Verify json was written with correct path pattern
        json_call = mock_backend.write_json.call_args
        assert json_call[0][1].endswith(".meta.json")

    def test_get_with_storage_backend_cache_hit(self, tmp_path, sample_ohlcv_data):
        """Test get() uses storage_backend for read (cache hit)."""
        mock_backend = MagicMock()

        # Mock exists to return True
        mock_backend.exists.return_value = True

        # Mock read_json to return valid metadata
        mock_backend.read_json.return_value = {
            "symbol": "SPY",
            "asset_type": "stock",
            "frequency": "1d",
            "source": "test",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "adjusted": True,
            "row_count": 10,
        }

        # Mock read_parquet to return DataFrame
        mock_backend.read_parquet.return_value = sample_ohlcv_data.data

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            storage_backend=mock_backend,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = cache.get("SPY", DataFrequency.DAILY, start, end)

        assert result is not None
        assert result.symbol == "SPY"
        assert mock_backend.exists.called
        assert mock_backend.read_json.called
        assert mock_backend.read_parquet.called

    def test_get_with_storage_backend_cache_miss(self, tmp_path):
        """Test get() returns None when cache miss."""
        mock_backend = MagicMock()
        mock_backend.exists.return_value = False

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            storage_backend=mock_backend,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        result = cache.get("SPY", DataFrequency.DAILY, start, end)

        assert result is None
        assert mock_backend.exists.called


class TestDataCacheStorageBackendInvalidateCleanup:
    """Test invalidate and cleanup with storage_backend."""

    def test_invalidate_with_storage_backend(self, tmp_path):
        """Test invalidate uses storage_backend."""
        mock_backend = MagicMock()
        mock_backend.list_files.return_value = [
            "abc123.meta.json",
            "def456.meta.json",
        ]
        mock_backend.read_json.side_effect = [
            {"symbol": "SPY", "frequency": "1d"},
            {"symbol": "QQQ", "frequency": "1d"},
        ]
        mock_backend.delete.return_value = True

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            storage_backend=mock_backend,
        )

        count = cache.invalidate(symbol="SPY")

        assert count == 1
        assert mock_backend.list_files.called
        # Should delete 2 files (parquet + meta) for SPY
        assert mock_backend.delete.call_count == 2

    def test_cleanup_with_storage_backend(self, tmp_path):
        """Test cleanup uses storage_backend."""
        mock_backend = MagicMock()
        mock_backend.list_files.return_value = ["old_cache.meta.json"]
        mock_backend.read_json.return_value = {
            "symbol": "SPY",
            "frequency": "1d",
            "fetched_at": "2020-01-01T00:00:00+00:00",  # Very old
        }
        mock_backend.delete.return_value = True

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            max_age_days=1,
            storage_backend=mock_backend,
        )

        count = cache.cleanup_expired()

        assert count == 1
        assert mock_backend.delete.call_count == 2  # parquet + meta

    def test_get_stats_with_storage_backend(self, tmp_path):
        """Test get_stats uses storage_backend."""
        mock_backend = MagicMock()
        mock_backend.list_files.return_value = ["a.parquet", "b.parquet"]
        mock_backend.get_stats.return_value = {
            "backend": "s3",
            "base_path": "bucket/cache",
            "local_cache_enabled": True,
        }

        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            storage_backend=mock_backend,
        )

        stats = cache.get_stats()

        assert stats["backend"] == "parquet"
        assert stats["storage_backend"] == "s3"
        assert stats["entry_count"] == 2


class TestDataCacheBackwardCompatibility:
    """Test backward compatibility without storage_backend."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCVData."""
        df = pl.DataFrame({
            "date": pl.date_range(
                datetime(2024, 1, 1),
                datetime(2024, 1, 10),
                eager=True
            ),
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000000 + i * 100000 for i in range(10)],
        })

        return OHLCVData(
            symbol="SPY",
            asset_type=AssetType.STOCK,
            frequency=DataFrequency.DAILY,
            data=df,
            source="test",
            fetched_at=datetime.now(timezone.utc),
            adjusted=True,
        )

    def test_local_put_get_roundtrip(self, tmp_path, sample_ohlcv_data):
        """Test put/get works without storage_backend (local mode)."""
        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
            max_age_days=1,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # Put data
        cache.put(sample_ohlcv_data, start, end)

        # Verify files created locally
        parquet_files = list((tmp_path / "cache").glob("*.parquet"))
        meta_files = list((tmp_path / "cache").glob("*.meta.json"))
        assert len(parquet_files) == 1
        assert len(meta_files) == 1

        # Get data
        result = cache.get("SPY", DataFrequency.DAILY, start, end)

        assert result is not None
        assert result.symbol == "SPY"
        assert len(result.data) == 10

    def test_local_invalidate(self, tmp_path, sample_ohlcv_data):
        """Test invalidate works without storage_backend."""
        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        cache.put(sample_ohlcv_data, start, end)

        # Verify files exist
        assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1

        # Invalidate
        count = cache.invalidate(symbol="SPY")

        assert count == 1
        assert len(list((tmp_path / "cache").glob("*.parquet"))) == 0

    def test_local_get_stats(self, tmp_path, sample_ohlcv_data):
        """Test get_stats works without storage_backend."""
        cache = DataCache(
            cache_dir=tmp_path / "cache",
            backend=CacheBackend.PARQUET,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        cache.put(sample_ohlcv_data, start, end)

        stats = cache.get_stats()

        assert stats["backend"] == "parquet"
        assert stats["entry_count"] == 1
        assert "cache_dir" in stats
        assert "storage_backend" not in stats  # No storage_backend key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
