"""
Data Cache Module.

Provides local caching for OHLCV data using DuckDB or Parquet files.
Reduces API calls and improves performance for repeated data access.

Supports StorageBackend for S3 integration (Parquet backend only).
"""

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import polars as pl

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

from .adapters import AssetType, DataFrequency, OHLCVData

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache storage backend."""

    DUCKDB = "duckdb"
    PARQUET = "parquet"


class DataCache:
    """
    Local cache for OHLCV data.

    Supports two backends:
    - DuckDB: Single file database, good for querying
    - Parquet: One file per symbol, good for portability

    Cache keys are generated from symbol, frequency, and date range.

    S3 Integration:
        Pass a StorageBackend instance to enable S3 storage (Parquet backend only).
        When storage_backend is provided, all Parquet I/O uses the backend.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".cache/data",
        backend: CacheBackend = CacheBackend.PARQUET,
        max_age_days: int = 1,
        storage_backend: Optional["StorageBackend"] = None,
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache files (used when storage_backend is None)
            backend: Cache storage backend (DUCKDB or PARQUET)
            max_age_days: Maximum age of cached data in days
            storage_backend: Optional StorageBackend for S3 integration.
                            When provided, Parquet I/O uses this backend.
                            DuckDB backend is not supported with storage_backend.

        Raises:
            ImportError: If DuckDB backend is selected but not installed
            ValueError: If DuckDB backend is used with storage_backend
        """
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.max_age_days = max_age_days
        self._storage_backend = storage_backend

        # Validate: DuckDB + storage_backend is not supported
        if backend == CacheBackend.DUCKDB and storage_backend is not None:
            raise ValueError(
                "DuckDB backend is not supported with storage_backend. "
                "Use PARQUET backend for S3 integration."
            )

        # Create local cache directory (needed even with storage_backend for temp ops)
        if storage_backend is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if backend == CacheBackend.DUCKDB:
            if duckdb is None:
                raise ImportError(
                    "duckdb is required for DuckDB backend. "
                    "Install with: pip install duckdb"
                )
            self._db_path = self.cache_dir / "cache.duckdb"
            self._init_duckdb()

    def _init_duckdb(self) -> None:
        """Initialize DuckDB schema."""
        with duckdb.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    cache_key VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    asset_type VARCHAR,
                    frequency VARCHAR,
                    source VARCHAR,
                    fetched_at TIMESTAMP,
                    adjusted BOOLEAN,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    data BLOB
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol
                ON ohlcv_cache(symbol, frequency)
            """)

    def _generate_cache_key(
        self,
        symbol: str,
        frequency: DataFrequency,
        start: datetime,
        end: datetime,
    ) -> str:
        """
        Generate unique cache key for data request.

        Args:
            symbol: Asset symbol
            frequency: Data frequency
            start: Start datetime
            end: End datetime

        Returns:
            SHA256-based cache key
        """
        key_parts = [
            symbol,
            frequency.value,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_parquet_path(self, cache_key: str) -> Path:
        """Get parquet file path for cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata file path for cache key."""
        return self.cache_dir / f"{cache_key}.meta.json"

    def get(
        self,
        symbol: str,
        frequency: DataFrequency,
        start: datetime,
        end: datetime,
    ) -> Optional[OHLCVData]:
        """
        Get cached data if available and not expired.

        Args:
            symbol: Asset symbol
            frequency: Data frequency
            start: Start datetime
            end: End datetime

        Returns:
            OHLCVData if cache hit, None if cache miss or expired
        """
        cache_key = self._generate_cache_key(symbol, frequency, start, end)

        if self.backend == CacheBackend.PARQUET:
            return self._get_parquet(cache_key, symbol, frequency)
        else:
            return self._get_duckdb(cache_key, symbol, frequency)

    def _get_parquet(
        self,
        cache_key: str,
        symbol: str,
        frequency: DataFrequency,
    ) -> Optional[OHLCVData]:
        """Get data from parquet cache."""
        import json

        parquet_rel_path = f"{cache_key}.parquet"
        meta_rel_path = f"{cache_key}.meta.json"

        # Use storage backend if available
        if self._storage_backend is not None:
            if not self._storage_backend.exists(parquet_rel_path):
                return None
            if not self._storage_backend.exists(meta_rel_path):
                return None

            try:
                metadata = self._storage_backend.read_json(meta_rel_path)
            except Exception:
                return None

            fetched_at = datetime.fromisoformat(metadata["fetched_at"])
            age_days = (datetime.now(timezone.utc) - fetched_at).days

            if age_days > self.max_age_days:
                logger.debug(f"Cache expired for {symbol} (age: {age_days} days)")
                return None

            df = self._storage_backend.read_parquet(parquet_rel_path)

            logger.debug(f"Cache hit for {symbol} (storage backend)")

            return OHLCVData(
                symbol=symbol,
                asset_type=AssetType(metadata["asset_type"]),
                frequency=frequency,
                data=df,
                source=metadata["source"],
                fetched_at=fetched_at,
                adjusted=metadata.get("adjusted", False),
            )

        # Local file system path
        parquet_path = self._get_parquet_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)

        if not parquet_path.exists() or not meta_path.exists():
            return None

        with open(meta_path) as f:
            metadata = json.load(f)

        fetched_at = datetime.fromisoformat(metadata["fetched_at"])
        age_days = (datetime.now(timezone.utc) - fetched_at).days

        if age_days > self.max_age_days:
            logger.debug(f"Cache expired for {symbol} (age: {age_days} days)")
            return None

        df = pl.read_parquet(parquet_path)

        logger.debug(f"Cache hit for {symbol}")

        return OHLCVData(
            symbol=symbol,
            asset_type=AssetType(metadata["asset_type"]),
            frequency=frequency,
            data=df,
            source=metadata["source"],
            fetched_at=fetched_at,
            adjusted=metadata.get("adjusted", False),
        )

    def _get_duckdb(
        self,
        cache_key: str,
        symbol: str,
        frequency: DataFrequency,
    ) -> Optional[OHLCVData]:
        """Get data from DuckDB cache."""
        with duckdb.connect(str(self._db_path)) as conn:
            result = conn.execute(
                """
                SELECT asset_type, source, fetched_at, adjusted, data
                FROM ohlcv_cache
                WHERE cache_key = ?
                """,
                [cache_key],
            ).fetchone()

            if result is None:
                return None

            asset_type, source, fetched_at, adjusted, data_blob = result

            age_days = (datetime.now(timezone.utc) - fetched_at).days
            if age_days > self.max_age_days:
                logger.debug(f"Cache expired for {symbol}")
                return None

            import io

            df = pl.read_parquet(io.BytesIO(data_blob))

            logger.debug(f"DuckDB cache hit for {symbol}")

            return OHLCVData(
                symbol=symbol,
                asset_type=AssetType(asset_type),
                frequency=frequency,
                data=df,
                source=source,
                fetched_at=fetched_at,
                adjusted=adjusted,
            )

    def put(
        self,
        data: OHLCVData,
        start: datetime,
        end: datetime,
    ) -> None:
        """
        Store data in cache.

        Args:
            data: OHLCVData to cache
            start: Start datetime of the request
            end: End datetime of the request
        """
        cache_key = self._generate_cache_key(data.symbol, data.frequency, start, end)

        if self.backend == CacheBackend.PARQUET:
            self._put_parquet(cache_key, data)
        else:
            self._put_duckdb(cache_key, data, start, end)

        logger.debug(f"Cached data for {data.symbol}")

    def _put_parquet(self, cache_key: str, data: OHLCVData) -> None:
        """Store data in parquet cache."""
        import json

        parquet_rel_path = f"{cache_key}.parquet"
        meta_rel_path = f"{cache_key}.meta.json"

        metadata = {
            "symbol": data.symbol,
            "asset_type": data.asset_type.value,
            "frequency": data.frequency.value,
            "source": data.source,
            "fetched_at": data.fetched_at.isoformat(),
            "adjusted": data.adjusted,
            "row_count": len(data.data),
        }

        # Use storage backend if available
        if self._storage_backend is not None:
            self._storage_backend.write_parquet(data.data, parquet_rel_path)
            self._storage_backend.write_json(metadata, meta_rel_path)
            return

        # Local file system path
        parquet_path = self._get_parquet_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)

        data.data.write_parquet(parquet_path)

        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    def _put_duckdb(
        self,
        cache_key: str,
        data: OHLCVData,
        start: datetime,
        end: datetime,
    ) -> None:
        """Store data in DuckDB cache."""
        import io

        buffer = io.BytesIO()
        data.data.write_parquet(buffer)
        data_blob = buffer.getvalue()

        with duckdb.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv_cache
                (cache_key, symbol, asset_type, frequency, source,
                 fetched_at, adjusted, start_date, end_date, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    cache_key,
                    data.symbol,
                    data.asset_type.value,
                    data.frequency.value,
                    data.source,
                    data.fetched_at,
                    data.adjusted,
                    start,
                    end,
                    data_blob,
                ],
            )

    def invalidate(
        self,
        symbol: Optional[str] = None,
        frequency: Optional[DataFrequency] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            symbol: Optional symbol to invalidate (None = all)
            frequency: Optional frequency to invalidate (None = all)

        Returns:
            Number of entries invalidated
        """
        if self.backend == CacheBackend.PARQUET:
            return self._invalidate_parquet(symbol, frequency)
        else:
            return self._invalidate_duckdb(symbol, frequency)

    def _invalidate_parquet(
        self,
        symbol: Optional[str],
        frequency: Optional[DataFrequency],
    ) -> int:
        """Invalidate parquet cache entries."""
        import json

        # Storage backend mode
        if self._storage_backend is not None:
            count = 0
            try:
                meta_files = self._storage_backend.list_files("", "*.meta.json")
                for meta_rel_path in meta_files:
                    try:
                        metadata = self._storage_backend.read_json(meta_rel_path)
                    except Exception:
                        continue

                    if symbol and metadata.get("symbol") != symbol:
                        continue
                    if frequency and metadata.get("frequency") != frequency.value:
                        continue

                    # Delete parquet and meta files
                    parquet_rel_path = meta_rel_path.replace(".meta.json", ".parquet")
                    self._storage_backend.delete(parquet_rel_path)
                    self._storage_backend.delete(meta_rel_path)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to invalidate via storage backend: {e}")
            return count

        # Local file system mode
        count = 0
        for meta_path in self.cache_dir.glob("*.meta.json"):
            with open(meta_path) as f:
                metadata = json.load(f)

            if symbol and metadata["symbol"] != symbol:
                continue
            if frequency and metadata["frequency"] != frequency.value:
                continue

            parquet_path = meta_path.with_suffix("").with_suffix(".parquet")
            if parquet_path.exists():
                parquet_path.unlink()
            meta_path.unlink()
            count += 1

        return count

    def _invalidate_duckdb(
        self,
        symbol: Optional[str],
        frequency: Optional[DataFrequency],
    ) -> int:
        """Invalidate DuckDB cache entries."""
        with duckdb.connect(str(self._db_path)) as conn:
            if symbol is None and frequency is None:
                result = conn.execute("DELETE FROM ohlcv_cache")
            elif symbol and frequency:
                result = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE symbol = ? AND frequency = ?",
                    [symbol, frequency.value],
                )
            elif symbol:
                result = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE symbol = ?", [symbol]
                )
            else:
                result = conn.execute(
                    "DELETE FROM ohlcv_cache WHERE frequency = ?", [frequency.value]
                )

            return result.fetchone()[0] if result else 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if self.backend == CacheBackend.PARQUET:
            return self._get_stats_parquet()
        else:
            return self._get_stats_duckdb()

    def _get_stats_parquet(self) -> dict:
        """Get parquet cache statistics."""
        # Storage backend mode
        if self._storage_backend is not None:
            try:
                parquet_files = self._storage_backend.list_files("", "*.parquet")
                backend_stats = self._storage_backend.get_stats()
                return {
                    "backend": "parquet",
                    "storage_backend": backend_stats.get("backend", "unknown"),
                    "entry_count": len(parquet_files),
                    "base_path": backend_stats.get("base_path", ""),
                    "local_cache_enabled": backend_stats.get("local_cache_enabled", False),
                }
            except Exception as e:
                logger.warning(f"Failed to get stats via storage backend: {e}")
                return {
                    "backend": "parquet",
                    "storage_backend": "error",
                    "error": str(e),
                }

        # Local file system mode
        parquet_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)

        return {
            "backend": "parquet",
            "entry_count": len(parquet_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def _get_stats_duckdb(self) -> dict:
        """Get DuckDB cache statistics."""
        with duckdb.connect(str(self._db_path)) as conn:
            result = conn.execute(
                "SELECT COUNT(*), SUM(LENGTH(data)) FROM ohlcv_cache"
            ).fetchone()

            entry_count = result[0] or 0
            total_size = result[1] or 0

        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "backend": "duckdb",
            "entry_count": entry_count,
            "data_size_mb": round(total_size / (1024 * 1024), 2),
            "db_file_size_mb": round(db_size / (1024 * 1024), 2),
            "db_path": str(self._db_path),
        }

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        if self.backend == CacheBackend.PARQUET:
            return self._cleanup_parquet()
        else:
            return self._cleanup_duckdb()

    def _cleanup_parquet(self) -> int:
        """Clean up expired parquet cache entries."""
        import json

        count = 0
        now = datetime.now(timezone.utc)

        # Storage backend mode
        if self._storage_backend is not None:
            try:
                meta_files = self._storage_backend.list_files("", "*.meta.json")
                for meta_rel_path in meta_files:
                    try:
                        metadata = self._storage_backend.read_json(meta_rel_path)
                    except Exception:
                        continue

                    fetched_at = datetime.fromisoformat(metadata["fetched_at"])
                    age_days = (now - fetched_at).days

                    if age_days > self.max_age_days:
                        parquet_rel_path = meta_rel_path.replace(".meta.json", ".parquet")
                        self._storage_backend.delete(parquet_rel_path)
                        self._storage_backend.delete(meta_rel_path)
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup via storage backend: {e}")
            return count

        # Local file system mode
        for meta_path in self.cache_dir.glob("*.meta.json"):
            with open(meta_path) as f:
                metadata = json.load(f)

            fetched_at = datetime.fromisoformat(metadata["fetched_at"])
            age_days = (now - fetched_at).days

            if age_days > self.max_age_days:
                parquet_path = meta_path.with_suffix("").with_suffix(".parquet")
                if parquet_path.exists():
                    parquet_path.unlink()
                meta_path.unlink()
                count += 1

        return count

    def _cleanup_duckdb(self) -> int:
        """Clean up expired DuckDB cache entries."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)

        with duckdb.connect(str(self._db_path)) as conn:
            result = conn.execute(
                "DELETE FROM ohlcv_cache WHERE fetched_at < ?", [cutoff]
            )
            return result.fetchone()[0] if result else 0
