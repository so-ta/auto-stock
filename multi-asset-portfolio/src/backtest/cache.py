"""
Backtest Cache - Signal and computation caching for performance optimization.

This module provides caching infrastructure for backtest operations:
- SignalCache: Cache signal computation results (memory + disk)
- DataCache: Cache fetched price data
- Incremental computation support

Design Principles:
1. Memory-first with disk fallback
2. Hash-based cache keys for deduplication
3. Thread-safe operations
4. Parquet format for disk persistence
5. Automatic cleanup to prevent unbounded growth
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""

    value: T
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0

    def touch(self) -> None:
        """Update access count."""
        self.access_count += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU (Least Recently Used) cache.

    Uses OrderedDict for O(1) access and LRU eviction.

    Args:
        max_size: Maximum number of entries (default: 1000)
        max_memory_mb: Maximum memory usage in MB (default: 512)
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 512,
    ) -> None:
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.touch()
                self._hits += 1
                return entry.value
            self._misses += 1
            return None

    def put(self, key: str, value: T, size_bytes: int = 0) -> None:
        """
        Put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Estimated size in bytes (for memory management)
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_memory -= old_entry.size_bytes

            # Create new entry
            entry = CacheEntry(value=value, size_bytes=size_bytes)
            self._cache[key] = entry
            self._current_memory += size_bytes

            # Evict if necessary
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds limits."""
        while (
            len(self._cache) > self._max_size
            or self._current_memory > self._max_memory_bytes
        ) and self._cache:
            # Remove oldest (first) item
            _, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size_bytes

    def set(self, key: str, value: T, size_bytes: int = 0) -> None:
        """Alias for put() to conform to CacheInterface."""
        self.put(key, value, size_bytes)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "memory_mb": self._current_memory / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


def align_cache_date(date: datetime, granularity_days: int = 7) -> datetime:
    """
    Align date to improve cache hit rate.

    Aligns dates to week boundaries (Monday) to ensure similar date ranges
    hit the same cache entry. This significantly improves hit rate for
    rolling window computations.

    Args:
        date: Date to align
        granularity_days: Alignment granularity (7 = weekly, default)

    Returns:
        Aligned date (Monday of the same week)

    Example:
        >>> align_cache_date(datetime(2024, 1, 3))  # Wednesday
        datetime(2024, 1, 1)  # Monday
    """
    if granularity_days == 7:
        # Align to Monday of the week
        return date - timedelta(days=date.weekday())
    else:
        # Align to nearest granularity boundary
        days_since_epoch = (date - datetime(1970, 1, 1)).days
        aligned_days = (days_since_epoch // granularity_days) * granularity_days
        return datetime(1970, 1, 1) + timedelta(days=aligned_days)


def generate_cache_key(
    symbol: str,
    signal_name: str,
    params: dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    align_dates: bool = True,
    granularity_days: int = 7,
) -> str:
    """
    Generate a unique cache key for signal computation.

    Args:
        symbol: Asset symbol
        signal_name: Signal class name
        params: Signal parameters
        start_date: Data start date
        end_date: Data end date
        align_dates: If True, align dates to improve hit rate (default: True)
        granularity_days: Date alignment granularity (default: 7 = weekly)

    Returns:
        Unique hash-based cache key

    Note:
        Date alignment improves cache hit rate for rolling window computations.
        Dates within the same week will generate the same cache key.
    """
    # Align dates to improve cache hit rate
    if align_dates:
        aligned_start = align_cache_date(start_date, granularity_days)
        aligned_end = align_cache_date(end_date, granularity_days)
    else:
        aligned_start = start_date
        aligned_end = end_date

    key_parts = [
        symbol,
        signal_name,
        str(sorted(params.items())),
        aligned_start.strftime("%Y%m%d"),
        aligned_end.strftime("%Y%m%d"),
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


class SignalCache:
    """
    Two-level cache for signal computation results.

    Level 1: In-memory LRU cache (fast)
    Level 2: Disk cache using Parquet (persistent)

    Features:
    - Automatic cleanup: Periodically removes expired entries
    - Size limit: Removes oldest entries when disk cache exceeds max size
    - Thread-safe: All operations are protected by locks

    Usage:
        cache = SignalCache(cache_dir="./cache")

        # Check cache
        result = cache.get(symbol, signal_name, params, start, end)
        if result is None:
            result = compute_signal(...)
            cache.put(symbol, signal_name, params, start, end, result)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_entries: int = 500,
        max_memory_mb: int = 256,
        enable_disk_cache: bool = True,
        max_disk_cache_mb: int = 500,
        cleanup_interval_seconds: int = 3600,
        max_age_days: int = 30,
    ) -> None:
        """
        Initialize signal cache.

        Args:
            cache_dir: Directory for disk cache (default: ./cache/signals)
            max_memory_entries: Max entries in memory cache
            max_memory_mb: Max memory usage in MB
            enable_disk_cache: Whether to use disk caching
            max_disk_cache_mb: Max disk cache size in MB (default: 500)
            cleanup_interval_seconds: Interval between cleanup checks (default: 3600 = 1 hour)
            max_age_days: Max age of cache entries in days (default: 30)
        """
        self._memory_cache: LRUCache[pd.Series] = LRUCache(
            max_size=max_memory_entries,
            max_memory_mb=max_memory_mb,
        )
        self._enable_disk = enable_disk_cache

        if cache_dir is None:
            cache_dir = Path("./cache/signals")
        self._cache_dir = Path(cache_dir)

        if self._enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-cleanup settings
        self._max_disk_cache_bytes = max_disk_cache_mb * 1024 * 1024
        self._cleanup_interval = cleanup_interval_seconds
        self._max_age_seconds = max_age_days * 24 * 3600
        self._last_cleanup = time.time()

        self._lock = threading.RLock()

        # Hit rate tracking
        self._hits = 0
        self._misses = 0
        self._memory_hits = 0
        self._disk_hits = 0
        self._log_interval = 100  # Log every N requests

        logger.debug(
            f"SignalCache initialized: dir={self._cache_dir}, disk={self._enable_disk}, "
            f"max_disk_mb={max_disk_cache_mb}, cleanup_interval={cleanup_interval_seconds}s"
        )

    def get(
        self,
        symbol: str,
        signal_name: str,
        params: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.Series | None:
        """
        Get cached signal result.

        Args:
            symbol: Asset symbol
            signal_name: Signal class name
            params: Signal parameters
            start_date: Data start date
            end_date: Data end date

        Returns:
            Cached Series or None if not found
        """
        # Periodic cleanup check
        self._maybe_cleanup()

        key = generate_cache_key(symbol, signal_name, params, start_date, end_date)

        # Try memory cache first
        result = self._memory_cache.get(key)
        if result is not None:
            self._hits += 1
            self._memory_hits += 1
            self._log_hit_rate_if_needed()
            return result

        # Try disk cache
        if self._enable_disk:
            result = self._read_from_disk(key)
            if result is not None:
                # Promote to memory cache
                size_bytes = result.memory_usage(deep=True)
                self._memory_cache.put(key, result, size_bytes)
                self._hits += 1
                self._disk_hits += 1
                self._log_hit_rate_if_needed()
                return result

        # Cache miss
        self._misses += 1
        self._log_hit_rate_if_needed()
        return None

    def _log_hit_rate_if_needed(self) -> None:
        """Log hit rate periodically."""
        total = self._hits + self._misses
        if total > 0 and total % self._log_interval == 0:
            hit_rate = self._hits / total * 100
            logger.info(
                f"SignalCache stats: hit_rate={hit_rate:.1f}% "
                f"(memory={self._memory_hits}, disk={self._disk_hits}, miss={self._misses})"
            )

    @property
    def hit_rate(self) -> float:
        """Get current cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get detailed cache statistics."""
        total = self._hits + self._misses
        return {
            "total_requests": total,
            "hits": self._hits,
            "misses": self._misses,
            "memory_hits": self._memory_hits,
            "disk_hits": self._disk_hits,
            "hit_rate": self.hit_rate,
            "memory_cache_size": len(self._memory_cache._cache),
        }

    def put(
        self,
        symbol: str,
        signal_name: str,
        params: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        scores: pd.Series,
    ) -> None:
        """
        Store signal result in cache.

        Args:
            symbol: Asset symbol
            signal_name: Signal class name
            params: Signal parameters
            start_date: Data start date
            end_date: Data end date
            scores: Signal scores to cache
        """
        key = generate_cache_key(symbol, signal_name, params, start_date, end_date)
        size_bytes = scores.memory_usage(deep=True)

        # Store in memory
        self._memory_cache.put(key, scores, size_bytes)

        # Store on disk
        if self._enable_disk:
            self._write_to_disk(key, scores)

    def _read_from_disk(self, key: str) -> pd.Series | None:
        """Read from disk cache."""
        file_path = self._cache_dir / f"{key}.parquet"
        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            if "scores" in df.columns:
                return df["scores"]
            return None
        except Exception as e:
            logger.debug(f"Failed to read cache file {file_path}: {e}")
            return None

    def _write_to_disk(self, key: str, scores: pd.Series) -> None:
        """Write to disk cache."""
        file_path = self._cache_dir / f"{key}.parquet"
        try:
            df = pd.DataFrame({"scores": scores})
            df.to_parquet(file_path, compression="snappy")
        except Exception as e:
            logger.debug(f"Failed to write cache file {file_path}: {e}")

    def clear_memory(self) -> None:
        """Clear memory cache only."""
        self._memory_cache.clear()

    def clear_all(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        if self._enable_disk and self._cache_dir.exists():
            for f in self._cache_dir.glob("*.parquet"):
                try:
                    f.unlink()
                except OSError:
                    pass

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        disk_files = 0
        disk_size_mb = 0.0
        if self._enable_disk and self._cache_dir.exists():
            files = list(self._cache_dir.glob("*.parquet"))
            disk_files = len(files)
            disk_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

        return {
            "memory": self._memory_cache.stats,
            "disk_files": disk_files,
            "disk_size_mb": disk_size_mb,
            "max_disk_size_mb": self._max_disk_cache_bytes / (1024 * 1024),
        }

    def _maybe_cleanup(self) -> None:
        """
        Check if cleanup is needed and perform if so.

        Called automatically on get() operations.
        Cleanup is performed if:
        - Time since last cleanup exceeds cleanup_interval
        """
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()
            self._enforce_size_limit()
            self._last_cleanup = current_time

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries from disk.

        Entries older than max_age_days are removed.

        Returns:
            Number of entries removed
        """
        if not self._enable_disk or not self._cache_dir.exists():
            return 0

        removed = 0
        current_time = time.time()

        with self._lock:
            for f in self._cache_dir.glob("*.parquet"):
                try:
                    # Check file age
                    file_mtime = f.stat().st_mtime
                    age_seconds = current_time - file_mtime

                    if age_seconds > self._max_age_seconds:
                        f.unlink()
                        removed += 1
                        logger.debug(f"Removed expired cache file: {f.name}")
                except OSError as e:
                    logger.debug(f"Failed to remove cache file {f}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache entries")

        return removed

    def _enforce_size_limit(self) -> int:
        """
        Enforce disk cache size limit by removing oldest entries.

        Returns:
            Number of entries removed
        """
        if not self._enable_disk or not self._cache_dir.exists():
            return 0

        removed = 0

        with self._lock:
            # Get all cache files with their stats
            files_with_stats = []
            total_size = 0

            for f in self._cache_dir.glob("*.parquet"):
                try:
                    stat = f.stat()
                    files_with_stats.append((f, stat.st_mtime, stat.st_size))
                    total_size += stat.st_size
                except OSError:
                    continue

            # Check if over limit
            if total_size <= self._max_disk_cache_bytes:
                return 0

            # Sort by modification time (oldest first)
            files_with_stats.sort(key=lambda x: x[1])

            # Remove oldest files until under limit
            for f, mtime, size in files_with_stats:
                if total_size <= self._max_disk_cache_bytes:
                    break

                try:
                    f.unlink()
                    total_size -= size
                    removed += 1
                    logger.debug(f"Removed cache file for size limit: {f.name}")
                except OSError as e:
                    logger.debug(f"Failed to remove cache file {f}: {e}")

        if removed > 0:
            logger.info(
                f"Removed {removed} cache entries to enforce size limit "
                f"({self._max_disk_cache_bytes / (1024 * 1024):.1f} MB)"
            )

        return removed

    def force_cleanup(self) -> dict[str, int]:
        """
        Force immediate cleanup (expired + size limit).

        Returns:
            Dictionary with cleanup statistics
        """
        expired = self.cleanup_expired()
        size_limited = self._enforce_size_limit()
        self._last_cleanup = time.time()

        return {
            "expired_removed": expired,
            "size_limited_removed": size_limited,
            "total_removed": expired + size_limited,
        }


class DataFrameCache:
    """
    Cache for DataFrames (price data, intermediate results).

    Optimized for larger data structures with compression.

    Features:
    - Automatic cleanup: Periodically removes expired entries
    - Size limit: Removes oldest entries when disk cache exceeds max size
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_entries: int = 100,
        max_memory_mb: int = 512,
        max_disk_cache_mb: int = 500,
        cleanup_interval_seconds: int = 3600,
        max_age_days: int = 30,
    ) -> None:
        """
        Initialize DataFrame cache.

        Args:
            cache_dir: Directory for disk cache
            max_memory_entries: Max entries in memory
            max_memory_mb: Max memory usage
            max_disk_cache_mb: Max disk cache size in MB (default: 500)
            cleanup_interval_seconds: Interval between cleanup checks (default: 3600)
            max_age_days: Max age of cache entries in days (default: 30)
        """
        self._memory_cache: LRUCache[pd.DataFrame] = LRUCache(
            max_size=max_memory_entries,
            max_memory_mb=max_memory_mb,
        )

        if cache_dir is None:
            cache_dir = Path("./cache/dataframes")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-cleanup settings
        self._max_disk_cache_bytes = max_disk_cache_mb * 1024 * 1024
        self._cleanup_interval = cleanup_interval_seconds
        self._max_age_seconds = max_age_days * 24 * 3600
        self._last_cleanup = time.time()
        self._lock = threading.RLock()

    def get(self, key: str) -> pd.DataFrame | None:
        """Get cached DataFrame."""
        # Periodic cleanup check
        self._maybe_cleanup()

        # Try memory first
        result = self._memory_cache.get(key)
        if result is not None:
            return result

        # Try disk
        file_path = self._cache_dir / f"{key}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                size_bytes = df.memory_usage(deep=True).sum()
                self._memory_cache.put(key, df, size_bytes)
                return df
            except Exception as e:
                logger.debug(f"Failed to read {file_path}: {e}")

        return None

    def _maybe_cleanup(self) -> None:
        """Check if cleanup is needed and perform if so."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()
            self._enforce_size_limit()
            self._last_cleanup = current_time

    def cleanup_expired(self) -> int:
        """Remove expired cache entries from disk."""
        removed = 0
        current_time = time.time()

        with self._lock:
            for f in self._cache_dir.glob("*.parquet"):
                try:
                    file_mtime = f.stat().st_mtime
                    age_seconds = current_time - file_mtime

                    if age_seconds > self._max_age_seconds:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass

        return removed

    def _enforce_size_limit(self) -> int:
        """Enforce disk cache size limit by removing oldest entries."""
        removed = 0

        with self._lock:
            files_with_stats = []
            total_size = 0

            for f in self._cache_dir.glob("*.parquet"):
                try:
                    stat = f.stat()
                    files_with_stats.append((f, stat.st_mtime, stat.st_size))
                    total_size += stat.st_size
                except OSError:
                    continue

            if total_size <= self._max_disk_cache_bytes:
                return 0

            files_with_stats.sort(key=lambda x: x[1])

            for f, mtime, size in files_with_stats:
                if total_size <= self._max_disk_cache_bytes:
                    break
                try:
                    f.unlink()
                    total_size -= size
                    removed += 1
                except OSError:
                    pass

        return removed

    def force_cleanup(self) -> dict[str, int]:
        """Force immediate cleanup."""
        expired = self.cleanup_expired()
        size_limited = self._enforce_size_limit()
        self._last_cleanup = time.time()
        return {
            "expired_removed": expired,
            "size_limited_removed": size_limited,
            "total_removed": expired + size_limited,
        }

    def put(self, key: str, df: pd.DataFrame, persist: bool = True) -> None:
        """Store DataFrame in cache."""
        size_bytes = df.memory_usage(deep=True).sum()
        self._memory_cache.put(key, df, size_bytes)

        if persist:
            file_path = self._cache_dir / f"{key}.parquet"
            try:
                df.to_parquet(file_path, compression="snappy")
            except Exception as e:
                logger.debug(f"Failed to write {file_path}: {e}")


class IncrementalCache:
    """
    Cache that supports incremental computation.

    Instead of recomputing entire history, only computes new data
    and appends to cached results.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize incremental cache."""
        if cache_dir is None:
            cache_dir = Path("./cache/incremental")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get_cached_range(self, key: str) -> tuple[datetime | None, datetime | None]:
        """
        Get the date range of cached data.

        Returns:
            (start_date, end_date) tuple, or (None, None) if not cached
        """
        meta_path = self._cache_dir / f"{key}.meta"
        if not meta_path.exists():
            return None, None

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return meta.get("start_date"), meta.get("end_date")
        except Exception:
            return None, None

    def get_cached_data(self, key: str) -> pd.Series | None:
        """Get cached data if available."""
        data_path = self._cache_dir / f"{key}.parquet"
        if not data_path.exists():
            return None

        try:
            df = pd.read_parquet(data_path)
            return df["data"]
        except Exception:
            return None

    def append_data(
        self,
        key: str,
        new_data: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Append new data to cache.

        If cache exists, merges with existing data.
        """
        with self._lock:
            existing = self.get_cached_data(key)

            if existing is not None:
                # Merge: prefer new data where overlapping
                combined = pd.concat([existing, new_data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                final_data = combined
            else:
                final_data = new_data

            # Save data
            data_path = self._cache_dir / f"{key}.parquet"
            df = pd.DataFrame({"data": final_data})
            df.to_parquet(data_path, compression="snappy")

            # Save metadata
            meta_path = self._cache_dir / f"{key}.meta"
            meta = {
                "start_date": final_data.index.min(),
                "end_date": final_data.index.max(),
                "updated_at": datetime.now(),
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)


# Vectorized computation helpers
def vectorized_momentum(
    close: pd.Series | np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Vectorized momentum calculation using NumPy.

    Faster than pd.Series.pct_change() for large arrays.

    Args:
        close: Close prices
        lookback: Lookback period

    Returns:
        Momentum values (returns over lookback period)
    """
    if isinstance(close, pd.Series):
        close = close.values

    n = len(close)
    result = np.empty(n)
    result[:lookback] = np.nan

    # Vectorized computation
    result[lookback:] = (close[lookback:] - close[:-lookback]) / close[:-lookback]

    return result


def vectorized_volatility(
    returns: pd.Series | np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Vectorized rolling volatility calculation.

    Uses efficient rolling window computation.

    Args:
        returns: Return series
        window: Rolling window size

    Returns:
        Rolling volatility values
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    n = len(returns)
    result = np.empty(n)
    result[:window] = np.nan

    # Use cumsum trick for efficiency
    cumsum = np.cumsum(returns)
    cumsum2 = np.cumsum(returns**2)

    for i in range(window, n):
        s1 = cumsum[i] - cumsum[i - window]
        s2 = cumsum2[i] - cumsum2[i - window]
        mean = s1 / window
        var = s2 / window - mean**2
        result[i] = np.sqrt(max(0, var))

    return result


def vectorized_sharpe(
    returns: np.ndarray,
    window: int,
    annualize: int = 252,
) -> np.ndarray:
    """
    Vectorized rolling Sharpe ratio calculation.

    Args:
        returns: Return array
        window: Rolling window
        annualize: Annualization factor

    Returns:
        Rolling Sharpe ratio
    """
    n = len(returns)
    result = np.empty(n)
    result[:window] = np.nan

    sqrt_ann = np.sqrt(annualize)

    for i in range(window, n):
        window_returns = returns[i - window:i]
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns, ddof=1)
        if std_ret > 0:
            result[i] = mean_ret / std_ret * sqrt_ann
        else:
            result[i] = 0.0

    return result


def batch_compute_signals(
    data_dict: dict[str, pd.DataFrame],
    signal_func: callable,
    **signal_params: Any,
) -> dict[str, pd.Series]:
    """
    Batch compute signals for multiple symbols.

    Optimized for vectorized operations across symbols.

    Args:
        data_dict: Dictionary of symbol -> DataFrame
        signal_func: Signal computation function
        **signal_params: Parameters for signal function

    Returns:
        Dictionary of symbol -> signal scores
    """
    results = {}
    for symbol, df in data_dict.items():
        try:
            result = signal_func(df, **signal_params)
            if isinstance(result, pd.Series):
                results[symbol] = result
            elif hasattr(result, "scores"):
                results[symbol] = result.scores
        except Exception as e:
            logger.debug(f"Signal computation failed for {symbol}: {e}")
            continue

    return results
