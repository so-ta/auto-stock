"""
Quality Cache Module - Quality check caching and validation.

This module handles caching of data quality check results:
- Cache invalidation based on universe and configuration changes
- Incremental quality checks for performance
- Pickle-based persistence
"""

from __future__ import annotations

import hashlib
import pickle
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import polars as pl
    from src.config.settings import Settings
    from src.data.quality_checker import DataQualityReport
    from src.utils.storage_backend import StorageBackend

from src.utils.hash_utils import compute_universe_hash

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckCache:
    """Cache for quality check results with invalidation support.

    Attributes:
        date: The date when the cache was created
        universe_hash: Hash of the universe (symbol list)
        quality_config_hash: Hash of quality check configuration
        reports: Quality reports for each symbol
        excluded_assets: List of excluded asset symbols
        last_bar_dates: Last bar date for each symbol (for incremental check)
    """

    date: datetime
    universe_hash: str
    quality_config_hash: str
    reports: Dict[str, "DataQualityReport"]
    excluded_assets: List[str]
    last_bar_dates: Dict[str, datetime] = field(default_factory=dict)


class QualityCacheManager:
    """
    Manages quality check caching for data preparation.

    Handles:
    - Cache path management
    - Hash computation for invalidation
    - Cache load/save operations
    - Incremental cache validation
    """

    def __init__(
        self,
        settings: "Settings",
        cache_dir: Path,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> None:
        """
        Initialize the quality cache manager.

        Args:
            settings: Application settings
            cache_dir: Base directory for cache files
            storage_backend: Optional StorageBackend for S3 support
        """
        self._settings = settings
        self._cache_dir = cache_dir
        self._storage_backend = storage_backend
        self._logger = logger

    def compute_universe_hash(self, universe: List[str]) -> str:
        """
        Compute a hash of the universe for cache invalidation.

        Args:
            universe: List of asset symbols

        Returns:
            Hash string
        """
        return compute_universe_hash(universe)

    def compute_quality_config_hash(self) -> str:
        """
        Compute a hash of quality check configuration for cache invalidation.

        Returns:
            Hash string
        """
        quality_settings = self._settings.data_quality
        config_str = f"{quality_settings.min_bars}|{quality_settings.max_gap_days}|"
        config_str += f"{quality_settings.max_missing_ratio}|{quality_settings.max_zero_ratio}|"
        config_str += f"{quality_settings.min_liquidity}|{quality_settings.stale_threshold_days}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_cache_path(self, date: datetime) -> Path:
        """
        Get the cache file path for a specific date.

        Args:
            date: Cache date

        Returns:
            Path to cache file
        """
        cache_subdir = self._cache_dir / "quality_cache"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        date_str = date.strftime("%Y%m%d")
        return cache_subdir / f"quality_cache_{date_str}.pkl"

    def load_cache(self, date: datetime) -> Optional[QualityCheckCache]:
        """
        Load quality check cache from disk.

        Args:
            date: Date to load cache for

        Returns:
            QualityCheckCache if found and valid, None otherwise
        """
        # Try StorageBackend first
        if self._storage_backend is not None:
            return self._load_cache_via_backend(date)

        # Fallback to local file
        cache_path = self.get_cache_path(date)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, QualityCheckCache):
                self._logger.debug(f"Loaded quality cache from {cache_path}")
                return cache
        except Exception as e:
            self._logger.warning(f"Failed to load quality cache: {e}")

        return None

    def _load_cache_via_backend(self, date: datetime) -> Optional[QualityCheckCache]:
        """Load cache via StorageBackend."""
        date_str = date.strftime("%Y%m%d")
        rel_path = f"quality_cache/quality_cache_{date_str}.pkl"

        try:
            if not self._storage_backend.exists(rel_path):
                return None

            cache = self._storage_backend.read_pickle(rel_path)
            if isinstance(cache, QualityCheckCache):
                self._logger.debug(f"Loaded quality cache via backend: {rel_path}")
                return cache
        except Exception as e:
            self._logger.warning(f"Failed to load quality cache via backend: {e}")

        return None

    def save_cache(
        self,
        date: datetime,
        universe_hash: str,
        quality_config_hash: str,
        reports: Dict[str, Any],
        excluded_assets: List[str],
        last_bar_dates: Dict[str, datetime],
    ) -> None:
        """
        Save quality check cache to disk.

        Args:
            date: Cache date
            universe_hash: Hash of universe
            quality_config_hash: Hash of quality configuration
            reports: Quality reports
            excluded_assets: Excluded asset list
            last_bar_dates: Last bar dates for each symbol
        """
        cache = QualityCheckCache(
            date=date,
            universe_hash=universe_hash,
            quality_config_hash=quality_config_hash,
            reports=reports,
            excluded_assets=excluded_assets,
            last_bar_dates=last_bar_dates,
        )

        # Save via StorageBackend if available
        if self._storage_backend is not None:
            self._save_cache_via_backend(date, cache)
        else:
            self._save_cache_local(date, cache)

    def _save_cache_local(self, date: datetime, cache: QualityCheckCache) -> None:
        """Save cache to local filesystem."""
        cache_path = self.get_cache_path(date)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            self._logger.debug(f"Saved quality cache to {cache_path}")
        except Exception as e:
            self._logger.warning(f"Failed to save quality cache: {e}")

    def _save_cache_via_backend(self, date: datetime, cache: QualityCheckCache) -> None:
        """Save cache via StorageBackend."""
        date_str = date.strftime("%Y%m%d")
        rel_path = f"quality_cache/quality_cache_{date_str}.pkl"

        try:
            self._storage_backend.write_pickle(cache, rel_path)
            self._logger.debug(f"Saved quality cache via backend: {rel_path}")
        except Exception as e:
            self._logger.warning(f"Failed to save quality cache via backend: {e}")

    def is_cache_valid(
        self,
        cache: QualityCheckCache,
        universe_hash: str,
        quality_config_hash: str,
    ) -> bool:
        """
        Check if cache is still valid.

        Cache is invalid if:
        - Universe hash changed (different symbols)
        - Quality config hash changed (different thresholds)

        Args:
            cache: Loaded cache
            universe_hash: Current universe hash
            quality_config_hash: Current config hash

        Returns:
            True if cache is valid
        """
        return (
            cache.universe_hash == universe_hash
            and cache.quality_config_hash == quality_config_hash
        )

    def get_last_bar_date(self, df: "pl.DataFrame") -> Optional[datetime]:
        """
        Get the last bar date from a DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            Last timestamp as datetime, or None if not available
        """
        if df is None or len(df) == 0:
            return None

        date_col = None
        for col in ["timestamp", "date", "datetime", "time"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return None

        last_val = df[date_col].max()
        if last_val is None:
            return None

        # Convert to datetime if needed
        if isinstance(last_val, str):
            return datetime.fromisoformat(last_val)
        return last_val
