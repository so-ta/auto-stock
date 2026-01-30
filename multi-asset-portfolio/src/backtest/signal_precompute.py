"""
Signal Precomputation Module - Batch signal computation and caching for backtest optimization.

This module pre-computes all signals and stores them in Parquet format,
enabling 5-10x faster backtests by eliminating redundant calculations.

Usage:
    # ローカルモード（後方互換性）
    precomputer = SignalPrecomputer(cache_dir=".cache/signals")
    precomputer.precompute_all(prices_df, config)

    # S3モード（StorageBackend使用）
    from src.utils.storage_backend import StorageBackend, StorageConfig
    backend = StorageBackend(StorageConfig(backend="s3", s3_bucket="my-bucket"))
    precomputer = SignalPrecomputer(storage_backend=backend)
    precomputer.precompute_all(prices_df, config)

    # During backtest
    mom_signal = precomputer.get_signal_at_date("momentum_20", "SPY", rebalance_date)
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend

logger = logging.getLogger(__name__)

# Version for cache invalidation when implementation changes
PRECOMPUTE_VERSION = "1.2.0"

# Signal classification for incremental computation
# Independent signals: computed from each ticker's price data only (no cross-asset dependency)
INDEPENDENT_SIGNALS = {
    "momentum_*",
    "rsi_*",
    "volatility_*",
    "zscore_*",
    "sharpe_*",
    "atr_*",
    "bollinger_*",
    "stochastic_*",
    "breakout_*",
    "donchian_*",
    "fifty_two_week_high_*",
}

# Relative signals: computed from cross-asset comparisons or rankings
RELATIVE_SIGNALS = {
    "sector_relative_*",
    "cross_asset_*",
    "momentum_factor",
    "sector_momentum",
    "sector_breadth",
    "market_breadth",
    "ranking_*",
    "lead_lag",  # Lead-Lag relationship signal (task_042_1)
    "short_term_reversal*",  # Short-term reversal signal (task_042_3)
}


@dataclass
class CacheValidationResult:
    """
    Result of cache validation (task_041_7a).

    Replaces the 4-element tuple return type of validate_cache_incremental()
    with a structured dataclass for better readability and extensibility.

    Attributes:
        can_use_cache: Whether cache can be used (True) or full recomputation needed (False)
        reason: Human-readable explanation of the validation result
        incremental_start: Start date for incremental time update (None if not needed)
        missing_signals: List of signals that need computation (None if none)
        new_tickers: List of newly added tickers that need computation (None if none)
    """

    can_use_cache: bool
    reason: str
    incremental_start: datetime | None = None
    missing_signals: list[str] | None = None
    new_tickers: list[str] | None = None  # task_041_7a: 銘柄差分検知


@dataclass
class PrecomputeMetadata:
    """
    Metadata for cache invalidation.

    Tracks all factors that should trigger cache regeneration:
    - Signal registry changes (new/removed signals)
    - Signal parameter changes
    - Universe (ticker list) changes
    - Price data changes
    - Library version changes

    Supports incremental updates:
    - 1-day increments don't trigger full recomputation
    - cached_start_date/cached_end_date track cached period
    """
    created_at: str
    signal_registry_hash: str  # Hash of registry.list_all()
    signal_config_hash: str    # Hash of signal parameters
    universe_hash: str         # Hash of ticker list
    prices_hash: str           # Simplified hash (row count + date range)
    version: str               # Library version

    # New fields for incremental update support
    cached_start_date: str = ""  # Cached period start (ISO format)
    cached_end_date: str = ""    # Cached period end (ISO format)
    ticker_count: int = 0        # Number of tickers (consistency check)

    def is_cache_valid(self, current: "PrecomputeMetadata") -> tuple[bool, str]:
        """
        Check if cache is valid against current metadata.

        Supports incremental updates:
        - If current data extends cached period by a few days, cache is still valid
        - Full recomputation triggered only for structural changes

        Args:
            current: Current metadata to compare against

        Returns:
            Tuple of (is_valid, reason)
        """
        if self.version != current.version:
            return False, f"Version changed: {self.version} -> {current.version}"

        if self.signal_registry_hash != current.signal_registry_hash:
            return False, "Signal registry changed (signals added/removed)"

        if self.signal_config_hash != current.signal_config_hash:
            return False, "Signal parameters changed"

        if self.universe_hash != current.universe_hash:
            return False, "Universe (ticker list) changed"

        # Check ticker count consistency
        if self.ticker_count > 0 and current.ticker_count > 0:
            if self.ticker_count != current.ticker_count:
                return False, f"Ticker count changed: {self.ticker_count} -> {current.ticker_count}"

        # Check date range - allow incremental extension
        if self.cached_start_date and current.cached_start_date:
            if self.cached_start_date != current.cached_start_date:
                return False, f"Start date changed: {self.cached_start_date} -> {current.cached_start_date}"

        # Allow cache to be valid if current end_date extends cached end_date
        # (i.e., new data was added at the end - incremental update case)
        if self.cached_end_date and current.cached_end_date:
            if current.cached_end_date < self.cached_end_date:
                return False, f"End date moved backward: {self.cached_end_date} -> {current.cached_end_date}"
            # If current end_date is same or later, cache is valid for existing period

        return True, "Cache is valid"

    def covers_period(self, start_date: str, end_date: str) -> bool:
        """
        Check if cache covers the requested period.

        Args:
            start_date: Requested start date (ISO format)
            end_date: Requested end date (ISO format)

        Returns:
            True if cache covers the entire requested period
        """
        if not self.cached_start_date or not self.cached_end_date:
            return False

        return self.cached_start_date <= start_date and self.cached_end_date >= end_date

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrecomputeMetadata":
        """Create from dictionary with backward compatibility."""
        return cls(
            created_at=data.get("created_at", ""),
            signal_registry_hash=data.get("signal_registry_hash", ""),
            signal_config_hash=data.get("signal_config_hash", ""),
            universe_hash=data.get("universe_hash", ""),
            prices_hash=data.get("prices_hash", ""),
            version=data.get("version", ""),
            # New fields with defaults for backward compatibility
            cached_start_date=data.get("cached_start_date", ""),
            cached_end_date=data.get("cached_end_date", ""),
            ticker_count=data.get("ticker_count", 0),
        )


class SignalPrecomputer:
    """
    Pre-compute and cache all signals for efficient backtesting.

    Instead of computing signals on-the-fly during backtest iterations,
    this class computes all signals once using vectorized operations
    and stores them in Parquet format for fast retrieval.

    Attributes:
        cache_dir: Directory for storing Parquet files
        _metadata: Cache metadata including data hash and timestamps
    """

    DEFAULT_CONFIG = {
        "momentum_periods": [20, 60, 120, 252],
        "volatility_periods": [20, 60],
        "rsi_periods": [14],
        "zscore_periods": [20, 60],
        "sharpe_periods": [60, 252],
    }

    def __init__(
        self,
        cache_dir: str | Path = ".cache/signals",
        storage_backend: "StorageBackend | None" = None,
    ) -> None:
        """
        Initialize SignalPrecomputer.

        Args:
            cache_dir: Directory for storing precomputed signal Parquet files (legacy mode).
                      Ignored if storage_backend is provided.
            storage_backend: StorageBackend instance for S3/local abstraction.
                           If provided, cache_dir is ignored and all operations
                           go through the backend.

        Usage:
            # ローカルモード（後方互換性）
            precomputer = SignalPrecomputer(cache_dir=".cache/signals")

            # S3モード
            from src.utils.storage_backend import StorageBackend, StorageConfig
            backend = StorageBackend(StorageConfig(backend="s3", s3_bucket="my-bucket"))
            precomputer = SignalPrecomputer(storage_backend=backend)
        """
        if storage_backend is not None:
            # StorageBackend経由での操作
            self._backend: "StorageBackend | None" = storage_backend
            self._cache_dir = Path(storage_backend.config.base_path)
            self._use_backend = True
        else:
            # 従来のローカルファイル操作（後方互換性）
            self._backend = None
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._use_backend = False

        self._metadata_file_name = "_metadata.json"
        self._metadata: dict[str, Any] = self._load_metadata()

        backend_type = storage_backend.config.backend if storage_backend else "local (legacy)"
        logger.debug(f"SignalPrecomputer initialized: backend={backend_type}, path={self._cache_dir}")

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from disk if exists."""
        if self._use_backend and self._backend is not None:
            # StorageBackend経由で読み込み
            try:
                if self._backend.exists(self._metadata_file_name):
                    return self._backend.read_json(self._metadata_file_name)
            except Exception as e:
                logger.warning(f"Failed to load metadata via backend: {e}")
            return {"prices_hash": None, "computed_at": None, "signals": []}
        else:
            # 従来のローカルファイル操作
            metadata_file = self._cache_dir / self._metadata_file_name
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {e}")
            return {"prices_hash": None, "computed_at": None, "signals": []}

    def _load_precompute_metadata(self) -> PrecomputeMetadata | None:
        """Load PrecomputeMetadata from disk if exists."""
        if self._use_backend and self._backend is not None:
            # StorageBackend経由で読み込み
            try:
                if self._backend.exists(self._metadata_file_name):
                    data = self._backend.read_json(self._metadata_file_name)
                    if "signal_registry_hash" in data:
                        return PrecomputeMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load precompute metadata via backend: {e}")
            return None
        else:
            # 従来のローカルファイル操作
            metadata_file = self._cache_dir / self._metadata_file_name
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        data = json.load(f)
                        if "signal_registry_hash" in data:
                            return PrecomputeMetadata.from_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to load precompute metadata: {e}")
            return None

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        if self._use_backend and self._backend is not None:
            # StorageBackend経由で書き込み
            try:
                self._backend.write_json(self._metadata, self._metadata_file_name)
            except Exception as e:
                logger.warning(f"Failed to save metadata via backend: {e}")
        else:
            # 従来のローカルファイル操作
            metadata_file = self._cache_dir / self._metadata_file_name
            try:
                with open(metadata_file, "w") as f:
                    json.dump(self._metadata, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save metadata: {e}")

    def _save_precompute_metadata(self, metadata: PrecomputeMetadata) -> None:
        """Save PrecomputeMetadata to disk."""
        if self._use_backend and self._backend is not None:
            # StorageBackend経由で書き込み
            try:
                self._backend.write_json(metadata.to_dict(), self._metadata_file_name)
            except Exception as e:
                logger.warning(f"Failed to save precompute metadata via backend: {e}")
        else:
            # 従来のローカルファイル操作
            metadata_file = self._cache_dir / self._metadata_file_name
            try:
                with open(metadata_file, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save precompute metadata: {e}")

    # =========================================================================
    # Parquet I/O ヘルパーメソッド（StorageBackend対応）
    # =========================================================================

    def _write_parquet(self, df: pl.DataFrame, filename: str) -> None:
        """
        Write Parquet file via StorageBackend or local filesystem.

        Args:
            df: DataFrame to write
            filename: Filename (e.g., "momentum_20.parquet")
        """
        if self._use_backend and self._backend is not None:
            self._backend.write_parquet(df, filename)
        else:
            output_path = self._cache_dir / filename
            df.write_parquet(output_path, compression="snappy")

    def _read_parquet(self, filename: str) -> pl.DataFrame:
        """
        Read Parquet file via StorageBackend or local filesystem.

        Args:
            filename: Filename (e.g., "momentum_20.parquet")

        Returns:
            DataFrame
        """
        if self._use_backend and self._backend is not None:
            return self._backend.read_parquet(filename)
        else:
            file_path = self._cache_dir / filename
            return pl.read_parquet(file_path)

    def _parquet_exists(self, filename: str) -> bool:
        """
        Check if Parquet file exists via StorageBackend or local filesystem.

        Args:
            filename: Filename (e.g., "momentum_20.parquet")

        Returns:
            True if file exists
        """
        if self._use_backend and self._backend is not None:
            return self._backend.exists(filename)
        else:
            return (self._cache_dir / filename).exists()

    def _delete_parquet(self, filename: str) -> bool:
        """
        Delete Parquet file via StorageBackend or local filesystem.

        Args:
            filename: Filename (e.g., "momentum_20.parquet")

        Returns:
            True if deletion succeeded
        """
        if self._use_backend and self._backend is not None:
            return self._backend.delete(filename)
        else:
            file_path = self._cache_dir / filename
            if file_path.exists():
                file_path.unlink()
                return True
            return False

    def _list_parquet_files(self) -> list[str]:
        """
        List all Parquet files via StorageBackend or local filesystem.

        Returns:
            List of filenames (without path)
        """
        if self._use_backend and self._backend is not None:
            files = self._backend.list_files(pattern="*.parquet")
            return [Path(f).name for f in files]
        else:
            return [f.name for f in self._cache_dir.glob("*.parquet")]

    def _compute_signal_registry_hash(self) -> str:
        """Compute hash of signal registry (list of all registered signals)."""
        try:
            from src.signals import SignalRegistry
            signal_names = sorted(SignalRegistry.list_all())
            hash_input = ",".join(signal_names)
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            logger.debug(f"Failed to compute signal registry hash: {e}")
            return hashlib.md5(b"default").hexdigest()

    def _compute_signal_config_hash(self, config: dict[str, Any]) -> str:
        """Compute hash of signal configuration (parameters)."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _compute_universe_hash(self, tickers: list[str]) -> str:
        """Compute hash of ticker universe."""
        sorted_tickers = sorted(tickers)
        hash_input = ",".join(sorted_tickers)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_required_signals(self, config: dict[str, Any]) -> list[str]:
        """
        Get list of required signal names from config.

        Args:
            config: Signal configuration

        Returns:
            List of signal names (e.g., ["momentum_20", "momentum_60", ...])
        """
        signals = []
        for period in config.get("momentum_periods", []):
            signals.append(f"momentum_{period}")
        for period in config.get("volatility_periods", []):
            signals.append(f"volatility_{period}")
        for period in config.get("rsi_periods", []):
            signals.append(f"rsi_{period}")
        for period in config.get("zscore_periods", []):
            signals.append(f"zscore_{period}")
        for period in config.get("sharpe_periods", []):
            signals.append(f"sharpe_{period}")
        return signals

    def classify_signal(self, signal_name: str) -> str:
        """
        Classify a signal as independent or relative.

        Independent signals: computed from each ticker's price data only.
        These can be incrementally computed for new tickers without
        affecting existing cached values.

        Relative signals: computed from cross-asset comparisons or rankings.
        Adding new tickers requires full recomputation.

        Args:
            signal_name: Name of the signal (e.g., "momentum_20", "sector_relative_strength")

        Returns:
            "independent" or "relative"
        """
        # Check RELATIVE_SIGNALS first to handle exact matches like "momentum_factor"
        # before INDEPENDENT_SIGNALS wildcard patterns like "momentum_*"
        for pattern in RELATIVE_SIGNALS:
            if fnmatch.fnmatch(signal_name, pattern):
                return "relative"

        for pattern in INDEPENDENT_SIGNALS:
            if fnmatch.fnmatch(signal_name, pattern):
                return "independent"

        # Default to relative (safer - triggers full recomputation)
        logger.warning(f"Unknown signal type: {signal_name}, treating as relative")
        return "relative"

    def classify_signals(self, signal_names: list[str]) -> dict[str, list[str]]:
        """
        Classify multiple signals into independent and relative groups.

        Args:
            signal_names: List of signal names

        Returns:
            Dictionary with keys "independent" and "relative",
            each containing list of signal names
        """
        result: dict[str, list[str]] = {"independent": [], "relative": []}
        for name in signal_names:
            classification = self.classify_signal(name)
            result[classification].append(name)
        return result

    def _compute_prices_hash(self, prices: pl.DataFrame) -> str:
        """
        Compute a hash of the prices DataFrame for cache validation.

        Args:
            prices: Price DataFrame with columns [timestamp, ticker, close, ...]

        Returns:
            MD5 hash string
        """
        hash_input = f"{prices.shape}_{prices.columns}"
        if "timestamp" in prices.columns:
            first_ts = prices.select("timestamp").head(1).item()
            last_ts = prices.select("timestamp").tail(1).item()
            hash_input += f"_{first_ts}_{last_ts}"
        if "close" in prices.columns:
            close_sum = prices.select(pl.col("close").sum()).item()
            hash_input += f"_{close_sum:.6f}"

        return hashlib.md5(hash_input.encode()).hexdigest()

    def is_cache_valid(self, prices_hash: str) -> bool:
        """
        Check if cached signals are valid for given price data (legacy method).

        Args:
            prices_hash: Hash of current price data

        Returns:
            True if cache is valid and can be used
        """
        cached_hash = self._metadata.get("prices_hash")
        if cached_hash != prices_hash:
            logger.debug(f"Cache invalid: hash mismatch ({cached_hash} != {prices_hash})")
            return False

        required_signals = [
            f"momentum_{p}" for p in self.DEFAULT_CONFIG["momentum_periods"]
        ] + [
            f"volatility_{p}" for p in self.DEFAULT_CONFIG["volatility_periods"]
        ]

        for signal in required_signals:
            if not self._parquet_exists(f"{signal}.parquet"):
                logger.debug(f"Cache invalid: missing signal {signal}")
                return False

        return True

    def _get_date_range(self, prices: pl.DataFrame) -> tuple[str, str]:
        """Extract date range from prices DataFrame."""
        if "timestamp" not in prices.columns:
            return "", ""

        try:
            timestamps = prices.select("timestamp").to_series()
            start_date = timestamps.min()
            end_date = timestamps.max()

            # Convert to ISO format string
            if hasattr(start_date, 'isoformat'):
                start_str = start_date.isoformat() if start_date else ""
                end_str = end_date.isoformat() if end_date else ""
            else:
                start_str = str(start_date) if start_date else ""
                end_str = str(end_date) if end_date else ""

            return start_str, end_str
        except Exception as e:
            logger.debug(f"Failed to extract date range: {e}")
            return "", ""

    def validate_cache(
        self,
        prices: pl.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate cache using full PrecomputeMetadata comparison.

        This method checks all factors that should trigger cache regeneration:
        - Version changes
        - Signal registry changes (new/removed signals)
        - Signal parameter changes
        - Universe (ticker list) changes
        - Date range changes (with incremental support)

        Supports incremental updates:
        - 1-day increments at the end don't trigger full recomputation
        - Only structural changes trigger recomputation

        Args:
            prices: Price DataFrame
            config: Signal configuration

        Returns:
            Tuple of (is_valid, reason)
        """
        if config is None:
            config = self.DEFAULT_CONFIG

        # Load cached metadata
        cached_metadata = self._load_precompute_metadata()
        if cached_metadata is None:
            return False, "No cached metadata found"

        # Compute current metadata
        tickers = prices.select("ticker").unique().to_series().to_list()
        start_date, end_date = self._get_date_range(prices)

        current_metadata = PrecomputeMetadata(
            created_at=datetime.now().isoformat(),
            signal_registry_hash=self._compute_signal_registry_hash(),
            signal_config_hash=self._compute_signal_config_hash(config),
            universe_hash=self._compute_universe_hash(tickers),
            prices_hash=self._compute_prices_hash(prices),
            version=PRECOMPUTE_VERSION,
            cached_start_date=start_date,
            cached_end_date=end_date,
            ticker_count=len(tickers),
        )

        # Compare
        is_valid, reason = cached_metadata.is_cache_valid(current_metadata)

        if not is_valid:
            logger.warning(f"Cache invalidated: {reason}")
        else:
            logger.debug("Cache validation passed")

        return is_valid, reason

    def validate_cache_incremental(
        self,
        prices: pl.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> CacheValidationResult:
        """
        Incremental cache validation with granular recomputation support (task_041_7a).

        This method determines whether cache can be used and if so,
        whether full computation or only incremental update is needed.
        Also detects newly added signals and new tickers that need computation.

        Returns:
            CacheValidationResult with:
            - can_use_cache=True, all None: Full cache hit - no computation needed
            - can_use_cache=True, incremental_start set: Time-based update needed
            - can_use_cache=True, missing_signals set: New signals need computation
            - can_use_cache=True, new_tickers set: New tickers need computation
            - can_use_cache=False: Full recomputation required

        Invalidation conditions (trigger full recomputation):
            - Version change
            - Start date moved backward (requesting earlier data)

        Incremental update conditions:
            - End date extended forward (new data at the end)
            - New signals added (only compute missing signals)
            - New tickers added (only compute for new tickers - independent signals only)
            - Tickers removed only (cache still valid, no recomputation)
        """
        if config is None:
            config = self.DEFAULT_CONFIG

        # Load cached metadata
        cached_metadata = self._load_precompute_metadata()
        if cached_metadata is None:
            return CacheValidationResult(
                can_use_cache=False,
                reason="No cached metadata found",
            )

        # Compute current metadata
        current_tickers = set(prices.select("ticker").unique().to_series().to_list())
        start_date, end_date = self._get_date_range(prices)

        # Version check - always requires full recomputation
        if cached_metadata.version != PRECOMPUTE_VERSION:
            return CacheValidationResult(
                can_use_cache=False,
                reason=f"Version changed: {cached_metadata.version} -> {PRECOMPUTE_VERSION}",
            )

        # ===== Ticker difference detection (task_041_7a) =====
        cached_tickers = set(self._metadata.get("tickers", []))

        new_tickers = current_tickers - cached_tickers
        removed_tickers = cached_tickers - current_tickers

        # Tickers removed only → cache still valid (no computation needed)
        if removed_tickers and not new_tickers:
            logger.info(
                f"Tickers removed only: {len(removed_tickers)} removed, cache still valid"
            )
            # Continue to check other conditions (time, signals)

        # New tickers added → incremental computation for new tickers
        # Note: Only works for independent signals; relative signals need full recompute
        new_tickers_list: list[str] | None = None
        if new_tickers:
            # Check if all required signals are independent
            required_signals = self._get_required_signals(config)
            classified = self.classify_signals(required_signals)

            if classified["relative"]:
                # Has relative signals → full recomputation required
                return CacheValidationResult(
                    can_use_cache=False,
                    reason=f"New tickers added but relative signals present: {classified['relative']}",
                )

            # All independent → can compute incrementally for new tickers
            new_tickers_list = list(new_tickers)
            logger.info(f"New tickers detected: {len(new_tickers)} (independent signals only)")

        # Date range checks
        if not cached_metadata.cached_start_date or not cached_metadata.cached_end_date:
            return CacheValidationResult(
                can_use_cache=False,
                reason="Cached metadata missing date range",
            )

        # Start date moved backward → full recomputation
        if start_date < cached_metadata.cached_start_date:
            return CacheValidationResult(
                can_use_cache=False,
                reason=f"Start date moved backward: {cached_metadata.cached_start_date} -> {start_date}",
            )

        # Start date moved forward → partial data loss, need full recomputation
        if start_date > cached_metadata.cached_start_date:
            return CacheValidationResult(
                can_use_cache=False,
                reason=f"Start date changed: {cached_metadata.cached_start_date} -> {start_date}",
            )

        # Detect missing signals (new signals added to config)
        cached_signals = set(self._metadata.get("signals", []))
        current_signals = set(self._get_required_signals(config))
        missing_signals_set = current_signals - cached_signals
        missing_signals = list(missing_signals_set) if missing_signals_set else None

        # Determine incremental time update
        incremental_start: datetime | None = None
        time_reason = ""

        # End date moved backward → cache covers requested period
        if end_date < cached_metadata.cached_end_date:
            logger.warning(
                f"Requested end date ({end_date}) is before cached end date "
                f"({cached_metadata.cached_end_date}). Using cache as-is."
            )
            time_reason = "Cache covers requested period (end date earlier)"

        # End date same → no time-based incremental needed
        elif end_date == cached_metadata.cached_end_date:
            time_reason = "Dates match exactly"

        # End date extended forward → incremental time update needed
        elif end_date > cached_metadata.cached_end_date:
            try:
                cached_end_str = cached_metadata.cached_end_date
                if "T" in cached_end_str:
                    incremental_start = datetime.fromisoformat(cached_end_str)
                else:
                    incremental_start = datetime.fromisoformat(cached_end_str + "T00:00:00")
                time_reason = f"Incremental update: {cached_metadata.cached_end_date} -> {end_date}"
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse cached end date: {e}")
                return CacheValidationResult(
                    can_use_cache=False,
                    reason=f"Failed to parse cached end date: {e}",
                )

        # Build final reason
        reasons = []
        if time_reason and time_reason != "Dates match exactly":
            reasons.append(time_reason)
        if missing_signals:
            reasons.append(f"New signals: {missing_signals}")
        if new_tickers_list:
            reasons.append(f"New tickers: {len(new_tickers_list)}")

        if reasons:
            reason = "; ".join(reasons)
            logger.info(f"Incremental update needed: {reason}")
        else:
            reason = "Full cache hit - dates match exactly"

        return CacheValidationResult(
            can_use_cache=True,
            reason=reason,
            incremental_start=incremental_start,
            missing_signals=missing_signals,
            new_tickers=new_tickers_list,
        )

    def precompute_all(
        self,
        prices: pl.DataFrame,
        config: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """
        Pre-compute all signals with full incremental support.

        This is the main entry point for signal pre-computation.
        Supports incremental updates to minimize computation:
        - Full recomputation when cache is invalid or force=True
        - New tickers computation when universe is expanded
        - Missing signals computation when new signals are added
        - Incremental update when only end date is extended
        - No computation when cache is fully valid

        Args:
            prices: Price DataFrame with columns [timestamp, ticker, close, high, low, volume]
            config: Signal configuration (periods, etc.). Uses DEFAULT_CONFIG if None.
            force: Force full recomputation even if cache is valid

        Returns:
            True if any computation was performed, False if cache was used as-is
        """
        config = config or self.DEFAULT_CONFIG

        # Force recomputation requested
        if force:
            logger.info("Force recomputation requested")
            return self._full_precompute(prices, config)

        # Incremental cache validation using CacheValidationResult
        result = self.validate_cache_incremental(prices, config)

        # Cache invalid - full recomputation needed
        if not result.can_use_cache:
            logger.info(f"Full recomputation needed: {result.reason}")
            return self._full_precompute(prices, config)

        # New tickers added - compute for new tickers (task_041_6a)
        if result.new_tickers:
            logger.info(f"Computing for {len(result.new_tickers)} new tickers: {result.new_tickers}")
            return self.precompute_for_new_tickers(prices, result.new_tickers, config)

        # New signals added - compute only missing signals
        if result.missing_signals:
            logger.info(f"Computing {len(result.missing_signals)} new signals: {result.missing_signals}")
            return self.precompute_missing_signals(prices, result.missing_signals, config)

        # Date range extended - incremental update
        if result.incremental_start is not None:
            logger.info(f"Incremental update from {result.incremental_start}")
            return self.precompute_incremental(prices, result.incremental_start, config)

        # Full cache hit - no computation needed
        logger.info("Cache is valid, no computation needed")
        return False

    def _full_precompute(
        self,
        prices: pl.DataFrame,
        config: dict[str, Any],
    ) -> bool:
        """
        Perform full signal precomputation.

        This is the internal method that does the actual computation.
        Called by precompute_all() when full recomputation is needed.

        Args:
            prices: Price DataFrame
            config: Signal configuration

        Returns:
            True if computation completed successfully
        """
        logger.info("Starting full signal precomputation...")
        start_time = datetime.now()

        tickers = prices.select("ticker").unique().to_series().to_list()
        logger.info(f"Computing signals for {len(tickers)} tickers")

        computed_signals = []

        for period in config.get("momentum_periods", [20, 60]):
            self._compute_and_save_momentum(prices, period, tickers)
            computed_signals.append(f"momentum_{period}")

        for period in config.get("volatility_periods", [20, 60]):
            self._compute_and_save_volatility(prices, period, tickers)
            computed_signals.append(f"volatility_{period}")

        for period in config.get("rsi_periods", [14]):
            self._compute_and_save_rsi(prices, period, tickers)
            computed_signals.append(f"rsi_{period}")

        for period in config.get("zscore_periods", [20, 60]):
            self._compute_and_save_zscore(prices, period, tickers)
            computed_signals.append(f"zscore_{period}")

        for period in config.get("sharpe_periods", [60, 252]):
            self._compute_and_save_sharpe(prices, period, tickers)
            computed_signals.append(f"sharpe_{period}")

        # Save PrecomputeMetadata for intelligent cache invalidation
        start_date, end_date = self._get_date_range(prices)
        metadata = PrecomputeMetadata(
            created_at=datetime.now().isoformat(),
            signal_registry_hash=self._compute_signal_registry_hash(),
            signal_config_hash=self._compute_signal_config_hash(config),
            universe_hash=self._compute_universe_hash(tickers),
            prices_hash=self._compute_prices_hash(prices),
            version=PRECOMPUTE_VERSION,
            cached_start_date=start_date,
            cached_end_date=end_date,
            ticker_count=len(tickers),
        )
        self._save_precompute_metadata(metadata)

        # Also save legacy metadata for backward compatibility
        self._metadata = {
            "prices_hash": metadata.prices_hash,
            "computed_at": metadata.created_at,
            "signals": computed_signals,
            "tickers": tickers,
            "config": config,
        }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Full signal precomputation completed in {elapsed:.2f}s ({len(computed_signals)} signals)")

        return True

    def precompute_incremental(
        self,
        prices: pl.DataFrame,
        start_from: datetime,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Incremental signal computation (new data only).

        This method computes signals only for new data (from start_from onwards)
        and appends them to existing cached Parquet files.

        To ensure correct signal values, this method includes sufficient
        historical data (lookback period) before start_from for calculations
        like momentum_252 which need 252 days of prior data.

        Args:
            prices: Full price DataFrame (must include history for lookback)
            start_from: Compute signals from this date onwards
            config: Signal configuration (uses DEFAULT_CONFIG if None)

        Returns:
            True if incremental computation succeeded
        """
        from datetime import timedelta

        if config is None:
            config = self.DEFAULT_CONFIG

        logger.info(f"Starting incremental signal precomputation from {start_from}")
        start_time = datetime.now()

        # Calculate maximum lookback needed across all signal types
        max_lookback = self._get_max_lookback(config)
        lookback_buffer = 10  # Extra buffer for edge cases

        # Calculate the earliest date we need for accurate signal computation
        lookback_start = start_from - timedelta(days=max_lookback + lookback_buffer)

        # Filter prices to required period (history + new data)
        prices_subset = prices.filter(
            pl.col("timestamp") >= lookback_start
        ).sort(["ticker", "timestamp"])

        if prices_subset.is_empty():
            logger.warning("No data in price subset for incremental computation")
            return False

        tickers = prices_subset.select("ticker").unique().to_series().to_list()
        logger.info(
            f"Incremental computation: {len(tickers)} tickers, "
            f"lookback_start={lookback_start.date()}, start_from={start_from.date()}"
        )

        computed_signals = []

        # Compute each signal type incrementally
        for period in config.get("momentum_periods", [20, 60]):
            success = self._compute_incremental_momentum(
                prices_subset, period, tickers, start_from
            )
            if success:
                computed_signals.append(f"momentum_{period}")

        for period in config.get("volatility_periods", [20, 60]):
            success = self._compute_incremental_volatility(
                prices_subset, period, tickers, start_from
            )
            if success:
                computed_signals.append(f"volatility_{period}")

        # RSI uses EMA-style calculation that depends on entire history.
        # For precision (diff < 1e-10), we must use full prices, not subset.
        for period in config.get("rsi_periods", [14]):
            success = self._compute_incremental_rsi(
                prices, period, tickers, start_from
            )
            if success:
                computed_signals.append(f"rsi_{period}")

        for period in config.get("zscore_periods", [20, 60]):
            success = self._compute_incremental_zscore(
                prices_subset, period, tickers, start_from
            )
            if success:
                computed_signals.append(f"zscore_{period}")

        # Sharpe uses rolling window calculation.
        # For precision (diff < 1e-10), we must use full prices, not subset.
        for period in config.get("sharpe_periods", [60, 252]):
            success = self._compute_incremental_sharpe(
                prices, period, tickers, start_from
            )
            if success:
                computed_signals.append(f"sharpe_{period}")

        # Update metadata with new end date
        new_end_date = prices.select("timestamp").max().item()
        self._update_metadata_end_date(new_end_date)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Incremental precomputation completed in {elapsed:.2f}s "
            f"({len(computed_signals)} signals updated)"
        )

        return True

    def precompute_missing_signals(
        self,
        prices: pl.DataFrame,
        missing_signals: list[str],
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Compute only newly added signals.

        This method computes only the signals that are missing from the cache,
        preserving existing cached signals. Used when new signals are added
        to the registry without requiring full recomputation.

        Args:
            prices: Full price DataFrame
            missing_signals: List of signal names to compute (e.g., ["momentum_120", "rsi_21"])
            config: Signal configuration (uses DEFAULT_CONFIG if None)

        Returns:
            True if all missing signals were computed successfully
        """
        if not missing_signals:
            logger.info("No missing signals to compute")
            return True

        config = config or self.DEFAULT_CONFIG

        logger.info(f"Computing {len(missing_signals)} missing signals: {missing_signals}")
        start_time = datetime.now()

        tickers = prices.select("ticker").unique().to_series().to_list()
        computed_signals = []

        for signal_name in missing_signals:
            try:
                logger.info(f"Computing new signal: {signal_name}")
                success = self._compute_and_save_signal(prices, signal_name, tickers)
                if success:
                    computed_signals.append(signal_name)
            except Exception as e:
                logger.error(f"Failed to compute signal {signal_name}: {e}")

        # Update metadata
        self._update_metadata_for_new_signals(computed_signals)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Missing signals computation completed in {elapsed:.2f}s "
            f"({len(computed_signals)}/{len(missing_signals)} signals computed)"
        )

        return len(computed_signals) == len(missing_signals)

    def _compute_and_save_signal(
        self,
        prices: pl.DataFrame,
        signal_name: str,
        tickers: list[str],
    ) -> bool:
        """
        Compute and save a single signal by name.

        Parses signal name (e.g., "momentum_20") to determine type and period,
        then calls the appropriate computation method.

        Args:
            prices: Price DataFrame
            signal_name: Signal name (e.g., "momentum_20", "rsi_14")
            tickers: List of tickers

        Returns:
            True if successful
        """
        # Parse signal name: "type_period" -> (type, period)
        parts = signal_name.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid signal name format: {signal_name}")
            return False

        signal_type, period_str = parts
        try:
            period = int(period_str)
        except ValueError:
            logger.warning(f"Invalid period in signal name: {signal_name}")
            return False

        # Dispatch to appropriate computation method
        if signal_type == "momentum":
            self._compute_and_save_momentum(prices, period, tickers)
        elif signal_type == "volatility":
            self._compute_and_save_volatility(prices, period, tickers)
        elif signal_type == "rsi":
            self._compute_and_save_rsi(prices, period, tickers)
        elif signal_type == "zscore":
            self._compute_and_save_zscore(prices, period, tickers)
        elif signal_type == "sharpe":
            self._compute_and_save_sharpe(prices, period, tickers)
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return False

        return True

    def _update_metadata_for_new_signals(self, new_signals: list[str]) -> None:
        """
        Update metadata after computing new signals.

        Updates:
        - signals list (adds new signals)
        - signal_registry_hash (to reflect current registry state)

        Args:
            new_signals: List of newly computed signal names
        """
        if not new_signals:
            return

        # Load existing metadata
        metadata = self._load_precompute_metadata()
        if metadata is None:
            logger.warning("Cannot update metadata: no existing metadata found")
            return

        # Update legacy metadata signals list
        current_signals = self._metadata.get("signals", [])
        updated_signals = list(set(current_signals) | set(new_signals))
        self._metadata["signals"] = updated_signals

        # Update signal_registry_hash to reflect current state
        new_registry_hash = self._compute_signal_registry_hash()

        # Create updated PrecomputeMetadata
        updated_metadata = PrecomputeMetadata(
            created_at=metadata.created_at,
            signal_registry_hash=new_registry_hash,
            signal_config_hash=metadata.signal_config_hash,
            universe_hash=metadata.universe_hash,
            prices_hash=metadata.prices_hash,
            version=metadata.version,
            cached_start_date=metadata.cached_start_date,
            cached_end_date=metadata.cached_end_date,
            ticker_count=metadata.ticker_count,
        )

        self._save_precompute_metadata(updated_metadata)
        self._save_metadata()

        logger.info(f"Metadata updated with {len(new_signals)} new signals")

    def precompute_for_new_tickers(
        self,
        prices: pl.DataFrame,
        new_tickers: list[str],
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Incremental computation for newly added tickers.

        - Independent signals: Compute only for new tickers, append to cache
        - Relative signals: Full recomputation for all tickers

        Args:
            prices: Full price DataFrame including new tickers
            new_tickers: List of newly added ticker symbols
            config: Signal configuration

        Returns:
            True if computation succeeded
        """
        if not new_tickers:
            logger.info("No new tickers to process")
            return True

        config = config or self.DEFAULT_CONFIG
        logger.info(f"Processing {len(new_tickers)} new tickers: {new_tickers}")
        start_time = datetime.now()

        all_signals = self._get_required_signals(config)
        all_tickers = prices.select("ticker").unique().to_series().to_list()
        independent_count = 0
        relative_count = 0

        for signal_name in all_signals:
            signal_type = self.classify_signal(signal_name)
            if signal_type == "independent":
                logger.debug(f"Computing {signal_name} for new tickers only")
                new_ticker_prices = prices.filter(pl.col("ticker").is_in(new_tickers))
                self._compute_and_append_signal(new_ticker_prices, signal_name, new_tickers)
                independent_count += 1
            else:
                logger.debug(f"Recomputing {signal_name} for all tickers (relative)")
                self._compute_and_save_signal(prices, signal_name, all_tickers)
                relative_count += 1

        self._update_metadata_tickers(all_tickers)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"New tickers completed in {elapsed:.2f}s: {independent_count} independent, {relative_count} relative")
        return True

    def _compute_and_append_signal(self, prices: pl.DataFrame, signal_name: str, tickers: list[str]) -> bool:
        """Compute signal for specific tickers and append to existing Parquet."""
        parts = signal_name.rsplit("_", 1)
        if len(parts) != 2:
            return False
        signal_type, period_str = parts
        try:
            period = int(period_str)
        except ValueError:
            return False

        new_signals = self._compute_signal_df(prices, signal_type, period, tickers)
        if new_signals is None or new_signals.is_empty():
            return True

        filename = f"{signal_name}.parquet"
        if self._parquet_exists(filename):
            existing = self._read_parquet(filename)
            existing_filtered = existing.filter(~pl.col("ticker").is_in(tickers))
            combined = pl.concat([existing_filtered, new_signals]).sort(["ticker", "timestamp"])
            self._write_parquet(combined, filename)
        else:
            self._write_parquet(new_signals, filename)
        return True

    def _compute_signal_df(self, prices: pl.DataFrame, signal_type: str, period: int, tickers: list[str]) -> pl.DataFrame | None:
        """Compute signal and return as DataFrame."""
        if signal_type == "momentum":
            return self._compute_momentum_df(prices, period, tickers)
        elif signal_type == "volatility":
            return self._compute_volatility_df(prices, period, tickers)
        elif signal_type == "rsi":
            return self._compute_rsi_df(prices, period, tickers)
        elif signal_type == "zscore":
            return self._compute_zscore_df(prices, period, tickers)
        elif signal_type == "sharpe":
            return self._compute_sharpe_df(prices, period, tickers)
        return None

    def _compute_momentum_df(self, prices: pl.DataFrame, period: int, tickers: list[str]) -> pl.DataFrame:
        """Compute momentum signal as DataFrame."""
        return (prices.sort(["ticker", "timestamp"])
                .with_columns([((pl.col("close") - pl.col("close").shift(period).over("ticker"))
                               / pl.col("close").shift(period).over("ticker")).alias("value")])
                .select(["timestamp", "ticker", "value"])
                .filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan()))

    def _compute_volatility_df(self, prices: pl.DataFrame, period: int, tickers: list[str]) -> pl.DataFrame:
        """Compute volatility signal as DataFrame."""
        results = []
        for ticker in tickers:
            td = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(td) < period + 1:
                continue
            close = td.select("close").to_series()
            ts = td.select("timestamp").to_series()
            vol = close.pct_change().rolling_std(period) * np.sqrt(252)
            results.append(pl.DataFrame({"timestamp": ts, "ticker": [ticker] * len(ts), "value": vol}))
        return pl.concat(results).filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan()) if results else pl.DataFrame({"timestamp": [], "ticker": [], "value": []})

    def _compute_rsi_df(self, prices: pl.DataFrame, period: int, tickers: list[str]) -> pl.DataFrame:
        """Compute RSI signal as DataFrame."""
        results = []
        for ticker in tickers:
            td = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(td) < period + 1:
                continue
            close = td.select("close").to_series().to_numpy()
            ts = td.select("timestamp").to_series()
            rsi = self._vectorized_rsi(close, period)
            results.append(pl.DataFrame({"timestamp": ts, "ticker": [ticker] * len(ts), "value": rsi}))
        return pl.concat(results).filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan()) if results else pl.DataFrame({"timestamp": [], "ticker": [], "value": []})

    def _compute_zscore_df(self, prices: pl.DataFrame, period: int, tickers: list[str]) -> pl.DataFrame:
        """Compute z-score signal as DataFrame."""
        results = []
        for ticker in tickers:
            td = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(td) < period + 1:
                continue
            close = td.select("close").to_series()
            ts = td.select("timestamp").to_series()
            zscore = (close - close.rolling_mean(period)) / close.rolling_std(period)
            results.append(pl.DataFrame({"timestamp": ts, "ticker": [ticker] * len(ts), "value": zscore}))
        return pl.concat(results).filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan()) if results else pl.DataFrame({"timestamp": [], "ticker": [], "value": []})

    def _compute_sharpe_df(self, prices: pl.DataFrame, period: int, tickers: list[str]) -> pl.DataFrame:
        """Compute Sharpe ratio signal as DataFrame."""
        results = []
        for ticker in tickers:
            td = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(td) < period + 1:
                continue
            close = td.select("close").to_series().to_numpy()
            ts = td.select("timestamp").to_series()
            sharpe = self._vectorized_sharpe(close, period)
            results.append(pl.DataFrame({"timestamp": ts, "ticker": [ticker] * len(ts), "value": sharpe}))
        return pl.concat(results).filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan()) if results else pl.DataFrame({"timestamp": [], "ticker": [], "value": []})

    def _update_metadata_tickers(self, tickers: list[str]) -> None:
        """Update metadata after adding new tickers."""
        metadata = self._load_precompute_metadata()
        if metadata is None:
            return
        updated = PrecomputeMetadata(
            created_at=metadata.created_at,
            signal_registry_hash=metadata.signal_registry_hash,
            signal_config_hash=metadata.signal_config_hash,
            universe_hash=self._compute_universe_hash(tickers),
            prices_hash=metadata.prices_hash,
            version=metadata.version,
            cached_start_date=metadata.cached_start_date,
            cached_end_date=metadata.cached_end_date,
            ticker_count=len(tickers),
        )
        # Update legacy metadata first (may be overwritten)
        self._metadata["tickers"] = tickers
        self._save_metadata()
        # Save PrecomputeMetadata last to ensure it's preserved
        self._save_precompute_metadata(updated)

    def _get_max_lookback(self, config: dict[str, Any]) -> int:
        """Calculate maximum lookback period needed across all signals."""
        all_periods = []
        all_periods.extend(config.get("momentum_periods", []))
        all_periods.extend(config.get("volatility_periods", []))
        all_periods.extend(config.get("rsi_periods", []))
        all_periods.extend(config.get("zscore_periods", []))
        all_periods.extend(config.get("sharpe_periods", []))
        return max(all_periods) if all_periods else 252

    def _update_metadata_end_date(self, new_end_date: datetime) -> None:
        """Update cached_end_date in metadata after incremental computation."""
        metadata = self._load_precompute_metadata()
        if metadata is None:
            logger.warning("Cannot update metadata: no existing metadata found")
            return

        # Convert to ISO format string
        if hasattr(new_end_date, 'isoformat'):
            end_date_str = new_end_date.isoformat()
        else:
            end_date_str = str(new_end_date)

        # Create updated metadata
        updated_metadata = PrecomputeMetadata(
            created_at=metadata.created_at,
            signal_registry_hash=metadata.signal_registry_hash,
            signal_config_hash=metadata.signal_config_hash,
            universe_hash=metadata.universe_hash,
            prices_hash=metadata.prices_hash,
            version=metadata.version,
            cached_start_date=metadata.cached_start_date,
            cached_end_date=end_date_str,
            ticker_count=metadata.ticker_count,
        )

        self._save_precompute_metadata(updated_metadata)
        logger.debug(f"Updated metadata end_date to {end_date_str}")

    def _append_to_parquet(
        self,
        signal_name: str,
        new_data: pl.DataFrame,
        start_from: datetime,
    ) -> bool:
        """
        Append new signal data to existing Parquet file.

        Args:
            signal_name: Signal name (e.g., "momentum_20")
            new_data: New signal data to append
            start_from: Only append data from this date onwards

        Returns:
            True if successful
        """
        filename = f"{signal_name}.parquet"

        # Filter new data to only include dates >= start_from
        new_data_filtered = new_data.filter(pl.col("timestamp") >= start_from)

        if new_data_filtered.is_empty():
            logger.debug(f"No new data to append for {signal_name}")
            return True

        if self._parquet_exists(filename):
            # Load existing data
            existing = self._read_parquet(filename)

            # Remove any existing data >= start_from (will be replaced)
            existing_filtered = existing.filter(pl.col("timestamp") < start_from)

            # Concatenate
            combined = pl.concat([existing_filtered, new_data_filtered])
            combined = combined.sort(["ticker", "timestamp"])
        else:
            combined = new_data_filtered.sort(["ticker", "timestamp"])

        # Save
        self._write_parquet(combined, filename)
        logger.debug(f"Appended {len(new_data_filtered)} rows to {signal_name}")

        return True

    def _compute_incremental_momentum(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        start_from: datetime,
    ) -> bool:
        """Compute momentum signal incrementally and append to cache."""
        # Compute momentum for all data (including lookback)
        result = (
            prices
            .sort(["ticker", "timestamp"])
            .with_columns([
                (
                    (pl.col("close") - pl.col("close").shift(period).over("ticker"))
                    / pl.col("close").shift(period).over("ticker")
                ).alias("value")
            ])
            .select(["timestamp", "ticker", "value"])
            .filter(pl.col("value").is_not_null())
            .filter(pl.col("value").is_not_nan())
        )

        return self._append_to_parquet(f"momentum_{period}", result, start_from)

    def _compute_incremental_volatility(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        start_from: datetime,
    ) -> bool:
        """Compute volatility signal incrementally and append to cache."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series()
            timestamps = ticker_data.select("timestamp").to_series()

            returns = close.pct_change()
            volatility = returns.rolling_std(period) * np.sqrt(252)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": volatility,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            return self._append_to_parquet(f"volatility_{period}", combined, start_from)
        return True

    def _compute_incremental_rsi(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        start_from: datetime,
    ) -> bool:
        """Compute RSI signal incrementally and append to cache.

        Note: RSI uses EMA-style calculation that depends on entire history.
        For precision (diff < 1e-10), we must compute from the very beginning
        of the price data, not just from lookback_start.
        """
        # Get the full price data from the original precomputer prices
        # RSI needs full history for EMA initialization to match exactly
        full_prices = prices.sort(["ticker", "timestamp"])

        results = []

        for ticker in tickers:
            ticker_data = full_prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series().to_numpy()
            timestamps = ticker_data.select("timestamp").to_series()

            rsi = self._vectorized_rsi(close, period)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": rsi,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            return self._append_to_parquet(f"rsi_{period}", combined, start_from)
        return True

    def _compute_incremental_zscore(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        start_from: datetime,
    ) -> bool:
        """Compute z-score signal incrementally and append to cache."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series()
            timestamps = ticker_data.select("timestamp").to_series()

            rolling_mean = close.rolling_mean(period)
            rolling_std = close.rolling_std(period)
            zscore = (close - rolling_mean) / rolling_std

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": zscore,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            return self._append_to_parquet(f"zscore_{period}", combined, start_from)
        return True

    def _compute_incremental_sharpe(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        start_from: datetime,
    ) -> bool:
        """Compute Sharpe ratio signal incrementally and append to cache."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series().to_numpy()
            timestamps = ticker_data.select("timestamp").to_series()

            sharpe = self._vectorized_sharpe(close, period)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": sharpe,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            return self._append_to_parquet(f"sharpe_{period}", combined, start_from)
        return True

    def _compute_and_save_momentum(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> None:
        """Compute momentum signal for all tickers and save to Parquet.

        Vectorized implementation (task_013_2) - 5-10x faster than sequential.
        Uses polars .over() for efficient group-wise calculation.
        """
        # Vectorized computation using polars .over() for group-wise operations
        result = (
            prices
            .sort(["ticker", "timestamp"])
            .with_columns([
                # Momentum = (close - close_shifted) / close_shifted
                (
                    (pl.col("close") - pl.col("close").shift(period).over("ticker"))
                    / pl.col("close").shift(period).over("ticker")
                ).alias("value")
            ])
            .select(["timestamp", "ticker", "value"])
        )

        # Filter out NaN values (from shift) and tickers with insufficient data
        # Count rows per ticker to filter those with < period+1 data points
        ticker_counts = prices.group_by("ticker").agg(pl.count().alias("cnt"))
        valid_tickers = (
            ticker_counts
            .filter(pl.col("cnt") >= period + 1)
            .select("ticker")
            .to_series()
            .to_list()
        )

        combined = (
            result
            .filter(pl.col("ticker").is_in(valid_tickers))
            .filter(pl.col("value").is_not_null())
            .filter(pl.col("value").is_not_nan())
        )

        if len(combined) > 0:
            self._write_parquet(combined, f"momentum_{period}.parquet")
            logger.debug(f"Saved momentum_{period} ({len(combined)} rows)")

    def _compute_momentum_sequential(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> pl.DataFrame:
        """Original sequential implementation for validation (task_013_2).

        This method is kept for precision validation against the vectorized version.
        Returns DataFrame instead of saving to file.
        """
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series()
            timestamps = ticker_data.select("timestamp").to_series()

            momentum = (close - close.shift(period)) / close.shift(period)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": momentum,
            })
            results.append(result_df)

        if results:
            return pl.concat(results).filter(
                pl.col("value").is_not_null() & ~pl.col("value").is_nan()
            )
        return pl.DataFrame({"timestamp": [], "ticker": [], "value": []})

    def _compute_and_save_volatility(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> None:
        """Compute volatility signal for all tickers and save to Parquet."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series()
            timestamps = ticker_data.select("timestamp").to_series()

            returns = close.pct_change()
            volatility = returns.rolling_std(period) * np.sqrt(252)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": volatility,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            self._write_parquet(combined, f"volatility_{period}.parquet")
            logger.debug(f"Saved volatility_{period} ({len(combined)} rows)")

    def _compute_and_save_rsi(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> None:
        """Compute RSI signal for all tickers and save to Parquet."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series().to_numpy()
            timestamps = ticker_data.select("timestamp").to_series()

            rsi = self._vectorized_rsi(close, period)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": rsi,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            self._write_parquet(combined, f"rsi_{period}.parquet")
            logger.debug(f"Saved rsi_{period} ({len(combined)} rows)")

    def _compute_and_save_zscore(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> None:
        """Compute z-score signal for all tickers and save to Parquet."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series()
            timestamps = ticker_data.select("timestamp").to_series()

            rolling_mean = close.rolling_mean(period)
            rolling_std = close.rolling_std(period)
            zscore = (close - rolling_mean) / rolling_std

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": zscore,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            self._write_parquet(combined, f"zscore_{period}.parquet")
            logger.debug(f"Saved zscore_{period} ({len(combined)} rows)")

    def _compute_and_save_sharpe(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
    ) -> None:
        """Compute rolling Sharpe ratio for all tickers and save to Parquet."""
        results = []

        for ticker in tickers:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < period + 1:
                continue

            close = ticker_data.select("close").to_series().to_numpy()
            timestamps = ticker_data.select("timestamp").to_series()

            sharpe = self._vectorized_sharpe(close, period)

            result_df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(timestamps),
                "value": sharpe,
            })
            results.append(result_df)

        if results:
            combined = pl.concat(results)
            self._write_parquet(combined, f"sharpe_{period}.parquet")
            logger.debug(f"Saved sharpe_{period} ({len(combined)} rows)")

    @staticmethod
    def _vectorized_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """
        Vectorized RSI calculation.

        Args:
            close: Close price array
            period: RSI period

        Returns:
            RSI values array
        """
        n = len(close)
        result = np.full(n, np.nan)

        if n < period + 1:
            return result

        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros(n - 1)
        avg_loss = np.zeros(n - 1)

        avg_gain[period - 1] = np.mean(gains[:period])
        avg_loss[period - 1] = np.mean(losses[:period])

        for i in range(period, n - 1):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.where(np.isinf(rs), 100, rsi)

        result[period:] = rsi[period - 1:]
        return result

    @staticmethod
    def _vectorized_sharpe(close: np.ndarray, period: int, annualize: int = 252) -> np.ndarray:
        """
        Vectorized rolling Sharpe ratio calculation.

        Args:
            close: Close price array
            period: Rolling window period
            annualize: Annualization factor

        Returns:
            Rolling Sharpe ratio array
        """
        n = len(close)
        result = np.full(n, np.nan)

        if n < period + 1:
            return result

        returns = np.diff(close) / close[:-1]
        sqrt_ann = np.sqrt(annualize)

        for i in range(period, n):
            window_returns = returns[i - period:i]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns, ddof=1)
            if std_ret > 0:
                result[i] = mean_ret / std_ret * sqrt_ann
            else:
                result[i] = 0.0

        return result

    def load_signal(self, signal_name: str, ticker: str | None = None) -> pl.DataFrame:
        """
        Load precomputed signal from Parquet.

        Args:
            signal_name: Signal name (e.g., "momentum_20", "volatility_60")
            ticker: Optional ticker to filter. If None, returns all tickers.

        Returns:
            DataFrame with columns [timestamp, ticker, value]
        """
        filename = f"{signal_name}.parquet"

        if not self._parquet_exists(filename):
            raise FileNotFoundError(f"Signal file not found: {filename}")

        df = self._read_parquet(filename)

        if ticker is not None:
            df = df.filter(pl.col("ticker") == ticker)

        return df

    def get_signal_at_date(
        self,
        signal_name: str,
        ticker: str,
        date: datetime,
    ) -> float | None:
        """
        Get signal value for a specific ticker at a specific date.

        This is the primary method used during backtest iterations.

        Args:
            signal_name: Signal name (e.g., "momentum_20")
            ticker: Asset ticker
            date: Target date

        Returns:
            Signal value or None if not found
        """
        try:
            df = self.load_signal(signal_name, ticker)

            date_pl = pl.lit(date).cast(pl.Datetime)
            exact = df.filter(pl.col("timestamp") == date_pl)

            if len(exact) > 0:
                return exact.select("value").item()

            before = df.filter(pl.col("timestamp") <= date_pl).sort("timestamp", descending=True)
            if len(before) > 0:
                return before.head(1).select("value").item()

            return None
        except Exception as e:
            logger.debug(f"Failed to get signal {signal_name} for {ticker} at {date}: {e}")
            return None

    def get_signals_at_date(
        self,
        signal_name: str,
        date: datetime,
    ) -> dict[str, float]:
        """
        Get signal values for all tickers at a specific date.

        Optimized for batch retrieval during backtest.

        Args:
            signal_name: Signal name (e.g., "momentum_20")
            date: Target date

        Returns:
            Dictionary of ticker -> signal value
        """
        try:
            df = self.load_signal(signal_name)

            date_pl = pl.lit(date).cast(pl.Datetime)

            latest_per_ticker = (
                df.filter(pl.col("timestamp") <= date_pl)
                .sort("timestamp", descending=True)
                .group_by("ticker")
                .first()
            )

            result = {}
            for row in latest_per_ticker.iter_rows(named=True):
                if row["value"] is not None and not np.isnan(row["value"]):
                    result[row["ticker"]] = row["value"]

            return result
        except Exception as e:
            logger.warning(f"Failed to get signals {signal_name} at {date}: {e}")
            return {}

    def list_cached_signals(self) -> list[str]:
        """List all cached signal names."""
        parquet_files = self._list_parquet_files()
        signals = [Path(f).stem for f in parquet_files if f != "_metadata.json"]
        return sorted(signals)

    def clear_cache(self) -> None:
        """Clear all cached signals."""
        for filename in self._list_parquet_files():
            self._delete_parquet(filename)

        # Delete metadata
        if self._use_backend and self._backend is not None:
            try:
                self._backend.delete(self._metadata_file_name)
            except Exception:
                pass
        else:
            metadata_file = self._cache_dir / self._metadata_file_name
            if metadata_file.exists():
                metadata_file.unlink()

        self._metadata = {"prices_hash": None, "computed_at": None, "signals": []}
        logger.info("Signal cache cleared")

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        parquet_files = self._list_parquet_files()

        # For backend mode, we can't easily get file sizes without downloading
        if self._use_backend and self._backend is not None:
            return {
                "cache_dir": str(self._cache_dir),
                "backend": self._backend.config.backend,
                "num_signals": len(parquet_files),
                "signals": [Path(f).stem for f in parquet_files],
                "prices_hash": self._metadata.get("prices_hash"),
                "computed_at": self._metadata.get("computed_at"),
            }
        else:
            files = list(self._cache_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in files)

            return {
                "cache_dir": str(self._cache_dir),
                "backend": "local (legacy)",
                "num_signals": len(files),
                "signals": [f.stem for f in files],
                "total_size_mb": total_size / (1024 * 1024),
                "prices_hash": self._metadata.get("prices_hash"),
                "computed_at": self._metadata.get("computed_at"),
            }

    def validate_momentum_vectorization(
        self,
        prices: pl.DataFrame,
        period: int,
        tickers: list[str],
        tolerance: float = 1e-10,
    ) -> dict[str, Any]:
        """Validate that vectorized momentum matches sequential implementation (task_013_2).

        This method compares the vectorized and sequential implementations
        to ensure numerical precision is maintained.

        Args:
            prices: Input price DataFrame with columns [timestamp, ticker, close]
            period: Momentum period
            tickers: List of tickers to validate
            tolerance: Maximum allowed numerical difference (default: 1e-10)

        Returns:
            Dictionary with validation results:
                - passed: bool - Whether validation passed
                - max_diff: float - Maximum absolute difference found
                - num_compared: int - Number of values compared
                - mismatches: int - Number of values exceeding tolerance
        """
        # Compute using sequential method
        sequential_result = self._compute_momentum_sequential(prices, period, tickers)

        # Compute using vectorized method (inline, not saving to file)
        vectorized_result = (
            prices
            .sort(["ticker", "timestamp"])
            .with_columns([
                (
                    (pl.col("close") - pl.col("close").shift(period).over("ticker"))
                    / pl.col("close").shift(period).over("ticker")
                ).alias("value")
            ])
            .select(["timestamp", "ticker", "value"])
        )

        # Filter valid tickers and values
        ticker_counts = prices.group_by("ticker").agg(pl.count().alias("cnt"))
        valid_tickers = (
            ticker_counts
            .filter(pl.col("cnt") >= period + 1)
            .select("ticker")
            .to_series()
            .to_list()
        )

        vectorized_result = (
            vectorized_result
            .filter(pl.col("ticker").is_in(valid_tickers))
            .filter(pl.col("value").is_not_null())
            .filter(pl.col("value").is_not_nan())
        )

        # Join and compare
        joined = sequential_result.join(
            vectorized_result,
            on=["timestamp", "ticker"],
            suffix="_vec",
        )

        if len(joined) == 0:
            return {
                "passed": True,
                "max_diff": 0.0,
                "num_compared": 0,
                "mismatches": 0,
                "message": "No data to compare",
            }

        # Calculate differences
        diffs = (joined["value"] - joined["value_vec"]).abs()
        max_diff = diffs.max()
        mismatches = (diffs > tolerance).sum()

        passed = max_diff <= tolerance

        return {
            "passed": passed,
            "max_diff": float(max_diff) if max_diff is not None else 0.0,
            "num_compared": len(joined),
            "mismatches": int(mismatches),
            "message": "PASS" if passed else f"FAIL: max_diff={max_diff:.2e} > tolerance={tolerance:.2e}",
        }

    # =========================================================================
    # Lead-Lag Signal Cache (task_042_1_opt)
    # =========================================================================

    def precompute_lead_lag(
        self,
        prices: pl.DataFrame,
        lookback: int = 60,
        lag_min: int = 1,
        lag_max: int = 5,
        min_correlation: float = 0.3,
        force: bool = False,
    ) -> bool:
        """
        Lead-Lagシグナルのプリコンピュートとキャッシュ保存。

        相関行列とLeader-Followerペアリストを保存。
        新期間追加時は増分更新。

        Args:
            prices: 価格データ（columns: timestamp, ticker, close）
            lookback: 相関計算期間
            lag_min: 最小ラグ
            lag_max: 最大ラグ
            min_correlation: 最小相関閾値
            force: 強制再計算

        Returns:
            True if computation completed successfully
        """
        import pandas as pd
        from src.signals.lead_lag import LeadLagSignal, LeadLagPair

        cache_filename = "lead_lag_cache.parquet"
        pairs_filename = "lead_lag_pairs.parquet"

        # キャッシュチェック
        if not force and self._parquet_exists(cache_filename) and self._parquet_exists(pairs_filename):
            cached_meta = self._load_precompute_metadata()
            if cached_meta is not None:
                # 期間拡張チェック
                _, current_end = self._get_date_range(prices)
                if cached_meta.cached_end_date:
                    cached_end = datetime.fromisoformat(str(cached_meta.cached_end_date))
                    if current_end <= cached_end:
                        logger.info("Lead-lag cache is valid, skipping computation")
                        return True

        logger.info(f"Computing lead-lag signal (lookback={lookback}, lag={lag_min}-{lag_max})")

        # Wide format に変換
        tickers = prices.select("ticker").unique().to_series().to_list()
        pivot_df = prices.pivot(
            index="timestamp", columns="ticker", values="close"
        ).to_pandas()

        if "timestamp" in pivot_df.columns:
            pivot_df = pivot_df.set_index("timestamp")
        pivot_df.index = pd.to_datetime(pivot_df.index)

        # 数値列のみ抽出
        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
        pivot_df = pivot_df[numeric_cols]

        # LeadLagSignal で計算
        signal = LeadLagSignal(
            lookback=lookback,
            lag_min=lag_min,
            lag_max=lag_max,
            min_correlation=min_correlation,
            use_numba=True,
            use_staged_filter=True,
        )

        result = signal.compute(pivot_df)

        # シグナル値をキャッシュ
        latest_timestamp = pivot_df.index[-1]
        records = [
            {
                "timestamp": latest_timestamp,
                "ticker": ticker,
                "value": float(score),
            }
            for ticker, score in result.scores.items()
        ]
        cache_df = pl.DataFrame(records)
        self._write_parquet(cache_df, cache_filename)

        # ペア情報をキャッシュ
        pairs_data = result.metadata.get("top_pairs", [])
        if pairs_data:
            pairs_records = [
                {
                    "leader": p["leader"],
                    "follower": p["follower"],
                    "lag": p["lag"],
                    "correlation": p["correlation"],
                }
                for p in pairs_data
            ]
            pairs_df = pl.DataFrame(pairs_records)
            self._write_parquet(pairs_df, pairs_filename)

        logger.info(f"Lead-lag cache saved: {len(cache_df)} signals, {len(pairs_data)} pairs")
        return True

    def load_lead_lag_cache(self) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """
        Lead-Lagキャッシュを読み込む。

        Returns:
            (signals_df, pairs_df) or (None, None) if not cached
        """
        cache_filename = "lead_lag_cache.parquet"
        pairs_filename = "lead_lag_pairs.parquet"

        signals = None
        pairs = None

        if self._parquet_exists(cache_filename):
            signals = self._read_parquet(cache_filename)

        if self._parquet_exists(pairs_filename):
            pairs = self._read_parquet(pairs_filename)

        return signals, pairs
