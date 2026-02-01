"""
Signal Precomputation Module - Batch signal computation and caching for backtest optimization.

This module pre-computes all signals and stores them in Parquet format,
enabling 5-10x faster backtests by eliminating redundant calculations.

Usage:
    # StorageBackend経由（S3必須）
    from src.utils.storage_backend import get_storage_backend
    backend = get_storage_backend()  # S3_BUCKET環境変数が必要
    precomputer = SignalPrecomputer(storage_backend=backend)
    precomputer.precompute_all(prices_df, config)

    # 明示的なS3バケット指定
    from src.utils.storage_backend import StorageBackend, StorageConfig
    backend = StorageBackend(StorageConfig(s3_bucket="my-bucket"))
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
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import polars as pl

from src.utils.hash_utils import compute_universe_hash, compute_config_hash, compute_hash
from src.utils.metrics import calculate_rolling_sharpe

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend
    import pandas as pd

logger = logging.getLogger(__name__)

# Version for cache invalidation when implementation changes
PRECOMPUTE_VERSION = "3.1.0"  # Bump for timeframe affinity-based variant filtering


class SignalType(Enum):
    """Signal classification for computation strategy."""
    INDEPENDENT = "independent"  # Single-ticker computation (price data only)
    RELATIVE = "relative"        # Cross-ticker comparison required
    EXTERNAL = "external"        # External data dependency (FINRA, SEC, etc.)

# Signal classification for incremental computation
# Independent signals: computed from each ticker's price data only (no cross-asset dependency)
INDEPENDENT_SIGNALS = {
    # Momentum family - exact names and patterns
    "momentum_*",
    "roc",        # Base signal name
    "roc_*",      # With variants
    # Mean reversion / oscillators - exact names and patterns
    "rsi",        # Base signal name
    "rsi_*",      # With variants
    "bollinger_*",
    "stochastic_*",
    "zscore_*",
    # Volatility - exact names and patterns
    "volatility_*",
    "atr",        # Base signal name
    "atr_*",      # With variants
    # Breakout - exact names and patterns
    "breakout_*",
    "donchian_*",
    "high_low_breakout",   # Base signal name
    "high_low_breakout_*", # With variants
    "range_breakout",      # Base signal name
    "range_breakout_*",    # With variants
    # Long-term
    "fifty_two_week_high_*",
    # Sharpe / risk-adjusted
    "sharpe_*",
    # Volume-based - exact names and patterns
    "obv_*",
    "vwap_*",
    "money_flow_*",
    "accumulation_distribution",   # Base signal name
    "accumulation_distribution_*", # With variants
    # Advanced technical
    "kama",       # Base signal name (no underscore)
    "kama_*",     # With variants
    "keltner_*",
    # Dual momentum
    "dual_momentum",    # Base signal name
    "dual_momentum_*",
    # Trend
    "adaptive_trend_*",
    "trend_*",
    # Additional patterns
    "macd_*",
    "williams_r_*",
    "cci_*",
    "mean_reversion_*",
    # Low Volatility Premium (factor signal)
    "low_vol_premium",
    "low_vol_premium_*",
    # Seasonality patterns (no cross-asset dependency)
    "day_of_week*",
    "turn_of_month*",
    "month_effect*",
    # Factor signals (single ticker computation)
    "value_factor*",
    "quality_factor*",
    "size_factor*",
    # Regime detection (single ticker)
    "regime_detector*",
}

# Relative signals: computed from cross-asset comparisons or rankings
RELATIVE_SIGNALS = {
    "sector_relative_*",
    "cross_asset_*",
    "momentum_factor",
    "sector_momentum",
    "sector_momentum_*",
    "sector_breadth",
    "sector_breadth_*",
    "market_breadth",
    "market_breadth_*",
    "ranking_*",
    "lead_lag",    # Lead-Lag relationship signal (task_042_1)
    "lead_lag_*",  # With variants
    "short_term_reversal*",  # Short-term reversal signal (task_042_3)
    "correlation_regime",
    "correlation_regime_*",
    "return_dispersion",
    "return_dispersion_*",
    # Ensemble strategies (cross-asset)
    "multi_timeframe_momentum*",
    "timeframe_consensus*",
    "enhanced_sector_rotation*",
    "macro_regime_composite*",
    # Factor signals (cross-sectional ranking)
    "low_vol_factor*",
}

# External data dependent signals (require FINRA, SEC, or other external APIs)
EXTERNAL_SIGNALS = {
    "short_interest",
    "short_interest*",
    "short_interest_change",
    "short_interest_change*",
    "insider_trading",
    "insider_trading*",
    "yield_curve",
    "yield_curve_*",
    "enhanced_yield_curve",  # Enhanced Yield Curve (FRED data)
    "enhanced_yield_curve_*",
    "inflation_expectation",
    "inflation_expectation_*",
    "credit_spread",
    "credit_spread_*",
    "dollar_strength",
    "dollar_strength_*",
    "put_call_ratio",
    "put_call_ratio_*",
    "vix_sentiment",
    "vix_sentiment_*",
    "vix_term_structure",
    "vix_term_structure_*",
    "fear_greed_composite",
    "fear_greed_composite_*",
    # VIX signal
    "vix_signal",
    "vix_signal_*",
}

# External ETF tickers for macro signals
EXTERNAL_ETF_TICKERS = {
    "bond_etf": ["TLT", "SHY", "TIP", "IEF", "LQD", "HYG"],
    "sector_etf": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
    "vix": ["VXX", "UVXY", "SVXY"],
    "currency": ["UUP", "FXE", "FXY"],
}


# Signal classification mapping: signal_name -> SignalType
# This mapping is used to determine computation strategy for each registered signal
SIGNAL_CLASSIFICATION: dict[str, SignalType] = {}


def _init_signal_classification() -> dict[str, SignalType]:
    """
    Initialize signal classification from SignalRegistry.

    This function is called lazily to avoid circular imports.
    Maps each registered signal to its computation type.
    """
    global SIGNAL_CLASSIFICATION
    if SIGNAL_CLASSIFICATION:
        return SIGNAL_CLASSIFICATION

    try:
        from src.signals import SignalRegistry

        for signal_name in SignalRegistry.list_all():
            # Check if matches any external signal pattern
            if signal_name in EXTERNAL_SIGNALS:
                SIGNAL_CLASSIFICATION[signal_name] = SignalType.EXTERNAL
            # Check if matches any relative signal pattern
            elif any(fnmatch.fnmatch(signal_name, p) for p in RELATIVE_SIGNALS):
                SIGNAL_CLASSIFICATION[signal_name] = SignalType.RELATIVE
            # Check if matches any independent signal pattern
            elif any(fnmatch.fnmatch(signal_name, p) for p in INDEPENDENT_SIGNALS):
                SIGNAL_CLASSIFICATION[signal_name] = SignalType.INDEPENDENT
            else:
                # Default classification based on metadata
                meta = SignalRegistry.get_metadata(signal_name)
                category = meta.get("category", "")

                if category in ("sentiment", "macro"):
                    # Macro/sentiment signals often need external data
                    SIGNAL_CLASSIFICATION[signal_name] = SignalType.EXTERNAL
                elif category in ("sector", "factor"):
                    # Factor/sector signals need cross-asset comparison
                    SIGNAL_CLASSIFICATION[signal_name] = SignalType.RELATIVE
                else:
                    # Default to independent (single-asset computation)
                    SIGNAL_CLASSIFICATION[signal_name] = SignalType.INDEPENDENT
    except Exception as e:
        logger.warning(f"Failed to initialize signal classification: {e}")

    return SIGNAL_CLASSIFICATION


@dataclass
class PeriodVariant:
    """Period variant definition for signal precomputation.

    Research-backed optimal periods for different time horizons.
    """
    name: str
    period: int
    description: str


# Default period variants (research-backed values for performance maximization)
DEFAULT_PERIOD_VARIANTS = [
    PeriodVariant("short", 5, "Ultra-short-term (1 week) - mean reversion"),
    PeriodVariant("medium", 20, "Standard (1 month)"),
    PeriodVariant("long", 60, "Medium-term (3 months)"),
    PeriodVariant("half_year", 126, "6 months (highest return in research - 702% over 10 years)"),
    PeriodVariant("yearly", 252, "Annual (highest Sharpe ratio)"),
]


# Parameter names that indicate period/lookback (for variant detection)
PERIOD_PARAM_NAMES = {"lookback", "period", "window", "vol_lookback", "n_periods", "rsi_period"}


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
        has_relative_signals: Whether relative signals are in the config (require full universe)
    """

    can_use_cache: bool
    reason: str
    incremental_start: datetime | None = None
    missing_signals: list[str] | None = None
    new_tickers: list[str] | None = None  # task_041_7a: 銘柄差分検知
    has_relative_signals: bool = False  # task_cache_fix: relative signalsの有無


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


@dataclass
class CacheStats:
    """
    Cache statistics for performance monitoring (task_cache_fix).

    Tracks cache hits, misses, and reasons for better observability.
    """
    hits: int = 0
    misses: int = 0
    miss_reasons: dict[str, int] = field(default_factory=dict)
    signals_loaded: int = 0
    signals_computed: int = 0
    tickers_reused: int = 0
    tickers_computed: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self, reason: str) -> None:
        """Record a cache miss with reason."""
        self.misses += 1
        self.miss_reasons[reason] = self.miss_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "miss_reasons": self.miss_reasons,
            "signals_loaded": self.signals_loaded,
            "signals_computed": self.signals_computed,
            "tickers_reused": self.tickers_reused,
            "tickers_computed": self.tickers_computed,
        }


class SignalPrecomputer:
    """
    Pre-compute and cache all signals for efficient backtesting (Unified Mode v3.0).

    Instead of computing signals on-the-fly during backtest iterations,
    this class computes all signals once using vectorized operations
    and stores them in Parquet format for fast retrieval.

    Unified Mode (v3.0):
    - Uses SignalRegistry for all 60+ signals
    - Generates period variants for signals with lookback/period parameters
    - Results in ~230 signal variants (60 base signals × ~4 variants average)
    - Signal naming: {base_name}_{variant} (e.g., momentum_return_short)

    Supports all 64 signals registered in SignalRegistry:
    - Independent signals: Computed per-ticker from price data
    - Relative signals: Require cross-ticker comparison (universe-level)
    - External signals: Require external data (FINRA, SEC, etc.)

    Configuration via config/default.yaml signal_precompute section:
    - period_variants: short=5, medium=20, long=60, half_year=126, yearly=252
    - enabled_variants: which variants to compute
    - custom_periods: signal-specific period overrides

    Attributes:
        cache_dir: Directory for storing Parquet files
        _metadata: Cache metadata including data hash and timestamps
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        storage_backend: "StorageBackend | None" = None,
        progress_callback: "Callable[[str], None] | None" = None,
    ) -> None:
        """
        Initialize SignalPrecomputer.

        Args:
            cache_dir: Directory for storing precomputed signal Parquet files (legacy mode).
                      Ignored if storage_backend is provided.
            storage_backend: StorageBackend instance for S3/local abstraction.
                           If provided, cache_dir is ignored and all operations
                           go through the backend.
            progress_callback: Optional callback function called after each signal
                             computation. Receives signal name (e.g., "momentum_20").

        Usage:
            # StorageBackend経由（S3必須）
            from src.utils.storage_backend import get_storage_backend
            backend = get_storage_backend()
            precomputer = SignalPrecomputer(storage_backend=backend)

            # 明示的なS3バケット指定
            from src.utils.storage_backend import StorageBackend, StorageConfig
            backend = StorageBackend(StorageConfig(s3_bucket="my-bucket"))
            precomputer = SignalPrecomputer(storage_backend=backend)
        """
        if storage_backend is not None:
            # StorageBackend経由での操作
            self._backend: "StorageBackend | None" = storage_backend
            self._cache_dir = Path(storage_backend.config.base_path)
            self._use_backend = True
        else:
            # 従来のローカルファイル操作
            self._backend = None
            if cache_dir is None:
                from src.config.settings import get_cache_path
                cache_dir = get_cache_path("signals")
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._use_backend = False

        self._metadata_file_name = "_metadata.json"
        self._metadata: dict[str, Any] = self._load_metadata()

        # Cache statistics for observability (task_cache_fix)
        self._cache_stats = CacheStats()

        # Progress callback for UI integration
        self._progress_callback = progress_callback

        # External data clients (lazy initialized)
        self._finra_client: Any = None
        self._sec_client: Any = None

        # External ETF price data cache
        self._external_etf_prices: dict[str, pl.DataFrame] = {}

        s3_bucket = storage_backend.config.s3_bucket if storage_backend else None
        logger.debug(f"SignalPrecomputer initialized: s3_bucket={s3_bucket}, path={self._cache_dir}")

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
            return compute_hash(",".join(signal_names), truncate=0)
        except Exception as e:
            logger.debug(f"Failed to compute signal registry hash: {e}")
            return compute_hash("default", truncate=0)

    def _compute_signal_config_hash(self, config: dict[str, Any]) -> str:
        """Compute hash of signal configuration (parameters)."""
        return compute_config_hash(config, truncate=0, algorithm="md5")

    def _compute_universe_hash(self, tickers: list[str]) -> str:
        """Compute hash of ticker universe."""
        return compute_universe_hash(tickers, truncate=0)

    def _get_required_signals(self, config: dict[str, Any]) -> list[str]:
        """
        Get list of required signal names.

        Returns all signal variants from SignalRegistry (unified mode).

        Args:
            config: Signal configuration (unused, kept for API compatibility)

        Returns:
            List of signal names (e.g., ["momentum_return_short", "momentum_return_medium", ...])
        """
        return self._get_unified_signal_names()

    def _get_unified_signal_names(self) -> list[str]:
        """
        Get list of expected signal names in unified mode.

        Generates names for all signals with their period variants, filtered by
        each signal's timeframe affinity configuration.

        - Signals with period params: {signal_name}_{variant} (e.g., momentum_return_short)
        - Signals without period params: {signal_name} (e.g., vix_sentiment)

        Only variants that match the signal's timeframe affinity are included.

        Returns:
            List of expected signal names
        """
        try:
            from src.signals import SignalRegistry
            from src.config.settings import get_settings

            settings = get_settings()
            signal_settings = settings.signal_precompute
            enabled_variants = signal_settings.enabled_variants
            custom_periods = signal_settings.custom_periods
            variant_periods = signal_settings.period_variants.to_dict()

            signal_names = []

            for base_name in SignalRegistry.list_all():
                signal_cls = SignalRegistry.get(base_name)
                specs = signal_cls.parameter_specs()
                period_param = self._find_period_param(specs)

                if period_param is None:
                    # No period parameter - single signal
                    signal_names.append(base_name)
                elif base_name in custom_periods:
                    # Custom periods defined
                    for period in custom_periods[base_name]:
                        signal_names.append(f"{base_name}_{period}")
                else:
                    # Standard period variants filtered by timeframe affinity
                    tf_config = signal_cls.timeframe_config()

                    for variant in enabled_variants:
                        # Check if variant is supported
                        if not tf_config.supports_variant(variant):
                            continue

                        # Check if period is within bounds
                        period = variant_periods.get(variant)
                        if period is None:
                            continue
                        if not (tf_config.min_period <= period <= tf_config.max_period):
                            continue

                        signal_names.append(f"{base_name}_{variant}")

            return signal_names

        except Exception as e:
            logger.warning(f"Failed to get unified signal names: {e}")
            # Return empty list - will trigger full recomputation
            return []

    def classify_signal(self, signal_name: str) -> str:
        """
        Classify a signal as independent, relative, or external.

        Independent signals: computed from each ticker's price data only.
        These can be incrementally computed for new tickers without
        affecting existing cached values.

        Relative signals: computed from cross-asset comparisons or rankings.
        Adding new tickers requires full recomputation.

        External signals: depend on external data sources (FRED, VIX, etc.)
        Treated as relative for incremental computation purposes.

        Args:
            signal_name: Name of the signal (e.g., "momentum_20", "sector_relative_strength")

        Returns:
            "independent", "relative", or "external"
        """
        # Check EXTERNAL_SIGNALS first (these need special handling)
        for pattern in EXTERNAL_SIGNALS:
            if fnmatch.fnmatch(signal_name, pattern):
                return "external"

        # Check RELATIVE_SIGNALS to handle exact matches like "momentum_factor"
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
        Classify multiple signals into independent, relative, and external groups.

        Args:
            signal_names: List of signal names

        Returns:
            Dictionary with keys "independent", "relative", and "external",
            each containing list of signal names
        """
        result: dict[str, list[str]] = {"independent": [], "relative": [], "external": []}
        for name in signal_names:
            classification = self.classify_signal(name)
            result[classification].append(name)
        return result

    def _compute_prices_hash(self, prices: pl.DataFrame) -> str:
        """
        Compute a hash of the prices DataFrame for cache validation.

        This method intentionally excludes price values (like close_sum) from the hash
        to prevent cache invalidation due to:
        - Dividend adjustments (adjusted close values change retroactively)
        - Stock splits (historical prices are restated)
        - Minor floating-point differences across data sources
        - Small data corrections by data providers

        The hash is based on structural properties:
        - Shape (row count, column count)
        - Column names
        - Date range (first and last timestamps)
        - Ticker count

        This approach provides ~99% cache stability while still detecting
        significant structural changes that require recomputation.

        Args:
            prices: Price DataFrame with columns [timestamp, ticker, close, ...]

        Returns:
            MD5 hash string
        """
        # Base structural information
        hash_parts = [f"shape={prices.shape}", f"cols={sorted(prices.columns)}"]

        # Date range (critical for time-based cache validation)
        if "timestamp" in prices.columns:
            try:
                timestamps = prices.select("timestamp")
                first_ts = timestamps.head(1).item()
                last_ts = timestamps.tail(1).item()
                hash_parts.append(f"dates={first_ts}_{last_ts}")
            except Exception:
                pass

        # Ticker count (for universe change detection)
        if "ticker" in prices.columns:
            ticker_count = prices.select("ticker").n_unique()
            hash_parts.append(f"tickers={ticker_count}")

        # Sample a few specific values for sanity check (optional, adds minimal sensitivity)
        # Using only the count of non-null close values instead of sum
        if "close" in prices.columns:
            close_count = prices.select(pl.col("close").is_not_null().sum()).item()
            hash_parts.append(f"close_count={close_count}")

        hash_input = "|".join(hash_parts)
        return hashlib.md5(hash_input.encode()).hexdigest()

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
            config: Signal configuration (unused, kept for API compatibility)

        Returns:
            Tuple of (is_valid, reason)
        """
        if config is None:
            config = {}

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
            self._cache_stats.record_miss(reason)
            logger.info(f"Cache validation MISS: {reason}")
            logger.info(
                f"  Hash comparison: "
                f"version={cached_metadata.version == current_metadata.version}, "
                f"registry={cached_metadata.signal_registry_hash == current_metadata.signal_registry_hash}, "
                f"config={cached_metadata.signal_config_hash == current_metadata.signal_config_hash}, "
                f"universe={cached_metadata.universe_hash == current_metadata.universe_hash}"
            )
        else:
            self._cache_stats.record_hit()
            logger.info("Cache validation HIT: Cache is valid")

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
            config = {}

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

        # ===== Ticker difference detection (task_041_7a enhanced) =====
        cached_tickers = set(self._metadata.get("tickers", []))

        new_tickers = current_tickers - cached_tickers
        removed_tickers = cached_tickers - current_tickers

        # Check if relative signals are required
        required_signals = self._get_required_signals(config)
        classified = self.classify_signals(required_signals)
        has_relative_signals = bool(classified["relative"])

        # Tickers removed only → cache still valid (no computation needed)
        if removed_tickers and not new_tickers:
            logger.info(
                f"Tickers removed only: {len(removed_tickers)} removed, cache still valid"
            )
            # Continue to check other conditions (time, signals)

        # New tickers added → incremental computation for new tickers
        new_tickers_list: list[str] | None = None
        if new_tickers:
            if has_relative_signals:
                # Has relative signals → full recomputation required for relative signals
                # But we can still use cache for independent signals!
                # This is a partial cache hit scenario
                logger.info(
                    f"New tickers detected: {len(new_tickers)}. "
                    f"Independent signals: incremental update for new tickers. "
                    f"Relative signals ({len(classified['relative'])}): full recomputation."
                )
                new_tickers_list = list(new_tickers)
            else:
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

        # Start date moved forward → cache still covers this period, no recomputation needed
        if start_date > cached_metadata.cached_start_date:
            logger.info(
                f"Start date moved forward: {cached_metadata.cached_start_date} -> {start_date}. "
                f"Cache still valid (covers requested period)."
            )
            # Continue with validation - cache can be used

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
            has_relative_signals=has_relative_signals,
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
            config: Signal configuration (unused, kept for API compatibility)
            force: Force full recomputation even if cache is valid

        Returns:
            True if any computation was performed, False if cache was used as-is
        """
        # config is kept for API compatibility but no longer used
        if config is None:
            config = {}

        # Force recomputation requested
        if force:
            logger.info("Force recomputation requested")
            return self._full_precompute(prices, config)

        # Incremental cache validation using CacheValidationResult
        result = self.validate_cache_incremental(prices, config)

        # Cache invalid - full recomputation needed
        if not result.can_use_cache:
            self._cache_stats.record_miss(result.reason)
            logger.info(f"Cache MISS - Full recomputation needed: {result.reason}")
            computed = self._full_precompute(prices, config)
            self._log_cache_summary()
            return computed

        # New tickers added - compute for new tickers (task_041_6a)
        if result.new_tickers:
            self._cache_stats.tickers_computed = len(result.new_tickers)
            existing_tickers = prices.select("ticker").n_unique() - len(result.new_tickers)
            self._cache_stats.tickers_reused = existing_tickers
            logger.info(f"Cache PARTIAL HIT - Computing for {len(result.new_tickers)} new tickers")
            computed = self.precompute_for_new_tickers(
                prices, result.new_tickers, config,
                has_relative_signals=result.has_relative_signals
            )
            self._log_cache_summary()
            return computed

        # New signals added - compute only missing signals
        if result.missing_signals:
            self._cache_stats.signals_computed = len(result.missing_signals)
            cached_signals = len(self._metadata.get("signals", []))
            self._cache_stats.signals_loaded = cached_signals
            logger.info(f"Cache PARTIAL HIT - Computing {len(result.missing_signals)} new signals")
            computed = self.precompute_missing_signals(prices, result.missing_signals, config)
            self._log_cache_summary()
            return computed

        # Date range extended - incremental update
        if result.incremental_start is not None:
            self._cache_stats.record_hit()
            logger.info(f"Cache HIT with incremental update from {result.incremental_start}")
            computed = self.precompute_incremental(prices, result.incremental_start, config)
            self._log_cache_summary()
            return computed

        # Full cache hit - no computation needed
        self._cache_stats.record_hit()
        self._cache_stats.signals_loaded = len(self._metadata.get("signals", []))
        self._cache_stats.tickers_reused = len(self._metadata.get("tickers", []))
        logger.info("Cache FULL HIT - No computation needed")
        self._log_cache_summary()
        return False

    def _log_cache_summary(self) -> None:
        """Log cache statistics summary at INFO level."""
        stats = self._cache_stats.to_dict()
        logger.info(
            f"Cache Statistics Summary: "
            f"hit_rate={stats['hit_rate']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"signals_loaded={stats['signals_loaded']}, signals_computed={stats['signals_computed']}, "
            f"tickers_reused={stats['tickers_reused']}, tickers_computed={stats['tickers_computed']}"
        )
        if stats['miss_reasons']:
            logger.info(f"Cache miss reasons: {stats['miss_reasons']}")

    def reset_cache_stats(self) -> None:
        """Reset cache statistics (useful for testing or new backtest runs)."""
        self._cache_stats = CacheStats()

    def _full_precompute(
        self,
        prices: pl.DataFrame,
        config: dict[str, Any],
    ) -> bool:
        """
        Perform full signal precomputation.

        This is the internal method that does the actual computation.
        Called by precompute_all() when full recomputation is needed.

        Uses unified mode (v3.0): All signals from SignalRegistry with period variants.

        Args:
            prices: Price DataFrame
            config: Signal configuration (unused, kept for API compatibility)

        Returns:
            True if computation completed successfully
        """
        start_time = datetime.now()

        tickers = prices.select("ticker").unique().to_series().to_list()
        logger.info(f"Computing signals for {len(tickers)} tickers")

        # Unified mode (v3.0): All signals with period variants
        logger.info("Starting full signal precomputation (unified mode)...")

        # Load signal precompute settings
        from src.config.settings import get_settings
        try:
            settings = get_settings()
            signal_settings = settings.signal_precompute
        except Exception as e:
            logger.warning(f"Failed to load signal_precompute settings: {e}, using defaults")
            from src.config.settings import SignalPrecomputeSettings
            signal_settings = SignalPrecomputeSettings()

        # Compute all signals using unified mode
        computed_signals = self._compute_all_signals_unified(prices, tickers, signal_settings)

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

        # Update cache statistics
        self._cache_stats.signals_computed = len(computed_signals)
        self._cache_stats.tickers_computed = len(tickers)

        logger.info(f"Full signal precomputation completed in {elapsed:.2f}s ({len(computed_signals)} signals, {len(tickers)} tickers)")

        return True

    def _find_period_param(self, specs: list[Any]) -> Any | None:
        """
        Find the primary period parameter from a signal's parameter specs.

        Args:
            specs: List of ParameterSpec from signal.parameter_specs()

        Returns:
            ParameterSpec for the period parameter, or None if not found
        """
        for spec in specs:
            if spec.name in PERIOD_PARAM_NAMES and spec.searchable:
                return spec
        # Fall back to first period-like param even if not searchable
        for spec in specs:
            if spec.name in PERIOD_PARAM_NAMES:
                return spec
        return None

    def _get_period_for_variant(
        self,
        signal_name: str,
        variant_name: str,
        period_param: Any,
        settings: Any,  # SignalPrecomputeSettings
    ) -> int:
        """
        Get the period value for a specific variant.

        Args:
            signal_name: Base signal name (e.g., "momentum_return")
            variant_name: Variant name (e.g., "short", "medium", "long")
            period_param: ParameterSpec for the period parameter
            settings: SignalPrecomputeSettings instance

        Returns:
            Period value for this variant
        """
        # Check custom periods first
        if signal_name in settings.custom_periods:
            custom = settings.custom_periods[signal_name]
            # Map variant index to custom period
            variant_index = settings.enabled_variants.index(variant_name)
            if variant_index < len(custom):
                return custom[variant_index]

        # Use standard period variants
        variant_periods = settings.period_variants.to_dict()
        return variant_periods.get(variant_name, period_param.default)

    def _compute_all_signals_unified(
        self,
        prices: pl.DataFrame,
        tickers: list[str],
        settings: Any,  # SignalPrecomputeSettings
    ) -> list[str]:
        """
        Compute all signals in unified mode with period variants.

        This method implements the unified computation approach:
        1. Iterate through all registered signals
        2. For signals with period parameters: generate all enabled variants
        3. For signals without period parameters: compute once
        4. Save each signal variant to Parquet

        Args:
            prices: Price DataFrame with columns [timestamp, ticker, close, high, low, volume]
            tickers: List of ticker symbols
            settings: SignalPrecomputeSettings instance

        Returns:
            List of computed signal names (including variants)
        """
        from src.signals import SignalRegistry

        computed_signals: list[str] = []
        signal_names = SignalRegistry.list_all()
        total_signals = len(signal_names)

        logger.info(f"Computing {total_signals} registered signals with period variants...")
        logger.info(f"Enabled variants: {settings.enabled_variants}")

        # Initialize signal classification
        classification = _init_signal_classification()

        # Track statistics
        signals_with_variants = 0
        signals_without_variants = 0
        total_variants = 0

        for idx, signal_name in enumerate(signal_names):
            try:
                signal_cls = SignalRegistry.get(signal_name)
                specs = signal_cls.parameter_specs()
                period_param = self._find_period_param(specs)

                sig_type = classification.get(signal_name, SignalType.INDEPENDENT)

                if period_param is None:
                    # No period parameter - compute single signal
                    signals_without_variants += 1
                    success = self._compute_registry_signal(
                        signal_name, prices, tickers, sig_type
                    )
                    if success:
                        computed_signals.append(signal_name)
                        total_variants += 1
                    if self._progress_callback:
                        self._progress_callback(signal_name)
                else:
                    # Has period parameter - compute variants based on timeframe affinity
                    signals_with_variants += 1

                    # Get timeframe configuration from signal class
                    tf_config = signal_cls.timeframe_config()

                    # Check for custom periods
                    if signal_name in settings.custom_periods:
                        custom_periods = settings.custom_periods[signal_name]
                        for i, period in enumerate(custom_periods):
                            variant_name = f"{signal_name}_{period}"
                            success = self._compute_registry_signal_with_params(
                                signal_name, prices, tickers, sig_type,
                                {period_param.name: period}, variant_name
                            )
                            if success:
                                computed_signals.append(variant_name)
                                total_variants += 1
                            if self._progress_callback:
                                self._progress_callback(variant_name)
                    else:
                        # Use enabled variants filtered by timeframe affinity
                        for variant in settings.enabled_variants:
                            # Check if this variant is supported by the signal
                            if not tf_config.supports_variant(variant):
                                logger.debug(
                                    f"Skipping {signal_name}_{variant}: variant not supported "
                                    f"(affinity={tf_config.affinity.value}, supported={tf_config.supported_variants})"
                                )
                                continue

                            period = self._get_period_for_variant(
                                signal_name, variant, period_param, settings
                            )

                            # Validate period against signal's bounds
                            if not (tf_config.min_period <= period <= tf_config.max_period):
                                logger.debug(
                                    f"Skipping {signal_name}_{variant}: period {period} outside "
                                    f"range [{tf_config.min_period}, {tf_config.max_period}]"
                                )
                                continue

                            variant_name = f"{signal_name}_{variant}"
                            success = self._compute_registry_signal_with_params(
                                signal_name, prices, tickers, sig_type,
                                {period_param.name: period}, variant_name
                            )
                            if success:
                                computed_signals.append(variant_name)
                                total_variants += 1
                            if self._progress_callback:
                                self._progress_callback(variant_name)

            except Exception as e:
                logger.warning(f"Failed to compute signal {signal_name}: {e}")
                continue

            # Log progress every 10 signals
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{total_signals} base signals processed")

        logger.info(
            f"Unified computation complete: "
            f"{signals_with_variants} signals with variants + "
            f"{signals_without_variants} signals without variants = "
            f"{total_variants} total signal variants"
        )

        return computed_signals

    def _compute_registry_signal_with_params(
        self,
        signal_name: str,
        prices: pl.DataFrame,
        tickers: list[str],
        signal_type: SignalType,
        params: dict[str, Any],
        output_name: str,
    ) -> bool:
        """
        Compute a signal from SignalRegistry with custom parameters.

        Args:
            signal_name: Registered signal name
            prices: Price DataFrame
            tickers: List of tickers
            signal_type: Signal classification type
            params: Custom parameters to override defaults
            output_name: Output file name (e.g., "momentum_return_short")

        Returns:
            True if computation succeeded
        """
        from src.signals import SignalRegistry

        try:
            signal_cls = SignalRegistry.get(signal_name)
            signal = signal_cls(**params)

            results = []

            if signal_type == SignalType.INDEPENDENT:
                # Compute per-ticker
                for ticker in tickers:
                    try:
                        ticker_prices = self._get_ticker_prices_as_pandas(prices, ticker)
                        if ticker_prices is None or len(ticker_prices) < 30:
                            continue

                        result = signal.compute(ticker_prices)
                        if result is not None and result.scores is not None:
                            df = self._signal_result_to_polars(result, ticker)
                            if df is not None and len(df) > 0:
                                results.append(df)
                    except Exception as e:
                        logger.debug(f"Failed to compute {output_name} for {ticker}: {e}")
                        continue

            elif signal_type == SignalType.RELATIVE:
                # Compute with full universe context
                results = self._compute_relative_signal(signal_name, signal, prices, tickers)

            elif signal_type == SignalType.EXTERNAL:
                # Compute with external data (or fallback)
                results = self._compute_external_signal(signal_name, signal, prices, tickers)

            if results:
                combined = pl.concat(results)
                combined = combined.filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan())
                if len(combined) > 0:
                    self._write_parquet(combined, f"{output_name}.parquet")
                    logger.debug(f"Saved {output_name} ({len(combined)} rows)")
                    return True

            return False

        except Exception as e:
            logger.warning(f"Failed to compute registry signal {output_name}: {e}")
            return False

    def _compute_registry_signal(
        self,
        signal_name: str,
        prices: pl.DataFrame,
        tickers: list[str],
        signal_type: SignalType,
    ) -> bool:
        """
        Compute a signal from SignalRegistry and save to Parquet.

        Args:
            signal_name: Registered signal name
            prices: Price DataFrame
            tickers: List of tickers
            signal_type: Signal classification type

        Returns:
            True if computation succeeded
        """
        from src.signals import SignalRegistry

        try:
            signal_cls = SignalRegistry.get(signal_name)
            signal = signal_cls()

            results = []

            if signal_type == SignalType.INDEPENDENT:
                # Compute per-ticker
                for ticker in tickers:
                    try:
                        ticker_prices = self._get_ticker_prices_as_pandas(prices, ticker)
                        if ticker_prices is None or len(ticker_prices) < 30:
                            continue

                        result = signal.compute(ticker_prices)
                        if result is not None and result.scores is not None:
                            df = self._signal_result_to_polars(result, ticker)
                            if df is not None and len(df) > 0:
                                results.append(df)
                    except Exception as e:
                        logger.debug(f"Failed to compute {signal_name} for {ticker}: {e}")
                        continue

            elif signal_type == SignalType.RELATIVE:
                # Compute with full universe context
                results = self._compute_relative_signal(signal_name, signal, prices, tickers)

            elif signal_type == SignalType.EXTERNAL:
                # Compute with external data (or fallback)
                results = self._compute_external_signal(signal_name, signal, prices, tickers)

            if results:
                combined = pl.concat(results)
                combined = combined.filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan())
                if len(combined) > 0:
                    self._write_parquet(combined, f"{signal_name}.parquet")
                    logger.debug(f"Saved {signal_name} ({len(combined)} rows)")
                    return True

            return False

        except Exception as e:
            logger.warning(f"Failed to compute registry signal {signal_name}: {e}")
            return False

    def _get_ticker_prices_as_pandas(
        self,
        prices: pl.DataFrame,
        ticker: str,
    ) -> "pd.DataFrame | None":
        """
        Extract ticker prices and convert to pandas DataFrame for Signal.compute().

        Args:
            prices: Polars DataFrame with all prices
            ticker: Ticker symbol

        Returns:
            Pandas DataFrame with DatetimeIndex, or None if insufficient data
        """
        import pandas as pd

        try:
            ticker_data = prices.filter(pl.col("ticker") == ticker).sort("timestamp")
            if len(ticker_data) < 30:
                return None

            pdf = ticker_data.to_pandas()
            if "timestamp" in pdf.columns:
                pdf = pdf.set_index("timestamp")
                pdf.index = pd.to_datetime(pdf.index)

            return pdf
        except Exception:
            return None

    def _signal_result_to_polars(
        self,
        result: Any,  # SignalResult
        ticker: str,
    ) -> pl.DataFrame | None:
        """
        Convert SignalResult to Polars DataFrame format.

        Args:
            result: SignalResult with scores Series
            ticker: Ticker symbol

        Returns:
            Polars DataFrame with [timestamp, ticker, value] columns
        """
        try:
            scores = result.scores
            if scores is None or len(scores) == 0:
                return None

            # Convert pandas Series to Polars DataFrame
            import pandas as pd

            if isinstance(scores.index, pd.DatetimeIndex):
                timestamps = scores.index.to_list()
            else:
                timestamps = list(scores.index)

            df = pl.DataFrame({
                "timestamp": timestamps,
                "ticker": [ticker] * len(scores),
                "value": scores.values,
            })

            # Cast timestamp to consistent type
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

            return df
        except Exception as e:
            logger.debug(f"Failed to convert signal result: {e}")
            return None

    def _compute_relative_signal(
        self,
        signal_name: str,
        signal: Any,  # Signal instance
        prices: pl.DataFrame,
        tickers: list[str],
    ) -> list[pl.DataFrame]:
        """
        Compute relative signal requiring cross-ticker comparison.

        For signals like sector_relative_strength, cross_asset_momentum, etc.,
        we need access to the full universe for comparison.

        Args:
            signal_name: Signal name
            signal: Signal instance
            prices: Full universe price DataFrame
            tickers: List of all tickers

        Returns:
            List of DataFrames with computed signals
        """
        import pandas as pd

        results = []

        # Convert prices to dict of pandas DataFrames for cross-asset access
        universe_prices: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            pdf = self._get_ticker_prices_as_pandas(prices, ticker)
            if pdf is not None:
                universe_prices[ticker] = pdf

        # Compute for each ticker with universe context
        for ticker in tickers:
            if ticker not in universe_prices:
                continue

            ticker_data = universe_prices[ticker]

            try:
                # Check if signal has universe-aware compute method
                if hasattr(signal, "compute_with_universe"):
                    result = signal.compute_with_universe(ticker_data, universe_prices)
                else:
                    # Fall back to standard compute (may be less accurate)
                    result = signal.compute(ticker_data)

                if result is not None and result.scores is not None:
                    df = self._signal_result_to_polars(result, ticker)
                    if df is not None and len(df) > 0:
                        results.append(df)
            except Exception as e:
                logger.debug(f"Failed to compute relative signal {signal_name} for {ticker}: {e}")

        return results

    def _compute_external_signal(
        self,
        signal_name: str,
        signal: Any,  # Signal instance
        prices: pl.DataFrame,
        tickers: list[str],
    ) -> list[pl.DataFrame]:
        """
        Compute external data dependent signal with fallback.

        For signals requiring FINRA, SEC, or other external data,
        attempts to fetch external data and falls back to proxy if unavailable.

        Args:
            signal_name: Signal name
            signal: Signal instance
            prices: Price DataFrame
            tickers: List of tickers

        Returns:
            List of DataFrames with computed signals
        """
        results = []

        for ticker in tickers:
            ticker_data = self._get_ticker_prices_as_pandas(prices, ticker)
            if ticker_data is None:
                continue

            try:
                # Try to compute with external data
                if signal_name in ("short_interest", "short_interest_change"):
                    result = self._compute_short_interest_signal(signal, ticker_data, ticker)
                elif signal_name == "insider_trading":
                    result = self._compute_insider_signal(signal, ticker_data, ticker)
                else:
                    # For other external signals, use standard compute (proxy mode)
                    result = signal.compute(ticker_data)

                if result is not None and result.scores is not None:
                    df = self._signal_result_to_polars(result, ticker)
                    if df is not None and len(df) > 0:
                        results.append(df)
            except Exception as e:
                logger.debug(f"Failed to compute external signal {signal_name} for {ticker}: {e}")

        return results

    def _compute_short_interest_signal(
        self,
        signal: Any,
        ticker_data: "pd.DataFrame",
        ticker: str,
    ) -> Any:
        """
        Compute short interest signal using FINRA data or fallback.

        Args:
            signal: ShortInterestSignal instance
            ticker_data: Price DataFrame
            ticker: Ticker symbol

        Returns:
            SignalResult
        """
        try:
            # Try to use FINRA client if available
            if self._finra_client is None:
                try:
                    from src.data.finra import FINRAClient
                    self._finra_client = FINRAClient(
                        cache_dir=self._cache_dir / "finra" if not self._use_backend else None,
                        cache_enabled=True,
                    )
                except Exception:
                    self._finra_client = False  # Mark as unavailable

            if self._finra_client and self._finra_client is not False:
                # Try to fetch short interest data
                try:
                    if hasattr(signal, "compute_with_finra_client"):
                        return signal.compute_with_finra_client(ticker_data, ticker)
                except Exception as e:
                    logger.debug(f"FINRA data unavailable for {ticker}: {e}")

            # Fallback to proxy computation
            return signal.compute(ticker_data)
        except Exception:
            return signal.compute(ticker_data)

    def _compute_insider_signal(
        self,
        signal: Any,
        ticker_data: "pd.DataFrame",
        ticker: str,
    ) -> Any:
        """
        Compute insider trading signal using SEC data or fallback.

        Args:
            signal: InsiderTradingSignal instance
            ticker_data: Price DataFrame
            ticker: Ticker symbol

        Returns:
            SignalResult
        """
        try:
            # Try to use SEC client if available
            if self._sec_client is None:
                try:
                    from src.data.sec_edgar import SECEdgarClient
                    self._sec_client = SECEdgarClient()
                except Exception:
                    self._sec_client = False  # Mark as unavailable

            if self._sec_client and self._sec_client is not False:
                # Try to fetch SEC data
                try:
                    if hasattr(signal, "compute_from_sec_data"):
                        return signal.compute_from_sec_data(ticker_data, self._sec_client, ticker)
                except Exception as e:
                    logger.debug(f"SEC data unavailable for {ticker}: {e}")

            # Fallback to proxy computation
            return signal.compute(ticker_data)
        except Exception:
            return signal.compute(ticker_data)

    def get_registered_signal_count(self) -> int:
        """
        Get the total number of registered signals.

        Returns:
            Number of signals in SignalRegistry
        """
        try:
            from src.signals import SignalRegistry
            return len(SignalRegistry.list_all())
        except Exception:
            return 11  # Fallback to legacy count

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

        Uses unified mode: Computes all signals from SignalRegistry.

        Args:
            prices: Full price DataFrame (must include history for lookback)
            start_from: Compute signals from this date onwards
            config: Signal configuration (unused, kept for API compatibility)

        Returns:
            True if incremental computation succeeded
        """
        from datetime import timedelta
        from src.signals import SignalRegistry
        from src.config.settings import get_settings, SignalPrecomputeSettings

        logger.info(f"Starting incremental signal precomputation from {start_from}")
        start_time = datetime.now()

        # Load signal precompute settings
        try:
            settings = get_settings()
            signal_settings = settings.signal_precompute
        except Exception as e:
            logger.warning(f"Failed to load signal_precompute settings: {e}, using defaults")
            signal_settings = SignalPrecomputeSettings()

        # Calculate maximum lookback from period variants
        max_lookback = self._get_max_lookback_unified(signal_settings)
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

        # Initialize signal classification
        classification = _init_signal_classification()
        computed_signals = []

        # Iterate through all registered signals
        for signal_name in SignalRegistry.list_all():
            try:
                signal_cls = SignalRegistry.get(signal_name)
                specs = signal_cls.parameter_specs()
                period_param = self._find_period_param(specs)
                sig_type = classification.get(signal_name, SignalType.INDEPENDENT)

                if period_param is None:
                    # No period parameter - compute single signal incrementally
                    success = self._compute_incremental_unified_signal(
                        signal_name, prices, tickers, sig_type, {}, signal_name, start_from
                    )
                    if success:
                        computed_signals.append(signal_name)
                elif signal_name in signal_settings.custom_periods:
                    # Custom periods
                    for period in signal_settings.custom_periods[signal_name]:
                        variant_name = f"{signal_name}_{period}"
                        success = self._compute_incremental_unified_signal(
                            signal_name, prices, tickers, sig_type,
                            {period_param.name: period}, variant_name, start_from
                        )
                        if success:
                            computed_signals.append(variant_name)
                else:
                    # Standard period variants - filtered by timeframe affinity
                    tf_config = signal_cls.timeframe_config()
                    for variant in signal_settings.enabled_variants:
                        # Check if this variant is supported by the signal
                        if not tf_config.supports_variant(variant):
                            logger.debug(
                                f"Skipping {signal_name}_{variant}: variant not supported "
                                f"(affinity={tf_config.affinity.value})"
                            )
                            continue

                        period = self._get_period_for_variant(
                            signal_name, variant, period_param, signal_settings
                        )

                        # Validate period against signal's bounds
                        if not (tf_config.min_period <= period <= tf_config.max_period):
                            logger.debug(
                                f"Skipping {signal_name}_{variant}: period {period} outside "
                                f"range [{tf_config.min_period}, {tf_config.max_period}]"
                            )
                            continue

                        variant_name = f"{signal_name}_{variant}"
                        success = self._compute_incremental_unified_signal(
                            signal_name, prices, tickers, sig_type,
                            {period_param.name: period}, variant_name, start_from
                        )
                        if success:
                            computed_signals.append(variant_name)
            except Exception as e:
                logger.warning(f"Failed to incrementally compute signal {signal_name}: {e}")
                continue

        # Update metadata with new end date
        new_end_date = prices.select("timestamp").max().item()
        self._update_metadata_end_date(new_end_date)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Incremental precomputation completed in {elapsed:.2f}s "
            f"({len(computed_signals)} signals updated)"
        )

        return True

    def _compute_incremental_unified_signal(
        self,
        signal_name: str,
        prices: pl.DataFrame,
        tickers: list[str],
        signal_type: SignalType,
        params: dict[str, Any],
        output_name: str,
        start_from: datetime,
    ) -> bool:
        """
        Compute a signal incrementally and append to existing cache.

        Args:
            signal_name: Registered signal name
            prices: Price DataFrame
            tickers: List of tickers
            signal_type: Signal classification type
            params: Custom parameters to override defaults
            output_name: Output file name (e.g., "momentum_return_short")
            start_from: Only append data from this date onwards

        Returns:
            True if computation succeeded
        """
        from src.signals import SignalRegistry

        try:
            signal_cls = SignalRegistry.get(signal_name)
            signal = signal_cls(**params)

            results = []

            if signal_type == SignalType.INDEPENDENT:
                # Compute per-ticker
                for ticker in tickers:
                    try:
                        ticker_prices = self._get_ticker_prices_as_pandas(prices, ticker)
                        if ticker_prices is None or len(ticker_prices) < 30:
                            continue

                        result = signal.compute(ticker_prices)
                        if result is not None and result.scores is not None:
                            df = self._signal_result_to_polars(result, ticker)
                            if df is not None and len(df) > 0:
                                results.append(df)
                    except Exception as e:
                        logger.debug(f"Failed to compute {output_name} for {ticker}: {e}")
                        continue

            elif signal_type == SignalType.RELATIVE:
                # Compute with full universe context
                results = self._compute_relative_signal(signal_name, signal, prices, tickers)

            elif signal_type == SignalType.EXTERNAL:
                # Compute with external data (or fallback)
                results = self._compute_external_signal(signal_name, signal, prices, tickers)

            if results:
                combined = pl.concat(results)
                combined = combined.filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan())
                if len(combined) > 0:
                    return self._append_to_parquet(output_name, combined, start_from)

            return False

        except Exception as e:
            logger.warning(f"Failed to compute incremental signal {output_name}: {e}")
            return False

    def _get_max_lookback_unified(self, settings: Any) -> int:
        """Calculate maximum lookback period from unified mode settings."""
        variant_periods = settings.period_variants.to_dict()
        return max(variant_periods.values()) if variant_periods else 252

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

        Uses unified mode: Computes signals using SignalRegistry.

        Args:
            prices: Full price DataFrame
            missing_signals: List of signal names to compute (e.g., ["momentum_return_short"])
            config: Signal configuration (unused, kept for API compatibility)

        Returns:
            True if all missing signals were computed successfully
        """
        if not missing_signals:
            logger.info("No missing signals to compute")
            return True

        logger.info(f"Computing {len(missing_signals)} missing signals: {missing_signals}")
        start_time = datetime.now()

        tickers = prices.select("ticker").unique().to_series().to_list()
        computed_signals = []

        # Initialize signal classification
        classification = _init_signal_classification()

        for signal_full_name in missing_signals:
            try:
                logger.info(f"Computing new signal: {signal_full_name}")
                success = self._compute_unified_signal_by_name(
                    prices, signal_full_name, tickers, classification
                )
                if success:
                    computed_signals.append(signal_full_name)
            except Exception as e:
                logger.error(f"Failed to compute signal {signal_full_name}: {e}")

        # Update metadata
        self._update_metadata_for_new_signals(computed_signals)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Missing signals computation completed in {elapsed:.2f}s "
            f"({len(computed_signals)}/{len(missing_signals)} signals computed)"
        )

        return len(computed_signals) == len(missing_signals)

    def _compute_unified_signal_by_name(
        self,
        prices: pl.DataFrame,
        signal_full_name: str,
        tickers: list[str],
        classification: dict[str, SignalType],
    ) -> bool:
        """
        Compute a signal by its full name using unified mode.

        Parses signal name (e.g., "momentum_return_short") to determine base name and variant,
        then calls the appropriate computation method.

        Args:
            prices: Price DataFrame
            signal_full_name: Full signal name (e.g., "momentum_return_short", "vix_sentiment")
            tickers: List of tickers
            classification: Signal classification mapping

        Returns:
            True if successful
        """
        from src.signals import SignalRegistry
        from src.config.settings import get_settings, SignalPrecomputeSettings

        try:
            settings = get_settings()
            signal_settings = settings.signal_precompute
        except Exception:
            signal_settings = SignalPrecomputeSettings()

        # Try to find base signal and variant
        # Format can be: "base_signal_variant" or "base_signal" (no variant)
        enabled_variants = signal_settings.enabled_variants

        base_name = None
        variant = None
        params = {}

        # Check if it matches any registered signal directly (no variant)
        if signal_full_name in SignalRegistry.list_all():
            base_name = signal_full_name
        else:
            # Try to parse variant suffix
            for var in enabled_variants:
                if signal_full_name.endswith(f"_{var}"):
                    potential_base = signal_full_name[:-len(f"_{var}")]
                    if potential_base in SignalRegistry.list_all():
                        base_name = potential_base
                        variant = var
                        break

            # Try custom period format: "signal_name_123"
            if base_name is None:
                parts = signal_full_name.rsplit("_", 1)
                if len(parts) == 2:
                    potential_base, period_str = parts
                    try:
                        period = int(period_str)
                        if potential_base in SignalRegistry.list_all():
                            base_name = potential_base
                            signal_cls = SignalRegistry.get(base_name)
                            specs = signal_cls.parameter_specs()
                            period_param = self._find_period_param(specs)
                            if period_param:
                                params = {period_param.name: period}
                    except ValueError:
                        pass

        if base_name is None:
            logger.warning(f"Could not parse signal name: {signal_full_name}")
            return False

        sig_type = classification.get(base_name, SignalType.INDEPENDENT)

        # If variant is set, resolve the period
        if variant and not params:
            signal_cls = SignalRegistry.get(base_name)
            specs = signal_cls.parameter_specs()
            period_param = self._find_period_param(specs)
            if period_param:
                period = self._get_period_for_variant(
                    base_name, variant, period_param, signal_settings
                )
                params = {period_param.name: period}

        return self._compute_registry_signal_with_params(
            base_name, prices, tickers, sig_type, params, signal_full_name
        )

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
        has_relative_signals: bool = False,
    ) -> bool:
        """
        Incremental computation for newly added tickers.

        Uses unified mode: Computes all signals from SignalRegistry.

        - Independent signals: Compute only for new tickers, append to cache
        - Relative signals: Full recomputation for all tickers (if has_relative_signals=True)

        This method provides significant cache efficiency:
        - For 100 existing tickers + 5 new tickers with only independent signals:
          Only 5% of computation needed instead of 100%
        - For mixed signals: Independent signals still benefit from incremental update

        Args:
            prices: Full price DataFrame including new tickers
            new_tickers: List of newly added ticker symbols
            config: Signal configuration (unused, kept for API compatibility)
            has_relative_signals: Whether relative signals are in the config

        Returns:
            True if computation succeeded
        """
        from src.signals import SignalRegistry
        from src.config.settings import get_settings, SignalPrecomputeSettings

        if not new_tickers:
            logger.info("No new tickers to process")
            return True

        start_time = datetime.now()

        # Load signal precompute settings
        try:
            settings = get_settings()
            signal_settings = settings.signal_precompute
        except Exception as e:
            logger.warning(f"Failed to load signal_precompute settings: {e}, using defaults")
            signal_settings = SignalPrecomputeSettings()

        all_tickers = prices.select("ticker").unique().to_series().to_list()
        existing_tickers = [t for t in all_tickers if t not in new_tickers]

        # Initialize signal classification
        classification = _init_signal_classification()

        independent_count = 0
        relative_count = 0
        independent_reused = 0

        logger.info(
            f"Processing {len(new_tickers)} new tickers "
            f"(existing: {len(existing_tickers)}, total: {len(all_tickers)})"
        )

        # Iterate through all registered signals
        for signal_name in SignalRegistry.list_all():
            try:
                signal_cls = SignalRegistry.get(signal_name)
                specs = signal_cls.parameter_specs()
                period_param = self._find_period_param(specs)
                sig_type = classification.get(signal_name, SignalType.INDEPENDENT)

                # Determine variants to compute
                if period_param is None:
                    variants = [(signal_name, {})]
                elif signal_name in signal_settings.custom_periods:
                    variants = [
                        (f"{signal_name}_{period}", {period_param.name: period})
                        for period in signal_settings.custom_periods[signal_name]
                    ]
                else:
                    # Filter variants by timeframe affinity
                    tf_config = signal_cls.timeframe_config()
                    variants = []
                    for variant in signal_settings.enabled_variants:
                        # Check if this variant is supported by the signal
                        if not tf_config.supports_variant(variant):
                            logger.debug(
                                f"Skipping {signal_name}_{variant}: variant not supported "
                                f"(affinity={tf_config.affinity.value})"
                            )
                            continue

                        period = self._get_period_for_variant(
                            signal_name, variant, period_param, signal_settings
                        )

                        # Validate period against signal's bounds
                        if not (tf_config.min_period <= period <= tf_config.max_period):
                            logger.debug(
                                f"Skipping {signal_name}_{variant}: period {period} outside "
                                f"range [{tf_config.min_period}, {tf_config.max_period}]"
                            )
                            continue

                        variants.append((
                            f"{signal_name}_{variant}",
                            {period_param.name: period}
                        ))

                for output_name, params in variants:
                    signal_class = self.classify_signal(output_name)
                    if signal_class == "independent":
                        # Independent signals: only compute for new tickers
                        logger.debug(
                            f"Computing {output_name} for {len(new_tickers)} new tickers "
                            f"(reusing {len(existing_tickers)} cached)"
                        )
                        new_ticker_prices = prices.filter(pl.col("ticker").is_in(new_tickers))
                        self._compute_and_append_unified_signal(
                            signal_name, new_ticker_prices, new_tickers,
                            sig_type, params, output_name
                        )
                        independent_count += 1
                        independent_reused += len(existing_tickers)
                    else:
                        # Relative signals: require full recomputation
                        logger.debug(
                            f"Recomputing {output_name} for all {len(all_tickers)} tickers (relative signal)"
                        )
                        self._compute_registry_signal_with_params(
                            signal_name, prices, all_tickers, sig_type, params, output_name
                        )
                        relative_count += 1

            except Exception as e:
                logger.warning(f"Failed to compute signal {signal_name} for new tickers: {e}")
                continue

        self._update_metadata_tickers(all_tickers)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Calculate cache efficiency
        total_signals = independent_count + relative_count
        total_possible = total_signals * len(all_tickers)
        actually_computed = (independent_count * len(new_tickers)) + (relative_count * len(all_tickers))
        cache_efficiency = 1.0 - (actually_computed / total_possible) if total_possible > 0 else 0.0

        logger.info(
            f"New tickers computation completed in {elapsed:.2f}s: "
            f"independent={independent_count} (reused {independent_reused} cached values), "
            f"relative={relative_count}, "
            f"cache efficiency={cache_efficiency:.1%}"
        )
        return True

    def _compute_and_append_unified_signal(
        self,
        signal_name: str,
        prices: pl.DataFrame,
        tickers: list[str],
        signal_type: SignalType,
        params: dict[str, Any],
        output_name: str,
    ) -> bool:
        """Compute signal for specific tickers and append to existing Parquet (unified mode)."""
        from src.signals import SignalRegistry

        try:
            signal_cls = SignalRegistry.get(signal_name)
            signal = signal_cls(**params)

            results = []

            if signal_type == SignalType.INDEPENDENT:
                for ticker in tickers:
                    try:
                        ticker_prices = self._get_ticker_prices_as_pandas(prices, ticker)
                        if ticker_prices is None or len(ticker_prices) < 30:
                            continue

                        result = signal.compute(ticker_prices)
                        if result is not None and result.scores is not None:
                            df = self._signal_result_to_polars(result, ticker)
                            if df is not None and len(df) > 0:
                                results.append(df)
                    except Exception as e:
                        logger.debug(f"Failed to compute {output_name} for {ticker}: {e}")
                        continue
            else:
                # For relative/external signals, compute normally
                return self._compute_registry_signal_with_params(
                    signal_name, prices, tickers, signal_type, params, output_name
                )

            if not results:
                return True

            new_signals = pl.concat(results)
            new_signals = new_signals.filter(pl.col("value").is_not_null() & ~pl.col("value").is_nan())

            if new_signals.is_empty():
                return True

            filename = f"{output_name}.parquet"
            if self._parquet_exists(filename):
                existing = self._read_parquet(filename)
                existing_filtered = existing.filter(~pl.col("ticker").is_in(tickers))
                # Ensure timestamp columns have consistent types (use milliseconds)
                if "timestamp" in existing_filtered.columns and "timestamp" in new_signals.columns:
                    existing_filtered = existing_filtered.with_columns(
                        pl.col("timestamp").cast(pl.Datetime("ms"))
                    )
                    new_signals = new_signals.with_columns(
                        pl.col("timestamp").cast(pl.Datetime("ms"))
                    )
                combined = pl.concat([existing_filtered, new_signals]).sort(["ticker", "timestamp"])
                self._write_parquet(combined, filename)
            else:
                self._write_parquet(new_signals, filename)

            return True

        except Exception as e:
            logger.warning(f"Failed to compute and append signal {output_name}: {e}")
            return False

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

        if n < period + 1:
            return np.full(n, np.nan)

        # Calculate returns from close prices
        returns = np.diff(close) / close[:-1]

        # Use unified metrics module
        rolling_sharpe = calculate_rolling_sharpe(
            returns, window=period, annualization_factor=annualize
        )

        # Prepend NaN to align with original close array length
        result = np.empty(n)
        result[0] = np.nan
        result[1:] = rolling_sharpe

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
        """Get cache statistics including hit/miss rates."""
        parquet_files = self._list_parquet_files()

        # Base stats from CacheStats object
        base_stats = self._cache_stats.to_dict()

        # For backend mode, we can't easily get file sizes without downloading
        if self._use_backend and self._backend is not None:
            return {
                **base_stats,
                "cache_dir": str(self._cache_dir),
                "backend": "s3",  # S3 is required
                "num_signals": len(parquet_files),
                "signals": [Path(f).stem for f in parquet_files],
                "prices_hash": self._metadata.get("prices_hash"),
                "computed_at": self._metadata.get("computed_at"),
            }
        else:
            files = list(self._cache_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in files)

            return {
                **base_stats,
                "cache_dir": str(self._cache_dir),
                "backend": "local (fallback)",
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
