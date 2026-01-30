"""
Data Preparation Module - Handles data fetching, cutoff, and quality checks.

Extracted from pipeline.py for better modularity (QA-003-P1).
This module handles:
1. Data fetching via MultiSourceAdapter
2. Data cutoff for backtesting (prevents future data leakage)
3. Quality checks and asset exclusion
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

    from src.config.settings import Settings
    from src.data.quality_checker import DataQualityReport
    from src.utils.logger import AuditLogger

# Universe expansion support (optional - imported on demand)
try:
    from src.data.universe_loader import UniverseLoader
    from src.data.adapters.multi_source_adapter import MultiSourceAdapter
    from src.data.currency_converter import CurrencyConverter
    from src.data.calendar_manager import CalendarManager
    UNIVERSE_EXPANSION_AVAILABLE = True
except ImportError:
    UNIVERSE_EXPANSION_AVAILABLE = False

logger = structlog.get_logger(__name__)


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
    reports: dict[str, "DataQualityReport"]
    excluded_assets: list[str]
    last_bar_dates: dict[str, datetime] = field(default_factory=dict)


@dataclass
class DataPreparationResult:
    """Result of data preparation step."""

    raw_data: dict[str, "pl.DataFrame"]
    quality_reports: dict[str, Any]
    excluded_assets: list[str]
    quality_summary: dict[str, list[str]]
    fetch_summary: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class DataPreparation:
    """
    Handles data preparation for the pipeline.

    Responsible for:
    - Fetching OHLCV data from various sources
    - Applying data cutoff for backtesting
    - Running quality checks and excluding bad data
    """

    def __init__(
        self,
        settings: "Settings",
        output_dir: Path | None = None,
        audit_logger: "AuditLogger | None" = None,
    ) -> None:
        """
        Initialize DataPreparation.

        Args:
            settings: Application settings
            output_dir: Directory for cache and outputs
            audit_logger: Optional audit logger for detailed logging
        """
        self._settings = settings
        self._output_dir = output_dir or Path("data/output")
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="data_preparation")

        # Data stores
        self._raw_data: dict[str, "pl.DataFrame"] = {}
        self._quality_reports: dict[str, Any] = {}
        self._excluded_assets: list[str] = []
        self._quality_summary: dict[str, list[str]] = {}
        self._fetch_summary: dict[str, list[str]] = {}
        self._warnings: list[str] = []

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    # =========================================================================
    # Quality Check Cache Methods
    # =========================================================================
    def _get_cache_dir(self) -> Path:
        """Get quality check cache directory."""
        cache_dir = self._output_dir / ".cache" / "quality"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _compute_universe_hash(self, universe: list[str]) -> str:
        """Compute hash of universe (symbol list).

        Args:
            universe: List of asset symbols

        Returns:
            MD5 hash of sorted symbols
        """
        sorted_symbols = sorted(universe)
        hash_input = ",".join(sorted_symbols).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()[:16]

    def _compute_quality_config_hash(self) -> str:
        """Compute hash of quality configuration.

        Returns:
            MD5 hash of quality config parameters
        """
        config = self._settings.data_quality
        config_dict = {
            "max_missing_rate": config.max_missing_rate,
            "max_consecutive_missing": config.max_consecutive_missing,
            "price_change_threshold": config.price_change_threshold,
            "min_volume_threshold": config.min_volume_threshold,
            "staleness_hours": config.staleness_hours,
            "ohlc_inconsistency_threshold": config.ohlc_inconsistency_threshold,
        }
        hash_input = str(sorted(config_dict.items())).encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()[:16]

    def _get_cache_path(self, date: datetime) -> Path:
        """Get cache file path for a specific date.

        Args:
            date: The date for the cache file

        Returns:
            Path to the cache file
        """
        date_str = date.strftime("%Y%m%d")
        return self._get_cache_dir() / f"{date_str}.pkl"

    def _load_quality_cache(self, date: datetime) -> QualityCheckCache | None:
        """Load quality check cache from disk.

        Args:
            date: The date to load cache for

        Returns:
            QualityCheckCache if valid cache exists, None otherwise
        """
        cache_path = self._get_cache_path(date)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, QualityCheckCache):
                return cache
        except (pickle.PickleError, EOFError, TypeError) as e:
            self._logger.warning(f"Failed to load quality cache: {e}")

        return None

    def _save_quality_cache(
        self,
        date: datetime,
        universe_hash: str,
        quality_config_hash: str,
        reports: dict[str, Any],
        excluded_assets: list[str],
        last_bar_dates: dict[str, datetime],
    ) -> None:
        """Save quality check cache to disk.

        Args:
            date: The date for the cache
            universe_hash: Hash of the universe
            quality_config_hash: Hash of quality configuration
            reports: Quality reports for each symbol
            excluded_assets: List of excluded asset symbols
            last_bar_dates: Last bar date for each symbol
        """
        cache = QualityCheckCache(
            date=date,
            universe_hash=universe_hash,
            quality_config_hash=quality_config_hash,
            reports=reports,
            excluded_assets=excluded_assets,
            last_bar_dates=last_bar_dates,
        )

        cache_path = self._get_cache_path(date)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._logger.debug(f"Saved quality cache to {cache_path}")
        except (pickle.PickleError, OSError) as e:
            self._logger.warning(f"Failed to save quality cache: {e}")

    def _get_last_bar_date(self, df: "pl.DataFrame") -> datetime | None:
        """Get the last bar date from a DataFrame.

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

    def fetch_data(
        self,
        universe: list[str],
        as_of_date: datetime,
        skip_fetch: bool = False,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Fetch data for all assets in universe.

        Supports two modes:
        - Legacy mode: Uses StockAdapter/CryptoAdapter (backward compatible)
        - Universe expansion mode: Uses MultiSourceAdapter with batch processing

        Args:
            universe: List of asset symbols to fetch
            as_of_date: Reference date for data fetching
            skip_fetch: If True, skip actual fetching (dry run)

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if skip_fetch:
            self._logger.info("Skipping data fetch (dry run)")
            # Return existing data if already injected (e.g., by BacktestEngine)
            # This allows external callers to pre-populate _raw_data
            return self._raw_data

        return self._fetch_data_expanded(universe, as_of_date)

    def _fetch_data_expanded(
        self,
        universe: list[str],
        as_of_date: datetime,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Expanded data fetch using UniverseLoader, MultiSourceAdapter,
        CurrencyConverter, and CalendarManager.

        New data flow:
        1. UniverseLoader -> ticker list
        2. MultiSourceAdapter -> batch fetch with progress
        3. CurrencyConverter -> USD conversion
        4. CalendarManager -> common calendar alignment
        5. Quality checks with summary report
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Calculate date range
        train_days = self.settings.walk_forward.train_period_days
        test_days = self.settings.walk_forward.test_period_days
        total_days = train_days + test_days

        end_date = as_of_date
        start_date = end_date - timedelta(days=total_days)

        # Get settings
        batch_size = getattr(self.settings.data, 'batch_size', 50)
        base_currency = getattr(self.settings.data, 'base_currency', 'USD')
        parallel_workers = getattr(self.settings.data, 'parallel_workers', 4)

        self._logger.info(
            "Fetching data (expanded mode)",
            asset_count=len(universe),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            batch_size=batch_size,
            parallel_workers=parallel_workers,
        )

        # Initialize components
        adapter = MultiSourceAdapter(settings=self.settings)
        converter = CurrencyConverter(base_currency=base_currency)
        calendar_mgr = CalendarManager()

        # Track results by category
        fetch_results: dict[str, list] = {
            "success": [],
            "failed": [],
            "converted": [],
            "aligned": [],
        }

        # Batch processing with progress
        batches = [
            universe[i:i + batch_size]
            for i in range(0, len(universe), batch_size)
        ]

        progress_iter = tqdm(batches, desc="Fetching data") if tqdm else batches

        for batch in progress_iter:
            try:
                # Fetch batch using MultiSourceAdapter
                batch_data = adapter.fetch_batch(
                    symbols=batch,
                    start_date=start_date,
                    end_date=end_date,
                    parallel=parallel_workers > 1,
                    max_workers=parallel_workers,
                )

                # Process each symbol in batch
                for symbol, ohlcv_data in batch_data.items():
                    try:
                        if ohlcv_data is None or ohlcv_data.data.is_empty():
                            fetch_results["failed"].append(symbol)
                            continue

                        df = ohlcv_data.data

                        # Currency conversion if needed
                        if ohlcv_data.currency != base_currency:
                            df = converter.convert(
                                df=df,
                                from_currency=ohlcv_data.currency,
                                to_currency=base_currency,
                            )
                            fetch_results["converted"].append(symbol)

                        # Store raw data
                        self._raw_data[symbol] = df
                        fetch_results["success"].append(symbol)

                    except Exception as e:
                        self._logger.warning(
                            f"Failed to process {symbol}: {e}"
                        )
                        fetch_results["failed"].append(symbol)

            except Exception as e:
                self._logger.error(f"Batch fetch failed: {e}")
                fetch_results["failed"].extend(batch)

        # Calendar alignment for all fetched data
        if self._raw_data:
            try:
                aligned_data = calendar_mgr.align_to_common_calendar(
                    data_dict=self._raw_data,
                    method="ffill",
                )
                self._raw_data = aligned_data
                fetch_results["aligned"] = list(aligned_data.keys())
            except Exception as e:
                self._logger.warning(f"Calendar alignment failed: {e}")

        # Record failed assets
        if fetch_results["failed"]:
            self._warnings.append(
                f"Failed to fetch {len(fetch_results['failed'])} asset(s)"
            )

        # Store fetch summary
        self._fetch_summary = fetch_results

        self._logger.info(
            "Data fetch completed (expanded mode)",
            success=len(fetch_results["success"]),
            failed=len(fetch_results["failed"]),
            converted=len(fetch_results["converted"]),
            aligned=len(fetch_results["aligned"]),
        )

        return self._raw_data

    def apply_data_cutoff(self, cutoff_date: datetime) -> None:
        """
        Apply data cutoff filter to all raw data.

        This method ensures NO data after cutoff_date is used in any pipeline step.
        Critical for backtesting to prevent future data leakage.

        Args:
            cutoff_date: All data after this date will be removed
        """
        import polars as pl

        # Remove timezone info from cutoff_date to avoid comparison issues
        # between timezone-aware and timezone-naive datetimes
        if hasattr(cutoff_date, 'tzinfo') and cutoff_date.tzinfo is not None:
            cutoff_date_naive = cutoff_date.replace(tzinfo=None)
        else:
            cutoff_date_naive = cutoff_date

        filtered_count = 0
        total_removed_rows = 0

        for symbol, df in list(self._raw_data.items()):
            if df is None or len(df) == 0:
                continue

            original_len = len(df)

            # Handle both polars and pandas DataFrames
            if hasattr(df, 'filter'):  # Polars DataFrame
                # Find the timestamp/date column
                date_col = None
                for col in ['timestamp', 'date', 'datetime', 'time']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    # Convert cutoff to proper format for comparison
                    if df[date_col].dtype == pl.Date:
                        cutoff = (
                            cutoff_date_naive.date()
                            if hasattr(cutoff_date_naive, 'date')
                            else cutoff_date_naive
                        )
                        filtered_df = df.filter(pl.col(date_col) <= cutoff)
                    else:
                        # Assume datetime - use timezone-naive version
                        filtered_df = df.filter(pl.col(date_col) <= cutoff_date_naive)
                    self._raw_data[symbol] = filtered_df
                else:
                    # No date column found, try filtering by index if available
                    self._logger.warning(
                        f"No date column found for {symbol}, cannot apply cutoff"
                    )
                    continue

            else:  # Pandas DataFrame
                # Check if index is datetime
                if (
                    hasattr(df.index, 'tz_localize')
                    or str(df.index.dtype).startswith('datetime')
                ):
                    # Index is datetime - use timezone-naive version
                    mask = df.index <= cutoff_date_naive
                    filtered_df = df.loc[mask]
                    self._raw_data[symbol] = filtered_df
                else:
                    # Look for date column
                    date_col = None
                    for col in ['timestamp', 'date', 'datetime', 'time']:
                        if col in df.columns:
                            date_col = col
                            break

                    if date_col:
                        mask = df[date_col] <= cutoff_date_naive
                        filtered_df = df.loc[mask]
                        self._raw_data[symbol] = filtered_df
                    else:
                        self._logger.warning(
                            f"No date column found for {symbol}, cannot apply cutoff"
                        )
                        continue

            new_len = len(self._raw_data[symbol])
            removed = original_len - new_len

            if removed > 0:
                filtered_count += 1
                total_removed_rows += removed

        self._logger.info(
            "Applied data cutoff filter",
            cutoff_date=cutoff_date.isoformat(),
            assets_filtered=filtered_count,
            total_rows_removed=total_removed_rows,
        )

    def run_quality_check(
        self,
        use_cache: bool = True,
        cache_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Run quality checks on all raw data with incremental caching support.

        For large universes, generates a summary report instead of
        logging each asset individually.

        Cache invalidation occurs when:
        - Quality configuration (thresholds) changes
        - Universe (symbol list) changes

        Incremental check:
        - Previously excluded assets are reused if no new data
        - Only new bars (~20 days) are checked for existing assets
        - New symbols receive full quality check

        Args:
            use_cache: Whether to use caching (default: True)
            cache_date: Date for cache lookup (default: today)

        Returns:
            Dictionary mapping symbol to quality report
        """
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(self._settings)
        cache_date = cache_date or datetime.utcnow()

        # Compute hashes for cache invalidation
        current_universe = list(self._raw_data.keys())
        universe_hash = self._compute_universe_hash(current_universe)
        quality_config_hash = self._compute_quality_config_hash()

        # Try to load cache
        cache: QualityCheckCache | None = None
        cache_hit = False
        reused_count = 0
        checked_count = 0

        if use_cache:
            cache = self._load_quality_cache(cache_date)
            if cache is not None:
                # Validate cache
                if (
                    cache.universe_hash == universe_hash
                    and cache.quality_config_hash == quality_config_hash
                ):
                    cache_hit = True
                    self._logger.info(
                        "Quality cache valid",
                        date=cache_date.strftime("%Y-%m-%d"),
                        cached_symbols=len(cache.reports),
                    )
                else:
                    self._logger.info(
                        "Quality cache invalidated",
                        universe_changed=cache.universe_hash != universe_hash,
                        config_changed=cache.quality_config_hash != quality_config_hash,
                    )
                    cache = None

        self._quality_reports = {}
        self._excluded_assets = []

        # Track quality issues by category for summary
        self._quality_summary = {
            "ok": [],
            "insufficient_data": [],
            "high_missing_ratio": [],
            "low_liquidity": [],
            "other_exclusion": [],
        }

        # Track last bar dates for incremental cache
        last_bar_dates: dict[str, datetime] = {}

        for symbol, df in self._raw_data.items():
            current_last_bar = self._get_last_bar_date(df)
            if current_last_bar:
                last_bar_dates[symbol] = current_last_bar

            # Check if we can reuse cached result
            should_recheck = True
            if cache_hit and cache is not None and symbol in cache.reports:
                cached_last_bar = cache.last_bar_dates.get(symbol)
                if cached_last_bar and current_last_bar:
                    # Reuse if no new data (last bar same or older)
                    if current_last_bar <= cached_last_bar:
                        # Reuse cached report
                        report = cache.reports[symbol]
                        self._quality_reports[symbol] = report
                        should_recheck = False
                        reused_count += 1

            if should_recheck:
                # Run full quality check
                report = checker.check(df, symbol)
                self._quality_reports[symbol] = report
                checked_count += 1

            # Categorize result
            report = self._quality_reports[symbol]
            if report.is_excluded:
                self._excluded_assets.append(symbol)
                # Categorize exclusion reason
                reason = report.exclusion_reason or ""
                if "insufficient" in reason.lower():
                    self._quality_summary["insufficient_data"].append(symbol)
                elif "missing" in reason.lower():
                    self._quality_summary["high_missing_ratio"].append(symbol)
                elif "liquidity" in reason.lower() or "volume" in reason.lower():
                    self._quality_summary["low_liquidity"].append(symbol)
                else:
                    self._quality_summary["other_exclusion"].append(symbol)
            else:
                self._quality_summary["ok"].append(symbol)

            # Log individual asset only in debug mode for large universes
            if len(self._raw_data) <= 50 and self._audit_logger:
                self._audit_logger.log_data_quality(
                    symbol=symbol,
                    status="excluded" if report.is_excluded else "ok",
                    metrics=report.to_dict(),
                    excluded=report.is_excluded,
                    reason=report.exclusion_reason,
                )

        # Save cache
        if use_cache:
            self._save_quality_cache(
                date=cache_date,
                universe_hash=universe_hash,
                quality_config_hash=quality_config_hash,
                reports=self._quality_reports,
                excluded_assets=self._excluded_assets,
                last_bar_dates=last_bar_dates,
            )

        # Log summary for large universes
        if len(self._raw_data) > 50:
            self._logger.info(
                "Quality check summary",
                total=len(self._raw_data),
                ok=len(self._quality_summary["ok"]),
                insufficient_data=len(self._quality_summary["insufficient_data"]),
                high_missing_ratio=len(self._quality_summary["high_missing_ratio"]),
                low_liquidity=len(self._quality_summary["low_liquidity"]),
                other_exclusion=len(self._quality_summary["other_exclusion"]),
            )

            # Log excluded assets list
            if self._excluded_assets:
                self._logger.debug(
                    "Excluded assets",
                    count=len(self._excluded_assets),
                    assets=self._excluded_assets[:20],  # First 20 only
                )

        self._logger.info(
            "Quality check completed",
            total=len(self._raw_data),
            excluded=len(self._excluded_assets),
            cache_hit=cache_hit,
            reused=reused_count,
            checked=checked_count,
        )

        return self._quality_reports

    def get_result(self) -> DataPreparationResult:
        """
        Get the result of data preparation.

        Returns:
            DataPreparationResult with all prepared data
        """
        return DataPreparationResult(
            raw_data=self._raw_data,
            quality_reports=self._quality_reports,
            excluded_assets=self._excluded_assets,
            quality_summary=self._quality_summary,
            fetch_summary=self._fetch_summary,
            warnings=self._warnings,
        )

    def prepare(
        self,
        universe: list[str],
        as_of_date: datetime,
        data_cutoff_date: datetime | None = None,
        skip_fetch: bool = False,
    ) -> DataPreparationResult:
        """
        Run full data preparation pipeline.

        This is a convenience method that runs all preparation steps:
        1. Fetch data
        2. Apply cutoff (if specified)
        3. Run quality checks

        Args:
            universe: List of asset symbols
            as_of_date: Reference date
            data_cutoff_date: Optional cutoff date for backtesting
            skip_fetch: If True, skip data fetching

        Returns:
            DataPreparationResult with all prepared data
        """
        # Step 1: Fetch data
        self.fetch_data(universe, as_of_date, skip_fetch)

        # Step 2: Apply cutoff if specified
        if data_cutoff_date:
            self.apply_data_cutoff(data_cutoff_date)

        # Step 3: Quality check
        self.run_quality_check()

        return self.get_result()
