"""
Data Preparation Module - Handles data fetching, cutoff, and quality checks.

Extracted from pipeline.py for better modularity (QA-003-P1).
This module handles:
1. Data fetching via MultiSourceAdapter
2. Data cutoff for backtesting (prevents future data leakage)
3. Quality checks and asset exclusion

Cache operations are delegated to data_cache submodules for modularity.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import structlog

if TYPE_CHECKING:
    import polars as pl

    from src.config.settings import Settings
    from src.data.quality_checker import DataQualityReport
    from src.utils.logger import AuditLogger
    from src.utils.storage_backend import StorageBackend

from src.utils.hash_utils import compute_universe_hash

# Cache managers (extracted for modularity)
from src.orchestrator.data_cache.quality_cache import QualityCacheManager, QualityCheckCache
from src.orchestrator.data_cache.price_cache import PriceCacheManager

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


# Note: QualityCheckCache is now imported from data_cache.quality_cache


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
        storage_backend: "Optional[StorageBackend]" = None,
    ) -> None:
        """
        Initialize DataPreparation.

        Args:
            settings: Application settings
            output_dir: Directory for cache and outputs
            audit_logger: Optional audit logger for detailed logging
            storage_backend: Optional StorageBackend for S3 price cache support
        """
        self._settings = settings
        self._output_dir = output_dir or Path("data/output")
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="data_preparation")
        self._storage_backend = storage_backend

        # Data stores
        self._raw_data: dict[str, "pl.DataFrame"] = {}
        self._quality_reports: dict[str, Any] = {}
        self._excluded_assets: list[str] = []
        self._quality_summary: dict[str, list[str]] = {}
        self._fetch_summary: dict[str, list[str]] = {}
        self._warnings: list[str] = []

        # Cache managers (initialized lazily)
        self._quality_cache_manager: QualityCacheManager | None = None
        self._price_cache_manager: PriceCacheManager | None = None

    @property
    def quality_cache_manager(self) -> QualityCacheManager:
        """Get quality cache manager (lazy initialization)."""
        if self._quality_cache_manager is None:
            self._quality_cache_manager = QualityCacheManager(
                settings=self._settings,
                cache_dir=self._get_cache_dir(),
                storage_backend=self._storage_backend,
            )
        return self._quality_cache_manager

    @property
    def price_cache_manager(self) -> PriceCacheManager | None:
        """Get price cache manager (requires StorageBackend)."""
        if self._price_cache_manager is None and self._storage_backend is not None:
            self._price_cache_manager = PriceCacheManager(
                settings=self._settings,
                storage_backend=self._storage_backend,
            )
        return self._price_cache_manager

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    # =========================================================================
    # Quality Check Cache Methods
    # =========================================================================
    def _get_cache_dir(self) -> Path:
        """Get quality check cache directory (local mode only)."""
        from src.config.settings import get_cache_path
        cache_dir = Path(get_cache_path("quality"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_cache_subdir(self) -> str:
        """Get cache subdirectory for StorageBackend."""
        return "quality"

    def _compute_universe_hash(self, universe: list[str]) -> str:
        """Compute hash of universe (symbol list).

        Args:
            universe: List of asset symbols

        Returns:
            MD5 hash of sorted symbols
        """
        return compute_universe_hash(universe)

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
        """Load quality check cache from disk or StorageBackend.

        Args:
            date: The date to load cache for

        Returns:
            QualityCheckCache if valid cache exists, None otherwise
        """
        date_str = date.strftime("%Y%m%d")
        cache_key = f"{date_str}.pkl"

        try:
            if self._storage_backend is not None:
                # Use StorageBackend
                cache = self._storage_backend.read_pickle(
                    f"{self._get_cache_subdir()}/{cache_key}"
                )
            else:
                # Local file
                cache_path = self._get_cache_path(date)
                if not cache_path.exists():
                    return None
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)

            if isinstance(cache, QualityCheckCache):
                return cache
        except (pickle.PickleError, EOFError, TypeError, FileNotFoundError) as e:
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
        """Save quality check cache to disk or StorageBackend.

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

        date_str = date.strftime("%Y%m%d")
        cache_key = f"{date_str}.pkl"

        try:
            if self._storage_backend is not None:
                # Use StorageBackend
                self._storage_backend.write_pickle(
                    cache,
                    f"{self._get_cache_subdir()}/{cache_key}"
                )
                self._logger.debug(f"Saved quality cache via StorageBackend: {cache_key}")
            else:
                # Local file
                cache_path = self._get_cache_path(date)
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
        1. Load from cache (partial)
        2. Fetch missing symbols via MultiSourceAdapter
        3. CurrencyConverter -> USD conversion
        4. CalendarManager -> common calendar alignment
        5. Save to cache
        6. Quality checks with summary report
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

        # Step 1: Load from cache
        cached_prices, missing_symbols = self._load_from_cache_partial(
            universe, start_date, end_date
        )
        self._raw_data.update(cached_prices)

        # If all symbols are in cache, skip fetch
        if not missing_symbols:
            self._logger.info(f"All {len(universe)} symbols loaded from cache")
            return self._raw_data

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
            "from_cache": list(cached_prices.keys()),
        }

        # Batch processing with progress (only for missing symbols)
        batches = [
            missing_symbols[i:i + batch_size]
            for i in range(0, len(missing_symbols), batch_size)
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

                # Save batch to cache after processing
                self._save_batch_to_cache(batch_data, start_date, end_date)

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

        from_cache = len(fetch_results.get("from_cache", []))
        self._logger.info(
            "Data fetch completed (expanded mode)",
            from_cache=from_cache,
            fetched=len(fetch_results["success"]),
            failed=len(fetch_results["failed"]),
            converted=len(fetch_results["converted"]),
            aligned=len(fetch_results["aligned"]),
        )

        return self._raw_data

    def _load_from_cache_partial(
        self,
        universe: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> tuple[dict[str, "pl.DataFrame"], list[str]]:
        """
        キャッシュから価格データを部分的に読み込む。

        StorageBackend対応:
        - storage_backend指定時: S3/ローカルを透過的に操作
        - storage_backend未指定時: ローカルファイルシステムを直接操作

        Args:
            universe: シンボルのリスト
            start_date: 開始日
            end_date: 終了日

        Returns:
            (読み込めた価格データ, 読み込めなかったシンボルのリスト)
        """
        import polars as pl

        prices: dict[str, pl.DataFrame] = {}
        missing_symbols: list[str] = []

        # StorageBackend経由の場合
        if self._storage_backend is not None:
            return self._load_from_cache_via_backend(universe, start_date, end_date)

        # ローカルファイルシステムの場合
        from src.config.settings import get_cache_path
        cache_dir = Path(get_cache_path("prices"))

        if not cache_dir.exists():
            return {}, list(universe)

        # キャッシュファイルのマッピングを作成
        cache_symbol_map: dict[str, Path] = {}
        for pf in cache_dir.glob("*.parquet"):
            cache_name = pf.stem
            # ファイル名から元のシンボルを復元
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            cache_symbol_map[original_symbol] = pf

        for symbol in universe:
            pf = cache_symbol_map.get(symbol)
            if pf is None:
                missing_symbols.append(symbol)
                continue

            try:
                df = pl.read_parquet(pf)
                df = self._filter_by_date_range(df, start_date, end_date)

                if df is not None and len(df) > 0:
                    prices[symbol] = df
                else:
                    missing_symbols.append(symbol)

            except Exception as e:
                self._logger.debug(f"Failed to load {symbol} from cache: {e}")
                missing_symbols.append(symbol)

        if prices:
            self._logger.info(
                f"Loaded {len(prices)}/{len(universe)} symbols from cache"
            )

        return prices, missing_symbols

    def _load_from_cache_via_backend(
        self,
        universe: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> tuple[dict[str, "pl.DataFrame"], list[str]]:
        """StorageBackend経由でキャッシュを読み込む。"""
        import polars as pl

        prices: dict[str, pl.DataFrame] = {}
        missing_symbols: list[str] = []

        # キャッシュファイル一覧を取得
        cache_files = self._storage_backend.list_files("prices", "*.parquet")

        # シンボル -> ファイルパスのマッピング
        cache_symbol_map: dict[str, str] = {}
        for rel_path in cache_files:
            # prices/AAPL.parquet -> AAPL
            filename = rel_path.split("/")[-1] if "/" in rel_path else rel_path
            cache_name = filename.replace(".parquet", "")
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            cache_symbol_map[original_symbol] = f"prices/{filename}"

        for symbol in universe:
            rel_path = cache_symbol_map.get(symbol)
            if rel_path is None:
                # シンボル名からパスを構築して確認
                safe_symbol = symbol.replace("=", "_").replace("/", "_")
                rel_path = f"prices/{safe_symbol}.parquet"
                if not self._storage_backend.exists(rel_path):
                    missing_symbols.append(symbol)
                    continue

            try:
                df = self._storage_backend.read_parquet(rel_path)
                df = self._filter_by_date_range(df, start_date, end_date)

                if df is not None and len(df) > 0:
                    prices[symbol] = df
                else:
                    missing_symbols.append(symbol)

            except Exception as e:
                self._logger.debug(f"Failed to load {symbol} from backend: {e}")
                missing_symbols.append(symbol)

        if prices:
            self._logger.info(
                f"Loaded {len(prices)}/{len(universe)} symbols from cache (via backend)"
            )

        return prices, missing_symbols

    def _filter_by_date_range(
        self,
        df: "pl.DataFrame",
        start_date: datetime,
        end_date: datetime,
    ) -> "pl.DataFrame | None":
        """DataFrameを日付範囲でフィルタリング。"""
        import polars as pl

        # 日付列を特定
        date_col = None
        for col in ["timestamp", "date", "datetime", "time"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return None

        # 日付でフィルタリング
        if df[date_col].dtype == pl.Date:
            start_filter = start_date.date() if hasattr(start_date, 'date') else start_date
            end_filter = end_date.date() if hasattr(end_date, 'date') else end_date
        else:
            start_filter = start_date
            end_filter = end_date

        return df.filter(
            (pl.col(date_col) >= start_filter) &
            (pl.col(date_col) <= end_filter)
        )

    def _save_batch_to_cache(
        self,
        batch_data: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        バッチデータをキャッシュに保存する（既存データとマージ）。

        StorageBackend対応:
        - storage_backend指定時: S3/ローカルを透過的に操作
        - storage_backend未指定時: ローカルファイルシステムを直接操作

        Args:
            batch_data: シンボル -> OHLCVData の辞書
            start_date: データ開始日
            end_date: データ終了日
        """
        import polars as pl

        saved_count = 0
        merged_count = 0

        for symbol, ohlcv_data in batch_data.items():
            if ohlcv_data is None:
                continue

            # OHLCVData の data 属性を取得
            if hasattr(ohlcv_data, "data"):
                new_df = ohlcv_data.data
                if new_df is None or (hasattr(new_df, "is_empty") and new_df.is_empty()):
                    continue
            else:
                continue

            try:
                # ファイル名に使用できない文字を置換
                safe_symbol = symbol.replace("=", "_").replace("/", "_")
                rel_path = f"prices/{safe_symbol}.parquet"

                # StorageBackend経由の場合
                if self._storage_backend is not None:
                    merged = self._save_to_backend_with_merge(
                        new_df, rel_path, symbol
                    )
                    if merged:
                        merged_count += 1
                    else:
                        saved_count += 1
                    continue

                # ローカルファイルシステムの場合
                from src.config.settings import get_cache_path
                cache_dir = Path(get_cache_path("prices"))
                cache_path = cache_dir / f"{safe_symbol}.parquet"
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                # 既存のキャッシュがあればマージ
                if cache_path.exists():
                    try:
                        existing_df = pl.read_parquet(cache_path)
                        merged_df = self._merge_dataframes(existing_df, new_df)
                        if merged_df is not None:
                            merged_df.write_parquet(cache_path)
                            merged_count += 1
                            continue
                    except Exception as e:
                        self._logger.debug(f"Cache merge failed for {symbol}, overwriting: {e}")

                # Polars DataFrame の場合
                if hasattr(new_df, "write_parquet"):
                    new_df.write_parquet(cache_path)
                # Pandas DataFrame の場合
                elif hasattr(new_df, "to_parquet"):
                    new_df.to_parquet(cache_path)

                saved_count += 1
            except Exception as e:
                self._logger.debug(f"Failed to cache {symbol}: {e}")

        total = saved_count + merged_count
        if total > 0:
            backend_info = " (via backend)" if self._storage_backend else ""
            self._logger.info(
                f"Cached {total} symbols (new: {saved_count}, merged: {merged_count}){backend_info}"
            )

    def _save_to_backend_with_merge(
        self,
        new_df: "pl.DataFrame",
        rel_path: str,
        symbol: str,
    ) -> bool:
        """StorageBackend経由でキャッシュを保存（マージ対応）。

        Returns:
            True if merged with existing, False if new save
        """
        import polars as pl

        merged = False

        # 既存データがあればマージ
        if self._storage_backend.exists(rel_path):
            try:
                existing_df = self._storage_backend.read_parquet(rel_path)
                merged_df = self._merge_dataframes(existing_df, new_df)
                if merged_df is not None:
                    self._storage_backend.write_parquet(merged_df, rel_path)
                    merged = True
                    return merged
            except Exception as e:
                self._logger.debug(f"Merge failed for {symbol}, overwriting: {e}")

        # 新規保存
        self._storage_backend.write_parquet(new_df, rel_path)
        return merged

    def _merge_dataframes(
        self,
        existing_df: "pl.DataFrame",
        new_df: "pl.DataFrame",
    ) -> "pl.DataFrame | None":
        """2つのDataFrameをマージ（重複は新しいデータを優先）。"""
        import polars as pl

        # 日付列を特定
        date_col = None
        for col in ["timestamp", "date", "datetime", "time"]:
            if col in existing_df.columns and col in new_df.columns:
                date_col = col
                break

        if date_col is None:
            return None

        # マージ（重複は新しいデータを優先）
        combined = pl.concat([existing_df, new_df])
        combined = combined.unique(subset=[date_col], keep="last")
        combined = combined.sort(date_col)
        return combined

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
