"""
Price Cache Module - Price data caching with StorageBackend support.

This module handles caching of price data (OHLCV):
- StorageBackend integration for S3/local transparency
- Parquet format for efficient storage
- Date range filtering
- Batch save with existing data merge
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import polars as pl
    from src.config.settings import Settings
    from src.utils.storage_backend import StorageBackend

logger = logging.getLogger(__name__)


class PriceCacheManager:
    """
    Manages price data caching for data preparation.

    Handles:
    - Price data storage in Parquet format
    - StorageBackend integration for S3 support
    - Date range filtering
    - Incremental data merge
    """

    def __init__(
        self,
        settings: "Settings",
        storage_backend: "StorageBackend",
    ) -> None:
        """
        Initialize the price cache manager.

        Args:
            settings: Application settings
            storage_backend: StorageBackend for S3/local operations (required)
        """
        from src.utils.storage_backend import StorageBackend

        if not isinstance(storage_backend, StorageBackend):
            raise TypeError("storage_backend must be a StorageBackend instance")

        self._settings = settings
        self._storage_backend = storage_backend
        self._logger = logger

    def load_from_cache(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[Dict[str, "pl.DataFrame"], List[str]]:
        """
        Load price data from cache via StorageBackend.

        Args:
            universe: List of symbols to load
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Tuple of (loaded prices dict, missing symbols list)
        """
        import polars as pl

        prices: Dict[str, pl.DataFrame] = {}
        missing_symbols: List[str] = []

        # Get list of cached files
        cache_files = self._storage_backend.list_files("prices", "*.parquet")

        # Build symbol -> file path mapping
        cache_symbol_map: Dict[str, str] = {}
        for rel_path in cache_files:
            # prices/AAPL.parquet -> AAPL
            filename = rel_path.split("/")[-1] if "/" in rel_path else rel_path
            cache_name = filename.replace(".parquet", "")
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            cache_symbol_map[original_symbol] = f"prices/{filename}"

        for symbol in universe:
            rel_path = cache_symbol_map.get(symbol)
            if rel_path is None:
                # Build path from symbol name
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
    ) -> Optional["pl.DataFrame"]:
        """
        Filter DataFrame by date range.

        Args:
            df: Input DataFrame
            start_date: Start date
            end_date: End date

        Returns:
            Filtered DataFrame or None if no date column
        """
        import polars as pl

        # Identify date column
        date_col = None
        for col in ["timestamp", "date", "datetime", "time"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return None

        # Filter by date
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

    def save_batch(
        self,
        batch_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Save batch data to cache, merging with existing data.

        Args:
            batch_data: Symbol -> OHLCVData dictionary
            start_date: Data start date
            end_date: Data end date
        """
        import polars as pl

        for symbol, data in batch_data.items():
            if data is None:
                continue

            # Get DataFrame from OHLCVData or use directly
            if hasattr(data, 'df'):
                df = data.df
            elif isinstance(data, (pl.DataFrame,)):
                df = data
            else:
                continue

            if df is None or len(df) == 0:
                continue

            # Convert pandas to polars if needed
            if hasattr(df, 'reset_index'):  # pandas DataFrame
                df = pl.from_pandas(df.reset_index())

            # Safe symbol name for file
            safe_symbol = symbol.replace("=", "_").replace("/", "_")
            rel_path = f"prices/{safe_symbol}.parquet"

            # Merge with existing data
            merged_df = self._merge_with_existing(df, rel_path)

            # Save via StorageBackend
            try:
                self._storage_backend.write_parquet(merged_df, rel_path)
                self._logger.debug(f"Saved {symbol} to cache: {len(merged_df)} rows")
            except Exception as e:
                self._logger.warning(f"Failed to save {symbol} to cache: {e}")

    def _merge_with_existing(
        self,
        new_df: "pl.DataFrame",
        rel_path: str,
    ) -> "pl.DataFrame":
        """
        Merge new data with existing cached data.

        Args:
            new_df: New data to merge
            rel_path: Relative path to cache file

        Returns:
            Merged DataFrame
        """
        import polars as pl

        if not self._storage_backend.exists(rel_path):
            return new_df

        try:
            existing_df = self._storage_backend.read_parquet(rel_path)

            # Find date column
            date_col = None
            for col in ["timestamp", "date", "datetime", "time"]:
                if col in new_df.columns:
                    date_col = col
                    break

            if date_col is None:
                return new_df

            # Concatenate and remove duplicates
            combined = pl.concat([existing_df, new_df], how="diagonal")
            combined = combined.unique(subset=[date_col], keep="last")
            combined = combined.sort(date_col)

            return combined

        except Exception as e:
            self._logger.debug(f"Failed to merge with existing cache: {e}")
            return new_df

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear price cache.

        Args:
            symbol: Specific symbol to clear, or None for all

        Returns:
            Number of files deleted
        """
        count = 0

        if symbol is not None:
            safe_symbol = symbol.replace("=", "_").replace("/", "_")
            rel_path = f"prices/{safe_symbol}.parquet"
            if self._storage_backend.exists(rel_path):
                self._storage_backend.delete(rel_path)
                count = 1
        else:
            cache_files = self._storage_backend.list_files("prices", "*.parquet")
            for rel_path in cache_files:
                try:
                    self._storage_backend.delete(rel_path)
                    count += 1
                except Exception:
                    pass

        self._logger.info(f"Cleared {count} cache files")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the price cache.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = self._storage_backend.list_files("prices", "*.parquet")

        stats = {
            "total_symbols": len(cache_files),
            "symbols": [],
        }

        for rel_path in cache_files:
            filename = rel_path.split("/")[-1] if "/" in rel_path else rel_path
            cache_name = filename.replace(".parquet", "")
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            stats["symbols"].append(original_symbol)

        return stats
