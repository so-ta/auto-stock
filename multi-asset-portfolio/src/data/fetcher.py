"""
Data Fetcher - Abstract Base Class for Data Sources

This module defines the abstract interface for fetching OHLCV data
from various data sources (exchanges, APIs, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import structlog

if TYPE_CHECKING:
    from src.config.schemas import AssetClass, AssetMetadata, OHLCVData
    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class DataFetcherError(Exception):
    """Base exception for data fetcher errors."""

    pass


class DataNotFoundError(DataFetcherError):
    """Raised when requested data is not available."""

    pass


class RateLimitError(DataFetcherError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(DataFetcherError):
    """Raised when API authentication fails."""

    pass


class DataFetcher(ABC):
    """
    Abstract base class for data fetchers.

    All data source adapters must inherit from this class and implement
    the abstract methods.

    Example:
        >>> class BinanceFetcher(DataFetcher):
        ...     def fetch_ohlcv(self, symbol, start, end, timeframe):
        ...         # Implementation for Binance API
        ...         pass

    Attributes:
        name: Human-readable name of the data source
        asset_class: The asset class this fetcher handles
        settings: Global settings instance
    """

    def __init__(
        self,
        name: str,
        asset_class: "AssetClass",
        settings: "Settings | None" = None,
    ) -> None:
        """
        Initialize the data fetcher.

        Args:
            name: Human-readable name of the data source
            asset_class: The asset class this fetcher handles
            settings: Optional settings instance (uses global if not provided)
        """
        self.name = name
        self.asset_class = asset_class
        self._settings = settings
        self._logger = logger.bind(fetcher=name, asset_class=asset_class.value)

    @property
    def settings(self) -> "Settings":
        """Get settings, loading global if not set."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    # =========================================================================
    # Abstract Methods (Must Implement)
    # =========================================================================
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Asset symbol (e.g., 'BTCUSD', 'AAPL')
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Data timeframe ('1m', '5m', '1h', '1d', etc.)

        Returns:
            Polars DataFrame with columns:
                - timestamp: datetime
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: float

        Raises:
            DataNotFoundError: If symbol or data range is not available
            RateLimitError: If API rate limit is exceeded
            AuthenticationError: If authentication fails
            DataFetcherError: For other fetch errors
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols.

        Returns:
            List of available symbol strings
        """
        pass

    @abstractmethod
    def get_symbol_metadata(self, symbol: str) -> "AssetMetadata":
        """
        Get metadata for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            AssetMetadata instance

        Raises:
            DataNotFoundError: If symbol is not found
        """
        pass

    # =========================================================================
    # Optional Methods (Can Override)
    # =========================================================================
    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid and available.

        Args:
            symbol: Asset symbol to validate

        Returns:
            True if symbol is valid and available
        """
        try:
            available = self.get_available_symbols()
            return symbol in available
        except DataFetcherError:
            return False

    def get_latest_price(self, symbol: str) -> float | None:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Latest price or None if unavailable
        """
        try:
            now = datetime.utcnow()
            df = self.fetch_ohlcv(symbol, now, now, "1d")
            if len(df) > 0:
                return df["close"][-1]
        except DataFetcherError:
            pass
        return None

    def health_check(self) -> bool:
        """
        Check if the data source is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            symbols = self.get_available_symbols()
            return len(symbols) > 0
        except Exception as e:
            self._logger.warning("Health check failed", error=str(e))
            return False

    # =========================================================================
    # Utility Methods
    # =========================================================================
    def fetch_ohlcv_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of asset symbols
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results: dict[str, pl.DataFrame] = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, start, end, timeframe)
                results[symbol] = df
                self._logger.debug(
                    "Fetched data",
                    symbol=symbol,
                    rows=len(df),
                )
            except DataFetcherError as e:
                self._logger.warning(
                    "Failed to fetch data",
                    symbol=symbol,
                    error=str(e),
                )
                continue

        return results

    def to_standard_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure DataFrame has standard column names.

        This method normalizes column names from various sources to the
        standard format: timestamp, open, high, low, close, volume.

        Args:
            df: Input DataFrame with potentially non-standard columns

        Returns:
            DataFrame with standard column names
        """
        column_mapping = {
            # Timestamp variations
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "dt": "timestamp",
            "Date": "timestamp",
            "Datetime": "timestamp",
            # OHLCV variations
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }

        # Rename columns that exist in the mapping
        rename_dict = {
            old: new
            for old, new in column_mapping.items()
            if old in df.columns and old != new
        }

        if rename_dict:
            df = df.rename(rename_dict)

        return df

    def validate_dataframe(self, df: pl.DataFrame) -> bool:
        """
        Validate that a DataFrame has required columns and types.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}

        # Check columns exist
        if not required_columns.issubset(set(df.columns)):
            missing = required_columns - set(df.columns)
            self._logger.warning("Missing required columns", missing=list(missing))
            return False

        # Check for empty DataFrame
        if len(df) == 0:
            self._logger.warning("DataFrame is empty")
            return False

        # Check for null values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            null_count = df[col].null_count()
            if null_count > 0:
                self._logger.warning(
                    "Null values in price column",
                    column=col,
                    null_count=null_count,
                )
                return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', asset_class={self.asset_class.value})"


class CachedDataFetcher(DataFetcher):
    """
    Data fetcher with caching support.

    This class wraps another DataFetcher and adds caching functionality
    to reduce API calls and improve performance.
    """

    def __init__(
        self,
        fetcher: DataFetcher,
        cache_dir: str = "data/cache",
        cache_ttl_hours: int = 24,
    ) -> None:
        """
        Initialize cached fetcher.

        Args:
            fetcher: Underlying data fetcher to wrap
            cache_dir: Directory for cache files
            cache_ttl_hours: Cache time-to-live in hours
        """
        super().__init__(
            name=f"cached_{fetcher.name}",
            asset_class=fetcher.asset_class,
            settings=fetcher._settings,
        )
        self._fetcher = fetcher
        self._cache_dir = cache_dir
        self._cache_ttl_hours = cache_ttl_hours
        self._cache: dict[str, tuple[datetime, pl.DataFrame]] = {}

    def _get_cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> str:
        """Generate cache key."""
        return f"{symbol}_{start.isoformat()}_{end.isoformat()}_{timeframe}"

    def _is_cache_valid(self, cached_time: datetime) -> bool:
        """Check if cache entry is still valid."""
        from datetime import timedelta

        age = datetime.utcnow() - cached_time
        return age < timedelta(hours=self._cache_ttl_hours)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch with caching."""
        cache_key = self._get_cache_key(symbol, start, end, timeframe)

        # Check memory cache
        if cache_key in self._cache:
            cached_time, df = self._cache[cache_key]
            if self._is_cache_valid(cached_time):
                self._logger.debug("Cache hit", symbol=symbol, key=cache_key)
                return df

        # Fetch from underlying fetcher
        df = self._fetcher.fetch_ohlcv(symbol, start, end, timeframe)

        # Update cache
        self._cache[cache_key] = (datetime.utcnow(), df)
        self._logger.debug("Cache miss, fetched from source", symbol=symbol)

        return df

    def get_available_symbols(self) -> list[str]:
        """Delegate to underlying fetcher."""
        return self._fetcher.get_available_symbols()

    def get_symbol_metadata(self, symbol: str) -> "AssetMetadata":
        """Delegate to underlying fetcher."""
        return self._fetcher.get_symbol_metadata(symbol)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._logger.info("Cache cleared")


class MockDataFetcher(DataFetcher):
    """
    Mock data fetcher for testing.

    Generates synthetic OHLCV data for testing purposes.
    """

    def __init__(
        self,
        asset_class: "AssetClass",
        symbols: list[str] | None = None,
    ) -> None:
        """Initialize mock fetcher."""
        from src.config.schemas import AssetClass as AC

        super().__init__(
            name="mock",
            asset_class=asset_class if isinstance(asset_class, AC) else AC(asset_class),
        )
        self._symbols = symbols or ["TEST1", "TEST2", "TEST3"]

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Generate mock OHLCV data."""
        import numpy as np

        if symbol not in self._symbols:
            raise DataNotFoundError(f"Symbol not found: {symbol}")

        # Generate date range
        dates = pl.date_range(start, end, eager=True)

        n = len(dates)
        if n == 0:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        # Generate random walk prices
        np.random.seed(hash(symbol) % (2**32))
        returns = np.random.normal(0.0001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        opens = prices * (1 + np.random.uniform(-0.01, 0.01, n))
        highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, 0.02, n))
        lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, 0.02, n))
        volumes = np.random.uniform(1000, 10000, n)

        return pl.DataFrame(
            {
                "timestamp": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )

    def get_available_symbols(self) -> list[str]:
        """Return mock symbols."""
        return self._symbols.copy()

    def get_symbol_metadata(self, symbol: str) -> "AssetMetadata":
        """Return mock metadata."""
        from src.config.schemas import AssetMetadata

        if symbol not in self._symbols:
            raise DataNotFoundError(f"Symbol not found: {symbol}")

        return AssetMetadata(
            symbol=symbol,
            name=f"Mock Asset {symbol}",
            asset_class=self.asset_class,
            exchange="MOCK",
            currency="USD",
        )
