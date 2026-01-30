"""
Data Adapters Package.

Provides unified interfaces for fetching OHLCV data from various sources:
- Crypto: Binance, Coinbase via ccxt
- Stock: Yahoo Finance via yfinance
- FX: Forex data sources
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import polars as pl


class AssetType(Enum):
    """Asset type enumeration."""

    CRYPTO = "crypto"
    STOCK = "stock"
    FX = "fx"


class DataFrequency(Enum):
    """Data frequency enumeration."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class OHLCVData:
    """OHLCV data container with metadata."""

    symbol: str
    asset_type: AssetType
    frequency: DataFrequency
    data: pl.DataFrame
    source: str
    fetched_at: datetime
    adjusted: bool = False

    def validate(self) -> bool:
        """Validate OHLCV data structure."""
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        return required_cols.issubset(set(self.data.columns))


class BaseAdapter(ABC):
    """
    Abstract base class for all data adapters.

    All adapters must implement:
    - fetch_ohlcv: Fetch OHLCV data for a symbol
    - get_available_symbols: List available symbols
    - validate_symbol: Check if a symbol is valid
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize adapter with optional configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._rate_limit_remaining: Optional[int] = None

    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """Return the asset type this adapter handles."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the data source name."""
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> OHLCVData:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Asset symbol (e.g., "BTCUSD", "AAPL", "USDJPY")
            start: Start datetime
            end: End datetime
            frequency: Data frequency

        Returns:
            OHLCVData container with fetched data

        Raises:
            ValueError: If symbol is invalid
            ConnectionError: If data source is unavailable
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
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid and available
        """
        pass

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """Create an empty OHLCV DataFrame with correct schema."""
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


# Re-export adapters
from .crypto import CryptoAdapter
from .fx import FXAdapter
from .stock import StockAdapter
from .multi_source_adapter import MultiSourceAdapter, MultiSourceConfig

__all__ = [
    "AssetType",
    "DataFrequency",
    "OHLCVData",
    "BaseAdapter",
    "CryptoAdapter",
    "StockAdapter",
    "FXAdapter",
    "MultiSourceAdapter",
    "MultiSourceConfig",
]
