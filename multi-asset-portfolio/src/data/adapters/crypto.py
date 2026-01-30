"""
Crypto Data Adapter.

Fetches OHLCV data from cryptocurrency exchanges using ccxt.
Supports Binance, Coinbase, and other major exchanges.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import polars as pl

try:
    import ccxt
except ImportError:
    ccxt = None  # type: ignore

from . import AssetType, BaseAdapter, DataFrequency, OHLCVData

logger = logging.getLogger(__name__)


class CryptoAdapter(BaseAdapter):
    """
    Cryptocurrency data adapter using ccxt.

    Supports multiple exchanges with unified interface.
    Default exchange is Binance.
    """

    FREQUENCY_MAP = {
        DataFrequency.MINUTE_1: "1m",
        DataFrequency.MINUTE_5: "5m",
        DataFrequency.MINUTE_15: "15m",
        DataFrequency.MINUTE_30: "30m",
        DataFrequency.HOUR_1: "1h",
        DataFrequency.HOUR_4: "4h",
        DataFrequency.DAILY: "1d",
        DataFrequency.WEEKLY: "1w",
        DataFrequency.MONTHLY: "1M",
    }

    SUPPORTED_EXCHANGES = ["binance", "coinbase", "kraken", "bitflyer"]

    def __init__(
        self,
        exchange: str = "binance",
        config: Optional[dict] = None,
    ):
        """
        Initialize crypto adapter.

        Args:
            exchange: Exchange name (default: binance)
            config: Optional configuration with API keys

        Raises:
            ImportError: If ccxt is not installed
            ValueError: If exchange is not supported
        """
        super().__init__(config)

        if ccxt is None:
            raise ImportError(
                "ccxt is required for CryptoAdapter. Install with: pip install ccxt"
            )

        exchange = exchange.lower()
        if exchange not in self.SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Exchange '{exchange}' not supported. "
                f"Available: {self.SUPPORTED_EXCHANGES}"
            )

        self._exchange_name = exchange
        self._exchange = self._create_exchange(exchange)
        self._symbols_cache: Optional[list[str]] = None

    def _create_exchange(self, exchange: str) -> "ccxt.Exchange":
        """Create exchange instance with optional authentication."""
        exchange_class = getattr(ccxt, exchange)
        exchange_config = {
            "enableRateLimit": True,
            "timeout": 30000,
        }

        if self.config.get("api_key"):
            exchange_config["apiKey"] = self.config["api_key"]
        if self.config.get("secret"):
            exchange_config["secret"] = self.config["secret"]

        return exchange_class(exchange_config)

    @property
    def asset_type(self) -> AssetType:
        """Return crypto asset type."""
        return AssetType.CRYPTO

    @property
    def source_name(self) -> str:
        """Return exchange name as source."""
        return f"ccxt:{self._exchange_name}"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> OHLCVData:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH/USD")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            frequency: Data frequency

        Returns:
            OHLCVData with fetched data

        Raises:
            ValueError: If symbol format is invalid
            ConnectionError: If exchange is unavailable
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")

        timeframe = self.FREQUENCY_MAP.get(frequency)
        if timeframe is None:
            raise ValueError(f"Unsupported frequency: {frequency}")

        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        all_ohlcv: list[list] = []
        current_ts = start_ts

        try:
            while current_ts < end_ts:
                ohlcv = self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000,
                )

                if not ohlcv:
                    break

                filtered = [row for row in ohlcv if row[0] <= end_ts]
                all_ohlcv.extend(filtered)

                if len(ohlcv) < 1000:
                    break

                current_ts = ohlcv[-1][0] + 1

            if not all_ohlcv:
                df = self._create_empty_dataframe()
            else:
                df = pl.DataFrame(
                    {
                        "timestamp": [
                            datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
                            for row in all_ohlcv
                        ],
                        "open": [float(row[1]) for row in all_ohlcv],
                        "high": [float(row[2]) for row in all_ohlcv],
                        "low": [float(row[3]) for row in all_ohlcv],
                        "close": [float(row[4]) for row in all_ohlcv],
                        "volume": [float(row[5]) for row in all_ohlcv],
                    }
                ).sort("timestamp")

            logger.info(
                f"Fetched {len(df)} rows for {symbol} from {self._exchange_name}"
            )

            return OHLCVData(
                symbol=symbol,
                asset_type=self.asset_type,
                frequency=frequency,
                data=df,
                source=self.source_name,
                fetched_at=datetime.now(timezone.utc),
                adjusted=False,
            )

        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error fetching {symbol}: {e}") from e
        except ccxt.ExchangeError as e:
            raise ValueError(f"Exchange error for {symbol}: {e}") from e

    def get_available_symbols(self) -> list[str]:
        """
        Get list of available trading pairs.

        Returns:
            List of symbol strings (e.g., ["BTC/USDT", "ETH/USD"])
        """
        if self._symbols_cache is None:
            self._exchange.load_markets()
            self._symbols_cache = list(self._exchange.symbols)
        return self._symbols_cache

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is a valid trading pair.

        Args:
            symbol: Symbol to validate (e.g., "BTC/USDT")

        Returns:
            True if symbol is valid
        """
        if "/" not in symbol:
            return False

        try:
            self._exchange.load_markets()
            return symbol in self._exchange.symbols
        except Exception as e:
            logger.warning(f"Failed to validate symbol {symbol}: {e}")
            return False

    def get_exchange_info(self) -> dict:
        """Get exchange information and rate limits."""
        return {
            "name": self._exchange_name,
            "has_fetch_ohlcv": self._exchange.has.get("fetchOHLCV", False),
            "timeframes": list(self._exchange.timeframes.keys())
            if hasattr(self._exchange, "timeframes")
            else [],
            "rate_limit": self._exchange.rateLimit,
        }
