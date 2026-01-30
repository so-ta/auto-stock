"""
FX Data Adapter.

Fetches OHLCV data for forex pairs.
Uses Yahoo Finance as default source (free), with optional support for
premium providers.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import polars as pl

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

from . import AssetType, BaseAdapter, DataFrequency, OHLCVData

logger = logging.getLogger(__name__)


class FXAdapter(BaseAdapter):
    """
    Forex data adapter.

    Default implementation uses Yahoo Finance.
    Supports major and minor currency pairs.
    """

    FREQUENCY_MAP = {
        DataFrequency.MINUTE_1: "1m",
        DataFrequency.MINUTE_5: "5m",
        DataFrequency.MINUTE_15: "15m",
        DataFrequency.MINUTE_30: "30m",
        DataFrequency.HOUR_1: "1h",
        DataFrequency.DAILY: "1d",
        DataFrequency.WEEKLY: "1wk",
        DataFrequency.MONTHLY: "1mo",
    }

    MAJOR_PAIRS = [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
    ]

    MINOR_PAIRS = [
        "EURGBP",
        "EURJPY",
        "GBPJPY",
        "EURAUD",
        "EURCAD",
        "EURCHF",
        "AUDNZD",
        "AUDJPY",
        "CADJPY",
        "CHFJPY",
        "GBPAUD",
        "GBPCAD",
        "GBPCHF",
        "NZDJPY",
    ]

    def __init__(
        self,
        source: str = "yahoo",
        config: Optional[dict] = None,
    ):
        """
        Initialize FX adapter.

        Args:
            source: Data source (currently only "yahoo" supported)
            config: Optional configuration

        Raises:
            ImportError: If yfinance is not installed
            ValueError: If source is not supported
        """
        super().__init__(config)

        if source != "yahoo":
            raise ValueError(f"Source '{source}' not supported. Use 'yahoo'.")

        if yf is None:
            raise ImportError(
                "yfinance is required for FXAdapter. "
                "Install with: pip install yfinance"
            )

        self._source = source
        self._symbols_cache: Optional[list[str]] = None

    @property
    def asset_type(self) -> AssetType:
        """Return FX asset type."""
        return AssetType.FX

    @property
    def source_name(self) -> str:
        """Return source name."""
        return f"fx:{self._source}"

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize FX pair symbol for Yahoo Finance.

        Args:
            symbol: Raw symbol (e.g., "USDJPY", "USD/JPY")

        Returns:
            Yahoo Finance format (e.g., "USDJPY=X")
        """
        clean = symbol.upper().replace("/", "").replace("-", "")

        if len(clean) != 6:
            raise ValueError(
                f"Invalid FX pair format: {symbol}. "
                f"Expected 6 characters (e.g., 'USDJPY')"
            )

        if not clean.endswith("=X"):
            return f"{clean}=X"
        return clean

    def _denormalize_symbol(self, yahoo_symbol: str) -> str:
        """
        Convert Yahoo Finance symbol back to standard format.

        Args:
            yahoo_symbol: Yahoo format (e.g., "USDJPY=X")

        Returns:
            Standard format (e.g., "USDJPY")
        """
        return yahoo_symbol.replace("=X", "")

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> OHLCVData:
        """
        Fetch OHLCV data for FX pair.

        Args:
            symbol: Currency pair (e.g., "USDJPY", "EUR/USD")
            start: Start datetime
            end: End datetime
            frequency: Data frequency

        Returns:
            OHLCVData with fetched data

        Note:
            Volume data for FX is typically not meaningful from free sources.
            The volume field contains tick volume or estimated values.
        """
        normalized = self._normalize_symbol(symbol)
        interval = self.FREQUENCY_MAP.get(frequency)

        if interval is None:
            raise ValueError(f"Unsupported frequency: {frequency}")

        try:
            ticker = yf.Ticker(normalized)

            df_pandas = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                actions=False,
            )

            if df_pandas.empty:
                logger.warning(f"No data returned for {normalized}")
                df = self._create_empty_dataframe()
            else:
                df_pandas = df_pandas.reset_index()

                date_col = "Date" if "Date" in df_pandas.columns else "Datetime"

                df = pl.DataFrame(
                    {
                        "timestamp": pl.Series(df_pandas[date_col]).cast(pl.Datetime),
                        "open": pl.Series(df_pandas["Open"].values).cast(pl.Float64),
                        "high": pl.Series(df_pandas["High"].values).cast(pl.Float64),
                        "low": pl.Series(df_pandas["Low"].values).cast(pl.Float64),
                        "close": pl.Series(df_pandas["Close"].values).cast(pl.Float64),
                        "volume": pl.Series(df_pandas["Volume"].values).cast(
                            pl.Float64
                        ),
                    }
                ).sort("timestamp")

            standard_symbol = self._denormalize_symbol(normalized)
            logger.info(f"Fetched {len(df)} rows for {standard_symbol}")

            return OHLCVData(
                symbol=standard_symbol,
                asset_type=self.asset_type,
                frequency=frequency,
                data=df,
                source=self.source_name,
                fetched_at=datetime.now(timezone.utc),
                adjusted=False,
            )

        except Exception as e:
            logger.error(f"Failed to fetch {normalized}: {e}")
            raise ConnectionError(f"Failed to fetch FX data for {symbol}: {e}") from e

    def get_available_symbols(self) -> list[str]:
        """
        Get list of available FX pairs.

        Returns:
            List of major and minor currency pairs
        """
        if self._symbols_cache is None:
            self._symbols_cache = self.MAJOR_PAIRS + self.MINOR_PAIRS
        return self._symbols_cache

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if FX pair is available.

        Args:
            symbol: Currency pair to validate

        Returns:
            True if pair is valid and has data
        """
        try:
            normalized = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized)
            info = ticker.info

            return info is not None and "symbol" in info

        except Exception as e:
            logger.warning(f"Failed to validate {symbol}: {e}")
            return False

    def get_pair_info(self, symbol: str) -> dict:
        """
        Get information about a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Dictionary with pair info
        """
        clean = symbol.upper().replace("/", "").replace("-", "").replace("=X", "")

        if len(clean) != 6:
            return {"error": "Invalid pair format"}

        base = clean[:3]
        quote = clean[3:]

        return {
            "symbol": clean,
            "base_currency": base,
            "quote_currency": quote,
            "is_major": clean in self.MAJOR_PAIRS,
            "description": f"{base}/{quote} exchange rate",
        }

    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None,
    ) -> float:
        """
        Convert amount between currencies using latest or historical rate.

        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., "USD")
            to_currency: Target currency code (e.g., "JPY")
            date: Optional historical date (default: latest)

        Returns:
            Converted amount

        Raises:
            ValueError: If conversion pair is not available
        """
        if from_currency == to_currency:
            return amount

        pair = f"{from_currency}{to_currency}"
        inverse_pair = f"{to_currency}{from_currency}"

        use_inverse = False
        try:
            normalized = self._normalize_symbol(pair)
            yf.Ticker(normalized).info
        except Exception:
            pair = inverse_pair
            use_inverse = True

        if date is None:
            from datetime import timedelta

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=7)
        else:
            from datetime import timedelta

            start = date - timedelta(days=7)
            end = date + timedelta(days=1)

        ohlcv = self.fetch_ohlcv(pair, start, end, DataFrequency.DAILY)

        if ohlcv.data.is_empty():
            raise ValueError(f"No exchange rate data for {pair}")

        rate = ohlcv.data["close"][-1]

        if use_inverse:
            rate = 1.0 / rate

        return amount * rate
