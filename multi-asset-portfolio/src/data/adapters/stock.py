"""
Stock Data Adapter.

Fetches OHLCV data from stock markets using yfinance.
Supports automatic adjustment for splits and dividends.
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


class StockAdapter(BaseAdapter):
    """
    Stock market data adapter using yfinance.

    Supports US, Japanese, and international stocks.
    Provides adjusted and unadjusted price data.
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

    MARKET_SUFFIXES = {
        "JP": ".T",  # Tokyo Stock Exchange
        "US": "",  # US stocks (no suffix)
        "HK": ".HK",  # Hong Kong
        "UK": ".L",  # London
        "DE": ".DE",  # Germany
    }

    def __init__(
        self,
        market: str = "US",
        adjusted: bool = True,
        config: Optional[dict] = None,
    ):
        """
        Initialize stock adapter.

        Args:
            market: Market code (US, JP, HK, UK, DE)
            adjusted: Whether to use adjusted prices (default: True)
            config: Optional configuration

        Raises:
            ImportError: If yfinance is not installed
        """
        super().__init__(config)

        if yf is None:
            raise ImportError(
                "yfinance is required for StockAdapter. "
                "Install with: pip install yfinance"
            )

        self._market = market.upper()
        self._adjusted = adjusted
        self._suffix = self.MARKET_SUFFIXES.get(self._market, "")
        self._symbols_cache: Optional[list[str]] = None

    @property
    def asset_type(self) -> AssetType:
        """Return stock asset type."""
        return AssetType.STOCK

    @property
    def source_name(self) -> str:
        """Return source name."""
        return f"yfinance:{self._market}"

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for the market.

        Args:
            symbol: Raw symbol (e.g., "7203" for Toyota in JP)

        Returns:
            Normalized symbol with market suffix if needed
        """
        if self._suffix and not symbol.endswith(self._suffix):
            return f"{symbol}{self._suffix}"
        return symbol

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> OHLCVData:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "7203.T")
            start: Start datetime
            end: End datetime
            frequency: Data frequency

        Returns:
            OHLCVData with fetched data

        Raises:
            ValueError: If symbol is invalid or no data available
            ConnectionError: If Yahoo Finance is unavailable
        """
        normalized_symbol = self._normalize_symbol(symbol)
        interval = self.FREQUENCY_MAP.get(frequency)

        if interval is None:
            raise ValueError(f"Unsupported frequency: {frequency}")

        try:
            ticker = yf.Ticker(normalized_symbol)

            df_pandas = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=self._adjusted,
                actions=False,
            )

            if df_pandas.empty:
                logger.warning(f"No data returned for {normalized_symbol}")
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

            logger.info(
                f"Fetched {len(df)} rows for {normalized_symbol} "
                f"({'adjusted' if self._adjusted else 'unadjusted'})"
            )

            return OHLCVData(
                symbol=normalized_symbol,
                asset_type=self.asset_type,
                frequency=frequency,
                data=df,
                source=self.source_name,
                fetched_at=datetime.now(timezone.utc),
                adjusted=self._adjusted,
            )

        except Exception as e:
            logger.error(f"Failed to fetch {normalized_symbol}: {e}")
            raise ConnectionError(f"Failed to fetch {normalized_symbol}: {e}") from e

    def get_available_symbols(self) -> list[str]:
        """
        Get list of commonly traded symbols for the market.

        Note: Yahoo Finance doesn't provide a complete symbol list.
        This returns a curated list for the configured market.

        Returns:
            List of common symbols
        """
        if self._symbols_cache is not None:
            return self._symbols_cache

        common_symbols = {
            "US": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "TSLA",
                "JPM",
                "V",
                "JNJ",
                "WMT",
                "PG",
                "UNH",
                "HD",
                "MA",
                "DIS",
                "PYPL",
                "NFLX",
                "ADBE",
                "CRM",
            ],
            "JP": [
                "7203",
                "6758",
                "9984",
                "6861",
                "7974",
                "8306",
                "9432",
                "6501",
                "4502",
                "6902",
                "7267",
                "8035",
                "6367",
                "4063",
                "6594",
                "8316",
                "9433",
                "2914",
                "4503",
                "6752",
            ],
        }

        symbols = common_symbols.get(self._market, [])
        self._symbols_cache = [self._normalize_symbol(s) for s in symbols]
        return self._symbols_cache

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol exists on Yahoo Finance.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid and has data
        """
        normalized = self._normalize_symbol(symbol)

        try:
            ticker = yf.Ticker(normalized)
            info = ticker.info

            return info is not None and "symbol" in info

        except Exception as e:
            logger.warning(f"Failed to validate {normalized}: {e}")
            return False

    def get_ticker_info(self, symbol: str) -> dict:
        """
        Get detailed ticker information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with ticker info (name, sector, market cap, etc.)
        """
        normalized = self._normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        info = ticker.info or {}

        return {
            "symbol": normalized,
            "name": info.get("longName", info.get("shortName", "")),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", ""),
            "exchange": info.get("exchange", ""),
        }

    def fetch_with_corporate_actions(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> tuple[OHLCVData, pl.DataFrame]:
        """
        Fetch OHLCV data along with corporate actions.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime

        Returns:
            Tuple of (OHLCVData, corporate_actions DataFrame)
        """
        normalized = self._normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        dividends = ticker.dividends
        splits = ticker.splits

        actions_data = []

        if not dividends.empty:
            for date, value in dividends.items():
                if start <= date.to_pydatetime() <= end:
                    actions_data.append(
                        {
                            "date": date.to_pydatetime(),
                            "action_type": "dividend",
                            "value": float(value),
                        }
                    )

        if not splits.empty:
            for date, value in splits.items():
                if start <= date.to_pydatetime() <= end:
                    actions_data.append(
                        {
                            "date": date.to_pydatetime(),
                            "action_type": "split",
                            "value": float(value),
                        }
                    )

        if actions_data:
            actions_df = pl.DataFrame(actions_data).sort("date")
        else:
            actions_df = pl.DataFrame(
                schema={
                    "date": pl.Datetime,
                    "action_type": pl.Utf8,
                    "value": pl.Float64,
                }
            )

        ohlcv = self.fetch_ohlcv(symbol, start, end)

        return ohlcv, actions_df

    def fetch_dividends(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch dividend history for a symbol.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: date, dividend_amount
        """
        normalized = self._normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        dividends = ticker.dividends

        if dividends.empty:
            logger.info(f"No dividends found for {normalized}")
            return pl.DataFrame(
                schema={
                    "date": pl.Datetime,
                    "dividend_amount": pl.Float64,
                }
            )

        # Filter by date range
        dividend_data = []
        for date, value in dividends.items():
            div_date = date.to_pydatetime()
            # Make timezone-naive if needed
            if div_date.tzinfo is not None:
                div_date = div_date.replace(tzinfo=None)

            start_naive = start.replace(tzinfo=None) if start.tzinfo else start
            end_naive = end.replace(tzinfo=None) if end.tzinfo else end

            if start_naive <= div_date <= end_naive:
                dividend_data.append({
                    "date": div_date,
                    "dividend_amount": float(value),
                })

        if dividend_data:
            df = pl.DataFrame(dividend_data).sort("date")
            logger.info(f"Found {len(df)} dividends for {normalized}")
            return df
        else:
            return pl.DataFrame(
                schema={
                    "date": pl.Datetime,
                    "dividend_amount": pl.Float64,
                }
            )

    def fetch_ohlcv_with_dividends(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> tuple[OHLCVData, pl.DataFrame]:
        """
        Fetch OHLCV data along with dividend history.

        Convenience method that combines fetch_ohlcv and fetch_dividends.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            frequency: Data frequency

        Returns:
            Tuple of (OHLCVData, dividends DataFrame)
        """
        ohlcv = self.fetch_ohlcv(symbol, start, end, frequency)
        dividends = self.fetch_dividends(symbol, start, end)

        return ohlcv, dividends
