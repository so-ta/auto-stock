"""
FINRA API Client - Short Interest Data Retrieval.

Provides access to FINRA's Equity Short Interest data.

Academic Basis:
- Rapach et al. (2016): Aggregate short interest predicts market returns
- Boehmer et al. (2008): Stock-level short interest predicts returns

Data Source:
- FINRA API (free, biweekly updates)
- Archive available from 2014

Usage:
    client = FINRAClient()
    data = client.get_short_interest(symbol="AAPL", start_date=datetime(2024, 1, 1))
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ShortInterestRecord:
    """Single short interest data record."""

    symbol: str
    settlement_date: datetime
    short_interest: int  # Number of shares short
    avg_daily_volume: float  # Average daily share volume
    days_to_cover: float  # Short interest / avg daily volume
    change_from_previous: Optional[float] = None  # Change in SI from previous report


class FINRAClientError(Exception):
    """Exception raised for FINRA API errors."""

    pass


class FINRAClient:
    """
    FINRA API Client for Short Interest Data.

    Retrieves equity short interest data from FINRA's public API.
    Data is updated bi-weekly (mid-month and end-of-month settlement dates).

    Note: The FINRA API may require registration for production use.
    For development/testing, cached data or mock responses can be used.

    Example:
        client = FINRAClient()

        # Get short interest for a single symbol
        df = client.get_short_interest(symbol="AAPL")

        # Get short interest with date range
        df = client.get_short_interest(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1)
        )

        # Get days to cover ratio
        dtc = client.get_days_to_cover(symbol="AAPL")
    """

    BASE_URL = "https://api.finra.org/data/group/otcMarket/name/EquityShortInterest"
    CACHE_DIR = Path("data/cache/finra")

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        timeout: int = 30,
    ):
        """
        Initialize FINRA client.

        Args:
            api_key: Optional API key for authenticated access
            cache_enabled: Whether to cache API responses
            cache_dir: Directory for caching data
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.timeout = timeout

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.cache_dir / f"short_interest_{symbol.upper()}.parquet"

    def _load_from_cache(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            if start_date:
                df = df[df["settlement_date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["settlement_date"] <= pd.Timestamp(end_date)]
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        if not self.cache_enabled or df.empty:
            return

        try:
            cache_path = self._get_cache_path(symbol)

            # Merge with existing cache
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["symbol", "settlement_date"], keep="last"
                )
                df = df.sort_values("settlement_date")

            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    def _fetch_from_api(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from FINRA API.

        Note: This is a simplified implementation. The actual FINRA API
        may require different query parameters and authentication.
        """
        if not REQUESTS_AVAILABLE:
            raise FINRAClientError(
                "requests library not available. Install with: pip install requests"
            )

        # Build query filters
        filters = []
        if symbol:
            filters.append(
                {"fieldName": "symbolCode", "fieldValue": symbol.upper(), "compareType": "EQUAL"}
            )
        if start_date:
            filters.append(
                {
                    "fieldName": "settlementDate",
                    "fieldValue": start_date.strftime("%Y-%m-%d"),
                    "compareType": "GREATER_THAN_OR_EQUAL",
                }
            )
        if end_date:
            filters.append(
                {
                    "fieldName": "settlementDate",
                    "fieldValue": end_date.strftime("%Y-%m-%d"),
                    "compareType": "LESS_THAN_OR_EQUAL",
                }
            )

        # Build request payload
        payload = {
            "fields": [
                "symbolCode",
                "settlementDate",
                "shortInterest",
                "averageDailyVolume",
                "daysToCover",
            ],
            "limit": 10000,
            "offset": 0,
            "sortFields": [{"fieldName": "settlementDate", "order": "DESC"}],
        }
        if filters:
            payload["compareFilters"] = filters

        try:
            response = requests.post(
                self.BASE_URL,
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FINRA API request failed: {e}")
            raise FINRAClientError(f"API request failed: {e}")

    def get_short_interest(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get short interest data.

        Args:
            symbol: Stock symbol (e.g., "AAPL"). If None, returns all symbols.
            start_date: Start date for data range
            end_date: End date for data range
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - settlement_date: Date of short interest settlement
            - short_interest: Number of shares short
            - avg_daily_volume: Average daily trading volume
            - days_to_cover: Short interest / avg daily volume
            - short_interest_ratio: Short interest / outstanding shares (if available)

        Raises:
            FINRAClientError: If API request fails
        """
        # Try cache first
        if use_cache and symbol:
            cached = self._load_from_cache(symbol, start_date, end_date)
            if cached is not None and not cached.empty:
                logger.info(f"Loaded {len(cached)} records from cache for {symbol}")
                return cached

        # Fetch from API
        try:
            data = self._fetch_from_api(symbol, start_date, end_date)
        except FINRAClientError:
            # Fall back to cache if available
            if symbol:
                cached = self._load_from_cache(symbol, start_date, end_date)
                if cached is not None:
                    logger.warning(f"Using cached data for {symbol} due to API failure")
                    return cached
            raise

        if not data:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "settlement_date",
                    "short_interest",
                    "avg_daily_volume",
                    "days_to_cover",
                ]
            )

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Rename columns to standard names
        column_mapping = {
            "symbolCode": "symbol",
            "settlementDate": "settlement_date",
            "shortInterest": "short_interest",
            "averageDailyVolume": "avg_daily_volume",
            "daysToCover": "days_to_cover",
        }
        df = df.rename(columns=column_mapping)

        # Parse dates
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])

        # Sort by date
        df = df.sort_values("settlement_date")

        # Save to cache
        if symbol:
            self._save_to_cache(symbol, df)

        return df

    def get_days_to_cover(
        self,
        symbol: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Get the most recent days to cover ratio.

        Args:
            symbol: Stock symbol
            as_of_date: Get value as of this date (latest if None)

        Returns:
            Days to cover ratio, or None if not available
        """
        df = self.get_short_interest(symbol=symbol)

        if df.empty:
            return None

        if as_of_date:
            df = df[df["settlement_date"] <= pd.Timestamp(as_of_date)]

        if df.empty:
            return None

        return float(df.iloc[-1]["days_to_cover"])

    def get_short_interest_ratio(
        self,
        symbol: str,
        shares_outstanding: float,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Calculate short interest ratio (SI / shares outstanding).

        Args:
            symbol: Stock symbol
            shares_outstanding: Total shares outstanding
            as_of_date: Get value as of this date (latest if None)

        Returns:
            Short interest ratio (0-1), or None if not available
        """
        df = self.get_short_interest(symbol=symbol)

        if df.empty or shares_outstanding <= 0:
            return None

        if as_of_date:
            df = df[df["settlement_date"] <= pd.Timestamp(as_of_date)]

        if df.empty:
            return None

        short_interest = float(df.iloc[-1]["short_interest"])
        return short_interest / shares_outstanding

    def get_short_interest_change(
        self,
        symbol: str,
        periods: int = 1,
    ) -> Optional[float]:
        """
        Get short interest change from previous report(s).

        Args:
            symbol: Stock symbol
            periods: Number of periods to look back

        Returns:
            Percentage change in short interest, or None if not available
        """
        df = self.get_short_interest(symbol=symbol)

        if len(df) < periods + 1:
            return None

        current = df.iloc[-1]["short_interest"]
        previous = df.iloc[-1 - periods]["short_interest"]

        if previous == 0:
            return None

        return (current - previous) / previous

    def get_bulk_short_interest(
        self,
        symbols: List[str],
        as_of_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get short interest data for multiple symbols.

        Args:
            symbols: List of stock symbols
            as_of_date: Get values as of this date (latest if None)

        Returns:
            DataFrame with one row per symbol
        """
        results = []

        for symbol in symbols:
            try:
                df = self.get_short_interest(symbol=symbol)
                if df.empty:
                    continue

                if as_of_date:
                    df = df[df["settlement_date"] <= pd.Timestamp(as_of_date)]

                if df.empty:
                    continue

                latest = df.iloc[-1].to_dict()
                results.append(latest)
            except FINRAClientError as e:
                logger.warning(f"Failed to get short interest for {symbol}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def to_polars(self, df: pd.DataFrame) -> "pl.DataFrame":
        """Convert pandas DataFrame to polars DataFrame."""
        if not POLARS_AVAILABLE:
            raise ImportError("polars not available")
        return pl.from_pandas(df)


# Convenience function for quick access
def get_short_interest(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Quick function to get short interest data.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date

    Returns:
        Short interest DataFrame
    """
    client = FINRAClient()
    return client.get_short_interest(
        symbol=symbol, start_date=start_date, end_date=end_date
    )
