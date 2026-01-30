"""
Calendar Manager - Market Trading Calendar Management.

Provides trading calendar functionality for multi-market portfolio systems.

Features:
- Trading day detection for multiple markets (NYSE, JPX, LSE, XETR)
- Trading day range queries
- Common trading day calculation across markets
- Data alignment to common calendar

Supports:
- pandas_market_calendars (preferred, if available)
- exchange_calendars (alternative)
- Fallback implementation (weekday-based with basic holidays)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Market Code Definitions
# =============================================================================

MARKET_CODES = {
    "NYSE": "New York Stock Exchange (US)",
    "JPX": "Tokyo Stock Exchange (Japan)",
    "LSE": "London Stock Exchange (UK)",
    "XETR": "Frankfurt Stock Exchange (Germany)",
    "NASDAQ": "NASDAQ (US)",
    "HKEX": "Hong Kong Stock Exchange",
    "SSE": "Shanghai Stock Exchange (China)",
}

# Mapping to pandas_market_calendars names
PMC_CALENDAR_MAP = {
    "NYSE": "NYSE",
    "JPX": "JPX",
    "LSE": "LSE",
    "XETR": "XETR",
    "NASDAQ": "NASDAQ",
    "HKEX": "HKEX",
    "SSE": "SSE",
}

# Mapping to exchange_calendars names
EC_CALENDAR_MAP = {
    "NYSE": "XNYS",
    "JPX": "XTKS",
    "LSE": "XLON",
    "XETR": "XFRA",
    "NASDAQ": "XNAS",
    "HKEX": "XHKG",
    "SSE": "XSHG",
}


# =============================================================================
# Calendar Backend Abstraction
# =============================================================================

class CalendarBackend(ABC):
    """Abstract base class for calendar backends."""

    @abstractmethod
    def is_trading_day(self, market: str, dt: date) -> bool:
        """Check if the given date is a trading day."""
        pass

    @abstractmethod
    def get_trading_days(
        self, market: str, start: date, end: date
    ) -> List[date]:
        """Get list of trading days in the given range."""
        pass


class PandasMarketCalendarsBackend(CalendarBackend):
    """Backend using pandas_market_calendars library."""

    def __init__(self) -> None:
        try:
            import pandas_market_calendars as mcal
            self._mcal = mcal
            self._calendars: Dict[str, Any] = {}
        except ImportError:
            raise ImportError(
                "pandas_market_calendars is not installed. "
                "Install with: pip install pandas-market-calendars"
            )

    def _get_calendar(self, market: str) -> Any:
        """Get or create calendar for market."""
        if market not in self._calendars:
            calendar_name = PMC_CALENDAR_MAP.get(market, market)
            try:
                self._calendars[market] = self._mcal.get_calendar(calendar_name)
            except Exception as e:
                raise ValueError(f"Unknown market calendar: {market}") from e
        return self._calendars[market]

    def is_trading_day(self, market: str, dt: date) -> bool:
        """Check if the given date is a trading day."""
        cal = self._get_calendar(market)
        schedule = cal.schedule(
            start_date=dt,
            end_date=dt,
        )
        return len(schedule) > 0

    def get_trading_days(
        self, market: str, start: date, end: date
    ) -> List[date]:
        """Get list of trading days in the given range."""
        cal = self._get_calendar(market)
        schedule = cal.schedule(start_date=start, end_date=end)
        return [d.date() for d in schedule.index.tolist()]


class ExchangeCalendarsBackend(CalendarBackend):
    """Backend using exchange_calendars library."""

    def __init__(self) -> None:
        try:
            import exchange_calendars as ec
            self._ec = ec
            self._calendars: Dict[str, Any] = {}
        except ImportError:
            raise ImportError(
                "exchange_calendars is not installed. "
                "Install with: pip install exchange-calendars"
            )

    def _get_calendar(self, market: str) -> Any:
        """Get or create calendar for market."""
        if market not in self._calendars:
            calendar_name = EC_CALENDAR_MAP.get(market, market)
            try:
                self._calendars[market] = self._ec.get_calendar(calendar_name)
            except Exception as e:
                raise ValueError(f"Unknown market calendar: {market}") from e
        return self._calendars[market]

    def is_trading_day(self, market: str, dt: date) -> bool:
        """Check if the given date is a trading day."""
        cal = self._get_calendar(market)
        ts = pd.Timestamp(dt)
        return cal.is_session(ts)

    def get_trading_days(
        self, market: str, start: date, end: date
    ) -> List[date]:
        """Get list of trading days in the given range."""
        cal = self._get_calendar(market)
        sessions = cal.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end),
        )
        return [s.date() for s in sessions]


class FallbackCalendarBackend(CalendarBackend):
    """
    Fallback backend using weekday-based logic with basic holidays.

    Used when neither pandas_market_calendars nor exchange_calendars
    are available.
    """

    # Basic US holidays (approximate dates, does not handle observed days)
    US_HOLIDAYS: Set[tuple] = {
        (1, 1),    # New Year's Day
        (7, 4),    # Independence Day
        (12, 25),  # Christmas Day
    }

    # Basic Japan holidays
    JP_HOLIDAYS: Set[tuple] = {
        (1, 1),    # New Year's Day
        (1, 2),    # Bank Holiday
        (1, 3),    # Bank Holiday
        (12, 31),  # Year End
    }

    # Basic UK holidays
    UK_HOLIDAYS: Set[tuple] = {
        (1, 1),    # New Year's Day
        (12, 25),  # Christmas Day
        (12, 26),  # Boxing Day
    }

    # Basic German holidays
    DE_HOLIDAYS: Set[tuple] = {
        (1, 1),    # New Year's Day
        (12, 25),  # Christmas Day
        (12, 26),  # Second Christmas Day
    }

    MARKET_HOLIDAYS: Dict[str, Set[tuple]] = {
        "NYSE": US_HOLIDAYS,
        "NASDAQ": US_HOLIDAYS,
        "JPX": JP_HOLIDAYS,
        "LSE": UK_HOLIDAYS,
        "XETR": DE_HOLIDAYS,
    }

    def __init__(self) -> None:
        logger.warning(
            "Using fallback calendar backend. For accurate trading calendars, "
            "install pandas-market-calendars or exchange-calendars."
        )

    def is_trading_day(self, market: str, dt: date) -> bool:
        """Check if the given date is a trading day (weekday, not holiday)."""
        # Weekends are not trading days
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check basic holidays
        holidays = self.MARKET_HOLIDAYS.get(market, set())
        if (dt.month, dt.day) in holidays:
            return False

        return True

    def get_trading_days(
        self, market: str, start: date, end: date
    ) -> List[date]:
        """Get list of trading days in the given range."""
        trading_days = []
        current = start

        while current <= end:
            if self.is_trading_day(market, current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days


# =============================================================================
# Calendar Manager
# =============================================================================

class CalendarManager:
    """
    Market Trading Calendar Manager.

    Manages trading calendars for multiple markets and provides utilities
    for date alignment and validation.

    Usage:
        manager = CalendarManager()

        # Check if today is a trading day
        is_open = manager.is_trading_day("NYSE", date.today())

        # Get trading days in a range
        days = manager.get_trading_days("NYSE", date(2024, 1, 1), date(2024, 12, 31))

        # Get common trading days across markets
        common = manager.get_common_trading_days(
            ["NYSE", "JPX"],
            date(2024, 1, 1),
            date(2024, 3, 31)
        )

        # Align data to common calendar
        aligned = manager.align_to_common_calendar(data_dict)
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        """
        Initialize CalendarManager.

        Args:
            backend: Calendar backend to use ("pmc", "ec", or "fallback").
                    If None, auto-detects best available backend.
        """
        self._backend = self._create_backend(backend)
        self._trading_days_cache: Dict[str, Dict[tuple, List[date]]] = {}

    def _create_backend(self, backend: Optional[str]) -> CalendarBackend:
        """Create the appropriate calendar backend."""
        if backend == "pmc":
            return PandasMarketCalendarsBackend()
        elif backend == "ec":
            return ExchangeCalendarsBackend()
        elif backend == "fallback":
            return FallbackCalendarBackend()

        # Auto-detect
        try:
            return PandasMarketCalendarsBackend()
        except ImportError:
            pass

        try:
            return ExchangeCalendarsBackend()
        except ImportError:
            pass

        return FallbackCalendarBackend()

    def is_trading_day(
        self,
        market: str,
        dt: Union[date, datetime],
    ) -> bool:
        """
        Check if the given date is a trading day for the market.

        Args:
            market: Market code (NYSE, JPX, LSE, XETR, etc.)
            dt: Date to check (date or datetime object)

        Returns:
            True if the date is a trading day, False otherwise.

        Example:
            >>> manager = CalendarManager()
            >>> manager.is_trading_day("NYSE", date(2024, 7, 4))
            False  # Independence Day
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        return self._backend.is_trading_day(market, dt)

    def get_trading_days(
        self,
        market: str,
        start: Union[date, datetime],
        end: Union[date, datetime],
    ) -> List[date]:
        """
        Get list of trading days in the given date range.

        Args:
            market: Market code (NYSE, JPX, LSE, XETR, etc.)
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of trading days as date objects.

        Example:
            >>> manager = CalendarManager()
            >>> days = manager.get_trading_days("NYSE", date(2024, 1, 1), date(2024, 1, 31))
            >>> len(days)  # ~21 trading days in January
        """
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()

        # Check cache
        cache_key = (start, end)
        if market in self._trading_days_cache:
            if cache_key in self._trading_days_cache[market]:
                return self._trading_days_cache[market][cache_key]

        # Compute trading days
        trading_days = self._backend.get_trading_days(market, start, end)

        # Cache result
        if market not in self._trading_days_cache:
            self._trading_days_cache[market] = {}
        self._trading_days_cache[market][cache_key] = trading_days

        return trading_days

    def get_common_trading_days(
        self,
        markets: List[str],
        start: Union[date, datetime],
        end: Union[date, datetime],
    ) -> List[date]:
        """
        Get trading days common to all specified markets.

        Args:
            markets: List of market codes
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of dates that are trading days in ALL specified markets.

        Example:
            >>> manager = CalendarManager()
            >>> common = manager.get_common_trading_days(
            ...     ["NYSE", "JPX"],
            ...     date(2024, 1, 1),
            ...     date(2024, 1, 31)
            ... )
        """
        if not markets:
            return []

        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()

        # Get trading days for first market
        common_days = set(self.get_trading_days(markets[0], start, end))

        # Intersect with trading days of other markets
        for market in markets[1:]:
            market_days = set(self.get_trading_days(market, start, end))
            common_days &= market_days

        return sorted(common_days)

    def align_to_common_calendar(
        self,
        data: Dict[str, pd.DataFrame],
        markets: Optional[Dict[str, str]] = None,
        fill_method: str = "ffill",
        drop_all_nan: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple assets' data to a common trading calendar.

        This is essential for multi-asset portfolio analysis where different
        assets may trade on different calendars (e.g., US stocks vs Japan stocks).

        Args:
            data: Dictionary mapping symbol to DataFrame with DatetimeIndex.
            markets: Dictionary mapping symbol to market code (e.g., {"AAPL": "NYSE"}).
                    If None, assumes all assets are NYSE.
            fill_method: Method for filling missing values:
                        - "ffill": Forward fill (use previous day's value)
                        - "bfill": Backward fill
                        - None: No filling (keep NaN)
            drop_all_nan: If True, drop rows where all values are NaN.

        Returns:
            Dictionary of aligned DataFrames with common DatetimeIndex.

        Example:
            >>> data = {
            ...     "AAPL": us_data,
            ...     "7203.T": japan_data,
            ... }
            >>> markets = {"AAPL": "NYSE", "7203.T": "JPX"}
            >>> aligned = manager.align_to_common_calendar(data, markets)
        """
        if not data:
            return {}

        # Determine markets for each symbol
        if markets is None:
            markets = {symbol: "NYSE" for symbol in data.keys()}

        # Get date range from data
        all_dates: List[date] = []
        for symbol, df in data.items():
            if df.index.empty:
                continue
            all_dates.extend([d.date() for d in df.index])

        if not all_dates:
            return data

        start_date = min(all_dates)
        end_date = max(all_dates)

        # Get unique markets
        unique_markets = list(set(markets.values()))

        # Get common trading days
        common_days = self.get_common_trading_days(
            unique_markets, start_date, end_date
        )

        if not common_days:
            logger.warning("No common trading days found")
            return data

        # Create common index
        common_index = pd.DatetimeIndex([pd.Timestamp(d) for d in common_days])

        # Align each DataFrame
        aligned: Dict[str, pd.DataFrame] = {}

        for symbol, df in data.items():
            if df.empty:
                aligned[symbol] = df
                continue

            # Reindex to common calendar
            aligned_df = df.reindex(common_index)

            # Apply fill method
            if fill_method == "ffill":
                aligned_df = aligned_df.ffill()
            elif fill_method == "bfill":
                aligned_df = aligned_df.bfill()

            # Drop rows where all values are NaN (if requested)
            if drop_all_nan:
                aligned_df = aligned_df.dropna(how="all")

            aligned[symbol] = aligned_df

        return aligned

    def get_next_trading_day(
        self,
        market: str,
        dt: Union[date, datetime],
    ) -> date:
        """
        Get the next trading day after the given date.

        Args:
            market: Market code
            dt: Reference date

        Returns:
            Next trading day (may be the same day if it's a trading day).
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        current = dt
        max_iterations = 10  # Prevent infinite loop

        for _ in range(max_iterations):
            if self.is_trading_day(market, current):
                return current
            current += timedelta(days=1)

        # Fallback
        return current

    def get_previous_trading_day(
        self,
        market: str,
        dt: Union[date, datetime],
    ) -> date:
        """
        Get the previous trading day before the given date.

        Args:
            market: Market code
            dt: Reference date

        Returns:
            Previous trading day (may be the same day if it's a trading day).
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        current = dt
        max_iterations = 10  # Prevent infinite loop

        for _ in range(max_iterations):
            if self.is_trading_day(market, current):
                return current
            current -= timedelta(days=1)

        # Fallback
        return current

    def count_trading_days(
        self,
        market: str,
        start: Union[date, datetime],
        end: Union[date, datetime],
    ) -> int:
        """
        Count the number of trading days in the given range.

        Args:
            market: Market code
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Number of trading days.
        """
        return len(self.get_trading_days(market, start, end))

    def clear_cache(self) -> None:
        """Clear the trading days cache."""
        self._trading_days_cache.clear()

    @property
    def supported_markets(self) -> List[str]:
        """Get list of supported market codes."""
        return list(MARKET_CODES.keys())

    def __repr__(self) -> str:
        backend_name = type(self._backend).__name__
        return f"CalendarManager(backend={backend_name})"


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_manager: Optional[CalendarManager] = None


def get_calendar_manager() -> CalendarManager:
    """Get or create the default CalendarManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CalendarManager()
    return _default_manager


def is_trading_day(market: str, dt: Union[date, datetime]) -> bool:
    """Check if the given date is a trading day (uses default manager)."""
    return get_calendar_manager().is_trading_day(market, dt)


def get_trading_days(
    market: str,
    start: Union[date, datetime],
    end: Union[date, datetime],
) -> List[date]:
    """Get trading days in range (uses default manager)."""
    return get_calendar_manager().get_trading_days(market, start, end)


def align_to_common_calendar(
    data: Dict[str, pd.DataFrame],
    markets: Optional[Dict[str, str]] = None,
    fill_method: str = "ffill",
) -> Dict[str, pd.DataFrame]:
    """Align data to common calendar (uses default manager)."""
    return get_calendar_manager().align_to_common_calendar(
        data, markets, fill_method
    )
