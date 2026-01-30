"""Data module for multi-asset portfolio system."""

from .batch_fetcher import (
    BatchDataFetcher,
    BatchFetcherConfig,
    BatchFetchResult,
    FetchResult,
    create_batch_fetcher,
    quick_fetch,
)
from .cache import DataCache
from .calendar_manager import (
    CalendarManager,
    align_to_common_calendar,
    get_calendar_manager,
    get_trading_days,
    is_trading_day,
)
from .universe_loader import UniverseConfig, UniverseLoader, UniverseLoaderError
from .duckdb_layer import (
    DuckDBDataLayer,
    DuckDBLayerConfig,
    TableInfo,
    QueryResult,
    DUCKDB_AVAILABLE,
    create_duckdb_layer,
    quick_price_matrix,
    quick_query,
)
from .dividend_handler import (
    DividendHandler,
    DividendConfig,
    DividendData,
    TotalReturnResult,
    calculate_total_return,
    adjust_prices_for_splits,
)
from .finra import (
    FINRAClient,
    FINRAClientError,
    ShortInterestRecord,
    get_short_interest,
)
from .sec_edgar import (
    SECEdgarClient,
    SECEdgarClientConfig,
    SECEdgarError,
    InsiderTransaction,
    create_sec_client,
)

__all__ = [
    # Batch Fetcher
    "BatchDataFetcher",
    "BatchFetcherConfig",
    "BatchFetchResult",
    "FetchResult",
    "create_batch_fetcher",
    "quick_fetch",
    # Cache
    "DataCache",
    # Calendar
    "CalendarManager",
    "get_calendar_manager",
    "is_trading_day",
    "get_trading_days",
    "align_to_common_calendar",
    # Universe
    "UniverseConfig",
    "UniverseLoader",
    "UniverseLoaderError",
    # DuckDB Layer (HI-003)
    "DuckDBDataLayer",
    "DuckDBLayerConfig",
    "TableInfo",
    "QueryResult",
    "DUCKDB_AVAILABLE",
    "create_duckdb_layer",
    "quick_price_matrix",
    "quick_query",
    # Dividend Handler (SYNC-006)
    "DividendHandler",
    "DividendConfig",
    "DividendData",
    "TotalReturnResult",
    "calculate_total_return",
    "adjust_prices_for_splits",
    # FINRA Short Interest (task_042_5)
    "FINRAClient",
    "FINRAClientError",
    "ShortInterestRecord",
    "get_short_interest",
    # SEC EDGAR (task_042_4)
    "SECEdgarClient",
    "SECEdgarClientConfig",
    "SECEdgarError",
    "InsiderTransaction",
    "create_sec_client",
]
