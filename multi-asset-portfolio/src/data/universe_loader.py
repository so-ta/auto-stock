"""
Universe Loader - Asset Universe Management

This module provides functionality to load and manage the investment universe,
including stock indices (S&P 500, Nikkei 225), ETFs, commodities, and forex pairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)


class UniverseLoaderError(Exception):
    """Base exception for universe loader errors."""

    pass


class ConfigNotFoundError(UniverseLoaderError):
    """Raised when configuration file is not found."""

    pass


class FetchError(UniverseLoaderError):
    """Raised when fetching ticker data fails."""

    pass


@dataclass
class UniverseConfig:
    """Configuration for the investment universe."""

    us_stocks_enabled: bool = True
    japan_stocks_enabled: bool = False
    etfs_enabled: bool = True
    commodities_enabled: bool = True
    forex_enabled: bool = True
    filters: dict[str, Any] = field(default_factory=dict)

    # Custom ticker lists (override automatic fetching)
    custom_us_stocks: list[str] = field(default_factory=list)
    custom_japan_stocks: list[str] = field(default_factory=list)
    custom_etfs: list[str] = field(default_factory=list)
    custom_commodities: list[str] = field(default_factory=list)
    custom_forex: list[str] = field(default_factory=list)

    # Limits
    max_us_stocks: int = 100
    max_japan_stocks: int = 50
    max_tickers: int = 150  # Total universe limit

    # Regional allocation
    regional_allocation: dict[str, float] = field(
        default_factory=lambda: {"us": 0.70, "international": 0.30}
    )

    # Dynamic selection settings
    dynamic_selection: dict[str, Any] = field(default_factory=dict)


class UniverseLoader:
    """
    Loads and manages the investment universe.

    Provides methods to fetch tickers from various sources including:
    - S&P 500 constituents (from Wikipedia)
    - Nikkei 225 constituents
    - ETFs
    - Commodities (futures)
    - Forex pairs

    Example:
        >>> loader = UniverseLoader()
        >>> tickers = loader.get_all_tickers()
        >>> print(tickers["us_stocks"][:5])
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    """

    # Wikipedia URL for S&P 500 constituents
    SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Cache settings
    CACHE_DURATION_DAYS = 7

    # Default ETF tickers
    DEFAULT_ETFS = [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "IWM",   # Russell 2000
        "DIA",   # Dow Jones
        "VTI",   # Total US Market
        "EFA",   # EAFE (International Developed)
        "EEM",   # Emerging Markets
        "VNQ",   # Real Estate
        "TLT",   # Long-term Treasury
        "GLD",   # Gold
        "SLV",   # Silver
        "USO",   # Oil
        "XLF",   # Financials
        "XLK",   # Technology
        "XLE",   # Energy
        "XLV",   # Healthcare
    ]

    # Default commodity tickers (Yahoo Finance format)
    DEFAULT_COMMODITIES = [
        "GC=F",   # Gold Futures
        "SI=F",   # Silver Futures
        "CL=F",   # Crude Oil Futures
        "NG=F",   # Natural Gas Futures
        "HG=F",   # Copper Futures
        "ZC=F",   # Corn Futures
        "ZW=F",   # Wheat Futures
        "ZS=F",   # Soybean Futures
    ]

    # Default forex pairs (Yahoo Finance format)
    DEFAULT_FOREX = [
        "USDJPY=X",   # USD/JPY
        "EURUSD=X",   # EUR/USD
        "GBPUSD=X",   # GBP/USD
        "AUDUSD=X",   # AUD/USD
        "USDCAD=X",   # USD/CAD
        "USDCHF=X",   # USD/CHF
        "EURJPY=X",   # EUR/JPY
        "GBPJPY=X",   # GBP/JPY
    ]

    # Fallback S&P 500 top tickers (in case Wikipedia fetch fails)
    FALLBACK_SP500 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "TMO", "ACN", "ABT", "DHR", "NEE", "VZ", "ADBE", "CRM", "NKE",
        "CMCSA", "TXN", "PM", "INTC", "WFC", "AMD", "UPS", "RTX", "QCOM",
        "MS", "T", "BMY", "ORCL", "SPGI",
    ]

    # Nikkei 225 representative tickers
    NIKKEI225_TICKERS = [
        "7203",   # Toyota
        "6758",   # Sony
        "9984",   # SoftBank Group
        "6861",   # Keyence
        "8306",   # MUFG
        "9432",   # NTT
        "6098",   # Recruit
        "7267",   # Honda
        "6501",   # Hitachi
        "4063",   # Shin-Etsu Chemical
        "7974",   # Nintendo
        "8035",   # Tokyo Electron
        "6367",   # Daikin
        "9433",   # KDDI
        "4502",   # Takeda
        "6902",   # Denso
        "7751",   # Canon
        "8058",   # Mitsubishi Corp
        "6954",   # Fanuc
        "7832",   # Bandai Namco
        "9983",   # Fast Retailing
        "4519",   # Chugai Pharmaceutical
        "6702",   # Fujitsu
        "8031",   # Mitsui & Co
        "6857",   # Advantest
    ]

    def __init__(
        self,
        config_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the UniverseLoader.

        Args:
            config_path: Path to universe.yaml config file.
                        If None, uses default configuration.
            cache_dir: Directory for caching fetched data.
                      If None, uses a default cache location.
        """
        self._config_path = Path(config_path) if config_path else None
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/universe")
        self._config: UniverseConfig | None = None

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "UniverseLoader initialized",
            config_path=str(self._config_path),
            cache_dir=str(self._cache_dir),
        )

    def load_config(self) -> UniverseConfig:
        """
        Load universe configuration from YAML file.

        Returns:
            UniverseConfig with settings from file or defaults.
        """
        if self._config is not None:
            return self._config

        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    data = yaml.safe_load(f) or {}

                universe_data = data.get("universe", data)
                self._config = UniverseConfig(
                    us_stocks_enabled=universe_data.get("us_stocks_enabled", True),
                    japan_stocks_enabled=universe_data.get("japan_stocks_enabled", False),
                    etfs_enabled=universe_data.get("etfs_enabled", True),
                    commodities_enabled=universe_data.get("commodities_enabled", True),
                    forex_enabled=universe_data.get("forex_enabled", True),
                    filters=universe_data.get("filters", {}),
                    custom_us_stocks=universe_data.get("custom_us_stocks", []),
                    custom_japan_stocks=universe_data.get("custom_japan_stocks", []),
                    custom_etfs=universe_data.get("custom_etfs", []),
                    custom_commodities=universe_data.get("custom_commodities", []),
                    custom_forex=universe_data.get("custom_forex", []),
                    max_us_stocks=universe_data.get("max_us_stocks", 100),
                    max_japan_stocks=universe_data.get("max_japan_stocks", 50),
                )
                logger.info("Loaded universe config from file", path=str(self._config_path))
            except Exception as e:
                logger.warning(
                    "Failed to load config, using defaults",
                    error=str(e),
                    path=str(self._config_path),
                )
                self._config = UniverseConfig()
        else:
            logger.info("No config file found, using defaults")
            self._config = UniverseConfig()

        return self._config

    def _get_cache_path(self, name: str) -> Path:
        """Get the cache file path for a given data type."""
        return self._cache_dir / f"{name}_cache.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False

        try:
            with open(cache_path) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
            expiry = cached_time + timedelta(days=self.CACHE_DURATION_DAYS)
            return datetime.now() < expiry
        except Exception:
            return False

    def _load_from_cache(self, name: str) -> list[str] | None:
        """Load tickers from cache if valid."""
        cache_path = self._get_cache_path(name)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            logger.debug(f"Loaded {name} from cache", count=len(data.get("tickers", [])))
            return data.get("tickers", [])
        except Exception as e:
            logger.warning(f"Failed to load {name} cache", error=str(e))
            return None

    def _save_to_cache(self, name: str, tickers: list[str]) -> None:
        """Save tickers to cache."""
        cache_path = self._get_cache_path(name)
        try:
            with open(cache_path, "w") as f:
                json.dump(
                    {"timestamp": datetime.now().isoformat(), "tickers": tickers},
                    f,
                    indent=2,
                )
            logger.debug(f"Saved {name} to cache", count=len(tickers))
        except Exception as e:
            logger.warning(f"Failed to save {name} cache", error=str(e))

    def get_sp500_tickers(self, use_cache: bool = True) -> list[str]:
        """
        Get S&P 500 constituent tickers.

        Attempts to fetch from Wikipedia. Falls back to a predefined list
        if the fetch fails.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            List of S&P 500 ticker symbols.
        """
        config = self.load_config()

        # Use custom list if provided
        if config.custom_us_stocks:
            logger.info("Using custom US stocks list", count=len(config.custom_us_stocks))
            return config.custom_us_stocks[: config.max_us_stocks]

        # Check cache
        if use_cache:
            cached = self._load_from_cache("sp500")
            if cached:
                return cached[: config.max_us_stocks]

        # Try to fetch from Wikipedia
        try:
            logger.info("Fetching S&P 500 from Wikipedia")
            tables = pd.read_html(self.SP500_WIKI_URL)

            # The first table contains the S&P 500 list
            df = tables[0]

            # Find the ticker column (usually "Symbol" or "Ticker")
            ticker_col = None
            for col in df.columns:
                if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                    ticker_col = col
                    break

            if ticker_col is None:
                # Try first column as fallback
                ticker_col = df.columns[0]

            tickers = df[ticker_col].tolist()

            # Clean tickers (remove any notes like "BRK.B" -> "BRK-B")
            tickers = [str(t).replace(".", "-").strip() for t in tickers if pd.notna(t)]

            logger.info("Fetched S&P 500 tickers", count=len(tickers))
            self._save_to_cache("sp500", tickers)
            return tickers[: config.max_us_stocks]

        except Exception as e:
            logger.warning(
                "Failed to fetch S&P 500 from Wikipedia, using fallback",
                error=str(e),
            )
            return self.FALLBACK_SP500[: config.max_us_stocks]

    def get_nikkei225_tickers(self, use_cache: bool = True) -> list[str]:
        """
        Get Nikkei 225 constituent tickers.

        Returns tickers with .T suffix for Yahoo Finance compatibility.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            List of Nikkei 225 ticker symbols (e.g., "7203.T").
        """
        config = self.load_config()

        # Use custom list if provided
        if config.custom_japan_stocks:
            # Ensure .T suffix
            tickers = [
                t if t.endswith(".T") else f"{t}.T"
                for t in config.custom_japan_stocks
            ]
            logger.info("Using custom Japan stocks list", count=len(tickers))
            return tickers[: config.max_japan_stocks]

        # Check cache
        if use_cache:
            cached = self._load_from_cache("nikkei225")
            if cached:
                return cached[: config.max_japan_stocks]

        # Use predefined list (add .T suffix)
        tickers = [f"{code}.T" for code in self.NIKKEI225_TICKERS]

        logger.info("Using predefined Nikkei 225 tickers", count=len(tickers))
        self._save_to_cache("nikkei225", tickers)
        return tickers[: config.max_japan_stocks]

    def get_etf_tickers(self) -> list[str]:
        """
        Get ETF tickers.

        Returns:
            List of ETF ticker symbols.
        """
        config = self.load_config()

        if config.custom_etfs:
            logger.info("Using custom ETF list", count=len(config.custom_etfs))
            return config.custom_etfs

        return self.DEFAULT_ETFS.copy()

    def get_commodity_tickers(self) -> list[str]:
        """
        Get commodity futures tickers.

        Returns:
            List of commodity ticker symbols (Yahoo Finance format).
        """
        config = self.load_config()

        if config.custom_commodities:
            logger.info("Using custom commodities list", count=len(config.custom_commodities))
            return config.custom_commodities

        return self.DEFAULT_COMMODITIES.copy()

    def get_forex_pairs(self) -> list[str]:
        """
        Get forex pair tickers.

        Returns:
            List of forex pair symbols (Yahoo Finance format).
        """
        config = self.load_config()

        if config.custom_forex:
            logger.info("Using custom forex list", count=len(config.custom_forex))
            return config.custom_forex

        return self.DEFAULT_FOREX.copy()

    def apply_filters(
        self,
        tickers: list[str],
        category: str | None = None,
    ) -> list[str]:
        """
        Apply configured filters to a list of tickers.

        Args:
            tickers: List of ticker symbols.
            category: Optional category name for category-specific filters.

        Returns:
            Filtered list of tickers.
        """
        config = self.load_config()
        filters = config.filters

        if not filters:
            return tickers

        result = tickers.copy()

        # Apply exclude list
        exclude = filters.get("exclude", [])
        if exclude:
            result = [t for t in result if t not in exclude]

        # Apply include list (overrides everything else)
        include = filters.get("include", [])
        if include:
            result = [t for t in result if t in include]

        # Apply category-specific filters
        if category and category in filters:
            cat_filters = filters[category]
            if "exclude" in cat_filters:
                result = [t for t in result if t not in cat_filters["exclude"]]
            if "include" in cat_filters:
                result = [t for t in result if t in cat_filters["include"]]

        return result

    def get_all_tickers(self) -> dict[str, list[str]]:
        """
        Get all tickers organized by category.

        Returns:
            Dictionary with categories as keys and ticker lists as values.
            {
                "us_stocks": ["AAPL", "MSFT", ...],
                "japan_stocks": ["7203.T", ...],
                "etfs": ["SPY", ...],
                "commodities": ["GC=F", ...],
                "forex": ["USDJPY=X", ...]
            }
        """
        config = self.load_config()
        result: dict[str, list[str]] = {}

        if config.us_stocks_enabled:
            tickers = self.get_sp500_tickers()
            result["us_stocks"] = self.apply_filters(tickers, "us_stocks")
            logger.info("Loaded US stocks", count=len(result["us_stocks"]))

        if config.japan_stocks_enabled:
            tickers = self.get_nikkei225_tickers()
            result["japan_stocks"] = self.apply_filters(tickers, "japan_stocks")
            logger.info("Loaded Japan stocks", count=len(result["japan_stocks"]))

        if config.etfs_enabled:
            tickers = self.get_etf_tickers()
            result["etfs"] = self.apply_filters(tickers, "etfs")
            logger.info("Loaded ETFs", count=len(result["etfs"]))

        if config.commodities_enabled:
            tickers = self.get_commodity_tickers()
            result["commodities"] = self.apply_filters(tickers, "commodities")
            logger.info("Loaded commodities", count=len(result["commodities"]))

        if config.forex_enabled:
            tickers = self.get_forex_pairs()
            result["forex"] = self.apply_filters(tickers, "forex")
            logger.info("Loaded forex pairs", count=len(result["forex"]))

        total = sum(len(v) for v in result.values())
        logger.info("Loaded complete universe", total_tickers=total, categories=len(result))

        return result

    def get_flat_ticker_list(self) -> list[str]:
        """
        Get all tickers as a flat list.

        Returns:
            List of all ticker symbols across all categories.
        """
        all_tickers = self.get_all_tickers()
        flat_list = []
        for tickers in all_tickers.values():
            flat_list.extend(tickers)
        return flat_list

    def clear_cache(self) -> None:
        """Clear all cached ticker data."""
        for cache_file in self._cache_dir.glob("*_cache.json"):
            try:
                cache_file.unlink()
                logger.info("Deleted cache file", path=str(cache_file))
            except Exception as e:
                logger.warning("Failed to delete cache file", path=str(cache_file), error=str(e))

    # ═══════════════════════════════════════════════════════════════
    # Enhanced Filtering Methods (IMP-001: Universe Size Optimization)
    # ═══════════════════════════════════════════════════════════════

    def apply_liquidity_filter(
        self,
        tickers: list[str],
        price_data: dict[str, pd.DataFrame] | None = None,
        min_daily_volume: float = 10_000_000,
        min_price: float = 5.0,
        max_price: float = 10_000.0,
    ) -> list[str]:
        """
        Apply enhanced liquidity filters to ticker list.

        Args:
            tickers: List of ticker symbols
            price_data: Optional price data dict {ticker: DataFrame}
            min_daily_volume: Minimum daily dollar volume
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            Filtered list of tickers meeting liquidity criteria
        """
        if price_data is None:
            logger.warning("No price data provided for liquidity filter, returning all tickers")
            return tickers

        filtered = []
        for ticker in tickers:
            if ticker not in price_data:
                continue

            df = price_data[ticker]

            # Find close and volume columns (handle various formats)
            close_col = None
            volume_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if "close" in col_lower:
                    close_col = col
                elif "volume" in col_lower:
                    volume_col = col

            if close_col is None or volume_col is None:
                continue

            try:
                # Calculate average metrics
                close_prices = df[close_col].dropna()
                volumes = df[volume_col].dropna()

                if len(close_prices) < 20:
                    continue

                avg_price = close_prices.mean()
                avg_volume = volumes.mean()
                dollar_volume = avg_price * avg_volume

                # Apply filters
                if dollar_volume >= min_daily_volume and min_price <= avg_price <= max_price:
                    filtered.append(ticker)

            except Exception as e:
                logger.debug(f"Liquidity filter error for {ticker}: {e}")
                continue

        logger.info(
            "Applied liquidity filter",
            input_count=len(tickers),
            output_count=len(filtered),
            min_volume=min_daily_volume,
        )
        return filtered

    def apply_sector_diversification(
        self,
        tickers: list[str],
        sector_mapping: dict[str, str] | None = None,
        max_sector_weight: float = 0.15,
        min_sector_count: int = 8,
    ) -> list[str]:
        """
        Apply sector diversification constraints.

        Args:
            tickers: List of ticker symbols
            sector_mapping: Dict mapping ticker to sector name
            max_sector_weight: Maximum weight per sector (0-1)
            min_sector_count: Minimum number of sectors required

        Returns:
            Diversified list of tickers
        """
        if sector_mapping is None:
            # Use default sector hints from config or return as-is
            logger.warning("No sector mapping provided, skipping diversification")
            return tickers

        # Group tickers by sector
        sector_tickers: dict[str, list[str]] = {}
        for ticker in tickers:
            sector = sector_mapping.get(ticker, "unknown")
            if sector not in sector_tickers:
                sector_tickers[sector] = []
            sector_tickers[sector].append(ticker)

        # Calculate target count per sector
        total_target = len(tickers)
        max_per_sector = int(total_target * max_sector_weight)

        # Select tickers respecting sector limits
        selected = []
        for sector, sector_list in sector_tickers.items():
            # Limit tickers per sector
            selected.extend(sector_list[:max_per_sector])

        # Ensure minimum sector count
        active_sectors = len([s for s in sector_tickers if sector_tickers[s]])
        if active_sectors < min_sector_count:
            logger.warning(
                f"Only {active_sectors} sectors available, target was {min_sector_count}"
            )

        logger.info(
            "Applied sector diversification",
            input_count=len(tickers),
            output_count=len(selected),
            sectors=len(sector_tickers),
        )
        return selected

    def dynamic_ticker_selection(
        self,
        candidates: list[str],
        price_data: dict[str, pd.DataFrame],
        target_count: int = 150,
        momentum_weight: float = 0.4,
        liquidity_weight: float = 0.3,
        volatility_weight: float = 0.2,
        diversification_weight: float = 0.1,
        momentum_lookback: int = 60,
    ) -> list[str]:
        """
        Dynamically select optimal tickers based on multiple criteria.

        Args:
            candidates: List of candidate tickers
            price_data: Price data dict {ticker: DataFrame}
            target_count: Number of tickers to select
            momentum_weight: Weight for momentum score (0-1)
            liquidity_weight: Weight for liquidity score (0-1)
            volatility_weight: Weight for volatility score (0-1, lower vol = higher score)
            diversification_weight: Weight for diversification score (0-1)
            momentum_lookback: Days for momentum calculation

        Returns:
            Optimally selected list of tickers
        """
        import numpy as np

        scores: dict[str, float] = {}

        for ticker in candidates:
            if ticker not in price_data:
                continue

            df = price_data[ticker]

            # Find columns
            close_col = None
            volume_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if "close" in col_lower:
                    close_col = col
                elif "volume" in col_lower:
                    volume_col = col

            if close_col is None:
                continue

            try:
                closes = df[close_col].dropna()
                if len(closes) < momentum_lookback:
                    continue

                # Momentum score (higher = better)
                recent = closes.iloc[-momentum_lookback:]
                if len(recent) >= 2 and recent.iloc[0] > 0:
                    momentum = (recent.iloc[-1] / recent.iloc[0]) - 1
                else:
                    momentum = 0

                # Volatility score (lower vol = higher score)
                returns = closes.pct_change().dropna()
                volatility = returns.std() if len(returns) > 0 else 1.0
                vol_score = 1 / (1 + volatility * 10)  # Normalize

                # Liquidity score
                if volume_col is not None:
                    volumes = df[volume_col].dropna()
                    avg_volume = volumes.mean() if len(volumes) > 0 else 0
                    avg_price = closes.mean()
                    dollar_volume = avg_price * avg_volume
                    # Log scale normalization
                    liquidity_score = np.log10(max(dollar_volume, 1)) / 12  # Normalize to ~0-1
                else:
                    liquidity_score = 0.5

                # Combined score
                score = (
                    momentum_weight * max(min(momentum, 1), -1)  # Clip momentum
                    + volatility_weight * vol_score
                    + liquidity_weight * min(liquidity_score, 1)
                    + diversification_weight * 0.5  # Base diversification score
                )
                scores[ticker] = score

            except Exception as e:
                logger.debug(f"Scoring error for {ticker}: {e}")
                continue

        # Sort by score and select top N
        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [t[0] for t in sorted_tickers[:target_count]]

        logger.info(
            "Dynamic ticker selection complete",
            candidates=len(candidates),
            selected=len(selected),
            target=target_count,
        )
        return selected

    def load_optimized_universe(
        self,
        config_path: str = "config/universe_optimized.yaml",
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[str]:
        """
        Load optimized universe with all filters applied.

        Args:
            config_path: Path to optimized universe config
            price_data: Optional price data for dynamic filtering

        Returns:
            Optimized list of tickers
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Optimized config not found: {config_path}, using defaults")
            return self.get_flat_ticker_list()

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)

            universe_cfg = config.get("universe", {})
            filters = universe_cfg.get("filters", {})
            dynamic_cfg = universe_cfg.get("dynamic_selection", {})

            # Start with all candidates
            all_tickers = self.get_flat_ticker_list()
            logger.info(f"Starting with {len(all_tickers)} candidates")

            # Apply liquidity filter if price data available
            if price_data:
                all_tickers = self.apply_liquidity_filter(
                    all_tickers,
                    price_data,
                    min_daily_volume=filters.get("min_daily_volume", 10_000_000),
                    min_price=filters.get("min_price", 5.0),
                    max_price=filters.get("max_price", 10_000.0),
                )

            # Apply dynamic selection if enabled
            if dynamic_cfg.get("enabled", False) and price_data:
                all_tickers = self.dynamic_ticker_selection(
                    all_tickers,
                    price_data,
                    target_count=universe_cfg.get("max_tickers", 150),
                    momentum_weight=dynamic_cfg.get("momentum_weight", 0.4),
                    liquidity_weight=dynamic_cfg.get("liquidity_weight", 0.3),
                    volatility_weight=dynamic_cfg.get("volatility_weight", 0.2),
                    diversification_weight=dynamic_cfg.get("diversification_weight", 0.1),
                    momentum_lookback=dynamic_cfg.get("momentum_lookback", 60),
                )
            else:
                # Just limit to max_tickers
                max_tickers = universe_cfg.get("max_tickers", 150)
                all_tickers = all_tickers[:max_tickers]

            logger.info(f"Optimized universe: {len(all_tickers)} tickers")
            return all_tickers

        except Exception as e:
            logger.error(f"Failed to load optimized universe: {e}")
            return self.get_flat_ticker_list()[:150]
