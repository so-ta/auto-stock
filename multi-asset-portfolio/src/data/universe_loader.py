"""
Universe Loader - Asset Universe Management (v3.0)

This module provides functionality to load and manage the investment universe
from the unified asset_master.yaml file.

v3.0 features:
- Single source of truth: asset_master.yaml
- Dynamic taxonomy (extensible via YAML only)
- Named subsets for easy filtering
- Report generation support via group_by_taxonomy()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)


# =============================================================================
# Symbol Dataclass
# =============================================================================


@dataclass
class Symbol:
    """
    Represents a symbol with taxonomy classification and tags.

    Attributes:
        ticker: The ticker symbol (e.g., "AAPL", "7203.T")
        taxonomy: Key-value pairs for classification (market, asset_class, sector, etc.)
        tags: List of tags for filtering (sbi, quality, sp500, etc.)
    """

    ticker: str
    taxonomy: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Symbol":
        """Create Symbol from dictionary."""
        return cls(
            ticker=data["ticker"],
            taxonomy=data.get("taxonomy", {}),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Symbol to dictionary."""
        result: dict[str, Any] = {"ticker": self.ticker}
        if self.taxonomy:
            result["taxonomy"] = self.taxonomy
        if self.tags:
            result["tags"] = self.tags
        return result

    @property
    def market(self) -> str:
        """Get market from taxonomy."""
        return self.taxonomy.get("market", "unknown")

    @property
    def asset_class(self) -> str:
        """Get asset_class from taxonomy."""
        return self.taxonomy.get("asset_class", "equity")

    @property
    def sector(self) -> str | None:
        """Get sector from taxonomy."""
        return self.taxonomy.get("sector")

    @property
    def etf_category(self) -> str | None:
        """Get etf_category from taxonomy."""
        return self.taxonomy.get("etf_category")

    def has_tag(self, tag: str) -> bool:
        """Check if symbol has a specific tag."""
        return tag in self.tags

    def matches_taxonomy(self, **filters: str | list[str] | None) -> bool:
        """
        Check if symbol matches the given taxonomy filters.

        Args:
            **filters: Key-value pairs to match against taxonomy.
                       Values can be single strings or lists.

        Returns:
            True if all filters match.
        """
        for key, value in filters.items():
            if value is None:
                continue
            symbol_value = self.taxonomy.get(key)
            if isinstance(value, list):
                if symbol_value not in value:
                    return False
            elif symbol_value != value:
                return False
        return True


# =============================================================================
# Exceptions
# =============================================================================


class UniverseLoaderError(Exception):
    """Base exception for universe loader errors."""

    pass


class ConfigNotFoundError(UniverseLoaderError):
    """Raised when configuration file is not found."""

    pass


class InvalidVersionError(UniverseLoaderError):
    """Raised when the asset master version is not supported."""

    pass


# =============================================================================
# UniverseLoader
# =============================================================================


class UniverseLoader:
    """
    Loads and manages the investment universe from asset_master.yaml.

    Provides methods to:
    - Get all symbols or filter by criteria
    - Use named subsets (standard, sbi, japan, etc.)
    - Access taxonomy definitions
    - Group symbols for report generation

    Example:
        >>> loader = UniverseLoader()
        >>> symbols = loader.get_subset("standard")
        >>> print(f"Standard universe: {len(symbols)} symbols")

        >>> # Filter by taxonomy
        >>> us_equities = loader.filter_symbols(
        ...     taxonomy={"market": ["us"], "asset_class": ["equity"]}
        ... )

        >>> # Group for report
        >>> by_sector = loader.group_by_taxonomy(us_equities, "sector")
        >>> for sector, syms in by_sector.items():
        ...     print(f"{sector}: {len(syms)} symbols")
    """

    SUPPORTED_VERSIONS = ["3.0"]

    def __init__(
        self,
        master_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the UniverseLoader.

        Args:
            master_path: Path to asset_master.yaml. If None, uses default location.
        """
        if master_path is None:
            # Default to config/asset_master.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            master_path = project_root / "config" / "asset_master.yaml"

        self._master_path = Path(master_path)
        self._master_data: dict | None = None
        self._symbols_cache: list[Symbol] | None = None

        logger.debug(
            "UniverseLoader initialized",
            master_path=str(self._master_path),
        )

    def _load_master(self) -> dict:
        """
        Load and cache the master file.

        Returns:
            Parsed YAML data.

        Raises:
            ConfigNotFoundError: If the file doesn't exist.
            InvalidVersionError: If the version is not supported.
        """
        if self._master_data is not None:
            return self._master_data

        if not self._master_path.exists():
            raise ConfigNotFoundError(f"Asset master not found: {self._master_path}")

        with open(self._master_path) as f:
            data = yaml.safe_load(f) or {}

        # Validate version
        version = data.get("version", "1.0")
        if not any(version.startswith(v.split(".")[0]) for v in self.SUPPORTED_VERSIONS):
            raise InvalidVersionError(
                f"Asset master version {version} is not supported. "
                f"Supported versions: {self.SUPPORTED_VERSIONS}"
            )

        self._master_data = data
        logger.info(
            "Loaded asset master",
            version=version,
            name=data.get("name"),
            symbol_count=len(data.get("symbols", [])),
        )
        return data

    def _get_symbols_list(self) -> list[Symbol]:
        """Get all symbols as Symbol objects (cached)."""
        if self._symbols_cache is not None:
            return self._symbols_cache

        data = self._load_master()
        symbols_data = data.get("symbols", [])
        self._symbols_cache = [Symbol.from_dict(s) for s in symbols_data]
        return self._symbols_cache

    # =========================================================================
    # Public API: Symbol Access
    # =========================================================================

    def get_all_symbols(self) -> list[Symbol]:
        """
        Get all symbols in the master file.

        Returns:
            List of all Symbol objects.
        """
        return self._get_symbols_list().copy()

    def get_all_tickers(self) -> list[str]:
        """
        Get all ticker strings.

        Returns:
            List of ticker strings.
        """
        return [s.ticker for s in self._get_symbols_list()]

    def get_subset(self, subset_name: str) -> list[Symbol]:
        """
        Get symbols matching a named subset.

        Args:
            subset_name: Name of the subset (e.g., "standard", "sbi", "japan")

        Returns:
            List of Symbol objects matching the subset filters.

        Raises:
            KeyError: If the subset doesn't exist.
        """
        data = self._load_master()
        subsets = data.get("subsets", {})

        if subset_name not in subsets:
            available = list(subsets.keys())
            raise KeyError(
                f"Subset '{subset_name}' not found. Available: {available}"
            )

        subset_def = subsets[subset_name]
        filters = subset_def.get("filters", {})

        return self._apply_filters(filters)

    def get_subset_tickers(self, subset_name: str) -> list[str]:
        """
        Get ticker strings for a named subset.

        Args:
            subset_name: Name of the subset.

        Returns:
            List of ticker strings.
        """
        return [s.ticker for s in self.get_subset(subset_name)]

    def filter_symbols(
        self,
        tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        taxonomy: dict[str, list[str]] | None = None,
    ) -> list[Symbol]:
        """
        Filter symbols by tags and taxonomy.

        Args:
            tags: Include symbols with ANY of these tags.
            exclude_tags: Exclude symbols with ANY of these tags.
            taxonomy: Filter by taxonomy values (e.g., {"market": ["us", "japan"]}).

        Returns:
            List of matching Symbol objects.
        """
        filters = {}
        if tags:
            filters["tags"] = tags
        if exclude_tags:
            filters["exclude_tags"] = exclude_tags
        if taxonomy:
            filters["taxonomy"] = taxonomy

        return self._apply_filters(filters)

    def filter_tickers(
        self,
        tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        taxonomy: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """
        Filter ticker strings by criteria.

        Args:
            tags: Include symbols with ANY of these tags.
            exclude_tags: Exclude symbols with ANY of these tags.
            taxonomy: Filter by taxonomy values.

        Returns:
            List of matching ticker strings.
        """
        symbols = self.filter_symbols(tags, exclude_tags, taxonomy)
        return [s.ticker for s in symbols]

    def _apply_filters(self, filters: dict) -> list[Symbol]:
        """Apply filter dictionary to symbols."""
        all_symbols = self._get_symbols_list()
        result = []

        tags_filter = filters.get("tags", [])
        exclude_tags = filters.get("exclude_tags", [])
        taxonomy_filter = filters.get("taxonomy", {})

        for symbol in all_symbols:
            # Check exclude tags first
            if exclude_tags:
                if any(symbol.has_tag(t) for t in exclude_tags):
                    continue

            # Check required tags (any match)
            if tags_filter:
                if not any(symbol.has_tag(t) for t in tags_filter):
                    continue

            # Check taxonomy filters
            if taxonomy_filter:
                match = True
                for key, values in taxonomy_filter.items():
                    if not symbol.matches_taxonomy(**{key: values}):
                        match = False
                        break
                if not match:
                    continue

            result.append(symbol)

        return result

    # =========================================================================
    # Public API: Taxonomy
    # =========================================================================

    def get_taxonomy(self) -> dict[str, dict[str, str]]:
        """
        Get all taxonomy definitions.

        Returns:
            Dictionary with taxonomy categories and their value labels.
            Example: {"market": {"us": "US Markets", "japan": "Japanese Markets"}, ...}
        """
        data = self._load_master()
        return data.get("taxonomy", {})

    def get_taxonomy_keys(self) -> list[str]:
        """
        Get available taxonomy keys.

        Returns:
            List of taxonomy keys (e.g., ["market", "asset_class", "sector"]).
        """
        return list(self.get_taxonomy().keys())

    def get_taxonomy_label(self, key: str, value: str) -> str:
        """
        Get the display label for a taxonomy value.

        Args:
            key: Taxonomy key (e.g., "market").
            value: Taxonomy value (e.g., "us").

        Returns:
            Display label (e.g., "US Markets"), or the value itself if not found.
        """
        taxonomy = self.get_taxonomy()
        return taxonomy.get(key, {}).get(value, value)

    # =========================================================================
    # Public API: Grouping for Reports
    # =========================================================================

    def group_by_taxonomy(
        self,
        symbols: list[Symbol],
        taxonomy_key: str,
    ) -> dict[str, list[Symbol]]:
        """
        Group symbols by a taxonomy key.

        Args:
            symbols: List of symbols to group.
            taxonomy_key: Key to group by (e.g., "sector", "market").

        Returns:
            Dictionary mapping taxonomy values to lists of symbols.
            Example: {"technology": [AAPL, MSFT, ...], "healthcare": [JNJ, ...]}
        """
        groups: dict[str, list[Symbol]] = {}

        for symbol in symbols:
            value = symbol.taxonomy.get(taxonomy_key, "unknown")
            if value not in groups:
                groups[value] = []
            groups[value].append(symbol)

        return groups

    def group_by_tag(
        self,
        symbols: list[Symbol],
        tag: str,
    ) -> dict[bool, list[Symbol]]:
        """
        Group symbols by whether they have a specific tag.

        Args:
            symbols: List of symbols to group.
            tag: Tag to check.

        Returns:
            Dictionary with True/False keys mapping to symbol lists.
        """
        return {
            True: [s for s in symbols if s.has_tag(tag)],
            False: [s for s in symbols if not s.has_tag(tag)],
        }

    # =========================================================================
    # Public API: Subset Info
    # =========================================================================

    def get_available_subsets(self) -> list[dict]:
        """
        Get information about available subsets.

        Returns:
            List of subset info dictionaries with name, description, and count.
        """
        data = self._load_master()
        subsets_def = data.get("subsets", {})
        result = []

        for name, subset in subsets_def.items():
            try:
                symbols = self.get_subset(name)
                count = len(symbols)
            except Exception:
                count = 0

            result.append({
                "name": name,
                "description": subset.get("description", ""),
                "symbol_count": count,
            })

        return result

    def get_symbols_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of all symbols.

        Returns:
            Dictionary with counts by market, asset_class, etc.
        """
        symbols = self._get_symbols_list()

        by_market: dict[str, int] = {}
        by_asset_class: dict[str, int] = {}
        by_sector: dict[str, int] = {}

        for s in symbols:
            # Market
            market = s.market
            by_market[market] = by_market.get(market, 0) + 1

            # Asset class
            asset_class = s.asset_class
            by_asset_class[asset_class] = by_asset_class.get(asset_class, 0) + 1

            # Sector (only for equities with sector)
            sector = s.sector
            if sector:
                by_sector[sector] = by_sector.get(sector, 0) + 1

        return {
            "total": len(symbols),
            "by_market": by_market,
            "by_asset_class": by_asset_class,
            "by_sector": by_sector,
        }

    # =========================================================================
    # Backward Compatibility: Legacy Format Output
    # =========================================================================

    def load_standard_universe(self) -> dict[str, list[str]]:
        """
        Load standard universe in legacy format.

        This method provides backward compatibility with code expecting
        the old dictionary format.

        Returns:
            Dictionary with "us_stocks", "japan_stocks", "etfs" keys.
        """
        try:
            symbols = self.get_subset("standard")
        except KeyError:
            symbols = self.get_all_symbols()

        us_stocks = []
        japan_stocks = []
        etfs = []

        for s in symbols:
            if s.asset_class == "etf":
                etfs.append(s.ticker)
            elif s.market == "japan":
                japan_stocks.append(s.ticker)
            elif s.market == "us":
                us_stocks.append(s.ticker)

        return {
            "us_stocks": us_stocks,
            "japan_stocks": japan_stocks,
            "etfs": etfs,
        }

    def get_flat_ticker_list(self) -> list[str]:
        """
        Get all tickers as a flat list.

        Returns:
            List of all ticker strings.
        """
        return self.get_all_tickers()

    def reload(self) -> None:
        """Clear cache and reload the master file."""
        self._master_data = None
        self._symbols_cache = None
        self._load_master()
