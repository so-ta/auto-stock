"""
Config Service - Configuration management for backtest execution

Unified management of universes (via asset_master.yaml), cost profiles,
and default settings for backtesting.

Extended with CRUD operations for:
- Subsets (filter definitions)
- Symbols (add/remove/tag management)
- Taxonomy (key/value definitions)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass
class UniverseInfo:
    """Universe/subset information."""
    name: str  # Subset name or "all"
    file_path: str  # Path to asset_master.yaml
    symbol_count: int
    description: str = ""
    asset_types: List[str] = field(default_factory=list)
    markets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "symbol_count": self.symbol_count,
            "description": self.description,
            "asset_types": self.asset_types,
            "markets": self.markets,
        }


@dataclass
class CostProfile:
    """Cost profile for transaction costs."""
    name: str
    description: str
    spread_bps: float
    commission_bps: float
    slippage_bps: float
    total_bps: float = 0

    def __post_init__(self):
        self.total_bps = self.spread_bps + self.commission_bps + self.slippage_bps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "spread_bps": self.spread_bps,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_bps": self.total_bps,
        }


# Predefined cost profiles
COST_PROFILES: Dict[str, CostProfile] = {
    "zero": CostProfile(
        name="zero",
        description="No transaction costs (for testing)",
        spread_bps=0.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    ),
    "low": CostProfile(
        name="low",
        description="Low cost (major US stocks)",
        spread_bps=5.0,
        commission_bps=0.0,
        slippage_bps=5.0,
    ),
    "standard": CostProfile(
        name="standard",
        description="Standard cost (default)",
        spread_bps=10.0,
        commission_bps=5.0,
        slippage_bps=10.0,
    ),
    "high": CostProfile(
        name="high",
        description="High cost (small caps, emerging markets)",
        spread_bps=20.0,
        commission_bps=10.0,
        slippage_bps=20.0,
    ),
    "japan": CostProfile(
        name="japan",
        description="Japanese stocks",
        spread_bps=10.0,
        commission_bps=10.0,
        slippage_bps=10.0,
    ),
}


@dataclass
class FrequencyPreset:
    """Rebalance frequency preset."""
    name: str
    value: str  # daily, weekly, monthly, quarterly
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
        }


FREQUENCY_PRESETS: List[FrequencyPreset] = [
    FrequencyPreset("Daily", "daily", "Rebalance every trading day"),
    FrequencyPreset("Weekly", "weekly", "Rebalance every week"),
    FrequencyPreset("Monthly", "monthly", "Rebalance every month (default)"),
    FrequencyPreset("Quarterly", "quarterly", "Rebalance every quarter"),
]


@dataclass
class PeriodPreset:
    """Period preset for backtesting."""
    name: str
    years: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "years": self.years,
            "description": self.description,
        }


PERIOD_PRESETS: List[PeriodPreset] = [
    PeriodPreset("1 Year", 1, "Short-term validation"),
    PeriodPreset("3 Years", 3, "Standard validation"),
    PeriodPreset("5 Years", 5, "Long-term validation (recommended)"),
    PeriodPreset("10 Years", 10, "Extended validation"),
    PeriodPreset("Full History", 0, "All available data"),
]


class ConfigService:
    """
    Configuration service for backtest execution.

    Uses asset_master.yaml for universe/subset management.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ):
        """
        Initialize ConfigService.

        Args:
            project_root: Project root path.
            config_dir: Configuration directory path.
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.config_dir = config_dir or self.project_root / "config"
        self._master_path = self.config_dir / "asset_master.yaml"
        self._master_data: Optional[Dict] = None

        logger.info(f"ConfigService initialized: config_dir={self.config_dir}")

    def _get_registered_signal_count(self) -> int:
        """Get total number of registered signals from SignalRegistry."""
        try:
            from src.signals import SignalRegistry
            return len(SignalRegistry.list_all())
        except ImportError:
            return 0

    def _load_master(self) -> Dict[str, Any]:
        """Load and cache asset_master.yaml."""
        if self._master_data is not None:
            return self._master_data

        if not self._master_path.exists():
            logger.warning(f"Asset master not found: {self._master_path}")
            return {}

        with open(self._master_path, "r", encoding="utf-8") as f:
            self._master_data = yaml.safe_load(f) or {}

        return self._master_data

    def list_universes(self) -> List[UniverseInfo]:
        """
        List available universe subsets.

        Returns:
            List of UniverseInfo for each subset.
        """
        data = self._load_master()
        if not data:
            return []

        universes = []
        symbols = data.get("symbols", [])

        # Add "all" as the first option
        all_info = self._analyze_symbols(symbols)
        universes.append(UniverseInfo(
            name="all",
            file_path=str(self._master_path),
            symbol_count=len(symbols),
            description="All symbols in asset master",
            asset_types=all_info["asset_types"],
            markets=all_info["markets"],
        ))

        # Add each defined subset
        subsets = data.get("subsets", {})
        for name, subset_def in subsets.items():
            try:
                filtered = self._apply_subset_filters(symbols, subset_def.get("filters", {}))
                info = self._analyze_symbols(filtered)
                universes.append(UniverseInfo(
                    name=name,
                    file_path=str(self._master_path),
                    symbol_count=len(filtered),
                    description=subset_def.get("description", ""),
                    asset_types=info["asset_types"],
                    markets=info["markets"],
                ))
            except Exception as e:
                logger.warning(f"Failed to process subset {name}: {e}")

        return universes

    def get_universe(self, name: str) -> Optional[UniverseInfo]:
        """
        Get specific universe/subset info.

        Args:
            name: Subset name or "all".

        Returns:
            UniverseInfo or None.
        """
        universes = self.list_universes()
        for u in universes:
            if u.name == name:
                return u
        return None

    def get_universe_symbols(self, name: str) -> List[str]:
        """
        Get symbols for a specific subset.

        Args:
            name: Subset name or "all".

        Returns:
            List of ticker symbols.
        """
        data = self._load_master()
        if not data:
            return []

        symbols = data.get("symbols", [])

        if name == "all":
            return [s["ticker"] for s in symbols if "ticker" in s]

        subsets = data.get("subsets", {})
        if name not in subsets:
            return []

        filters = subsets[name].get("filters", {})
        filtered = self._apply_subset_filters(symbols, filters)
        return [s["ticker"] for s in filtered if "ticker" in s]

    def _apply_subset_filters(
        self,
        symbols: List[Dict],
        filters: Dict[str, Any],
    ) -> List[Dict]:
        """Apply subset filter criteria to symbols."""
        if not filters:
            return symbols

        result = []
        tags_filter = filters.get("tags", [])
        exclude_tags = filters.get("exclude_tags", [])
        taxonomy_filter = filters.get("taxonomy", {})

        for sym in symbols:
            sym_tags = sym.get("tags", [])
            sym_taxonomy = sym.get("taxonomy", {})

            # Check exclude tags
            if exclude_tags:
                if any(t in sym_tags for t in exclude_tags):
                    continue

            # Check required tags (any match)
            if tags_filter:
                if not any(t in sym_tags for t in tags_filter):
                    continue

            # Check taxonomy filters
            if taxonomy_filter:
                match = True
                for key, values in taxonomy_filter.items():
                    sym_value = sym_taxonomy.get(key)
                    if sym_value not in values:
                        match = False
                        break
                if not match:
                    continue

            result.append(sym)

        return result

    def _analyze_symbols(self, symbols: List[Dict]) -> Dict[str, List[str]]:
        """Analyze symbols to extract asset types and markets."""
        asset_types = set()
        markets = set()

        for sym in symbols:
            taxonomy = sym.get("taxonomy", {})

            # Market
            market = taxonomy.get("market", "")
            if market == "japan":
                markets.add("JP")
            elif market == "us":
                markets.add("US")
            elif market:
                markets.add(market.upper())

            # Asset class
            asset_class = taxonomy.get("asset_class", "equity")
            asset_types.add(asset_class)

        return {
            "asset_types": sorted(asset_types),
            "markets": sorted(markets),
        }

    def list_cost_profiles(self) -> List[CostProfile]:
        """Get all cost profiles."""
        return list(COST_PROFILES.values())

    def get_cost_profile(self, name: str) -> Optional[CostProfile]:
        """Get a specific cost profile by name."""
        return COST_PROFILES.get(name)

    def list_frequency_presets(self) -> List[FrequencyPreset]:
        """Get rebalance frequency presets."""
        return FREQUENCY_PRESETS

    def list_period_presets(self) -> List[PeriodPreset]:
        """Get period presets."""
        return PERIOD_PRESETS

    def get_default_settings(self) -> Dict[str, Any]:
        """Get default settings from default.yaml."""
        default_file = self.config_dir / "default.yaml"
        if not default_file.exists():
            return {}

        try:
            with open(default_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load default settings: {e}")
            return {}

    def get_backtest_defaults(self) -> Dict[str, Any]:
        """Get backtest-specific default values."""
        defaults = self.get_default_settings()

        backtest = defaults.get("backtest", {})
        cost_model = defaults.get("cost_model", {})
        rebalance = defaults.get("rebalance", {})

        return {
            "initial_capital": backtest.get("initial_capital", 100000.0),
            "transaction_cost_bps": backtest.get("transaction_cost_bps", 10.0),
            "slippage_bps": backtest.get("slippage_bps", 5.0),
            "frequency": rebalance.get("frequency", "monthly"),
            "spread_bps": cost_model.get("spread_bps", 10),
            "commission_bps": cost_model.get("commission_bps", 5),
            "slippage_model_bps": cost_model.get("slippage_bps", 10),
        }

    def estimate_backtest_duration(
        self,
        universe_size: int,
        start_date: str,
        end_date: str,
        frequency: str = "monthly",
    ) -> Dict[str, Any]:
        """
        Estimate backtest execution time.

        Args:
            universe_size: Number of symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Rebalance frequency.

        Returns:
            Estimation info dictionary.
        """
        from datetime import datetime

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days
        except ValueError:
            days = 365 * 5  # Default 5 years

        freq_multipliers = {
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
            "quarterly": 4,
        }
        annual_rebalances = freq_multipliers.get(frequency, 12)
        years = days / 365
        total_rebalances = int(annual_rebalances * years)

        # Heuristic estimation
        base_time_per_rebalance = 0.5
        symbol_factor = 0.01 * universe_size
        estimated_seconds = total_rebalances * (base_time_per_rebalance + symbol_factor)

        # Assume 70% reduction with cache
        with_cache_seconds = estimated_seconds * 0.3

        return {
            "total_rebalances": total_rebalances,
            "estimated_seconds": round(estimated_seconds, 1),
            "estimated_with_cache_seconds": round(with_cache_seconds, 1),
            "estimated_display": self._format_duration(with_cache_seconds),
        }

    def _format_duration(self, seconds: float) -> str:
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def validate_universe_file(self, name: str) -> Dict[str, Any]:
        """
        Validate universe/subset.

        Args:
            name: Subset name or "all".

        Returns:
            Validation result dictionary.
        """
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {},
        }

        symbols = self.get_universe_symbols(name)

        if not symbols:
            result["errors"].append(f"No symbols found for subset '{name}'")
            return result

        result["valid"] = True
        result["info"] = {
            "symbol_count": len(symbols),
            "symbols": symbols[:10],
            "has_more": len(symbols) > 10,
        }

        # Warnings
        if len(symbols) > 200:
            result["warnings"].append(
                f"Large universe ({len(symbols)} symbols) may slow down backtests"
            )

        if len(symbols) < 5:
            result["warnings"].append(
                f"Small universe ({len(symbols)} symbols) may not provide enough diversification"
            )

        return result

    # =========================================================================
    # Subset CRUD Operations
    # =========================================================================

    def get_subset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific subset definition.

        Args:
            name: Subset name.

        Returns:
            Subset definition dict or None.
        """
        data = self._load_master()
        subsets = data.get("subsets", {})
        return subsets.get(name)

    def create_subset(self, name: str, definition: Dict[str, Any]) -> bool:
        """
        Create a new subset.

        Args:
            name: Subset name (must be unique).
            definition: Subset definition with name, description, filters.

        Returns:
            True if created successfully.

        Raises:
            ValueError: If subset already exists.
        """
        data = self._load_master()
        subsets = data.setdefault("subsets", {})

        if name in subsets:
            raise ValueError(f"Subset '{name}' already exists")

        subsets[name] = definition
        logger.info(f"Created subset: {name}")
        return True

    def update_subset(self, name: str, definition: Dict[str, Any]) -> bool:
        """
        Update an existing subset.

        Args:
            name: Subset name.
            definition: New subset definition.

        Returns:
            True if updated successfully.

        Raises:
            ValueError: If subset doesn't exist.
        """
        data = self._load_master()
        subsets = data.get("subsets", {})

        if name not in subsets:
            raise ValueError(f"Subset '{name}' not found")

        subsets[name] = definition
        logger.info(f"Updated subset: {name}")
        return True

    def delete_subset(self, name: str) -> bool:
        """
        Delete a subset.

        Args:
            name: Subset name.

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If subset doesn't exist.
        """
        data = self._load_master()
        subsets = data.get("subsets", {})

        if name not in subsets:
            raise ValueError(f"Subset '{name}' not found")

        del subsets[name]
        logger.info(f"Deleted subset: {name}")
        return True

    def preview_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preview which symbols match given filter criteria.

        Args:
            filters: Filter definition (same format as subset filters).

        Returns:
            Dict with matched symbols and count.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        # Apply manual include/exclude
        manual_include = filters.pop("manual_include", [])
        manual_exclude = filters.pop("manual_exclude", [])

        filtered = self._apply_subset_filters(symbols, filters)
        tickers = [s["ticker"] for s in filtered if "ticker" in s]

        # Apply manual exclusions
        if manual_exclude:
            exclude_set = set(manual_exclude)
            tickers = [t for t in tickers if t not in exclude_set]

        # Apply manual inclusions
        if manual_include:
            all_tickers = {s["ticker"] for s in symbols if "ticker" in s}
            for ticker in manual_include:
                if ticker in all_tickers and ticker not in tickers:
                    tickers.append(ticker)

        return {
            "count": len(tickers),
            "symbols": sorted(tickers)[:100],  # Preview limited to 100
            "has_more": len(tickers) > 100,
        }

    # =========================================================================
    # Symbol CRUD Operations
    # =========================================================================

    def list_symbols(
        self,
        page: int = 1,
        size: int = 50,
        search: Optional[str] = None,
        market: Optional[str] = None,
        asset_class: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        List symbols with pagination and filters.

        Args:
            page: Page number (1-indexed).
            size: Page size.
            search: Search string (matches ticker).
            market: Filter by market.
            asset_class: Filter by asset class.
            tags: Filter by tags (any match).

        Returns:
            Dict with items, total, page info.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        # Apply filters
        filtered = []
        for sym in symbols:
            ticker = sym.get("ticker", "")
            sym_taxonomy = sym.get("taxonomy", {})
            sym_tags = sym.get("tags", [])

            # Search filter
            if search and search.upper() not in ticker.upper():
                continue

            # Market filter
            if market and sym_taxonomy.get("market") != market:
                continue

            # Asset class filter
            if asset_class and sym_taxonomy.get("asset_class") != asset_class:
                continue

            # Tags filter (any match)
            if tags and not any(t in sym_tags for t in tags):
                continue

            filtered.append(sym)

        total = len(filtered)
        total_pages = (total + size - 1) // size if size > 0 else 1

        # Pagination
        start = (page - 1) * size
        end = start + size
        items = filtered[start:end]

        return {
            "items": items,
            "total": total,
            "page": page,
            "size": size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }

    def get_symbol(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific symbol's details.

        Args:
            ticker: Ticker symbol.

        Returns:
            Symbol dict or None.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        for sym in symbols:
            if sym.get("ticker") == ticker:
                return sym
        return None

    def add_symbol(
        self,
        ticker: str,
        taxonomy: Dict[str, str],
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> bool:
        """
        Add a new symbol.

        Args:
            ticker: Ticker symbol.
            taxonomy: Taxonomy dict (market, asset_class, sector, etc.).
            tags: List of tags.
            name: Display name of the symbol.

        Returns:
            True if added successfully.

        Raises:
            ValueError: If symbol already exists.
        """
        data = self._load_master()
        symbols = data.setdefault("symbols", [])

        # Check for duplicate
        existing = {s.get("ticker") for s in symbols}
        if ticker in existing:
            raise ValueError(f"Symbol '{ticker}' already exists")

        new_symbol = {
            "ticker": ticker,
            "taxonomy": taxonomy,
            "tags": tags or [],
        }
        if name:
            new_symbol["name"] = name
        symbols.append(new_symbol)
        logger.info(f"Added symbol: {ticker}")
        return True

    def update_symbol(self, ticker: str, updates: Dict[str, Any]) -> bool:
        """
        Update a symbol's taxonomy, tags, or name.

        Args:
            ticker: Ticker symbol.
            updates: Dict with taxonomy, tags, and/or name to update.

        Returns:
            True if updated successfully.

        Raises:
            ValueError: If symbol doesn't exist.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        for sym in symbols:
            if sym.get("ticker") == ticker:
                if "taxonomy" in updates:
                    sym["taxonomy"] = updates["taxonomy"]
                if "tags" in updates:
                    sym["tags"] = updates["tags"]
                if "name" in updates:
                    if updates["name"]:
                        sym["name"] = updates["name"]
                    elif "name" in sym:
                        del sym["name"]
                logger.info(f"Updated symbol: {ticker}")
                return True

        raise ValueError(f"Symbol '{ticker}' not found")

    def delete_symbol(self, ticker: str) -> bool:
        """
        Delete a symbol.

        Args:
            ticker: Ticker symbol.

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If symbol doesn't exist.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        for i, sym in enumerate(symbols):
            if sym.get("ticker") == ticker:
                del symbols[i]
                logger.info(f"Deleted symbol: {ticker}")
                return True

        raise ValueError(f"Symbol '{ticker}' not found")

    def bulk_update_tags(
        self,
        tickers: List[str],
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> int:
        """
        Bulk update tags for multiple symbols.

        Args:
            tickers: List of ticker symbols.
            add_tags: Tags to add.
            remove_tags: Tags to remove.

        Returns:
            Number of symbols updated.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])
        ticker_set = set(tickers)
        updated = 0

        for sym in symbols:
            if sym.get("ticker") in ticker_set:
                current_tags = set(sym.get("tags", []))

                if add_tags:
                    current_tags.update(add_tags)

                if remove_tags:
                    current_tags -= set(remove_tags)

                sym["tags"] = sorted(current_tags)
                updated += 1

        logger.info(f"Bulk updated tags for {updated} symbols")
        return updated

    # =========================================================================
    # Taxonomy CRUD Operations
    # =========================================================================

    def get_taxonomy(self) -> Dict[str, Dict[str, str]]:
        """
        Get all taxonomy definitions.

        Returns:
            Dict of taxonomy keys with their code->label mappings.
        """
        data = self._load_master()
        return data.get("taxonomy", {})

    def get_taxonomy_key(self, key: str) -> Optional[Dict[str, str]]:
        """
        Get a specific taxonomy key's values.

        Args:
            key: Taxonomy key name.

        Returns:
            Dict of code->label mappings or None.
        """
        data = self._load_master()
        taxonomy = data.get("taxonomy", {})
        return taxonomy.get(key)

    def add_taxonomy_key(self, key: str, values: Dict[str, str]) -> bool:
        """
        Add a new taxonomy key.

        Args:
            key: Taxonomy key name.
            values: Dict of code->label mappings.

        Returns:
            True if added successfully.

        Raises:
            ValueError: If key already exists.
        """
        data = self._load_master()
        taxonomy = data.setdefault("taxonomy", {})

        if key in taxonomy:
            raise ValueError(f"Taxonomy key '{key}' already exists")

        taxonomy[key] = values
        logger.info(f"Added taxonomy key: {key}")
        return True

    def update_taxonomy_key(self, key: str, values: Dict[str, str]) -> bool:
        """
        Update a taxonomy key's values.

        Args:
            key: Taxonomy key name.
            values: New dict of code->label mappings.

        Returns:
            True if updated successfully.

        Raises:
            ValueError: If key doesn't exist.
        """
        data = self._load_master()
        taxonomy = data.get("taxonomy", {})

        if key not in taxonomy:
            raise ValueError(f"Taxonomy key '{key}' not found")

        taxonomy[key] = values
        logger.info(f"Updated taxonomy key: {key}")
        return True

    def delete_taxonomy_key(self, key: str) -> bool:
        """
        Delete a taxonomy key.

        Args:
            key: Taxonomy key name.

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If key doesn't exist or is in use.
        """
        data = self._load_master()
        taxonomy = data.get("taxonomy", {})

        if key not in taxonomy:
            raise ValueError(f"Taxonomy key '{key}' not found")

        # Check if in use
        symbols = data.get("symbols", [])
        in_use = sum(1 for s in symbols if key in s.get("taxonomy", {}))
        if in_use > 0:
            raise ValueError(
                f"Taxonomy key '{key}' is used by {in_use} symbols. "
                "Remove from symbols first."
            )

        del taxonomy[key]
        logger.info(f"Deleted taxonomy key: {key}")
        return True

    def add_taxonomy_value(self, key: str, code: str, label: str) -> bool:
        """
        Add a value to a taxonomy key.

        Args:
            key: Taxonomy key name.
            code: Value code.
            label: Value label.

        Returns:
            True if added successfully.

        Raises:
            ValueError: If key doesn't exist or code already exists.
        """
        data = self._load_master()
        taxonomy = data.get("taxonomy", {})

        if key not in taxonomy:
            raise ValueError(f"Taxonomy key '{key}' not found")

        if code in taxonomy[key]:
            raise ValueError(f"Value code '{code}' already exists in '{key}'")

        taxonomy[key][code] = label
        logger.info(f"Added taxonomy value: {key}.{code}")
        return True

    # =========================================================================
    # Save/Reload Operations
    # =========================================================================

    def save_master(self) -> bool:
        """
        Save the in-memory master data to asset_master.yaml.

        Returns:
            True if saved successfully.
        """
        if self._master_data is None:
            logger.warning("No changes to save")
            return False

        # Update timestamp
        from datetime import datetime
        self._master_data["generated_at"] = datetime.now().isoformat()

        # Create backup
        backup_path = self._master_path.with_suffix(".yaml.bak")
        if self._master_path.exists():
            import shutil
            shutil.copy2(self._master_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Write YAML
        with open(self._master_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._master_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"Saved asset master: {self._master_path}")
        return True

    def reload_master(self) -> None:
        """
        Clear the cached master data to force reload.
        """
        self._master_data = None
        logger.info("Master data cache cleared")

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags used across symbols.

        Returns:
            Sorted list of unique tags.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        all_tags: Set[str] = set()
        for sym in symbols:
            all_tags.update(sym.get("tags", []))

        return sorted(all_tags)

    def has_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes.

        This is a simple check - we compare the in-memory data with file.
        """
        if self._master_data is None:
            return False

        if not self._master_path.exists():
            return True

        try:
            with open(self._master_path, "r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}

            # Simple comparison (ignoring generated_at)
            mem_copy = copy.deepcopy(self._master_data)
            file_copy = copy.deepcopy(file_data)
            mem_copy.pop("generated_at", None)
            file_copy.pop("generated_at", None)

            return mem_copy != file_copy
        except Exception:
            return True

    # =========================================================================
    # Cache Status Operations
    # =========================================================================

    _signal_ticker_cache: Optional[Dict[str, Set[str]]] = None
    _signal_cache_loaded_at: Optional[float] = None

    def _load_signal_ticker_cache(self, force_reload: bool = False) -> Dict[str, Set[str]]:
        """
        Load and cache which tickers are present in each signal cache.

        Returns:
            Dict mapping signal name to set of tickers.
        """
        import time

        # Cache for 5 minutes
        if (
            not force_reload
            and self._signal_ticker_cache is not None
            and self._signal_cache_loaded_at is not None
            and time.time() - self._signal_cache_loaded_at < 300
        ):
            return self._signal_ticker_cache

        cache_dir = self.project_root / ".cache"
        signal_tickers: Dict[str, Set[str]] = {}

        try:
            import pandas as pd

            for f in cache_dir.glob("*.parquet"):
                if f.is_file() and not f.name.startswith("_"):
                    try:
                        df = pd.read_parquet(f)

                        # Check format: long format (timestamp, ticker, value) or wide format
                        if "ticker" in df.columns:
                            # Long format - get unique tickers from ticker column
                            tickers = set(df["ticker"].unique().tolist())
                        else:
                            # Wide format - columns are tickers
                            tickers = set(df.columns.tolist())
                            # Remove common non-ticker columns
                            for col in ["date", "Date", "timestamp", "index", "value"]:
                                tickers.discard(col)

                        signal_tickers[f.stem] = tickers
                    except Exception as e:
                        logger.warning(f"Failed to read signal cache {f.name}: {e}")
        except ImportError:
            logger.warning("pandas not available for signal cache loading")

        self._signal_ticker_cache = signal_tickers
        self._signal_cache_loaded_at = time.time()
        return signal_tickers

    def get_cache_status(self, tickers: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get cache status for symbols.

        Args:
            tickers: List of tickers to check. If None, check all symbols.

        Returns:
            Dict mapping ticker to cache status (has_price, signal_count, signals).
        """
        cache_dir = self.project_root / ".cache"
        prices_dir = cache_dir / "prices"

        # Load signal ticker cache
        signal_tickers = self._load_signal_ticker_cache()
        signal_names = sorted(signal_tickers.keys())

        # Get tickers to check
        if tickers is None:
            data = self._load_master()
            symbols = data.get("symbols", [])
            tickers = [s.get("ticker") for s in symbols if s.get("ticker")]

        result = {}
        for ticker in tickers:
            # Check price cache
            price_file = prices_dir / f"{ticker}.parquet"
            has_price = price_file.exists()

            # Check which signals have this ticker
            ticker_signals = [name for name, t_set in signal_tickers.items() if ticker in t_set]

            result[ticker] = {
                "has_price": has_price,
                "signal_count": len(ticker_signals),
                "total_signals": len(signal_names),
                "signals": ticker_signals,
            }

        return result

    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get cache summary statistics.

        Returns:
            Dict with cache statistics.
        """
        cache_dir = self.project_root / ".cache"
        prices_dir = cache_dir / "prices"

        # Count price cache files
        price_count = 0
        if prices_dir.exists():
            price_count = len(list(prices_dir.glob("*.parquet")))

        # Load signal ticker cache for detailed info
        signal_tickers = self._load_signal_ticker_cache()

        # Signal cache details
        signal_details = []
        for name in sorted(signal_tickers.keys()):
            signal_details.append({
                "name": name,
                "ticker_count": len(signal_tickers[name]),
            })

        # Get total symbols
        data = self._load_master()
        total_symbols = len(data.get("symbols", []))

        return {
            "total_symbols": total_symbols,
            "price_cache_count": price_count,
            "price_cache_coverage": round(price_count / total_symbols * 100, 1) if total_symbols > 0 else 0,
            "signal_caches": sorted(signal_tickers.keys()),
            "signal_details": signal_details,
            "total_signal_types": self._get_registered_signal_count(),  # 登録シグナル数
            "cached_signal_types": len(signal_tickers),  # キャッシュ済みシグナル数
        }

    def list_symbols_with_cache(
        self,
        page: int = 1,
        size: int = 50,
        search: Optional[str] = None,
        market: Optional[str] = None,
        asset_class: Optional[str] = None,
        tags: Optional[List[str]] = None,
        cache_filter: Optional[str] = None,
        signal_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List symbols with pagination, filters, and cache status.

        Args:
            page: Page number (1-indexed).
            size: Page size.
            search: Search string (matches ticker).
            market: Filter by market.
            asset_class: Filter by asset class.
            tags: Filter by tags (any match).
            cache_filter: "cached", "not_cached", or None for all (price cache).
            signal_filter: "has_signals", "no_signals", or None for all.

        Returns:
            Dict with items (including cache status), total, page info.
        """
        data = self._load_master()
        symbols = data.get("symbols", [])

        # Get cache status for all symbols
        cache_dir = self.project_root / ".cache"
        prices_dir = cache_dir / "prices"
        cached_tickers = set()
        if prices_dir.exists():
            for f in prices_dir.glob("*.parquet"):
                cached_tickers.add(f.stem)

        # Load signal ticker cache
        signal_tickers = self._load_signal_ticker_cache()
        registered_signal_count = self._get_registered_signal_count()
        cached_signal_count = len(signal_tickers)

        # Build a set of tickers that have signals
        tickers_with_signals: Dict[str, int] = {}
        for signal_name, ticker_set in signal_tickers.items():
            for ticker in ticker_set:
                tickers_with_signals[ticker] = tickers_with_signals.get(ticker, 0) + 1

        # Apply filters
        filtered = []
        for sym in symbols:
            ticker = sym.get("ticker", "")
            sym_taxonomy = sym.get("taxonomy", {})
            sym_tags = sym.get("tags", [])

            # Search filter
            if search and search.upper() not in ticker.upper():
                continue

            # Market filter
            if market and sym_taxonomy.get("market") != market:
                continue

            # Asset class filter
            if asset_class and sym_taxonomy.get("asset_class") != asset_class:
                continue

            # Tags filter (any match)
            if tags and not any(t in sym_tags for t in tags):
                continue

            # Cache filter (price)
            has_price_cache = ticker in cached_tickers
            if cache_filter == "cached" and not has_price_cache:
                continue
            if cache_filter == "not_cached" and has_price_cache:
                continue

            # Signal filter
            signal_count = tickers_with_signals.get(ticker, 0)
            has_signals = signal_count > 0
            if signal_filter == "has_signals" and not has_signals:
                continue
            if signal_filter == "no_signals" and has_signals:
                continue

            # Add cache status to symbol
            sym_with_cache = dict(sym)
            sym_with_cache["has_price_cache"] = has_price_cache
            sym_with_cache["signal_count"] = signal_count
            sym_with_cache["total_signal_types"] = registered_signal_count
            filtered.append(sym_with_cache)

        total = len(filtered)
        total_pages = (total + size - 1) // size if size > 0 else 1

        # Pagination
        start = (page - 1) * size
        end = start + size
        items = filtered[start:end]

        return {
            "items": items,
            "total": total,
            "page": page,
            "size": size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "cached_count": len(cached_tickers),
            "total_signal_types": registered_signal_count,
            "cached_signal_types": cached_signal_count,
        }


# Global singleton
_config_service: Optional[ConfigService] = None


def get_config_service(
    project_root: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> ConfigService:
    """
    Get ConfigService instance (singleton).

    Args:
        project_root: Project root path.
        config_dir: Configuration directory path.

    Returns:
        ConfigService instance.
    """
    global _config_service

    if _config_service is None:
        _config_service = ConfigService(
            project_root=project_root,
            config_dir=config_dir,
        )

    return _config_service
