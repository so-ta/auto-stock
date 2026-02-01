#!/usr/bin/env python3
"""
Build Asset Master v3.0 - Single Source of Truth for All Assets

Collects symbol information from existing universe files and generates
a unified asset_master.yaml with taxonomy and subset definitions.

Usage:
    python scripts/build_asset_master.py

This script is intended to be run once during migration.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_existing_asset_master() -> dict:
    """Load existing asset_master.yaml for name metadata."""
    path = CONFIG_DIR / "asset_master.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_sbi_universe() -> dict:
    """Load universe_sbi.yaml for comprehensive symbol list."""
    path = CONFIG_DIR / "universe_sbi.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_standard_universe() -> dict:
    """Load universe_standard.yaml for quality-filtered symbols."""
    path = CONFIG_DIR / "universe_standard.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_japan_all() -> list[str]:
    """Load universe_japan_all.yaml for Japan symbols."""
    path = CONFIG_DIR / "universe_japan_all.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("tickers", [])


def determine_asset_class(ticker: str, name_info: dict) -> str:
    """Determine asset class from ticker and metadata."""
    # Check subcategory from asset_master
    subcat = name_info.get("subcategory", "")
    if subcat:
        if subcat in ["equity_us", "sector"]:
            return "etf"
        if subcat in ["bond", "commodity", "international"]:
            return "etf"

    # ETF detection by ticker pattern
    etf_tickers = {
        "SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "IJH", "IJR",
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "TLT", "IEF", "BND", "AGG", "LQD", "HYG", "TIP", "SHY",
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "IAU", "GDX",
        "EFA", "EEM", "VWO", "VEA", "ACWI", "VT", "EWJ",
        "VNQ", "IYR", "SCHH", "MORT", "REM", "VNQI",
        "ARKK", "ARKG", "ARKF", "ARKQ", "ARKW",
        "VIXY", "VXX", "SQQQ", "TQQQ", "SPXU", "UPRO",
    }
    if ticker in etf_tickers:
        return "etf"

    # Commodity futures
    if "=F" in ticker:
        return "commodity"

    # Forex
    if "=X" in ticker:
        return "forex"

    # Crypto
    if "-USD" in ticker and ticker not in ["EURUSD", "GBPUSD"]:
        return "crypto"

    return "equity"


def determine_market(ticker: str) -> str:
    """Determine market from ticker suffix."""
    if ticker.endswith(".T"):
        return "japan"
    if ticker.endswith(".HK"):
        return "hong_kong"
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "korea"
    if "=X" in ticker or "-USD" in ticker:
        return "global"
    return "us"


def determine_sector(ticker: str, name_info: dict) -> str | None:
    """Determine sector from metadata. Returns None for non-equities."""
    # Sector ETFs
    sector_etfs = {
        "XLF": "financials", "XLK": "technology", "XLE": "energy",
        "XLV": "healthcare", "XLI": "industrials", "XLY": "consumer_discretionary",
        "XLP": "consumer_staples", "XLU": "utilities", "XLB": "materials",
        "XLRE": "real_estate", "XLC": "communication_services",
    }
    if ticker in sector_etfs:
        return sector_etfs[ticker]

    # Major US stocks - sector hints
    us_sectors = {
        # Technology
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "GOOG": "technology", "META": "technology", "NVDA": "technology",
        "AMD": "technology", "INTC": "technology", "CSCO": "technology",
        "ORCL": "technology", "CRM": "technology", "ADBE": "technology",
        "AVGO": "technology", "NFLX": "technology",
        # Healthcare
        "JNJ": "healthcare", "UNH": "healthcare", "MRK": "healthcare",
        "ABBV": "healthcare", "LLY": "healthcare", "TMO": "healthcare",
        "ABT": "healthcare",
        # Financials
        "JPM": "financials", "V": "financials", "MA": "financials",
        "BAC": "financials", "WFC": "financials", "GS": "financials",
        "MS": "financials", "C": "financials", "BRK-B": "financials",
        # Consumer Discretionary
        "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
        "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
        "DIS": "consumer_discretionary", "BA": "consumer_discretionary",
        # Consumer Staples
        "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
        "COST": "consumer_staples", "WMT": "consumer_staples",
        # Energy
        "XOM": "energy", "CVX": "energy",
    }
    return us_sectors.get(ticker)


def determine_etf_category(ticker: str, name_info: dict) -> str | None:
    """Determine ETF category from metadata."""
    subcat = name_info.get("subcategory", "")
    if subcat:
        return subcat

    # Infer from ticker
    if ticker in ["SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "IJH", "IJR"]:
        return "index"
    if ticker in ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]:
        return "sector"
    if ticker in ["TLT", "IEF", "BND", "AGG", "LQD", "HYG", "TIP", "SHY"]:
        return "bond"
    if ticker in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "IAU", "GDX"]:
        return "commodity"
    if ticker in ["EFA", "EEM", "VWO", "VEA", "ACWI", "VT", "EWJ"]:
        return "international"
    if ticker in ["VIXY", "VXX"]:
        return "volatility"
    if ticker in ["SQQQ", "TQQQ", "SPXU", "UPRO", "FAS", "FAZ", "SOXL", "SOXS"]:
        return "leveraged"

    return None


def build_symbol_entry(
    ticker: str,
    name_lookup: dict[str, dict],
    standard_tickers: set[str],
    sbi_tickers: set[str],
) -> dict:
    """Build a single symbol entry for the asset master."""
    name_info = name_lookup.get(ticker, {})

    market = determine_market(ticker)
    asset_class = determine_asset_class(ticker, name_info)
    sector = determine_sector(ticker, name_info)
    etf_category = determine_etf_category(ticker, name_info)

    # Build taxonomy
    taxonomy = {
        "market": market,
        "asset_class": asset_class,
    }
    if sector:
        taxonomy["sector"] = sector
    if etf_category:
        taxonomy["etf_category"] = etf_category

    # Build tags
    tags = []
    if ticker in sbi_tickers:
        tags.append("sbi")
    if ticker in standard_tickers:
        tags.append("quality")

    # Add index membership tags
    sp500_sample = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "V"}
    if ticker in sp500_sample:
        tags.append("sp500")

    entry = {
        "ticker": ticker,
        "taxonomy": taxonomy,
    }
    if tags:
        entry["tags"] = tags

    return entry


def build_name_lookup(existing_master: dict) -> dict[str, dict]:
    """Build ticker -> name info lookup from existing asset_master."""
    lookup = {}
    assets = existing_master.get("assets", {})

    for category, symbols in assets.items():
        if not isinstance(symbols, dict):
            continue
        for ticker, info in symbols.items():
            if isinstance(info, dict):
                lookup[ticker] = info

    return lookup


def collect_all_tickers(
    sbi_data: dict,
    standard_data: dict,
    japan_tickers: list[str],
) -> set[str]:
    """Collect all unique tickers from source files."""
    tickers = set()

    # From SBI
    for section in ["us_stocks", "japan_stocks", "hong_kong_stocks", "korea_stocks", "etfs"]:
        if section in sbi_data:
            section_data = sbi_data[section]
            if isinstance(section_data, dict) and "symbols" in section_data:
                tickers.update(section_data["symbols"])

    # From standard (_legacy section)
    legacy = standard_data.get("_legacy", {})
    for section in ["us_stocks", "japan_stocks", "etfs"]:
        if section in legacy:
            tickers.update(legacy[section])

    # From Japan all
    tickers.update(japan_tickers)

    return tickers


def get_standard_tickers(standard_data: dict) -> set[str]:
    """Get quality-filtered tickers from standard universe."""
    tickers = set()
    legacy = standard_data.get("_legacy", {})
    for section in ["us_stocks", "japan_stocks", "etfs"]:
        if section in legacy:
            tickers.update(legacy[section])
    return tickers


def get_sbi_tickers(sbi_data: dict) -> set[str]:
    """Get SBI tradeable tickers."""
    tickers = set()
    for section in ["us_stocks", "japan_stocks", "hong_kong_stocks", "korea_stocks", "etfs"]:
        if section in sbi_data:
            section_data = sbi_data[section]
            if isinstance(section_data, dict) and "symbols" in section_data:
                tickers.update(section_data["symbols"])
    return tickers


def build_asset_master() -> dict:
    """Build the complete asset master v3.0."""
    print("Loading existing files...")

    existing_master = load_existing_asset_master()
    sbi_data = load_sbi_universe()
    standard_data = load_standard_universe()
    japan_tickers = load_japan_all()

    name_lookup = build_name_lookup(existing_master)
    all_tickers = collect_all_tickers(sbi_data, standard_data, japan_tickers)
    standard_tickers = get_standard_tickers(standard_data)
    sbi_tickers = get_sbi_tickers(sbi_data)

    print(f"Found {len(all_tickers)} unique tickers")
    print(f"  - Standard (quality-filtered): {len(standard_tickers)}")
    print(f"  - SBI tradeable: {len(sbi_tickers)}")

    # Build taxonomy definitions
    taxonomy = {
        "market": {
            "us": "US Markets",
            "japan": "Japanese Markets",
            "hong_kong": "Hong Kong Markets",
            "korea": "Korean Markets",
            "global": "Global/Cross-border",
        },
        "asset_class": {
            "equity": "Individual Stocks",
            "etf": "Exchange Traded Funds",
            "bond": "Bonds",
            "commodity": "Commodities",
            "forex": "Foreign Exchange",
            "crypto": "Cryptocurrencies",
        },
        "sector": {
            "technology": "Technology",
            "healthcare": "Healthcare",
            "financials": "Financials",
            "industrials": "Industrials",
            "consumer_discretionary": "Consumer Discretionary",
            "consumer_staples": "Consumer Staples",
            "energy": "Energy",
            "materials": "Materials",
            "utilities": "Utilities",
            "real_estate": "Real Estate",
            "communication_services": "Communication Services",
        },
        "etf_category": {
            "index": "Index Tracking",
            "sector": "Sector ETFs",
            "bond": "Bond ETFs",
            "commodity": "Commodity ETFs",
            "international": "International ETFs",
            "real_estate": "Real Estate ETFs",
            "thematic": "Thematic ETFs",
            "volatility": "Volatility ETFs",
            "leveraged": "Leveraged/Inverse ETFs",
        },
    }

    # Build symbols list (only standard universe tickers for now to keep file manageable)
    # Use standard_tickers which are quality-filtered
    print("Building symbol entries...")
    symbols = []
    for ticker in sorted(standard_tickers):
        entry = build_symbol_entry(
            ticker,
            name_lookup,
            standard_tickers,
            sbi_tickers,
        )
        symbols.append(entry)

    print(f"Built {len(symbols)} symbol entries")

    # Build subset definitions
    subsets = {
        "standard": {
            "name": "Standard Universe",
            "description": "Quality-filtered subset for backtesting",
            "filters": {
                "tags": ["quality"],
            },
        },
        "sbi": {
            "name": "SBI Universe",
            "description": "SBI Securities tradeable symbols",
            "filters": {
                "tags": ["sbi"],
            },
        },
        "japan": {
            "name": "Japan Universe",
            "description": "Japanese market stocks",
            "filters": {
                "taxonomy": {
                    "market": ["japan"],
                },
            },
        },
        "us_equity": {
            "name": "US Equities",
            "description": "US individual stocks",
            "filters": {
                "taxonomy": {
                    "market": ["us"],
                    "asset_class": ["equity"],
                },
            },
        },
        "etf_only": {
            "name": "ETFs Only",
            "description": "All ETFs",
            "filters": {
                "taxonomy": {
                    "asset_class": ["etf"],
                },
            },
        },
    }

    # Build final document
    master = {
        "version": "3.0",
        "name": "Asset Master",
        "description": "Single source of truth for all tradeable assets",
        "generated_at": datetime.now().isoformat(),
        "taxonomy": taxonomy,
        "symbols": symbols,
        "subsets": subsets,
    }

    return master


def main():
    """Main entry point."""
    master = build_asset_master()

    output_path = CONFIG_DIR / "asset_master_v3.yaml"
    print(f"Writing to {output_path}...")

    with open(output_path, "w") as f:
        yaml.dump(
            master,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    print("Done!")
    print(f"  Total symbols: {len(master['symbols'])}")
    print(f"  Subsets defined: {list(master['subsets'].keys())}")


if __name__ == "__main__":
    main()
