#!/usr/bin/env python3
"""
Import Symbols from S3 Price Cache

Extracts ticker symbols from S3 price cache files and imports them into asset_master.yaml.

Usage:
    python scripts/import_symbols_from_s3.py
    python scripts/import_symbols_from_s3.py --dry-run
    python scripts/import_symbols_from_s3.py --preserve-tags
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import boto3
import yaml


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# S3 Configuration
S3_BUCKET = "stock-local-dev-014498665038"
S3_PREFIX = ".cache/prices/"


def get_s3_price_files() -> list[str]:
    """List all price cache files from S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Extract filename from key
            filename = key.replace(S3_PREFIX, "")
            if filename.endswith(".parquet"):
                files.append(filename)

    return files


def extract_ticker_from_filename(filename: str) -> str | None:
    """Extract ticker from filename (e.g., 1301.T.parquet → 1301.T)."""
    if filename.endswith(".parquet"):
        return filename[:-8]  # Remove .parquet
    return None


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


def determine_asset_class(ticker: str) -> str:
    """Determine asset class from ticker pattern."""
    # Commodity futures
    if "=F" in ticker:
        return "commodity"

    # Forex
    if "=X" in ticker:
        return "forex"

    # Crypto
    crypto_suffixes = ["-USD", "-BTC", "-ETH"]
    for suffix in crypto_suffixes:
        if suffix in ticker and ticker not in ["EURUSD", "GBPUSD"]:
            return "crypto"

    # Known ETF tickers (comprehensive list)
    etf_tickers = {
        # Index ETFs
        "SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "IJH", "IJR", "IVV",
        "VUG", "VTV", "VIG", "VYM", "SCHD", "SCHX", "SCHB", "SCHA",
        "RSP", "SPLG", "SPYG", "SPYV", "MDY", "SLY",
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "VGT", "VHT", "VFH", "VDE", "VIS", "VCR", "VDC", "VPU", "VAW", "VNQ",
        # Bond ETFs
        "TLT", "IEF", "BND", "AGG", "LQD", "HYG", "TIP", "SHY", "SHV", "GOVT",
        "VCIT", "VCSH", "VGIT", "VGSH", "MUB", "JNK", "EMB",
        # Commodity ETFs
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "IAU", "GDX", "GDXJ",
        "PPLT", "PALL", "WEAT", "CORN", "SOYB",
        # International ETFs
        "EFA", "EEM", "VWO", "VEA", "ACWI", "VT", "EWJ", "EWG", "EWU", "EWZ",
        "IEFA", "IEMG", "VGK", "VPL", "VXUS", "IXUS",
        # Real Estate ETFs
        "IYR", "SCHH", "MORT", "REM", "VNQI", "REET",
        # Thematic ETFs
        "ARKK", "ARKG", "ARKF", "ARKQ", "ARKW", "ARKX",
        "BOTZ", "ROBO", "LIT", "TAN", "ICLN", "PBW", "QCLN",
        # Volatility ETFs
        "VIXY", "VXX", "SVXY", "UVXY",
        # Leveraged ETFs
        "SQQQ", "TQQQ", "SPXU", "UPRO", "SPXL", "SPXS",
        "FAS", "FAZ", "SOXL", "SOXS", "LABU", "LABD",
        "TNA", "TZA", "UDOW", "SDOW", "NUGT", "DUST",
    }
    if ticker in etf_tickers:
        return "etf"

    # Detect potential ETFs by pattern (3-4 letter tickers that might be ETFs)
    # Only mark as equity by default - ETFs should be in the explicit list

    return "equity"


def determine_sector(ticker: str) -> str | None:
    """Determine sector for known US stocks and sector ETFs."""
    # Sector ETFs
    sector_etfs = {
        "XLF": "financials", "XLK": "technology", "XLE": "energy",
        "XLV": "healthcare", "XLI": "industrials", "XLY": "consumer_discretionary",
        "XLP": "consumer_staples", "XLU": "utilities", "XLB": "materials",
        "XLRE": "real_estate", "XLC": "communication_services",
        "VGT": "technology", "VHT": "healthcare", "VFH": "financials",
        "VDE": "energy", "VIS": "industrials", "VCR": "consumer_discretionary",
        "VDC": "consumer_staples", "VPU": "utilities", "VAW": "materials",
    }
    if ticker in sector_etfs:
        return sector_etfs[ticker]

    # Major US stocks by sector
    us_sectors = {
        # Technology
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "GOOG": "technology", "META": "technology", "NVDA": "technology",
        "AMD": "technology", "INTC": "technology", "CSCO": "technology",
        "ORCL": "technology", "CRM": "technology", "ADBE": "technology",
        "AVGO": "technology", "NFLX": "technology", "NOW": "technology",
        "INTU": "technology", "IBM": "technology", "AMAT": "technology",
        "MU": "technology", "LRCX": "technology", "KLAC": "technology",
        # Healthcare
        "JNJ": "healthcare", "UNH": "healthcare", "MRK": "healthcare",
        "ABBV": "healthcare", "LLY": "healthcare", "TMO": "healthcare",
        "ABT": "healthcare", "PFE": "healthcare", "DHR": "healthcare",
        "BMY": "healthcare", "AMGN": "healthcare", "GILD": "healthcare",
        # Financials
        "JPM": "financials", "V": "financials", "MA": "financials",
        "BAC": "financials", "WFC": "financials", "GS": "financials",
        "MS": "financials", "C": "financials", "BRK-B": "financials",
        "AXP": "financials", "SCHW": "financials", "BLK": "financials",
        # Consumer Discretionary
        "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
        "HD": "consumer_discretionary", "MCD": "consumer_discretionary",
        "DIS": "consumer_discretionary", "NKE": "consumer_discretionary",
        "SBUX": "consumer_discretionary", "LOW": "consumer_discretionary",
        # Consumer Staples
        "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
        "COST": "consumer_staples", "WMT": "consumer_staples", "PM": "consumer_staples",
        "MO": "consumer_staples", "CL": "consumer_staples",
        # Energy
        "XOM": "energy", "CVX": "energy", "COP": "energy",
        "SLB": "energy", "EOG": "energy", "PXD": "energy",
        # Industrials
        "BA": "industrials", "HON": "industrials", "UPS": "industrials",
        "UNP": "industrials", "CAT": "industrials", "RTX": "industrials",
        "DE": "industrials", "LMT": "industrials", "GE": "industrials",
        # Utilities
        "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
        "D": "utilities", "AEP": "utilities",
        # Materials
        "LIN": "materials", "APD": "materials", "SHW": "materials",
        "ECL": "materials", "NEM": "materials",
        # Real Estate
        "AMT": "real_estate", "PLD": "real_estate", "CCI": "real_estate",
        "EQIX": "real_estate", "PSA": "real_estate",
        # Communication Services
        "T": "communication_services", "VZ": "communication_services",
        "TMUS": "communication_services", "CMCSA": "communication_services",
        "CHTR": "communication_services",
    }
    return us_sectors.get(ticker)


def determine_etf_category(ticker: str) -> str | None:
    """Determine ETF category from ticker."""
    # Index ETFs
    index_etfs = {
        "SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "IJH", "IJR", "IVV",
        "VUG", "VTV", "VIG", "VYM", "SCHD", "SCHX", "SCHB", "SCHA",
        "RSP", "SPLG", "SPYG", "SPYV", "MDY", "SLY",
    }
    if ticker in index_etfs:
        return "index"

    # Sector ETFs
    sector_etfs = {
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "VGT", "VHT", "VFH", "VDE", "VIS", "VCR", "VDC", "VPU", "VAW",
    }
    if ticker in sector_etfs:
        return "sector"

    # Bond ETFs
    bond_etfs = {
        "TLT", "IEF", "BND", "AGG", "LQD", "HYG", "TIP", "SHY", "SHV", "GOVT",
        "VCIT", "VCSH", "VGIT", "VGSH", "MUB", "JNK", "EMB",
    }
    if ticker in bond_etfs:
        return "bond"

    # Commodity ETFs
    commodity_etfs = {
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "IAU", "GDX", "GDXJ",
        "PPLT", "PALL", "WEAT", "CORN", "SOYB",
    }
    if ticker in commodity_etfs:
        return "commodity"

    # International ETFs
    international_etfs = {
        "EFA", "EEM", "VWO", "VEA", "ACWI", "VT", "EWJ", "EWG", "EWU", "EWZ",
        "IEFA", "IEMG", "VGK", "VPL", "VXUS", "IXUS",
    }
    if ticker in international_etfs:
        return "international"

    # Real Estate ETFs
    realestate_etfs = {"VNQ", "IYR", "SCHH", "MORT", "REM", "VNQI", "REET"}
    if ticker in realestate_etfs:
        return "real_estate"

    # Thematic ETFs
    thematic_etfs = {
        "ARKK", "ARKG", "ARKF", "ARKQ", "ARKW", "ARKX",
        "BOTZ", "ROBO", "LIT", "TAN", "ICLN", "PBW", "QCLN",
    }
    if ticker in thematic_etfs:
        return "thematic"

    # Volatility ETFs
    volatility_etfs = {"VIXY", "VXX", "SVXY", "UVXY"}
    if ticker in volatility_etfs:
        return "volatility"

    # Leveraged ETFs
    leveraged_etfs = {
        "SQQQ", "TQQQ", "SPXU", "UPRO", "SPXL", "SPXS",
        "FAS", "FAZ", "SOXL", "SOXS", "LABU", "LABD",
        "TNA", "TZA", "UDOW", "SDOW", "NUGT", "DUST",
    }
    if ticker in leveraged_etfs:
        return "leveraged"

    return None


def load_existing_asset_master() -> dict:
    """Load existing asset_master.yaml."""
    path = CONFIG_DIR / "asset_master.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_existing_ticker_lookup(master: dict) -> dict[str, dict]:
    """Build a lookup of existing ticker info (tags, taxonomy details)."""
    lookup = {}
    for symbol in master.get("symbols", []):
        ticker = symbol.get("ticker")
        if ticker:
            lookup[ticker] = symbol
    return lookup


def build_symbol_entry(
    ticker: str,
    existing_lookup: dict[str, dict],
    preserve_tags: bool,
) -> dict:
    """Build a single symbol entry."""
    market = determine_market(ticker)
    asset_class = determine_asset_class(ticker)
    sector = determine_sector(ticker)
    etf_category = determine_etf_category(ticker)

    # Build taxonomy
    taxonomy = {
        "market": market,
        "asset_class": asset_class,
    }
    if sector:
        taxonomy["sector"] = sector
    if etf_category:
        taxonomy["etf_category"] = etf_category

    entry = {
        "ticker": ticker,
        "taxonomy": taxonomy,
    }

    # Preserve existing tags if requested and available
    if preserve_tags and ticker in existing_lookup:
        existing = existing_lookup[ticker]
        if "tags" in existing:
            entry["tags"] = existing["tags"]

    return entry


def import_symbols(dry_run: bool = False, preserve_tags: bool = False):
    """Import symbols from S3 price cache into asset_master.yaml."""
    print("Loading existing asset_master.yaml...")
    existing_master = load_existing_asset_master()
    existing_lookup = build_existing_ticker_lookup(existing_master)

    print(f"Existing symbols: {len(existing_lookup)}")

    print("Fetching price cache files from S3...")
    files = get_s3_price_files()
    print(f"Found {len(files)} files in S3 price cache")

    # Extract tickers
    tickers = set()
    for f in files:
        ticker = extract_ticker_from_filename(f)
        if ticker:
            tickers.add(ticker)

    print(f"Extracted {len(tickers)} unique tickers")

    # Count by market
    market_counts = {}
    for ticker in tickers:
        market = determine_market(ticker)
        market_counts[market] = market_counts.get(market, 0) + 1

    print("\nTickers by market:")
    for market, count in sorted(market_counts.items()):
        print(f"  {market}: {count}")

    if dry_run:
        print("\n[DRY RUN] Would import {} symbols".format(len(tickers)))
        return

    # Build new symbol entries
    print("\nBuilding symbol entries...")
    symbols = []
    for ticker in sorted(tickers):
        entry = build_symbol_entry(ticker, existing_lookup, preserve_tags)
        symbols.append(entry)

    # Preserve existing structure
    master = {
        "version": existing_master.get("version", "3.0"),
        "name": existing_master.get("name", "Asset Master"),
        "description": existing_master.get("description", "Single source of truth for all tradeable assets"),
        "generated_at": datetime.now().isoformat(),
        "taxonomy": existing_master.get("taxonomy", {
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
        }),
        "symbols": symbols,
        "subsets": existing_master.get("subsets", {
            "standard": {
                "name": "Standard Universe",
                "description": "Quality-filtered subset for backtesting",
                "filters": {"tags": ["quality"]},
            },
            "sbi": {
                "name": "SBI Universe",
                "description": "SBI Securities tradeable symbols",
                "filters": {"tags": ["sbi"]},
            },
            "japan": {
                "name": "Japan Universe",
                "description": "Japanese market stocks",
                "filters": {"taxonomy": {"market": ["japan"]}},
            },
            "us_equity": {
                "name": "US Equities",
                "description": "US individual stocks",
                "filters": {"taxonomy": {"market": ["us"], "asset_class": ["equity"]}},
            },
            "etf_only": {
                "name": "ETFs Only",
                "description": "All ETFs",
                "filters": {"taxonomy": {"asset_class": ["etf"]}},
            },
        }),
    }

    # Write output
    output_path = CONFIG_DIR / "asset_master.yaml"
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

    print("\nDone!")
    print(f"  Total symbols: {len(symbols)}")
    print(f"  New symbols: {len(tickers - set(existing_lookup.keys()))}")
    print(f"  Preserved from existing: {len(tickers & set(existing_lookup.keys()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Import symbols from S3 price cache into asset_master.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be imported",
    )
    parser.add_argument(
        "--preserve-tags",
        action="store_true",
        help="Preserve existing tags for symbols that already exist",
    )
    args = parser.parse_args()

    import_symbols(dry_run=args.dry_run, preserve_tags=args.preserve_tags)


if __name__ == "__main__":
    main()
