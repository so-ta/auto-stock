#!/usr/bin/env python3
"""
シグナルキャッシュ構築スクリプト

価格データキャッシュを使ってシグナルを事前計算し、
バックテスト高速化のためのキャッシュを構築する。

Usage:
    python scripts/build_signal_cache.py
    python scripts/build_signal_cache.py --market japan
    python scripts/build_signal_cache.py --universe config/universe_sbi.yaml
    python scripts/build_signal_cache.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default settings
DATA_CACHE_DIR = Path("data/cache/sbi_universe")
SIGNAL_CACHE_DIR = Path(".cache/signals")
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"


def load_universe(config_path: Path) -> dict[str, list[str]]:
    """Load universe from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = {}
    for key in ["us_stocks", "japan_stocks", "etfs"]:
        if key in config:
            data = config[key]
            if isinstance(data, dict) and "symbols" in data:
                result[key] = data["symbols"]
            elif isinstance(data, list):
                result[key] = data

    return result


def load_price_cache(
    symbols: list[str],
    cache_dir: Path,
    market_name: str,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """Load price data from cache."""
    market_dir = cache_dir / market_name
    if not market_dir.exists():
        logger.warning(f"Cache directory not found: {market_dir}")
        return {}

    prices = {}
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    for symbol in symbols:
        safe_symbol = symbol.replace("/", "_").replace("=", "_").replace(".", "_")
        filepath = market_dir / f"{safe_symbol}.parquet"

        if not filepath.exists():
            continue

        try:
            df = pd.read_parquet(filepath)

            # Normalize column names
            if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")

            # Filter by date range
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Normalize column names
            col_map = {}
            for col in df.columns:
                col_str = str(col).lower()
                if "close" in col_str:
                    col_map[col] = "close"
                elif "open" in col_str:
                    col_map[col] = "open"
                elif "high" in col_str:
                    col_map[col] = "high"
                elif "low" in col_str:
                    col_map[col] = "low"
                elif "volume" in col_str:
                    col_map[col] = "volume"
            if col_map:
                df = df.rename(columns=col_map)

            if len(df) > 0 and "close" in df.columns:
                prices[symbol] = df

        except Exception as e:
            logger.debug(f"Failed to load {symbol}: {e}")

    return prices


def convert_prices_to_polars(prices: Dict[str, pd.DataFrame]) -> "pl.DataFrame":
    """Convert dict of price DataFrames to polars DataFrame with ticker column."""
    import polars as pl

    dfs = []
    for ticker, df in prices.items():
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")

        # Create polars dataframe
        pdf = df.reset_index()
        pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})

        # Normalize column names
        col_map = {}
        for col in pdf.columns:
            col_lower = str(col).lower()
            if col_lower == "timestamp":
                col_map[col] = "timestamp"
            elif "close" in col_lower:
                col_map[col] = "close"
            elif "open" in col_lower:
                col_map[col] = "open"
            elif "high" in col_lower:
                col_map[col] = "high"
            elif "low" in col_lower:
                col_map[col] = "low"
            elif "volume" in col_lower:
                col_map[col] = "volume"
        pdf = pdf.rename(columns=col_map)

        # Add ticker column
        pdf["ticker"] = ticker

        # Select required columns
        required = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
        available = [c for c in required if c in pdf.columns]
        pdf = pdf[available]

        # Convert to polars and cast types for consistency
        pldf = pl.from_pandas(pdf)
        pldf = pldf.with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ])
        dfs.append(pldf)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs)


def build_signal_cache(
    prices: Dict[str, pd.DataFrame],
    signal_cache_dir: Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Build signal cache using SignalPrecomputer."""
    import polars as pl
    from src.backtest.signal_precompute import SignalPrecomputer

    signal_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building signal cache for {len(prices)} symbols...")
    logger.info(f"Signal cache directory: {signal_cache_dir}")

    # Convert prices dict to polars DataFrame
    logger.info("Converting price data to polars format...")
    prices_pl = convert_prices_to_polars(prices)

    if len(prices_pl) == 0:
        return {"success": False, "error": "No price data to process"}

    logger.info(f"Converted to polars DataFrame: {len(prices_pl)} rows, {prices_pl['ticker'].n_unique()} tickers")

    precomputer = SignalPrecomputer(cache_dir=str(signal_cache_dir))

    # Build config for signal generation
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "momentum_periods": [20, 60, 120, 252],
        "volatility_periods": [20, 60],
        "rsi_periods": [14],
        "zscore_periods": [20, 60],
        "sharpe_periods": [60, 252],
    }

    # Precompute all signals
    try:
        precomputer.precompute_all(prices_pl, config)
        logger.info("Signal precomputation completed")

        # Calculate stats from files
        parquet_files = list(signal_cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        stats = {
            "total_entries": len(parquet_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
        return {
            "success": True,
            "symbols": list(prices.keys()),
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Signal precomputation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


def verify_signal_cache(signal_cache_dir: Path) -> Dict[str, Any]:
    """Verify signal cache."""
    if not signal_cache_dir.exists():
        return {"exists": False, "error": "Cache directory not found"}

    parquet_files = list(signal_cache_dir.glob("*.parquet"))
    json_files = list(signal_cache_dir.glob("*.json"))

    total_size = sum(f.stat().st_size for f in parquet_files)

    # Check metadata
    metadata_path = signal_cache_dir / "precompute_metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")

    return {
        "exists": True,
        "parquet_files": len(parquet_files),
        "json_files": len(json_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Build signal cache")
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("config/universe_sbi.yaml"),
        help="Universe YAML file",
    )
    parser.add_argument(
        "--market",
        choices=["japan", "us", "etf", "all"],
        default="all",
        help="Market to process (default: all)",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=DATA_CACHE_DIR,
        help=f"Data cache directory (default: {DATA_CACHE_DIR})",
    )
    parser.add_argument(
        "--signal-cache",
        type=Path,
        default=SIGNAL_CACHE_DIR,
        help=f"Signal cache directory (default: {SIGNAL_CACHE_DIR})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=START_DATE,
        help=f"Start date (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=END_DATE,
        help=f"End date (default: {END_DATE})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify signal cache only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of symbols",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Signal Cache Builder")
    print("=" * 60)

    if args.verify:
        print(f"\nVerifying signal cache: {args.signal_cache}")
        result = verify_signal_cache(args.signal_cache)

        if not result["exists"]:
            print(f"  Error: {result.get('error', 'Unknown error')}")
            return

        print(f"  Parquet files: {result['parquet_files']}")
        print(f"  JSON files: {result['json_files']}")
        print(f"  Total size: {result['total_size_mb']} MB")

        if result["metadata"]:
            print(f"  Version: {result['metadata'].get('version', 'unknown')}")
            print(f"  Generated: {result['metadata'].get('generated_at', 'unknown')}")
            print(f"  Symbols: {result['metadata'].get('symbol_count', 'unknown')}")

        print("\n" + "=" * 60)
        print("Done!")
        return

    # Load universe
    print(f"\nLoading universe: {args.universe}")
    universe = load_universe(args.universe)

    # Determine markets to process
    market_map = {
        "japan": [("japan_stocks", "japan")],
        "us": [("us_stocks", "us")],
        "etf": [("etfs", "etf")],
        "all": [
            ("japan_stocks", "japan"),
            ("etfs", "etf"),
            ("us_stocks", "us"),
        ],
    }

    all_prices = {}

    for key, market_name in market_map[args.market]:
        if key not in universe:
            continue

        symbols = universe[key]
        if args.limit:
            symbols = symbols[:args.limit]

        print(f"\nLoading {market_name} prices ({len(symbols)} symbols)...")
        prices = load_price_cache(
            symbols,
            args.data_cache,
            market_name,
            args.start,
            args.end,
        )
        print(f"  Loaded: {len(prices)} symbols")
        all_prices.update(prices)

    if not all_prices:
        print("\nNo price data loaded. Please run build_sbi_cache.py first.")
        return

    print(f"\nTotal symbols with price data: {len(all_prices)}")

    # Build signal cache
    print("\nBuilding signal cache...")
    result = build_signal_cache(
        all_prices,
        args.signal_cache,
        args.start,
        args.end,
    )

    if result["success"]:
        print("\nSignal cache built successfully!")
        if "stats" in result:
            stats = result["stats"]
            print(f"  Cache entries: {stats.get('total_entries', 'unknown')}")
            print(f"  Total size: {stats.get('total_size_mb', 'unknown')} MB")
    else:
        print(f"\nSignal cache build failed: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
