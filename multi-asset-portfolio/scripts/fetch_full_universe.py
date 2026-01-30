#!/usr/bin/env python3
"""
Full Universe Data Fetcher - task_020_4
905銘柄のデータ取得スクリプト
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.batch_fetcher import BatchDataFetcher, BatchFetcherConfig


def load_tickers_from_universe(universe_path: Path) -> list[str]:
    """universe_full.yaml から銘柄リストを取得"""
    with open(universe_path, "r") as f:
        config = yaml.safe_load(f)

    universe = config.get("universe", config)
    tickers = []

    # US Stocks
    us_stocks = universe.get("us_stocks", {})
    if us_stocks.get("enabled", False):
        tickers.extend(us_stocks.get("tickers", []))
        print(f"  US Stocks: {len(us_stocks.get('tickers', []))} tickers")

    # Japan Stocks (.T suffix)
    japan_stocks = universe.get("japan_stocks", {})
    if japan_stocks.get("enabled", False):
        suffix = japan_stocks.get("suffix", ".T")
        jp_tickers = [f"{code}{suffix}" for code in japan_stocks.get("tickers", [])]
        tickers.extend(jp_tickers)
        print(f"  Japan Stocks: {len(jp_tickers)} tickers")

    # ETFs
    etfs = universe.get("etfs", {})
    if etfs.get("enabled", False):
        etf_count = 0
        for category_name, category_tickers in etfs.get("categories", {}).items():
            tickers.extend(category_tickers)
            etf_count += len(category_tickers)
        print(f"  ETFs: {etf_count} tickers")

    # Forex
    forex = universe.get("forex", {})
    if forex.get("enabled", False):
        forex_count = 0
        for pair_type, pairs in forex.get("pairs", {}).items():
            tickers.extend(pairs)
            forex_count += len(pairs)
        print(f"  Forex: {forex_count} pairs")

    # Crypto (disabled by default)
    crypto = universe.get("crypto", {})
    if crypto.get("enabled", False):
        tickers.extend(crypto.get("tickers", []))
        print(f"  Crypto: {len(crypto.get('tickers', []))} tickers")

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    return unique_tickers


def main():
    print("=" * 60)
    print("Full Universe Data Fetcher - task_020_4")
    print("=" * 60)
    print()

    start_time = time.time()

    # Paths
    universe_path = project_root / "config" / "universe_full.yaml"
    cache_dir = project_root / "cache" / "price_data"
    results_dir = project_root / "results"

    # Ensure directories exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load tickers
    print("Loading tickers from universe_full.yaml...")
    tickers = load_tickers_from_universe(universe_path)
    print(f"\nTotal unique tickers: {len(tickers)}")
    print()

    # Configure fetcher
    config = BatchFetcherConfig(
        max_concurrent=10,
        rate_limit_per_sec=2.0,
        retry_count=3,
        cache_dir=str(cache_dir),
        cache_max_age_days=1,  # Use cache for today's data
    )

    fetcher = BatchDataFetcher.from_config(config)

    # Date range
    start_date = "2010-01-01"
    end_date = "2024-12-31"

    print(f"Fetching data from {start_date} to {end_date}")
    print(f"Cache directory: {cache_dir}")
    print(f"Max concurrent: {config.max_concurrent}")
    print(f"Rate limit: {config.rate_limit_per_sec}/sec")
    print()

    # Progress callback
    progress_data = {"last_report": 0}

    def progress_callback(completed: int, total: int, ticker: str):
        # Report every 50 tickers or at key milestones
        if completed - progress_data["last_report"] >= 50 or completed == total:
            elapsed = time.time() - start_time
            pct = (completed / total) * 100
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0

            print(
                f"Progress: {completed}/{total} ({pct:.1f}%) - "
                f"Rate: {rate:.1f}/sec - ETA: {eta/60:.1f}min - "
                f"Last: {ticker}"
            )
            progress_data["last_report"] = completed

    # Fetch all data
    print("Starting data fetch...")
    print("-" * 60)

    import asyncio

    result = asyncio.run(
        fetcher.fetch_all(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )
    )

    print("-" * 60)
    print()

    # Calculate results
    elapsed_time = time.time() - start_time
    success_count = result.success_count
    failed_count = result.failed_count
    cached_count = result.cached_count

    # Get failed tickers
    failed_tickers = result.failed_tickers

    # Calculate cache size
    cache_size_mb = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            cache_size_mb += f.stat().st_size / (1024 * 1024)

    # Print summary
    print("=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Total tickers:    {len(tickers)}")
    print(f"Successfully fetched: {success_count}")
    print(f"From cache:       {cached_count}")
    print(f"Failed:           {failed_count}")
    print(f"Success rate:     {(success_count/len(tickers)*100):.1f}%")
    print(f"Elapsed time:     {elapsed_time/60:.1f} minutes")
    print(f"Cache size:       {cache_size_mb:.1f} MB")
    print()

    if failed_tickers:
        print(f"Failed tickers ({len(failed_tickers)}):")
        for t in failed_tickers[:20]:
            print(f"  - {t}")
        if len(failed_tickers) > 20:
            print(f"  ... and {len(failed_tickers) - 20} more")
        print()

    # Generate report
    report = {
        "task_id": "task_020_4",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "max_concurrent": config.max_concurrent,
            "rate_limit_per_sec": config.rate_limit_per_sec,
            "retry_count": config.retry_count,
        },
        "results": {
            "total_tickers": len(tickers),
            "success_count": success_count,
            "cached_count": cached_count,
            "failed_count": failed_count,
            "success_rate_pct": round(success_count / len(tickers) * 100, 2),
            "elapsed_time_sec": round(elapsed_time, 2),
            "cache_size_mb": round(cache_size_mb, 2),
        },
        "failed_tickers": failed_tickers,
        "categories": {
            "us_stocks": 501,
            "japan_stocks": 233,
            "etfs": 119,
            "forex": 37,
        },
    }

    report_path = results_dir / "data_fetch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_path}")
    print()

    return 0 if failed_count < len(tickers) * 0.1 else 1  # Fail if >10% failed


if __name__ == "__main__":
    sys.exit(main())
