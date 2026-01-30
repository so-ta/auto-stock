#!/usr/bin/env python3
"""
Universe Quality Filter - Filter stocks by quality criteria.

NOTE: This script outputs to config/universe_filtered.yaml as intermediate file.
      For production backtests, use config/universe_standard.yaml instead.
      See config/README_UNIVERSE.md for details.

Filtering Criteria:
- Minimum 2 years of data (500+ trading days)
- Average volume >= 100,000 (liquidity)
- Missing rate <= 5%
- OHLC consistency error rate <= 1.0%

Usage:
    python scripts/quality_filter_universe.py

Output:
    - results/quality_filter_report.json
    - config/universe_filtered.yaml (intermediate, use universe_standard.yaml for production)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Quality thresholds
MIN_TRADING_DAYS = 500  # ~2 years
MIN_AVG_VOLUME = 100_000
MAX_MISSING_RATE = 0.05  # 5%
MAX_OHLC_ERROR_RATE = 0.01  # 1.0%


def find_column(df: pl.DataFrame, keywords: list[str]) -> str | None:
    """Find column name containing any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def check_quality(ticker: str, df: pl.DataFrame) -> dict:
    """
    Check quality of a single ticker's data.

    Args:
        ticker: Stock ticker symbol
        df: Price DataFrame with columns [Date, Open, High, Low, Close, Volume]

    Returns:
        Quality check result dictionary
    """
    result = {
        "ticker": ticker,
        "total_days": len(df),
        "missing_rate": 0.0,
        "avg_volume": 0.0,
        "ohlc_error_rate": 0.0,
        "ohlc_errors": 0,
        "passed": False,
        "fail_reasons": [],
    }

    if len(df) == 0:
        result["fail_reasons"].append("no_data")
        return result

    # 1. Trading days check
    if len(df) < MIN_TRADING_DAYS:
        result["fail_reasons"].append(f"insufficient_data ({len(df)} < {MIN_TRADING_DAYS})")

    # 2. Missing rate check
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    total_cells = len(df) * len(df.columns)
    result["missing_rate"] = total_nulls / total_cells if total_cells > 0 else 0

    if result["missing_rate"] > MAX_MISSING_RATE:
        result["fail_reasons"].append(f"high_missing_rate ({result['missing_rate']:.2%} > {MAX_MISSING_RATE:.0%})")

    # 3. Volume check (handle tuple-style column names)
    volume_col = find_column(df, ["volume", "vol"])

    if volume_col:
        avg_vol = df[volume_col].mean()
        result["avg_volume"] = float(avg_vol) if avg_vol is not None else 0
        if result["avg_volume"] < MIN_AVG_VOLUME:
            result["fail_reasons"].append(f"low_volume ({result['avg_volume']:,.0f} < {MIN_AVG_VOLUME:,})")
    else:
        result["fail_reasons"].append("no_volume_column")

    # 4. OHLC consistency check
    ohlc_errors = check_ohlc_consistency(df)
    result["ohlc_errors"] = ohlc_errors
    result["ohlc_error_rate"] = ohlc_errors / len(df) if len(df) > 0 else 0

    if result["ohlc_error_rate"] > MAX_OHLC_ERROR_RATE:
        result["fail_reasons"].append(f"ohlc_errors ({result['ohlc_error_rate']:.2%} > {MAX_OHLC_ERROR_RATE:.0%})")

    result["passed"] = len(result["fail_reasons"]) == 0
    return result


def check_ohlc_consistency(df: pl.DataFrame) -> int:
    """
    Check OHLC price consistency.

    Rules:
    - High >= Low
    - High >= Open
    - High >= Close
    - Low <= Open
    - Low <= Close

    Returns:
        Number of rows with errors
    """
    # Find column names (handle tuple-style names like "('Open', 'KMX')")
    open_col = find_column(df, ["open"])
    high_col = find_column(df, ["high"])
    low_col = find_column(df, ["low"])
    close_col = find_column(df, ["close"])

    if not all([open_col, high_col, low_col, close_col]):
        return 0  # Cannot check without OHLC columns

    # Count errors
    errors = df.filter(
        (pl.col(high_col) < pl.col(low_col)) |
        (pl.col(high_col) < pl.col(open_col)) |
        (pl.col(high_col) < pl.col(close_col)) |
        (pl.col(low_col) > pl.col(open_col)) |
        (pl.col(low_col) > pl.col(close_col))
    )

    return len(errors)


def run_quality_filter() -> dict:
    """
    Run quality filter on all cached price data.

    Returns:
        Quality filter report dictionary
    """
    cache_dir = Path("cache/price_data")

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    parquet_files = list(cache_dir.glob("*.parquet"))
    total_files = len(parquet_files)

    logger.info(f"Found {total_files} parquet files to check")

    results = []
    passed = []
    failed = []
    fail_reason_counts = {}

    for i, parquet_file in enumerate(parquet_files, 1):
        ticker = parquet_file.stem

        if i % 100 == 0:
            logger.info(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")

        try:
            df = pl.read_parquet(parquet_file)
            result = check_quality(ticker, df)
            results.append(result)

            if result["passed"]:
                passed.append(ticker)
            else:
                failed.append(ticker)
                for reason in result["fail_reasons"]:
                    reason_key = reason.split(" ")[0]
                    fail_reason_counts[reason_key] = fail_reason_counts.get(reason_key, 0) + 1

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "total_days": 0,
                "missing_rate": 1.0,
                "avg_volume": 0,
                "ohlc_error_rate": 1.0,
                "passed": False,
                "fail_reasons": [f"read_error: {str(e)}"],
            })
            failed.append(ticker)
            fail_reason_counts["read_error"] = fail_reason_counts.get("read_error", 0) + 1

    # Sort passed tickers
    passed.sort()
    failed.sort()

    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "thresholds": {
            "min_trading_days": MIN_TRADING_DAYS,
            "min_avg_volume": MIN_AVG_VOLUME,
            "max_missing_rate": MAX_MISSING_RATE,
            "max_ohlc_error_rate": MAX_OHLC_ERROR_RATE,
        },
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(results) * 100 if results else 0,
        },
        "fail_reason_counts": fail_reason_counts,
        "passed_tickers": passed,
        "failed_tickers": failed,
        "details": results,
    }

    return report


def save_report(report: dict) -> None:
    """Save quality filter report to JSON."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "quality_filter_report.json"

    # Save full report (without details for smaller file)
    report_summary = {k: v for k, v in report.items() if k != "details"}
    with open(report_path, "w") as f:
        json.dump(report_summary, f, indent=2)

    logger.info(f"Report saved to {report_path}")

    # Save full report with details
    full_report_path = results_dir / "quality_filter_report_full.json"
    with open(full_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Full report saved to {full_report_path}")


def save_filtered_universe(passed_tickers: list[str]) -> None:
    """Save filtered universe to YAML config."""
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    universe_path = config_dir / "universe_filtered.yaml"

    universe_config = {
        "description": "Quality-filtered universe from 889 stocks",
        "generated_at": datetime.now().isoformat(),
        "total_tickers": len(passed_tickers),
        "thresholds": {
            "min_trading_days": MIN_TRADING_DAYS,
            "min_avg_volume": MIN_AVG_VOLUME,
            "max_missing_rate": MAX_MISSING_RATE,
            "max_ohlc_error_rate": MAX_OHLC_ERROR_RATE,
        },
        "tickers": passed_tickers,
    }

    with open(universe_path, "w") as f:
        yaml.dump(universe_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Filtered universe saved to {universe_path}")


def print_summary(report: dict) -> None:
    """Print quality filter summary."""
    print("\n" + "=" * 60)
    print("QUALITY FILTER SUMMARY")
    print("=" * 60)

    summary = report["summary"]
    print(f"\nTotal tickers checked: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
    print(f"Failed: {summary['failed']} ({100 - summary['pass_rate']:.1f}%)")

    print("\nFailure reasons breakdown:")
    for reason, count in sorted(report["fail_reason_counts"].items(), key=lambda x: -x[1]):
        pct = count / summary["total"] * 100
        print(f"  - {reason}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    logger.info("Starting universe quality filter...")

    report = run_quality_filter()

    save_report(report)

    save_filtered_universe(report["passed_tickers"])

    print_summary(report)

    logger.info("Quality filter completed!")

    return report


if __name__ == "__main__":
    main()
