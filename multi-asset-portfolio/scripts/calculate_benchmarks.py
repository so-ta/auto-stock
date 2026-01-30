#!/usr/bin/env python3
"""
Benchmark calculation script for 15-year backtest comparison.

Calculates:
1. SPY (S&P 500) buy-and-hold returns
2. 60/40 Portfolio (SPY 60%, TLT 40%) returns

Usage:
    python scripts/calculate_benchmarks.py --start 2010-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data for a symbol."""
    from src.data.adapters import DataFrequency, StockAdapter
    from src.data.cache import DataCache

    cache = DataCache(max_age_days=7)

    # Try cache first
    cached = cache.get(symbol, DataFrequency.DAILY, start_date, end_date)
    if cached is not None:
        df = cached.data.to_pandas() if hasattr(cached.data, "to_pandas") else cached.data
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    # Fetch from API
    adapter = StockAdapter()
    ohlcv = adapter.fetch_ohlcv(symbol, start_date, end_date)
    df = ohlcv.data.to_pandas() if hasattr(ohlcv.data, "to_pandas") else ohlcv.data

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    # Cache for future use
    cache.put(ohlcv, start_date, end_date)

    return df


def calculate_metrics(portfolio_values: pd.Series) -> dict[str, float]:
    """Calculate performance metrics from portfolio value series."""
    if len(portfolio_values) < 2:
        return {}

    # Calculate returns
    returns = portfolio_values.pct_change().dropna()

    # Total return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    # Annualized return
    days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    years = days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = ann_return / volatility if volatility > 0 else 0

    # Max drawdown
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min()

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = ann_return / downside_std if downside_std > 0 else 0

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
    }


def calculate_yearly_returns(portfolio_values: pd.Series) -> dict[str, float]:
    """Calculate year-by-year returns."""
    yearly = {}
    for year in range(portfolio_values.index[0].year, portfolio_values.index[-1].year + 1):
        year_data = portfolio_values[portfolio_values.index.year == year]
        if len(year_data) >= 2:
            year_return = (year_data.iloc[-1] / year_data.iloc[0]) - 1
            yearly[str(year)] = float(year_return)
    return yearly


def calculate_spy_benchmark(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Calculate SPY buy-and-hold benchmark."""
    print("Calculating SPY benchmark...")

    # Fetch SPY data
    spy_data = fetch_data("SPY", start_date, end_date)

    if spy_data.empty:
        return {"error": "Failed to fetch SPY data"}

    # Calculate portfolio value (buy and hold)
    close_prices = spy_data["close"]
    shares = initial_capital / close_prices.iloc[0]
    portfolio_values = close_prices * shares

    # Calculate metrics
    metrics = calculate_metrics(portfolio_values)
    yearly = calculate_yearly_returns(portfolio_values)

    return {
        "name": "SPY (S&P 500)",
        "strategy": "buy_and_hold",
        "initial_capital": initial_capital,
        "final_value": float(portfolio_values.iloc[-1]),
        "metrics": metrics,
        "yearly_returns": yearly,
    }


def calculate_60_40_benchmark(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000.0,
    rebalance_frequency: str = "monthly",
) -> dict[str, Any]:
    """Calculate 60/40 portfolio (SPY 60%, TLT 40%) benchmark."""
    print("Calculating 60/40 benchmark...")

    # Fetch data
    spy_data = fetch_data("SPY", start_date, end_date)
    tlt_data = fetch_data("TLT", start_date, end_date)

    if spy_data.empty or tlt_data.empty:
        return {"error": "Failed to fetch data"}

    # Align data
    spy_close = spy_data["close"]
    tlt_close = tlt_data["close"]

    # Get common dates
    common_dates = spy_close.index.intersection(tlt_close.index)
    spy_close = spy_close.loc[common_dates]
    tlt_close = tlt_close.loc[common_dates]

    # Calculate returns
    spy_returns = spy_close.pct_change()
    tlt_returns = tlt_close.pct_change()

    # 60/40 portfolio returns
    portfolio_returns = 0.6 * spy_returns + 0.4 * tlt_returns
    portfolio_returns = portfolio_returns.fillna(0)

    # Calculate portfolio values
    portfolio_values = initial_capital * (1 + portfolio_returns).cumprod()

    # Calculate metrics
    metrics = calculate_metrics(portfolio_values)
    yearly = calculate_yearly_returns(portfolio_values)

    return {
        "name": "60/40 Portfolio (SPY/TLT)",
        "strategy": "fixed_weight",
        "weights": {"SPY": 0.6, "TLT": 0.4},
        "initial_capital": initial_capital,
        "final_value": float(portfolio_values.iloc[-1]),
        "metrics": metrics,
        "yearly_returns": yearly,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate benchmark returns")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--output", type=str, default="results/benchmark_comparison.json")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"Calculating benchmarks for {args.start} to {args.end}")
    print(f"Initial capital: ${args.capital:,.2f}")

    results = {
        "period": {
            "start": args.start,
            "end": args.end,
        },
        "initial_capital": args.capital,
        "benchmarks": [],
    }

    # Calculate SPY benchmark
    spy_result = calculate_spy_benchmark(start_date, end_date, args.capital)
    results["benchmarks"].append(spy_result)

    # Calculate 60/40 benchmark
    portfolio_60_40 = calculate_60_40_benchmark(start_date, end_date, args.capital)
    results["benchmarks"].append(portfolio_60_40)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 60)

    for bench in results["benchmarks"]:
        if "error" in bench:
            print(f"\n{bench['name']}: ERROR - {bench['error']}")
            continue

        print(f"\n{bench['name']}:")
        print(f"  Final Value: ${bench['final_value']:,.2f}")
        print(f"  Total Return: {bench['metrics']['total_return']:.2%}")
        print(f"  Annual Return: {bench['metrics']['annualized_return']:.2%}")
        print(f"  Volatility: {bench['metrics']['volatility']:.2%}")
        print(f"  Sharpe Ratio: {bench['metrics']['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {bench['metrics']['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {bench['metrics']['max_drawdown']:.2%}")
        print(f"  Calmar Ratio: {bench['metrics']['calmar_ratio']:.2f}")


if __name__ == "__main__":
    main()
