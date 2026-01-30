#!/usr/bin/env python3
"""
15-Year Backtest with Optimized Parameters

Executes a 15-year backtest (2010-2025) using the optimized parameters
from Bayesian optimization (Sharpe 0.835).

Output: results/hierarchical_ensemble_15year.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.result import BacktestResult, DailySnapshot
from src.backtest.simulator import PortfolioSimulator


# ============================================================================
# Optimized Parameters (from task_013_6 Bayesian Optimization)
# ============================================================================
OPTIMAL_PARAMS = {
    "momentum_lookback": 119,
    "rsi_period": 5,
    "bollinger_period": 11,
    "trend_weight": 0.311,
    "reversion_weight": 0.495,
    "macro_weight": 0.379,
    "beta": 2.307,
    "top_n": 4,
    "w_asset_max": 0.223,
}

# ============================================================================
# Backtest Configuration
# ============================================================================
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_BPS = 10.0  # 10 basis points

# Universe
UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "EFA", "EEM", "IWM", "VNQ"]

# Sub-periods for analysis
SUB_PERIODS = {
    "2010-2012 GFC Recovery": ("2010-01-01", "2012-12-31"),
    "2013-2017 Low Vol Bull": ("2013-01-01", "2017-12-31"),
    "2018-2019 Vol Spike": ("2018-01-01", "2019-12-31"),
    "2020-2021 COVID": ("2020-01-01", "2021-12-31"),
    "2022-2025 Rate Hike": ("2022-01-01", "2025-01-01"),
}


def fetch_data_yfinance(tickers: list, start: str, end: str) -> dict:
    """Fetch data using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf

    data = {}
    print(f"Fetching data for {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 0:
                # Handle MultiIndex columns (yfinance returns ('Close', 'TICKER') format)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Standardize column names
                df.columns = [str(c).lower() for c in df.columns]
                if 'adj close' in df.columns:
                    df['close'] = df['adj close']
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: No data")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    return data


def calculate_signals(prices_df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Calculate combined signal using optimized parameters.

    Returns signal in [-1, +1] range.
    """
    close = prices_df['close']

    # 1. Momentum Signal (trend layer)
    mom_lookback = params["momentum_lookback"]
    momentum = close.pct_change(periods=mom_lookback)
    momentum_signal = np.tanh(momentum * 10)  # Scale and normalize

    # 2. RSI Signal (reversion layer)
    rsi_period = params["rsi_period"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_signal = (50 - rsi) / 50  # Inverted for mean reversion

    # 3. Bollinger Signal (reversion layer)
    bb_period = params["bollinger_period"]
    bb_ma = close.rolling(window=bb_period).mean()
    bb_std = close.rolling(window=bb_period).std()
    upper = bb_ma + 2 * bb_std
    lower = bb_ma - 2 * bb_std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bollinger_signal = 1 - 2 * pct_b  # Inverted for mean reversion

    # Combine signals with weights
    trend_weight = params["trend_weight"]
    reversion_weight = params["reversion_weight"]

    # Normalize weights
    total_weight = trend_weight + reversion_weight
    trend_w = trend_weight / total_weight
    reversion_w = reversion_weight / total_weight

    # Combined signal
    combined = (
        trend_w * momentum_signal +
        reversion_w * 0.5 * (rsi_signal + bollinger_signal)
    )

    return combined.clip(-1, 1).fillna(0)


def calculate_target_weights(
    signals: dict,
    prices: dict,
    params: dict,
    date: pd.Timestamp,
) -> dict:
    """
    Calculate target weights based on signals and parameters.
    """
    # Get signals for this date
    date_signals = {}
    for ticker, signal_series in signals.items():
        if date in signal_series.index:
            date_signals[ticker] = signal_series.loc[date]
        else:
            # Use most recent available signal
            valid = signal_series.loc[:date].dropna()
            if len(valid) > 0:
                date_signals[ticker] = valid.iloc[-1]
            else:
                date_signals[ticker] = 0.0

    # Sort by signal strength and select top_n
    top_n = params["top_n"]
    sorted_tickers = sorted(date_signals.keys(), key=lambda x: date_signals[x], reverse=True)
    selected = sorted_tickers[:top_n]

    # Calculate weights using softmax with beta
    beta = params["beta"]
    selected_signals = [date_signals[t] for t in selected]

    # Apply softmax
    exp_signals = np.exp(np.array(selected_signals) * beta)
    raw_weights = exp_signals / exp_signals.sum()

    # Apply max weight constraint
    w_max = params["w_asset_max"]
    weights = {}
    total = 0.0

    for i, ticker in enumerate(selected):
        w = min(raw_weights[i], w_max)
        weights[ticker] = w
        total += w

    # Normalize
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    # Add cash if needed
    invested = sum(weights.values())
    if invested < 1.0:
        weights["CASH"] = 1.0 - invested

    return weights


def get_rebalance_dates(start: str, end: str, frequency: str = "monthly") -> list:
    """Get rebalance dates."""
    dates = pd.date_range(start=start, end=end, freq="B")

    if frequency == "monthly":
        # First trading day of each month
        rebalance_dates = []
        current_month = None
        for d in dates:
            if d.month != current_month:
                rebalance_dates.append(d)
                current_month = d.month
        return rebalance_dates
    elif frequency == "weekly":
        return dates[::5].tolist()
    else:
        return dates.tolist()


def run_backtest(
    data: dict,
    params: dict,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 10.0,
) -> BacktestResult:
    """
    Run the backtest with given parameters.
    """
    # Calculate signals for all tickers
    signals = {}
    for ticker, df in data.items():
        if len(df) > params["momentum_lookback"]:
            signals[ticker] = calculate_signals(df, params)

    # Get all trading dates
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    rebalance_dates = get_rebalance_dates(start_date, end_date, "monthly")
    rebalance_set = set(rebalance_dates)

    # Initialize simulator
    sim = PortfolioSimulator(
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
    )

    # Initialize result
    result = BacktestResult(
        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
        rebalance_frequency="monthly",
        initial_capital=initial_capital,
    )

    prev_value = initial_capital
    cumulative_return = 0.0

    for date in all_dates:
        # Get prices for this date
        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = df.loc[date, 'close']

        if not prices:
            continue

        # Rebalance if it's a rebalance date
        if date in rebalance_set and len(prices) > 0:
            target_weights = calculate_target_weights(signals, data, params, date)
            if target_weights:
                try:
                    sim.rebalance(target_weights, prices, date.to_pydatetime())
                    result.num_rebalances += 1
                except Exception as e:
                    pass  # Skip failed rebalances

        # Update value
        current_value = sim.get_portfolio_value(prices)

        if current_value > 0:
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
            cumulative_return = (current_value - initial_capital) / initial_capital

            snapshot = DailySnapshot(
                date=date.to_pydatetime(),
                weights=sim.get_current_weights(prices),
                portfolio_value=current_value,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                cash_weight=sim.cash / current_value if current_value > 0 else 0.0,
            )
            result.snapshots.append(snapshot)
            prev_value = current_value

    # Calculate metrics
    result.total_trades = sim.trades_count
    result.transaction_costs = sim.transaction_costs_total
    result.calculate_metrics()

    return result


def calculate_benchmark_returns(data: dict, ticker: str, start: str, end: str) -> dict:
    """Calculate benchmark returns (buy and hold)."""
    if ticker not in data:
        return {}

    df = data[ticker]
    df = df.loc[start:end]

    if len(df) < 2:
        return {}

    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    total_return = (final_price - initial_price) / initial_price

    returns = df['close'].pct_change().dropna()
    n_years = len(df) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def calculate_60_40_benchmark(data: dict, start: str, end: str) -> dict:
    """Calculate 60/40 portfolio benchmark."""
    if "SPY" not in data or "TLT" not in data:
        return {}

    spy = data["SPY"].loc[start:end]['close']
    tlt = data["TLT"].loc[start:end]['close']

    # Align indices
    common_index = spy.index.intersection(tlt.index)
    spy = spy.loc[common_index]
    tlt = tlt.loc[common_index]

    if len(spy) < 2:
        return {}

    # Calculate returns
    spy_ret = spy.pct_change()
    tlt_ret = tlt.pct_change()

    # 60/40 portfolio returns
    portfolio_ret = 0.6 * spy_ret + 0.4 * tlt_ret
    portfolio_ret = portfolio_ret.dropna()

    cumulative = (1 + portfolio_ret).cumprod()
    total_return = cumulative.iloc[-1] - 1

    n_years = len(portfolio_ret) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = portfolio_ret.std() * np.sqrt(252)
    sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0

    # Max drawdown
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def analyze_sub_periods(result: BacktestResult, sub_periods: dict) -> dict:
    """Analyze performance across sub-periods."""
    df = result.to_dataframe()
    if len(df) == 0:
        return {}

    analysis = {}

    for period_name, (start, end) in sub_periods.items():
        period_df = df.loc[start:end]
        if len(period_df) < 20:
            continue

        returns = period_df['daily_return']
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

        n_days = len(returns)
        n_years = n_days / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.02) / volatility if volatility > 0 else 0

        # Max drawdown
        values = period_df['portfolio_value']
        running_max = values.expanding().max()
        drawdowns = (values - running_max) / running_max
        max_dd = drawdowns.min()

        analysis[period_name] = {
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "n_days": int(n_days),
        }

    return analysis


def calculate_spy_excess_periods(result: BacktestResult, spy_data: pd.DataFrame) -> float:
    """Calculate percentage of time strategy outperforms SPY."""
    df = result.to_dataframe()
    if len(df) == 0 or len(spy_data) == 0:
        return 0.0

    # Align dates
    spy_returns = spy_data['close'].pct_change()
    strategy_returns = df['daily_return']

    common_dates = strategy_returns.index.intersection(spy_returns.index)

    spy_aligned = spy_returns.loc[common_dates]
    strategy_aligned = strategy_returns.loc[common_dates]

    # Calculate rolling 252-day returns
    spy_rolling = spy_aligned.rolling(252).apply(lambda x: (1 + x).prod() - 1, raw=False)
    strategy_rolling = strategy_aligned.rolling(252).apply(lambda x: (1 + x).prod() - 1, raw=False)

    valid_mask = spy_rolling.notna() & strategy_rolling.notna()
    if valid_mask.sum() == 0:
        return 0.0

    excess = (strategy_rolling[valid_mask] > spy_rolling[valid_mask]).mean()
    return float(excess)


def main():
    """Main entry point."""
    print("=" * 70)
    print("  15-Year Backtest with Optimized Parameters")
    print("=" * 70)
    print()

    # Display parameters
    print("Optimal Parameters:")
    for k, v in OPTIMAL_PARAMS.items():
        print(f"  {k}: {v}")
    print()

    # Fetch data
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Universe: {', '.join(UNIVERSE)}")
    print()

    data = fetch_data_yfinance(UNIVERSE, START_DATE, END_DATE)

    if len(data) < 3:
        print("ERROR: Insufficient data fetched")
        return

    print()

    # Run backtest
    print("Running backtest...")
    result = run_backtest(
        data=data,
        params=OPTIMAL_PARAMS,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    # Print summary
    print()
    print(result.summary())

    # Calculate benchmarks
    print("Calculating benchmarks...")
    spy_benchmark = calculate_benchmark_returns(data, "SPY", START_DATE, END_DATE)
    benchmark_60_40 = calculate_60_40_benchmark(data, START_DATE, END_DATE)

    # Analyze sub-periods
    print("Analyzing sub-periods...")
    sub_period_analysis = analyze_sub_periods(result, SUB_PERIODS)

    # Calculate SPY excess periods
    spy_excess_pct = 0.0
    if "SPY" in data:
        spy_excess_pct = calculate_spy_excess_periods(result, data["SPY"])

    # Prepare output
    output = {
        "metadata": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "universe": UNIVERSE,
            "rebalance_frequency": "monthly",
            "run_timestamp": datetime.now().isoformat(),
        },
        "optimal_params": OPTIMAL_PARAMS,
        "performance": {
            "final_value": result.final_value,
            "total_return": result.total_return,
            "total_return_pct": result.total_return * 100,
            "annualized_return": result.annualized_return,
            "annualized_return_pct": result.annualized_return * 100,
            "volatility": result.volatility,
            "volatility_pct": result.volatility * 100,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio if result.sortino_ratio != float("inf") else None,
            "calmar_ratio": result.calmar_ratio if result.calmar_ratio != float("inf") else None,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown * 100,
            "var_95": result.var_95,
            "expected_shortfall": result.expected_shortfall,
            "win_rate": result.win_rate,
        },
        "trading_stats": {
            "total_trades": result.total_trades,
            "num_rebalances": result.num_rebalances,
            "transaction_costs": result.transaction_costs,
        },
        "benchmarks": {
            "SPY": spy_benchmark,
            "60/40": benchmark_60_40,
        },
        "sub_period_analysis": sub_period_analysis,
        "validation": {
            "sharpe_gte_1": result.sharpe_ratio >= 1.0,
            "mdd_lt_20pct": abs(result.max_drawdown) < 0.20,
            "spy_excess_pct": spy_excess_pct,
            "spy_excess_gte_70pct": spy_excess_pct >= 0.70,
        },
    }

    # Save results
    output_path = PROJECT_ROOT / "results" / "hierarchical_ensemble_15year.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"Results saved to: {output_path}")

    # Print validation
    print()
    print("-" * 70)
    print("  VALIDATION")
    print("-" * 70)
    print(f"  Sharpe Ratio >= 1.0:  {'PASS' if result.sharpe_ratio >= 1.0 else 'FAIL'} ({result.sharpe_ratio:.3f})")
    print(f"  MDD < 20%:            {'PASS' if abs(result.max_drawdown) < 0.20 else 'FAIL'} ({result.max_drawdown * 100:.2f}%)")
    print(f"  SPY Excess >= 70%:    {'PASS' if spy_excess_pct >= 0.70 else 'FAIL'} ({spy_excess_pct * 100:.1f}%)")
    print()

    # Print benchmark comparison
    print("-" * 70)
    print("  BENCHMARK COMPARISON")
    print("-" * 70)
    print(f"  {'Metric':<25} {'Strategy':>12} {'SPY':>12} {'60/40':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Total Return':<25} {result.total_return * 100:>11.2f}% {spy_benchmark.get('total_return', 0) * 100:>11.2f}% {benchmark_60_40.get('total_return', 0) * 100:>11.2f}%")
    print(f"  {'Ann. Return':<25} {result.annualized_return * 100:>11.2f}% {spy_benchmark.get('annualized_return', 0) * 100:>11.2f}% {benchmark_60_40.get('annualized_return', 0) * 100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<25} {result.sharpe_ratio:>12.3f} {spy_benchmark.get('sharpe_ratio', 0):>12.3f} {benchmark_60_40.get('sharpe_ratio', 0):>12.3f}")
    print(f"  {'Max Drawdown':<25} {result.max_drawdown * 100:>11.2f}% {spy_benchmark.get('max_drawdown', 0) * 100:>11.2f}% {benchmark_60_40.get('max_drawdown', 0) * 100:>11.2f}%")
    print()

    # Print sub-period analysis
    if sub_period_analysis:
        print("-" * 70)
        print("  SUB-PERIOD ANALYSIS")
        print("-" * 70)
        print(f"  {'Period':<25} {'Return':>10} {'Sharpe':>10} {'MDD':>10}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
        for period, metrics in sub_period_analysis.items():
            print(f"  {period:<25} {metrics['annualized_return'] * 100:>9.2f}% {metrics['sharpe_ratio']:>10.3f} {metrics['max_drawdown'] * 100:>9.2f}%")

    print()
    print("=" * 70)
    print("  Backtest Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
