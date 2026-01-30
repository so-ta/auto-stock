#!/usr/bin/env python3
"""
Unified Full Portfolio Backtest Script

全頻度（daily/weekly/monthly）対応の統合バックテストスクリプト。
旧スクリプト run_full_backtest_weekly.py, run_full_backtest_monthly.py を統合。

Usage:
    python scripts/run_full_backtest.py --frequency weekly
    python scripts/run_full_backtest.py --frequency monthly
    python scripts/run_full_backtest.py --frequency daily
    python scripts/run_full_backtest.py --frequency weekly --strategy momentum --top-n 50
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl


# =============================================================================
# 共通関数
# =============================================================================


def load_universe(config_path: str = "config/universe_standard.yaml") -> list[str]:
    """標準ユニバースを読み込み"""
    path = Path(config_path)
    if not path.exists():
        # フォールバック: universe.yaml
        alt_path = Path("config/universe.yaml")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Universe file not found: {config_path}")

    with open(path, "r") as f:
        universe = yaml.safe_load(f)

    # 複数のキー形式に対応
    tickers = universe.get("tickers", universe.get("passed_tickers", universe.get("all_symbols", [])))
    return tickers


def load_price_data(tickers: list[str], cache_dir: str = "cache/price_data") -> dict:
    """価格データを読み込み"""
    cache_path = Path(cache_dir)
    price_data = {}

    for ticker in tickers:
        safe_ticker = ticker.replace("=", "_").replace("/", "_")
        path = cache_path / f"{safe_ticker}.parquet"
        if path.exists():
            try:
                df = pl.read_parquet(path)
                price_data[ticker] = df
            except Exception as e:
                print(f"Warning: Failed to load {ticker}: {e}")

    return price_data


def calculate_metrics(returns: np.ndarray, frequency: str = "weekly", risk_free_rate: float = 0.02) -> dict:
    """パフォーマンス指標を計算"""
    if len(returns) == 0:
        return {}

    # 頻度に応じた年間期間数
    periods_per_year = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 52)

    # 基本指標
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    if n_years > 0:
        annual_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annual_return = 0.0

    # ボラティリティ
    annual_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0.0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
        sortino = excess_return / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino = 0.0

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
        "n_periods": int(n_periods),
        "n_years": float(n_years),
    }


def calculate_yearly_returns(returns: np.ndarray, dates: list) -> dict:
    """年別リターンを計算"""
    if len(returns) != len(dates):
        return {}

    yearly = {}
    current_year = None
    year_returns = []

    for ret, date in zip(returns, dates):
        year = date.year if hasattr(date, "year") else int(str(date)[:4])

        if current_year != year:
            if current_year is not None and year_returns:
                yearly[str(current_year)] = float(np.prod(1 + np.array(year_returns)) - 1)
            current_year = year
            year_returns = []

        year_returns.append(ret)

    # 最後の年
    if current_year is not None and year_returns:
        yearly[str(current_year)] = float(np.prod(1 + np.array(year_returns)) - 1)

    return yearly


def convert_to_price_dataframe(price_data: dict):
    """価格データをpandas DataFrameに変換"""
    import pandas as pd

    all_closes = {}
    for ticker, df in price_data.items():
        try:
            # カラム名を検出（('Date', '') や ('Close', 'XXX') 形式に対応）
            date_col = None
            close_col = None
            for col in df.columns:
                col_lower = col.lower()
                if "date" in col_lower or "timestamp" in col_lower:
                    date_col = col
                elif "close" in col_lower:
                    close_col = col

            if date_col is None or close_col is None:
                continue

            # Polars to Pandas変換
            closes_df = df.select([date_col, close_col]).to_pandas()
            closes_df.columns = ["date", "close"]
            closes_df["date"] = pd.to_datetime(closes_df["date"])
            closes_df = closes_df.set_index("date")["close"]
            all_closes[ticker] = closes_df
        except Exception:
            continue

    if not all_closes:
        return None

    price_df = pd.DataFrame(all_closes)
    price_df = price_df.sort_index()
    return price_df


# =============================================================================
# バックテスト戦略
# =============================================================================


def run_momentum_backtest(
    price_data: dict,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    frequency: str = "weekly",
    top_n: int = 50,
    lookback_days: int = 60,
) -> dict:
    """モメンタム戦略バックテスト"""
    import pandas as pd

    print(f"Running momentum backtest: {len(price_data)} tickers")
    print(f"Period: {start_date} to {end_date}")
    print(f"Frequency: {frequency}, Top N: {top_n}")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 価格データをDataFrameに変換
    price_df = convert_to_price_dataframe(price_data)
    if price_df is None:
        return {"error": "No valid price data"}

    # 期間フィルタ
    price_df = price_df[(price_df.index >= start_dt) & (price_df.index <= end_dt)]
    price_df = price_df.ffill()
    price_df = price_df.dropna(axis=1, how="all")

    print(f"Price matrix: {price_df.shape[0]} days x {price_df.shape[1]} tickers")

    # リバランス日を生成
    if frequency == "daily":
        rebalance_dates = price_df.index[::1]
    elif frequency == "weekly":
        rebalance_dates = price_df.resample("W-FRI").last().index
    elif frequency == "monthly":
        rebalance_dates = price_df.resample("ME").last().index
    else:
        rebalance_dates = price_df.resample("W-FRI").last().index

    rebalance_dates = [d for d in rebalance_dates if d in price_df.index]
    print(f"Rebalance dates: {len(rebalance_dates)}")

    # バックテスト実行
    portfolio_returns = []
    portfolio_dates = []

    for i, rebal_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # モメンタムスコア計算
        lookback_start = rebal_date - pd.Timedelta(days=lookback_days)
        hist_data = price_df[(price_df.index >= lookback_start) & (price_df.index <= rebal_date)]

        if len(hist_data) < lookback_days // 2:
            continue

        # リスク調整済みモメンタム
        momentum_scores = {}
        for ticker in price_df.columns:
            if ticker in hist_data.columns:
                prices = hist_data[ticker].dropna()
                if len(prices) >= 2:
                    ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                    vol = prices.pct_change().std()
                    if vol > 0 and not np.isnan(ret):
                        momentum_scores[ticker] = ret / vol

        if not momentum_scores:
            continue

        # 上位N銘柄を選択
        sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [t[0] for t in sorted_tickers[:top_n] if t[1] > 0]

        if not top_tickers:
            continue

        # 均等ウェイト
        weight = 1.0 / len(top_tickers)
        current_weights = {t: weight for t in top_tickers}

        # 次のリバランス日までのリターン計算
        period_data = price_df[(price_df.index > rebal_date) & (price_df.index <= next_date)]

        if len(period_data) == 0:
            continue

        # ポートフォリオリターン
        period_return = 0.0
        for ticker, w in current_weights.items():
            if ticker in period_data.columns:
                ticker_prices = period_data[ticker].dropna()
                if len(ticker_prices) >= 2:
                    ticker_ret = (ticker_prices.iloc[-1] / ticker_prices.iloc[0]) - 1
                    period_return += w * ticker_ret

        portfolio_returns.append(period_return)
        portfolio_dates.append(next_date)

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(rebalance_dates)} rebalances")

    returns_arr = np.array(portfolio_returns)

    # 指標計算
    metrics = calculate_metrics(returns_arr, frequency)
    yearly_returns = calculate_yearly_returns(returns_arr, portfolio_dates)

    # 最終資産額
    initial_capital = 1_000_000
    final_value = initial_capital * np.prod(1 + returns_arr)

    return {
        "backtest_type": f"momentum_{frequency}",
        "strategy": "momentum",
        "start_date": start_date,
        "end_date": end_date,
        "frequency": frequency,
        "n_tickers": len(price_data),
        "n_rebalances": len(portfolio_returns),
        "initial_capital": initial_capital,
        "final_value": float(final_value),
        "metrics": metrics,
        "yearly_returns": yearly_returns,
        "parameters": {
            "top_n": top_n,
            "lookback_days": lookback_days,
        },
    }


def run_engine_backtest(
    price_data: dict,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    frequency: str = "monthly",
) -> dict:
    """バックテストエンジン使用"""
    import pandas as pd

    print(f"Running engine backtest: {len(price_data)} tickers")
    print(f"Period: {start_date} to {end_date}")
    print(f"Frequency: {frequency}")

    # エンジンインポート
    try:
        from src.backtest.streaming_engine import StreamingBacktestEngine

        use_streaming = True
    except ImportError:
        from src.backtest.fast_engine import FastBacktestConfig, FastBacktestEngine

        use_streaming = False

    if use_streaming:
        engine = StreamingBacktestEngine(
            chunk_size=50,
            temp_dir="cache/streaming_temp",
            save_interval=5,
        )

        def progress_callback(chunk_idx: int, total_chunks: int, chunk_tickers: list):
            pct = (chunk_idx / total_chunks) * 100
            print(f"Progress: {chunk_idx}/{total_chunks} ({pct:.1f}%)")

        result = engine.run(
            tickers=list(price_data.keys()),
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=frequency,
            progress_callback=progress_callback,
        )

        return result.to_dict()

    else:
        # FastBacktestEngineで代替
        price_df = convert_to_price_dataframe(price_data)
        if price_df is None:
            return {"error": "No valid price data"}

        price_df = price_df.ffill().bfill()
        print(f"Price matrix: {price_df.shape[0]} days x {price_df.shape[1]} tickers")

        config = FastBacktestConfig(
            start_date=datetime.strptime(start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(end_date, "%Y-%m-%d"),
            rebalance_frequency=frequency,
            initial_capital=100000.0,
            transaction_cost_bps=10.0,
        )

        engine = FastBacktestEngine(config)
        result = engine.run(price_df)

        return {
            "backtest_type": f"engine_{frequency}",
            "strategy": "engine",
            "total_return": result.total_return,
            "annual_return": getattr(result, "annualized_return", getattr(result, "annual_return", 0.0)),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "volatility": result.volatility,
            "num_trades": result.total_trades,
            "final_value": result.final_value,
            "num_rebalances": result.num_rebalances,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end_date,
        }


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified Full Portfolio Backtest")
    parser.add_argument(
        "--frequency",
        "-f",
        choices=["daily", "weekly", "monthly"],
        default="weekly",
        help="Rebalance frequency (default: weekly)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["momentum", "engine"],
        default="momentum",
        help="Backtest strategy (default: momentum)",
    )
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        help="Start date (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date (default: 2024-12-31)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top assets for momentum strategy (default: 50)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback days for momentum (default: 60)",
    )
    parser.add_argument(
        "--universe",
        default="config/universe_standard.yaml",
        help="Universe config file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: results/backtest_full_{frequency}.json)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Full Portfolio Backtest - {args.frequency.upper()}")
    print("=" * 60)

    start_time = time.time()

    # 1. ユニバース読み込み
    print("\n[1/4] Loading universe...")
    tickers = load_universe(args.universe)
    print(f"  Loaded {len(tickers)} tickers")

    # 2. 価格データ読み込み
    print("\n[2/4] Loading price data...")
    price_data = load_price_data(tickers)
    print(f"  Loaded {len(price_data)} tickers with price data")

    if not price_data:
        print("ERROR: No price data loaded!")
        sys.exit(1)

    # 3. バックテスト実行
    print("\n[3/4] Running backtest...")
    if args.strategy == "momentum":
        result_dict = run_momentum_backtest(
            price_data=price_data,
            start_date=args.start_date,
            end_date=args.end_date,
            frequency=args.frequency,
            top_n=args.top_n,
            lookback_days=args.lookback,
        )
    else:
        result_dict = run_engine_backtest(
            price_data=price_data,
            start_date=args.start_date,
            end_date=args.end_date,
            frequency=args.frequency,
        )

    # メタデータ追加
    elapsed = time.time() - start_time
    result_dict["elapsed_seconds"] = elapsed
    result_dict["timestamp"] = datetime.now().isoformat()
    result_dict["num_tickers"] = len(price_data)

    # 4. 結果保存
    print("\n[4/4] Saving results...")
    output_path = args.output or f"results/backtest_full_{args.frequency}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"  Saved to {output_path}")

    # 結果表示
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics = result_dict.get("metrics", result_dict)
    print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Annual Return:   {metrics.get('annual_return', 0)*100:.2f}%")
    print(f"Max Drawdown:    {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sortino Ratio:   {metrics.get('sortino_ratio', 0):.3f}")
    print(f"Calmar Ratio:    {metrics.get('calmar_ratio', 0):.3f}")

    if "yearly_returns" in result_dict:
        print("\nYearly Returns:")
        for year, ret in sorted(result_dict["yearly_returns"].items()):
            print(f"  {year}: {ret*100:+.2f}%")

    if "final_value" in result_dict:
        print(f"\nFinal Value: ¥{result_dict['final_value']:,.0f}")

    print(f"\nExecution Time: {elapsed:.1f}s")

    # 目標達成チェック
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)
    sharpe = metrics.get("sharpe_ratio", 0)
    annual_ret = metrics.get("annual_return", 0)

    sharpe_ok = sharpe >= 1.0
    return_ok = annual_ret >= 0.10

    print(f"Sharpe >= 1.0:      {'PASS' if sharpe_ok else 'FAIL'} ({sharpe:.3f})")
    print(f"Annual Ret >= 10%:  {'PASS' if return_ok else 'FAIL'} ({annual_ret*100:.2f}%)")

    return result_dict


if __name__ == "__main__":
    main()
