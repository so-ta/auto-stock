#!/usr/bin/env python3
"""
日次バックテスト（828銘柄、15年）【BT-001】

task_028_1: 15年間の日次リバランスバックテストを実行
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.factory import BacktestEngineFactory, recommend_engine
from src.backtest.base import UnifiedBacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_universe(config_path: str) -> list[str]:
    """ユニバース設定を読み込み"""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    tickers = []
    for category, info in data.get("universe", {}).items():
        if info.get("enabled", True) and "tickers" in info:
            tickers.extend(info["tickers"])

    return tickers


def run_backtest():
    """バックテスト実行"""
    print("=" * 60)
    print("  BT-001: 日次バックテスト（15年）")
    print("=" * 60)
    print()

    # パラメータ
    config_path = "config/universe_full.yaml"
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 12, 31)
    initial_capital = 1_000_000.0
    transaction_cost_bps = 10.0
    rebalance_frequency = "daily"
    output_path = "results/backtest_daily_15y.json"

    # ユニバース読み込み
    print(f"Loading universe from: {config_path}")
    universe = load_universe(config_path)
    print(f"Total symbols: {len(universe)}")
    print()

    # 推奨エンジン確認
    period_days = (end_date - start_date).days
    recommended = recommend_engine(len(universe), period_days)
    print(f"Recommended engine: {recommended}")
    print()

    # 統一設定
    config = UnifiedBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=5.0,
    )

    print("Backtest Configuration:")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Initial Capital: ${initial_capital:,.0f}")
    print(f"  Rebalance: {rebalance_frequency}")
    print(f"  Transaction Cost: {transaction_cost_bps} bps")
    print()

    # エンジン生成（自動選択）
    print("Creating engine (auto-select)...")
    start_time = time.time()

    try:
        # 大規模データのため vectorbt を明示的に使用
        engine = BacktestEngineFactory.create(
            mode="vectorbt",  # 大規模データに最適
            config=config,
        )
        print(f"Engine: {engine.__class__.__name__}")
    except Exception as e:
        logger.warning(f"VectorBT engine failed: {e}, trying streaming...")
        try:
            engine = BacktestEngineFactory.create(
                mode="streaming",
                config=config,
            )
            print(f"Engine: {engine.__class__.__name__}")
        except Exception as e2:
            logger.error(f"Streaming engine also failed: {e2}")
            print(f"Error: Could not create any engine - {e2}")
            return 1

    print()
    print("Starting backtest... (this may take a while)")
    print("-" * 60)

    try:
        # yfinanceで価格データを取得
        import yfinance as yf
        import pandas as pd

        print("Fetching price data via yfinance...")
        print("(This will take a while for 734 symbols over 15 years)")
        print()

        # バッチで取得（yfinanceは複数銘柄を一度に取得可能）
        # ただし大量だと失敗するので分割
        batch_size = 50
        prices = {}
        failed_symbols = []

        for i in range(0, len(universe), batch_size):
            batch = universe[i : i + batch_size]
            print(f"  Batch {i // batch_size + 1}/{(len(universe) + batch_size - 1) // batch_size}: {batch[0]}...")

            try:
                # 一括ダウンロード
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True,
                )

                if data.empty:
                    failed_symbols.extend(batch)
                    continue

                # 各銘柄のデータを抽出
                if len(batch) == 1:
                    # 単一銘柄の場合
                    symbol = batch[0]
                    if len(data) > 100:
                        prices[symbol] = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                else:
                    # 複数銘柄の場合
                    for symbol in batch:
                        try:
                            if "Close" in data.columns.get_level_values(0):
                                df = pd.DataFrame({
                                    "Open": data["Open"][symbol] if symbol in data["Open"].columns else None,
                                    "High": data["High"][symbol] if symbol in data["High"].columns else None,
                                    "Low": data["Low"][symbol] if symbol in data["Low"].columns else None,
                                    "Close": data["Close"][symbol] if symbol in data["Close"].columns else None,
                                    "Volume": data["Volume"][symbol] if symbol in data["Volume"].columns else None,
                                }).dropna()

                                if len(df) > 100:
                                    prices[symbol] = df
                                else:
                                    failed_symbols.append(symbol)
                            else:
                                failed_symbols.append(symbol)
                        except Exception:
                            failed_symbols.append(symbol)

            except Exception as e:
                logger.warning(f"Batch fetch failed: {e}")
                failed_symbols.extend(batch)

        print(f"Successfully fetched: {len(prices)} symbols")
        print(f"Failed: {len(failed_symbols)} symbols")

        if len(prices) < 50:
            print("Error: Not enough price data")
            return 1

        # 有効なユニバースに更新
        universe = list(prices.keys())
        print(f"Active universe: {len(universe)} symbols")
        print()

        # バックテスト実行
        print("Running backtest...")
        if hasattr(engine, "run"):
            # VectorBTStyleEngine.run(universe, prices, config, weights_func)
            result = engine.run(universe, prices, config, None)
        else:
            print("Error: Engine has no run method")
            return 1

        elapsed = time.time() - start_time
        print(f"Backtest completed in {elapsed:.1f} seconds")
        print()

        # 結果表示
        print("=" * 60)
        print("  RESULTS")
        print("=" * 60)

        if hasattr(result, "total_return"):
            print(f"  Total Return:      {result.total_return * 100:>10.2f}%")
            print(f"  Annual Return:     {result.annual_return * 100:>10.2f}%")
            print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.3f}")
            print(f"  Max Drawdown:      {result.max_drawdown * 100:>10.2f}%")
            if hasattr(result, "volatility"):
                print(f"  Volatility:        {result.volatility * 100:>10.2f}%")
            if hasattr(result, "win_rate"):
                print(f"  Win Rate:          {result.win_rate * 100:>10.2f}%")
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print()

        # 結果保存
        Path("results").mkdir(exist_ok=True)

        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {
                "total_return": getattr(result, "total_return", 0),
                "annual_return": getattr(result, "annual_return", 0),
                "sharpe_ratio": getattr(result, "sharpe_ratio", 0),
                "max_drawdown": getattr(result, "max_drawdown", 0),
            }

        # メタデータ追加
        result_dict["metadata"] = {
            "task_id": "task_028_1",
            "description": "BT-001: Daily Backtest 15Y",
            "universe_size": len(universe),
            "period": f"{start_date.date()} to {end_date.date()}",
            "rebalance_frequency": rebalance_frequency,
            "initial_capital": initial_capital,
            "transaction_cost_bps": transaction_cost_bps,
            "engine": engine.__class__.__name__,
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")
        print()
        print("=" * 60)
        print("  BACKTEST COMPLETE")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.exception("Backtest failed")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_backtest())
