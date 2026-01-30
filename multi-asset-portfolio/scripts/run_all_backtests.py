#!/usr/bin/env python3
"""
全頻度バックテスト実行スクリプト

既存のrun_backtest_with_checkpoint()を使用して、
日次・週次・月次のバックテストを順次実行。

Usage:
    # 全て実行（ローカルキャッシュ）
    python scripts/run_all_backtests.py

    # 特定の頻度のみ
    python scripts/run_all_backtests.py --frequency monthly

    # チェックポイントから再開
    python scripts/run_all_backtests.py --frequency weekly --resume checkpoints/weekly/cp_0050.pkl

    # S3キャッシュモード
    AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx \\
    python scripts/run_all_backtests.py --s3

    # S3 + 特定頻度
    python scripts/run_all_backtests.py --s3 --frequency daily
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageConfig

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "backtest_all.log"),
    ]
)
logger = logging.getLogger(__name__)

# 実行順序（計算コスト昇順）
FREQUENCIES = ["monthly", "weekly", "daily"]

# バックテスト設定
CONFIG = {
    "start_date": "2010-01-01",
    "end_date": "2025-01-01",
    "initial_capital": 1_000_000,
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
    "max_weight": 0.20,
    "risk_free_rate": 0.02,
    "checkpoint_interval": 10,
}

UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG",
    "GLD", "SLV", "USO", "UNG", "DBA", "XLF", "XLE", "XLK", "XLV", "XLI",
]


def fetch_prices(universe: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """価格データを取得"""
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf

    logger.info(f"{len(universe)}銘柄の価格データを取得中...")
    prices = {}

    for ticker in universe:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                prices[ticker] = df
        except Exception as e:
            logger.warning(f"{ticker}: 取得失敗 ({e})")

    logger.info(f"価格データ取得完了: {len(prices)}/{len(universe)}銘柄")
    return prices


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """最新のチェックポイントを探す"""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("cp_*.pkl"))
    return checkpoints[-1] if checkpoints else None


def run_backtest(
    frequency: str,
    prices: Dict[str, pd.DataFrame],
    resume_from: Optional[Path] = None,
    storage_config: Optional["StorageConfig"] = None,
) -> dict:
    """単一頻度のバックテストを実行"""
    from src.config.settings import Settings, StorageSettings
    from src.orchestrator.unified_executor import UnifiedExecutor

    logger.info(f"{'='*60}")
    logger.info(f"バックテスト開始: {frequency}")
    if storage_config:
        logger.info(f"ストレージ: S3 ({storage_config.s3_bucket})")
    else:
        logger.info("ストレージ: ローカル")
    logger.info(f"{'='*60}")

    # Settings作成（storage_configがあれば反映）
    settings = None
    if storage_config:
        storage_settings = StorageSettings(
            backend=storage_config.backend,
            base_path=storage_config.base_path,
            s3_bucket=storage_config.s3_bucket,
            s3_prefix=storage_config.s3_prefix,
            s3_region=storage_config.s3_region,
            local_cache_enabled=storage_config.local_cache_enabled,
            local_cache_path=storage_config.local_cache_path,
            local_cache_ttl_hours=storage_config.local_cache_ttl_hours,
        )
        settings = Settings(storage=storage_settings)

    executor = UnifiedExecutor(settings=settings)
    checkpoint_dir = RESULTS_DIR / "checkpoints" / frequency

    # 再開指定がなければ最新チェックポイントを探す
    if resume_from is None:
        resume_from = find_latest_checkpoint(checkpoint_dir)
        if resume_from:
            logger.info(f"チェックポイントから再開: {resume_from}")

    result = executor.run_backtest_with_checkpoint(
        universe=list(prices.keys()),
        prices=prices,
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        frequency=frequency,
        initial_capital=CONFIG["initial_capital"],
        transaction_cost_bps=CONFIG["transaction_cost_bps"],
        slippage_bps=CONFIG["slippage_bps"],
        max_weight=CONFIG["max_weight"],
        risk_free_rate=CONFIG["risk_free_rate"],
        checkpoint_interval=CONFIG["checkpoint_interval"],
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )

    summary = {
        "frequency": frequency,
        "sharpe_ratio": round(result.sharpe_ratio, 4) if result.sharpe_ratio else None,
        "annual_return": round(result.annual_return, 4) if result.annual_return else None,
        "max_drawdown": round(result.max_drawdown, 4) if result.max_drawdown else None,
    }

    result_file = RESULTS_DIR / f"backtest_{frequency}_15y.json"
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"結果保存: {result_file}")
    logger.info(f"Sharpe: {summary['sharpe_ratio']}, Return: {summary['annual_return']}")

    return summary


def setup_storage_config(use_s3: bool) -> Optional["StorageConfig"]:
    """ストレージ設定をセットアップ"""
    if not use_s3:
        return None

    from src.utils.storage_backend import StorageConfig

    # S3バケット名（環境変数 or デフォルト）
    s3_bucket = os.environ.get("BACKTEST_S3_BUCKET", "stock-local-dev-014498665038")
    s3_prefix = os.environ.get("BACKTEST_S3_PREFIX", ".cache")

    storage_config = StorageConfig(
        backend="s3",
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        local_cache_enabled=True,
        local_cache_path="/tmp/.backtest_cache",
        local_cache_ttl_hours=24,
    )

    logger.info(f"S3キャッシュモード有効: s3://{s3_bucket}/{s3_prefix}")
    return storage_config


def main():
    parser = argparse.ArgumentParser(description="全頻度バックテスト実行")
    parser.add_argument("--frequency", "-f", type=str, help="特定の頻度のみ実行")
    parser.add_argument("--resume", "-r", type=str, help="チェックポイントファイルパス")
    parser.add_argument("--s3", action="store_true", help="S3キャッシュモードを使用")
    args = parser.parse_args()

    # S3ストレージ設定
    storage_config = setup_storage_config(args.s3)

    # 価格データを一度だけ取得
    prices = fetch_prices(UNIVERSE, CONFIG["start_date"], CONFIG["end_date"])
    if len(prices) < len(UNIVERSE) * 0.5:
        logger.error(f"価格データ不足: {len(prices)}/{len(UNIVERSE)}銘柄")
        return

    # 実行対象の頻度
    frequencies = [args.frequency] if args.frequency else FREQUENCIES
    resume_from = Path(args.resume) if args.resume else None

    for freq in frequencies:
        if freq not in FREQUENCIES:
            logger.warning(f"不明な頻度: {freq}")
            continue

        try:
            run_backtest(
                freq,
                prices,
                resume_from if freq == frequencies[0] else None,
                storage_config=storage_config,
            )
        except KeyboardInterrupt:
            logger.warning(f"{freq}が中断されました。再開コマンド:")
            logger.warning(f"  python scripts/run_all_backtests.py -f {freq}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"{freq}でエラー: {e}")
            continue

    logger.info("全バックテスト完了")


if __name__ == "__main__":
    main()
