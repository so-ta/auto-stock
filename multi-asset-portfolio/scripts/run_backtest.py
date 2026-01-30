#!/usr/bin/env python3
"""
Unified Backtest Runner (v3.0)

単一エントリーポイントからすべてのバックテスト機能を実行。
S3キャッシュが必須です。

Required Environment Variables:
    BACKTEST_S3_BUCKET  S3バケット名 (default: stock-local-dev-014498665038)
    BACKTEST_S3_PREFIX  S3プレフィックス (default: .cache)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  AWS認証情報

Usage:
    # 標準バックテスト（monthly）
    python scripts/run_backtest.py

    # 全頻度実行
    python scripts/run_backtest.py -f all

    # テストモード（5銘柄×1年）
    python scripts/run_backtest.py --test

    # チェックポイント付き
    python scripts/run_backtest.py -f daily --checkpoint-interval 10

    # チェックポイントから再開
    python scripts/run_backtest.py -f daily --resume checkpoints/cp_0080.pkl

    # カスタム期間（15年相当）
    python scripts/run_backtest.py -f monthly --start 2010-01-01 --end 2025-01-01

    # S3を一時的に無効化（開発用）
    python scripts/run_backtest.py --no-s3 --test

Options:
    -f, --frequency     リバランス頻度 (daily/weekly/monthly/all) [default: monthly]
    -t, --test          テストモード（5銘柄×1年）
    --start             開始日 (YYYY-MM-DD)
    --end               終了日 (YYYY-MM-DD)
    -o, --output        出力パス
    --checkpoint-interval   N回毎にチェックポイント保存
    --resume            チェックポイントから再開
    --checkpoint-dir    チェックポイント保存ディレクトリ
    --no-s3             S3キャッシュを無効化（開発用、非推奨）
    --cache-dir         ローカルキャッシュディレクトリ
    -v, --verbose       詳細出力
    --json              JSON形式出力

Standard v3.0 Parameters:
    - initial_capital: 1,000,000
    - transaction_cost_bps: 10 (0.1%)
    - slippage_bps: 5 (0.05%)
    - period: 2010-01-01 to 2024-12-31
    - universe: config/universe_standard.yaml
    - engine: VectorBTStyleEngine with Pipeline weights
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.base import UnifiedBacktestResult
from src.orchestrator.unified_executor import UnifiedExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 統一規格パラメータ (Backtest Standard v3.0)
# =============================================================================

STANDARD_CONFIG = {
    "version": "3.0",
    "initial_capital": 1_000_000,
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
    "start_date": "2010-01-01",
    "end_date": "2024-12-31",
    "universe_file": "config/universe_standard.yaml",
    "cache_dir": "data/cache/standard_universe",
    "risk_free_rate": 0.02,
    "max_weight": 0.20,
}

# テスト用パラメータ（5銘柄×1年）
TEST_CONFIG = {
    **STANDARD_CONFIG,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "universe_file": None,
}

TEST_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


# =============================================================================
# Universe Loading
# =============================================================================

def load_universe(config: Dict[str, Any], test_mode: bool = False) -> List[str]:
    """ユニバースを読み込む"""
    if test_mode:
        logger.info(f"Using test universe: {TEST_UNIVERSE}")
        return TEST_UNIVERSE

    universe_path = Path(PROJECT_ROOT) / config["universe_file"]

    if not universe_path.exists():
        fallback_paths = [
            Path(PROJECT_ROOT) / "config" / "universe.yaml",
        ]
        for path in fallback_paths:
            if path.exists():
                logger.warning(f"Standard universe not found, using fallback: {path.name}")
                universe_path = path
                break
        else:
            raise FileNotFoundError(f"No universe file found. Expected: {config['universe_file']}")

    with open(universe_path, "r") as f:
        data = yaml.safe_load(f)

    tickers = data.get("tickers", [])
    logger.info(f"Loaded {len(tickers)} tickers from {universe_path.name}")

    return tickers


# =============================================================================
# Price Data Loading
# =============================================================================

def load_prices(
    universe: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """価格データを読み込む"""
    logger.info(f"Loading prices for {len(universe)} symbols...")

    if cache_dir:
        cache_path = Path(PROJECT_ROOT) / cache_dir
        if cache_path.exists():
            prices = _load_from_cache(universe, cache_path, start_date, end_date)
            if prices:
                return prices

    prices = _fetch_from_yfinance(universe, start_date, end_date)
    return prices


def _load_from_cache(
    universe: List[str],
    cache_path: Path,
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, pd.DataFrame]]:
    """キャッシュから価格データを読み込む"""
    try:
        parquet_files = list(cache_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {cache_path}")
            return None

        prices = {}
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        cache_symbol_map = {}
        for pf in parquet_files:
            cache_name = pf.stem
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            cache_symbol_map[original_symbol] = pf

        for symbol in universe:
            pf = cache_symbol_map.get(symbol)
            if pf is None:
                continue

            try:
                df = pd.read_parquet(pf)

                new_cols = {}
                for col in df.columns:
                    col_str = str(col)
                    if col_str.startswith("('") and "'" in col_str:
                        import ast
                        try:
                            parsed = ast.literal_eval(col_str)
                            if isinstance(parsed, tuple) and len(parsed) > 0:
                                base_name = parsed[0]
                                if base_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                    new_cols[col] = base_name.lower()
                                elif base_name == 'Date' or base_name == '':
                                    new_cols[col] = 'Date'
                                else:
                                    new_cols[col] = base_name
                            else:
                                new_cols[col] = col
                        except (ValueError, SyntaxError):
                            new_cols[col] = col
                    elif isinstance(col, tuple):
                        new_cols[col] = col[0].lower() if col[0] else str(col)
                    else:
                        new_cols[col] = col
                df = df.rename(columns=new_cols)

                if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")

                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                if len(df) > 0:
                    prices[symbol] = df

            except Exception as e:
                logger.debug(f"Failed to load {symbol}: {e}")
                continue

        coverage = len(prices) / len(universe) if universe else 0
        logger.info(f"Loaded {len(prices)}/{len(universe)} symbols from cache ({coverage:.1%})")

        if len(prices) >= len(universe) * 0.8:
            return prices

        logger.warning(f"Cache coverage too low ({coverage:.1%}), falling back to yfinance")

    except Exception as e:
        logger.warning(f"Failed to load from cache: {e}")

    return None


def _fetch_from_yfinance(
    universe: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """yfinanceで価格データを取得"""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    logger.info("Fetching data from yfinance...")

    tickers_str = " ".join(universe)
    data = yf.download(
        tickers_str,
        start=start_date,
        end=end_date,
        progress=False,
        group_by="ticker",
    )

    prices = {}
    for symbol in universe:
        try:
            if len(universe) == 1:
                df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                df = data[symbol][["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.dropna()
            if len(df) > 0:
                prices[symbol] = df
        except (KeyError, TypeError):
            logger.warning(f"Failed to get data for {symbol}")

    logger.info(f"Fetched {len(prices)}/{len(universe)} symbols")
    return prices


# =============================================================================
# Storage Configuration
# =============================================================================

class S3ConfigurationError(Exception):
    """S3設定エラー"""
    pass


def validate_s3_config() -> tuple[str, str]:
    """
    S3設定を検証する

    Returns:
        tuple: (s3_bucket, s3_prefix)

    Raises:
        S3ConfigurationError: S3設定が不正な場合
    """
    s3_bucket = os.environ.get("BACKTEST_S3_BUCKET", "stock-local-dev-014498665038")
    s3_prefix = os.environ.get("BACKTEST_S3_PREFIX", ".cache")

    # AWS認証情報の確認
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_profile = os.environ.get("AWS_PROFILE")

    # 認証情報がない場合は ~/.aws/credentials を確認
    has_credentials = bool(aws_access_key and aws_secret_key) or bool(aws_profile)

    if not has_credentials:
        # ~/.aws/credentials ファイルの存在確認
        credentials_path = Path.home() / ".aws" / "credentials"
        if not credentials_path.exists():
            raise S3ConfigurationError(
                "S3 cache is required but AWS credentials are not configured.\n"
                "Please set one of the following:\n"
                "  1. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                "  2. Environment variable: AWS_PROFILE\n"
                "  3. AWS credentials file: ~/.aws/credentials\n"
                "\n"
                "For development/testing without S3, use --no-s3 flag (not recommended for production)."
            )

    return s3_bucket, s3_prefix


def setup_storage_config(disable_s3: bool = False, cache_dir: Optional[str] = None) -> Optional["StorageConfig"]:
    """
    ストレージ設定をセットアップ

    Args:
        disable_s3: S3を無効化するかどうか（開発用）
        cache_dir: ローカルキャッシュディレクトリ

    Returns:
        StorageConfig: S3ストレージ設定（disable_s3=TrueならNone）

    Raises:
        S3ConfigurationError: S3設定が不正な場合（disable_s3=False時）
    """
    if disable_s3:
        logger.warning("S3 cache is DISABLED. This is not recommended for production use.")
        return None

    # S3設定の検証
    s3_bucket, s3_prefix = validate_s3_config()

    from src.utils.storage_backend import StorageConfig

    storage_config = StorageConfig(
        backend="s3",
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        local_cache_enabled=True,
        local_cache_path=cache_dir or "/tmp/.backtest_cache",
        local_cache_ttl_hours=24,
    )

    logger.info(f"S3 cache mode enabled: s3://{s3_bucket}/{s3_prefix}")
    return storage_config


# =============================================================================
# Backtest Execution (v3.0 - Pipeline統合)
# =============================================================================

def run_backtest(
    frequency: str,
    output_path: str,
    test_mode: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    checkpoint_interval: int = 0,
    resume_from: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    storage_config: Optional["StorageConfig"] = None,
    json_output: bool = False,
) -> UnifiedBacktestResult:
    """
    Pipeline統合バックテスト実行

    Pipelineの全最適化機能（NCO、Kelly、レジーム適応等）を使用。
    チェックポイント機能により中断・再開が可能。

    Args:
        frequency: リバランス頻度 (daily/weekly/monthly)
        output_path: 結果出力パス
        test_mode: テストモード
        start_date: 開始日（オプション）
        end_date: 終了日（オプション）
        checkpoint_interval: チェックポイント保存間隔（0=無効）
        resume_from: 再開するチェックポイントファイル
        checkpoint_dir: チェックポイント保存ディレクトリ
        storage_config: S3ストレージ設定
        json_output: JSON形式で出力

    Returns:
        UnifiedBacktestResult: バックテスト結果
    """
    config_dict = (TEST_CONFIG if test_mode else STANDARD_CONFIG).copy()

    if start_date:
        config_dict["start_date"] = start_date
    if end_date:
        config_dict["end_date"] = end_date

    if not json_output:
        logger.info("=" * 60)
        logger.info(f"UNIFIED BACKTEST RUNNER (v{config_dict['version']})")
        logger.info(f"Frequency: {frequency}")
        logger.info(f"Period: {config_dict['start_date']} to {config_dict['end_date']}")
        logger.info(f"Initial Capital: ${config_dict['initial_capital']:,}")
        logger.info(f"Transaction Cost: {config_dict['transaction_cost_bps']} bps")
        logger.info(f"Slippage: {config_dict['slippage_bps']} bps")
        if storage_config:
            logger.info(f"Storage: S3 ({storage_config.s3_bucket})")
        else:
            logger.info("Storage: Local")
        logger.info("=" * 60)

    # ユニバース読み込み
    universe = load_universe(config_dict, test_mode=test_mode)

    # 価格データ読み込み
    prices = load_prices(
        universe,
        config_dict["start_date"],
        config_dict["end_date"],
        config_dict.get("cache_dir"),
    )

    # 有効な銘柄のみに絞る
    valid_universe = [s for s in universe if s in prices]
    logger.info(f"Valid universe: {len(valid_universe)} symbols")

    if len(valid_universe) == 0:
        raise ValueError("No valid symbols in universe after price loading")

    # UnifiedExecutor でバックテスト実行
    # Settings作成（storage_configがあれば反映）
    settings = None
    if storage_config:
        from src.config.settings import Settings, StorageSettings
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

    logger.info("Initializing UnifiedExecutor (Pipeline-integrated)...")
    executor = UnifiedExecutor(settings=settings)

    logger.info("Running Pipeline-integrated backtest...")
    start_time = datetime.now()

    # Checkpoint support
    if checkpoint_interval > 0 or resume_from:
        logger.info(
            f"Checkpoint mode: interval={checkpoint_interval}, "
            f"resume={resume_from}, dir={checkpoint_dir}"
        )
        result = executor.run_backtest_with_checkpoint(
            universe=valid_universe,
            prices=prices,
            start_date=config_dict["start_date"],
            end_date=config_dict["end_date"],
            frequency=frequency,
            initial_capital=config_dict["initial_capital"],
            transaction_cost_bps=config_dict["transaction_cost_bps"],
            slippage_bps=config_dict["slippage_bps"],
            max_weight=config_dict["max_weight"],
            risk_free_rate=config_dict["risk_free_rate"],
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=Path(PROJECT_ROOT) / checkpoint_dir,
            resume_from=Path(resume_from) if resume_from else None,
        )
    else:
        result = executor.run_backtest(
            universe=valid_universe,
            prices=prices,
            start_date=config_dict["start_date"],
            end_date=config_dict["end_date"],
            frequency=frequency,
            initial_capital=config_dict["initial_capital"],
            transaction_cost_bps=config_dict["transaction_cost_bps"],
            slippage_bps=config_dict["slippage_bps"],
            max_weight=config_dict["max_weight"],
            risk_free_rate=config_dict["risk_free_rate"],
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Backtest completed in {elapsed:.2f} seconds")

    # 結果保存
    save_result(result, output_path, frequency, config_dict)

    # サマリ出力
    if json_output:
        print_json_summary(result, frequency, config_dict)
    else:
        print_summary(result, frequency)

    return result


def save_result(
    result: UnifiedBacktestResult,
    output_path: str,
    frequency: str,
    config_dict: Dict[str, Any],
) -> None:
    """結果を保存"""
    output_file = Path(PROJECT_ROOT) / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "standard_version": config_dict["version"],
        "generated_at": datetime.now().isoformat(),
        "frequency": frequency,
        "mode": "pipeline_integrated",
        "config": {
            "initial_capital": config_dict["initial_capital"],
            "transaction_cost_bps": config_dict["transaction_cost_bps"],
            "slippage_bps": config_dict["slippage_bps"],
            "start_date": config_dict["start_date"],
            "end_date": config_dict["end_date"],
        },
        "engine": result.engine_name,
        "metrics": {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "win_rate": result.win_rate,
        },
        "trading_stats": {
            "n_days": result.n_days,
            "n_rebalances": result.n_rebalances,
            "total_turnover": result.total_turnover,
            "total_transaction_costs": result.total_transaction_costs,
        },
        "values": {
            "initial_value": result.initial_value,
            "final_value": result.final_value,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Result saved to: {output_file}")


def print_summary(result: UnifiedBacktestResult, frequency: str) -> None:
    """サマリを出力"""
    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY ({frequency.upper()}) - Pipeline Integrated")
    print("=" * 60)
    print(f"  Engine:          {result.engine_name}")
    print(f"  Period:          {result.n_days} trading days")
    print("-" * 60)
    print(f"  Total Return:    {result.total_return * 100:>10.2f}%")
    print(f"  Annual Return:   {result.annual_return * 100:>10.2f}%")
    print(f"  Volatility:      {result.volatility * 100:>10.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>10.3f}")
    print(f"  Max Drawdown:    {result.max_drawdown * 100:>10.2f}%")
    print("-" * 60)
    print(f"  Initial Value:   ${result.initial_value:>12,.2f}")
    print(f"  Final Value:     ${result.final_value:>12,.2f}")
    print("=" * 60 + "\n")


def print_json_summary(result: UnifiedBacktestResult, frequency: str, config: Dict[str, Any]) -> None:
    """JSON形式でサマリを出力"""
    output = {
        "frequency": frequency,
        "version": config["version"],
        "metrics": {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar_ratio": result.calmar_ratio,
            "win_rate": result.win_rate,
        },
        "values": {
            "initial_value": result.initial_value,
            "final_value": result.final_value,
        },
    }
    print(json.dumps(output, indent=2))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Backtest Runner (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--frequency", "-f",
        choices=["daily", "weekly", "monthly", "all"],
        default="monthly",
        help="Rebalance frequency (default: monthly)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: results/backtest_{frequency}.json)",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode (5 symbols, 1 year)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides standard config.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Overrides standard config.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON format output",
    )
    # Checkpoint support
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save checkpoint every N rebalances (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file (e.g., checkpoints/cp_0080.pkl)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)",
    )
    # Storage options
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Disable S3 cache (development only, not recommended)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local cache directory",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # S3ストレージ設定（必須）
    try:
        storage_config = setup_storage_config(
            disable_s3=args.no_s3,
            cache_dir=args.cache_dir
        )
    except S3ConfigurationError as e:
        logger.error(str(e))
        sys.exit(1)

    # 実行対象の頻度
    frequencies = ["daily", "weekly", "monthly"] if args.frequency == "all" else [args.frequency]

    results = {}
    for freq in frequencies:
        output = args.output or f"results/backtest_{freq}.json"
        if args.frequency == "all":
            output = f"results/backtest_{freq}.json"

        try:
            result = run_backtest(
                freq, output,
                test_mode=args.test,
                start_date=args.start,
                end_date=args.end,
                checkpoint_interval=args.checkpoint_interval,
                resume_from=args.resume,
                checkpoint_dir=args.checkpoint_dir,
                storage_config=storage_config,
                json_output=args.json,
            )
            results[freq] = result
        except Exception as e:
            logger.error(f"Failed to run {freq} backtest: {e}")
            import traceback
            traceback.print_exc()
            if args.frequency != "all":
                raise

    # 全頻度比較（--frequency all オプション時）
    if args.frequency == "all" and len(results) > 1 and not args.json:
        print("\n" + "=" * 60)
        print("  FREQUENCY COMPARISON (Pipeline Integrated)")
        print("=" * 60)
        print(f"  {'Frequency':<10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10}")
        print("-" * 60)
        for freq, res in results.items():
            print(
                f"  {freq:<10} "
                f"{res.annual_return * 100:>9.2f}% "
                f"{res.sharpe_ratio:>10.3f} "
                f"{res.max_drawdown * 100:>9.2f}%"
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
