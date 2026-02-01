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

    # カスタムベンチマーク
    python scripts/run_backtest.py -f monthly --benchmarks SPY,QQQ,DIA,IWM

    # レポート生成をスキップ
    python scripts/run_backtest.py -f monthly --no-report

Options:
    -f, --frequency     リバランス頻度 (daily/weekly/monthly/all) [default: monthly]
    -t, --test          テストモード（5銘柄×1年）
    --start             開始日 (YYYY-MM-DD)
    --end               終了日 (YYYY-MM-DD)
    -o, --output        出力パス
    --checkpoint-interval   N回毎にチェックポイント保存
    --resume            チェックポイントから再開
    --checkpoint-dir    チェックポイント保存ディレクトリ
    --cache-dir         ローカルキャッシュディレクトリ
    -v, --verbose       詳細出力
    --json              JSON形式出力

Report Options:
    --no-report         レポート生成をスキップ
    --benchmarks        比較ベンチマーク (カンマ区切り) [default: SPY,QQQ]
    --report-dir        レポート出力ディレクトリ [default: reports]

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

from dotenv import load_dotenv
import pandas as pd
import yaml

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

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
    "lookback_days": 300,  # シグナル・共分散計算用のルックバック期間（営業日）
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
    """価格データを読み込む

    Args:
        universe: シンボルのリスト
        start_date: 開始日
        end_date: 終了日
        cache_dir: キャッシュディレクトリ（読み込み・保存に使用）

    Returns:
        シンボル -> DataFrameの辞書
    """
    logger.info(f"Loading prices for {len(universe)} symbols...")

    prices = {}
    missing_symbols = list(universe)

    # キャッシュから読み込み（部分的でもOK）
    if cache_dir:
        cache_path = Path(PROJECT_ROOT) / cache_dir
        if cache_path.exists():
            cached_prices, missing_symbols = _load_from_cache_partial(
                universe, cache_path, start_date, end_date
            )
            prices.update(cached_prices)

            if not missing_symbols:
                logger.info(f"All {len(prices)} symbols loaded from cache")
                return prices

    # 不足分をyfinanceから取得
    if missing_symbols:
        fetched_prices = _fetch_from_yfinance(
            missing_symbols, start_date, end_date, cache_dir=cache_dir
        )
        prices.update(fetched_prices)

    return prices


def _load_from_cache_partial(
    universe: List[str],
    cache_path: Path,
    start_date: str,
    end_date: str,
) -> tuple[Dict[str, pd.DataFrame], List[str]]:
    """キャッシュから価格データを部分的に読み込む

    Args:
        universe: シンボルのリスト
        cache_path: キャッシュディレクトリパス
        start_date: 開始日
        end_date: 終了日

    Returns:
        (読み込めた価格データ, 読み込めなかったシンボルのリスト)
    """
    prices = {}
    missing_symbols = []
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    try:
        parquet_files = list(cache_path.glob("*.parquet"))
        if not parquet_files:
            logger.debug(f"No parquet files found in {cache_path}")
            return {}, list(universe)

        cache_symbol_map = {}
        for pf in parquet_files:
            cache_name = pf.stem
            original_symbol = cache_name.replace("_X", "=X") if cache_name.endswith("_X") else cache_name
            cache_symbol_map[original_symbol] = pf

        for symbol in universe:
            pf = cache_symbol_map.get(symbol)
            if pf is None:
                missing_symbols.append(symbol)
                continue

            try:
                df = pd.read_parquet(pf)

                # カラム名の正規化
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

                # 日付範囲でフィルタリング
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                if len(df) > 0:
                    prices[symbol] = df
                else:
                    # キャッシュはあるが要求された期間のデータがない
                    missing_symbols.append(symbol)

            except Exception as e:
                logger.debug(f"Failed to load {symbol}: {e}")
                missing_symbols.append(symbol)

        if prices:
            logger.info(f"Loaded {len(prices)}/{len(universe)} symbols from cache")
        if missing_symbols:
            logger.debug(f"Missing from cache: {len(missing_symbols)} symbols")

    except Exception as e:
        logger.warning(f"Failed to load from cache: {e}")
        return {}, list(universe)

    return prices, missing_symbols


def _fetch_from_yfinance(
    universe: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, pd.DataFrame]:
    """yfinanceで価格データを取得（バッチ処理対応）

    大量銘柄を小さいバッチに分割してダウンロードし、
    スレッド制限エラーを回避する。

    Args:
        universe: シンボルのリスト
        start_date: 開始日
        end_date: 終了日
        cache_dir: キャッシュ保存先ディレクトリ（指定時は取得後にキャッシュ保存）
        batch_size: 1回のダウンロードで取得する銘柄数（default: 100）

    Returns:
        シンボル -> DataFrameの辞書
    """
    import time

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    prices = {}
    total_symbols = len(universe)
    n_batches = (total_symbols + batch_size - 1) // batch_size

    logger.info(f"Fetching {total_symbols} symbols from yfinance in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_symbols)
        batch = universe[start_idx:end_idx]

        logger.info(f"Batch {batch_idx + 1}/{n_batches}: fetching {len(batch)} symbols...")

        try:
            tickers_str = " ".join(batch)
            data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
                threads=False,  # シングルスレッドでスレッド制限を回避
            )

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = data[symbol][["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna()
                    if len(df) > 0:
                        prices[symbol] = df
                except (KeyError, TypeError):
                    pass  # 個別のエラーは無視

            # バッチ間で少し待機（API制限回避）
            if batch_idx < n_batches - 1:
                time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Batch {batch_idx + 1} failed: {e}")
            continue

    logger.info(f"Fetched {len(prices)}/{total_symbols} symbols")

    # キャッシュに保存
    if cache_dir and prices:
        _save_to_cache(prices, cache_dir)

    return prices


def _save_to_cache(
    prices: Dict[str, pd.DataFrame],
    cache_dir: str,
) -> None:
    """
    取得した価格データをキャッシュに保存（既存データとマージ）

    Args:
        prices: シンボル -> DataFrameの辞書
        cache_dir: キャッシュ保存先ディレクトリ（PROJECT_ROOTからの相対パス）
    """
    cache_path = Path(PROJECT_ROOT) / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    merged_count = 0
    for symbol, new_df in prices.items():
        try:
            # ファイル名に使用できない文字を置換
            safe_symbol = symbol.replace("=", "_").replace("/", "_")
            parquet_path = cache_path / f"{safe_symbol}.parquet"

            # 既存のキャッシュがあればマージ
            if parquet_path.exists():
                try:
                    existing_df = pd.read_parquet(parquet_path)

                    # 既存データのカラム名を正規化
                    if not all(c in ['open', 'high', 'low', 'close', 'volume'] for c in existing_df.columns):
                        # 古い形式のデータは無視して新しいデータで上書き
                        new_df.to_parquet(parquet_path)
                        saved_count += 1
                        continue

                    # DatetimeIndexに変換
                    if not isinstance(existing_df.index, pd.DatetimeIndex):
                        if "Date" in existing_df.columns:
                            existing_df["Date"] = pd.to_datetime(existing_df["Date"])
                            existing_df = existing_df.set_index("Date")

                    # マージ（重複は新しいデータを優先）
                    combined = pd.concat([existing_df, new_df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    combined.to_parquet(parquet_path)
                    merged_count += 1
                except Exception:
                    # マージに失敗したら新しいデータで上書き
                    new_df.to_parquet(parquet_path)
                    saved_count += 1
            else:
                new_df.to_parquet(parquet_path)
                saved_count += 1
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")

    total = saved_count + merged_count
    logger.info(f"Cached {total}/{len(prices)} symbols (new: {saved_count}, merged: {merged_count})")


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
                "  3. AWS credentials file: ~/.aws/credentials"
            )

    return s3_bucket, s3_prefix


def setup_storage_config(cache_dir: Optional[str] = None) -> "StorageConfig":
    """
    ストレージ設定をセットアップ

    Args:
        cache_dir: ローカルキャッシュディレクトリ

    Returns:
        StorageConfig: S3ストレージ設定

    Raises:
        S3ConfigurationError: S3設定が不正な場合
    """
    # S3設定の検証
    s3_bucket, s3_prefix = validate_s3_config()

    from src.utils.storage_backend import StorageConfig

    storage_config = StorageConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        base_path=cache_dir or ".cache",
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
    generate_report_flag: bool = True,
    benchmarks: Optional[List[str]] = None,
    report_dir: str = "reports",
    trading_filter: Optional[str] = None,
    universe_file: Optional[str] = None,
    max_assets: Optional[int] = None,
    save_archive: bool = True,
    archive_name: Optional[str] = None,
    archive_tags: Optional[List[str]] = None,
    archive_description: str = "",
) -> UnifiedBacktestResult:
    """
    Pipeline統合バックテスト実行

    Pipelineの全最適化機能（NCO、Kelly、レジーム適応等）を使用。
    チェックポイント機能により中断・再開が可能。

    v2.3: シグナル事前計算による40倍高速化
    - シグナルは事前計算されてキャッシュされる
    - 15年バックテスト: 21時間 → 30分

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
        max_assets: 最大アセット数（Top-Nフィルター）。大規模ユニバース向け最適化。

    Returns:
        UnifiedBacktestResult: バックテスト結果
    """
    config_dict = (TEST_CONFIG if test_mode else STANDARD_CONFIG).copy()

    if start_date:
        config_dict["start_date"] = start_date
    if end_date:
        config_dict["end_date"] = end_date
    if universe_file:
        config_dict["universe_file"] = universe_file

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
        logger.info("Precomputed Signals: Enabled (40x speedup)")
        if trading_filter:
            logger.info(f"Trading Filter: {trading_filter} (signals use full universe)")
        if max_assets:
            logger.info(f"Max Assets: {max_assets} (Top-N filter for large universe optimization)")
        logger.info("=" * 60)

    # ユニバース読み込み
    universe = load_universe(config_dict, test_mode=test_mode)

    # 価格データ読み込み（ルックバック期間を考慮）
    # シグナル・共分散計算のため、バックテスト開始日より前のデータも必要
    lookback_days = config_dict.get("lookback_days", 300)
    backtest_start = pd.Timestamp(config_dict["start_date"])
    data_start = (backtest_start - pd.Timedelta(days=int(lookback_days * 1.5))).strftime("%Y-%m-%d")

    if not json_output:
        logger.info(f"Data fetch range: {data_start} to {config_dict['end_date']} (lookback: {lookback_days} days)")

    prices = load_prices(
        universe,
        data_start,  # ルックバック期間を考慮した開始日
        config_dict["end_date"],
        config_dict.get("cache_dir"),
    )

    # 有効な銘柄のみに絞る（バックテスト開始日以前にデータがある銘柄）
    valid_universe = []
    excluded_symbols = []
    for s in universe:
        if s not in prices:
            continue
        df = prices[s]
        if isinstance(df.index, pd.DatetimeIndex):
            data_start_date = df.index.min()
        else:
            continue
        # バックテスト開始日の30日前までにデータが必要
        required_date = backtest_start - pd.Timedelta(days=30)
        if data_start_date <= required_date:
            valid_universe.append(s)
        else:
            excluded_symbols.append((s, data_start_date.strftime("%Y-%m-%d")))

    if excluded_symbols:
        logger.info(f"Excluded {len(excluded_symbols)} symbols with insufficient history")
        if len(excluded_symbols) <= 10:
            for sym, start in excluded_symbols:
                logger.debug(f"  {sym}: data starts {start}")

    logger.info(f"Valid universe: {len(valid_universe)} symbols")

    if len(valid_universe) == 0:
        raise ValueError("No valid symbols in universe after price loading")

    # 有効な銘柄の価格データのみを使用
    prices = {s: prices[s] for s in valid_universe}

    # 取引フィルター適用（シグナル計算は全銘柄、取引は指定銘柄のみ）
    trading_universe = valid_universe  # デフォルトは全銘柄
    if trading_filter:
        import fnmatch
        trading_universe = [s for s in valid_universe if fnmatch.fnmatch(s, trading_filter)]
        logger.info(f"Trading universe (filter: {trading_filter}): {len(trading_universe)} symbols")
        logger.info(f"Signal universe (full): {len(valid_universe)} symbols")

        if len(trading_universe) == 0:
            raise ValueError(f"No symbols match trading filter: {trading_filter}")

    # データの日付範囲をチェック
    backtest_start_ts = pd.Timestamp(config_dict["start_date"])
    backtest_end_ts = pd.Timestamp(config_dict["end_date"])

    # 最初の銘柄でデータ範囲を確認
    sample_df = prices[valid_universe[0]]
    if isinstance(sample_df.index, pd.DatetimeIndex):
        data_min_date = sample_df.index.min()
        data_max_date = sample_df.index.max()
    else:
        data_min_date = pd.Timestamp(sample_df["timestamp"].min()) if "timestamp" in sample_df.columns else None
        data_max_date = pd.Timestamp(sample_df["timestamp"].max()) if "timestamp" in sample_df.columns else None

    if data_max_date is not None:
        logger.info(f"Available data range: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}")

        # バックテスト期間にデータがあるかチェック
        if backtest_start_ts > data_max_date:
            logger.error(
                f"No data available for backtest period. "
                f"Requested: {config_dict['start_date']} to {config_dict['end_date']}, "
                f"Available: up to {data_max_date.strftime('%Y-%m-%d')}. "
                f"Skipping backtest."
            )
            raise ValueError(
                f"Backtest start date ({config_dict['start_date']}) is after available data "
                f"(ends {data_max_date.strftime('%Y-%m-%d')}). Please use an earlier date range."
            )

        if backtest_end_ts > data_max_date:
            logger.warning(
                f"Backtest end date ({config_dict['end_date']}) is after available data "
                f"(ends {data_max_date.strftime('%Y-%m-%d')}). "
                f"Adjusting end date to {data_max_date.strftime('%Y-%m-%d')}."
            )
            config_dict["end_date"] = data_max_date.strftime("%Y-%m-%d")

    # UnifiedExecutor でバックテスト実行
    # Settings作成（storage_configがあれば反映）
    settings = None
    if storage_config:
        from src.config.settings import Settings, StorageSettings
        storage_settings = StorageSettings(
            s3_bucket=storage_config.s3_bucket,
            s3_prefix=storage_config.s3_prefix,
            s3_region=storage_config.s3_region,
            base_path=storage_config.base_path,
            local_cache_ttl_hours=storage_config.local_cache_ttl_hours,
        )
        settings = Settings(storage=storage_settings)

    # ロギングの統一初期化（structlog + PipelineLogCollector連携）
    # これによりCLI/Viewerで同じログ設定が使用される
    from src.utils.logger import setup_logging, set_log_collector
    from src.utils.pipeline_log_collector import PipelineLogCollector

    # PipelineLogCollectorを初期化
    log_collector = PipelineLogCollector(run_id=f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    set_log_collector(log_collector)

    # Settingsがなければデフォルトをロード
    if settings is None:
        from src.config.settings import load_settings_from_yaml
        settings = load_settings_from_yaml()

    # structlogを設定（標準loggingからのブリッジも含む）
    setup_logging(settings=settings, enable_log_collector=True)

    logger.info("Initializing UnifiedExecutor (Pipeline-integrated)...")
    executor = UnifiedExecutor(settings=settings)

    # Progress tracking setup (always enabled)
    from src.utils.progress_tracker import ProgressTracker
    progress_dir = Path(PROJECT_ROOT) / "results" / ".progress"
    progress_tracker = ProgressTracker(
        progress_dir=progress_dir,
        universe_size=len(valid_universe),
        frequency=frequency,
    )

    # ProgressTrackerとLogCollectorを連携
    log_collector.attach_progress_tracker(progress_tracker)

    logger.info(f"Progress tracking: {progress_tracker.progress_file}")

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
            trading_universe=trading_universe if trading_filter else None,
            max_assets=max_assets,
            progress_tracker=progress_tracker,
        )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Backtest completed in {elapsed:.2f} seconds")

    # Calculate reliability score based on execution logs
    # 統一初期化で作成したlog_collectorを使用（executor.log_collectorは互換性のため残す）
    active_log_collector = log_collector if log_collector is not None else executor.log_collector
    if active_log_collector is not None:
        try:
            from src.backtest.reliability import ReliabilityCalculator
            calculator = ReliabilityCalculator()
            assessment = calculator.calculate(active_log_collector)
            result.reliability = assessment.to_dict()
            if not assessment.is_reliable:
                logger.warning(
                    f"Low reliability score: {assessment.score:.0%} ({assessment.level})"
                )
        except Exception as e:
            logger.warning(f"Failed to calculate reliability: {e}")

    # 結果保存
    save_result(result, output_path, frequency, config_dict)

    # サマリ出力
    if json_output:
        print_json_summary(result, frequency, config_dict)
    else:
        print_summary(result, frequency)

    # レポート生成
    if generate_report_flag and not json_output:
        try:
            report_benchmarks = benchmarks or ["SPY", "QQQ", "^N225"]
            # 配分手法を取得（default.yamlから）
            allocation_method = "nco"  # デフォルト
            try:
                default_config_path = Path(PROJECT_ROOT) / "config" / "default.yaml"
                if default_config_path.exists():
                    with open(default_config_path, "r") as f:
                        default_cfg = yaml.safe_load(f)
                        allocation_method = default_cfg.get("asset_allocation", {}).get("method", "nco")
            except Exception:
                pass  # デフォルトのncoを使用

            generate_report(
                result, frequency, config_dict, report_benchmarks, report_dir,
                universe=valid_universe,
                prices=prices,
                allocation_method=allocation_method,
            )
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")

    # アーカイブ保存
    if save_archive:
        try:
            from src.analysis.result_store import BacktestResultStore

            store = BacktestResultStore()

            # デフォルト名: 条件がわかるように生成
            if archive_name:
                name = archive_name
            else:
                # 例: "Monthly | 2010-2024 | 50 symbols"
                start_year = config_dict["start_date"][:4]
                end_year = config_dict["end_date"][:4]
                n_symbols = len(valid_universe)
                name = f"{frequency.capitalize()} | {start_year}-{end_year} | {n_symbols} symbols"

            # デフォルトタグ: 頻度、期間、テストモードなど
            if archive_tags:
                tags = archive_tags
            else:
                tags = [frequency]
                if test_mode:
                    tags.append("test")
                if trading_filter:
                    tags.append(f"filter:{trading_filter}")
                if max_assets:
                    tags.append(f"top{max_assets}")

            # デフォルト説明: 実行条件の詳細
            if not archive_description:
                desc_parts = [
                    f"Frequency: {frequency}",
                    f"Period: {config_dict['start_date']} to {config_dict['end_date']}",
                    f"Universe: {len(valid_universe)} symbols",
                    f"Initial Capital: ${config_dict['initial_capital']:,.0f}",
                    f"Cost: {config_dict['transaction_cost_bps']}bps + {config_dict['slippage_bps']}bps slippage",
                ]
                if trading_filter:
                    desc_parts.append(f"Trading Filter: {trading_filter}")
                if max_assets:
                    desc_parts.append(f"Max Assets: {max_assets}")
                archive_description = " | ".join(desc_parts)

            # Get log collector from executor for log persistence
            log_collector = executor.log_collector

            archive_id = store.save(
                result=result,
                name=name,
                description=archive_description,
                tags=tags,
                universe=valid_universe,
                log_collector=log_collector,
            )
            logger.info(f"Archive saved: {archive_id}")
            if not json_output:
                print(f"\n  Archive ID: {archive_id}")
        except Exception as e:
            logger.warning(f"Failed to save archive: {e}")

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
# Report Generation
# =============================================================================

def fetch_benchmark_data(
    benchmarks: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """ベンチマーク価格データを取得"""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for benchmark comparison")
        return {}

    benchmark_data = {}
    for ticker in benchmarks:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                benchmark_data[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    return benchmark_data


def calculate_benchmark_metrics(
    prices: pd.Series,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """ベンチマークのメトリクスを計算"""
    import numpy as np

    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return {}

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(252)

    # シャープレシオ: (年率リターン - リスクフリーレート) / 年率ボラティリティ
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
    }


def generate_chartjs_data(
    portfolio_values: pd.Series,
    benchmark_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Chart.js用のデータを生成

    Returns:
        Dict containing:
        - cumulative_return: 累積リターン%データ
        - drawdown: ドローダウン%データ
        - yearly: 年別パフォーマンスデータ
    """
    import numpy as np

    if len(portfolio_values) == 0:
        return {}

    pv_start = portfolio_values.index.min()
    pv_end = portfolio_values.index.max()

    # 日付ラベル（間引いて表示）
    # データ量を減らすため、週次でサンプリング
    pv_weekly = portfolio_values.resample('W').last().dropna()

    # 累積リターン%（ポートフォリオ）
    cumulative_returns = ((pv_weekly / pv_weekly.iloc[0]) - 1) * 100
    dates = [d.strftime('%Y-%m-%d') for d in cumulative_returns.index]
    portfolio_cum_ret = [round(v, 2) for v in cumulative_returns.values]

    # ベンチマーク累積リターン%
    benchmark_cum_rets = {}
    for ticker, df in benchmark_data.items():
        close_col = "adj close" if "adj close" in df.columns else "close"
        if close_col in df.columns:
            prices = df[close_col].dropna()
            prices = prices[(prices.index >= pv_start) & (prices.index <= pv_end)]
            if len(prices) > 0:
                prices_weekly = prices.resample('W').last().dropna()
                # ポートフォリオの日付に合わせる
                prices_aligned = prices_weekly.reindex(cumulative_returns.index, method='ffill')
                if len(prices_aligned.dropna()) > 0:
                    first_valid = prices_aligned.dropna().iloc[0]
                    cum_ret = ((prices_aligned / first_valid) - 1) * 100
                    benchmark_cum_rets[ticker] = [round(v, 2) if not np.isnan(v) else None for v in cum_ret.values]

    # ドローダウン計算
    cumulative = (1 + portfolio_values.pct_change().fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = ((cumulative - running_max) / running_max) * 100
    dd_weekly = drawdowns.resample('W').last().dropna()
    drawdown_data = [round(v, 2) for v in dd_weekly.values]

    # 年別パフォーマンス
    yearly_returns = {}
    for year in sorted(set(portfolio_values.index.year)):
        year_data = portfolio_values[portfolio_values.index.year == year]
        if len(year_data) > 1:
            year_return = ((year_data.iloc[-1] / year_data.iloc[0]) - 1) * 100
            yearly_returns[str(year)] = round(year_return, 2)

    # ベンチマーク年別
    benchmark_yearly = {}
    for ticker, df in benchmark_data.items():
        close_col = "adj close" if "adj close" in df.columns else "close"
        if close_col in df.columns:
            prices = df[close_col].dropna()
            prices = prices[(prices.index >= pv_start) & (prices.index <= pv_end)]
            ticker_yearly = {}
            for year in sorted(set(prices.index.year)):
                year_data = prices[prices.index.year == year]
                if len(year_data) > 1:
                    year_return = ((year_data.iloc[-1] / year_data.iloc[0]) - 1) * 100
                    ticker_yearly[str(year)] = round(year_return, 2)
            if ticker_yearly:
                benchmark_yearly[ticker] = ticker_yearly

    return {
        "dates": dates,
        "portfolio_cumulative": portfolio_cum_ret,
        "benchmark_cumulative": benchmark_cum_rets,
        "drawdown": drawdown_data,
        "yearly_labels": list(yearly_returns.keys()),
        "yearly_portfolio": list(yearly_returns.values()),
        "yearly_benchmarks": benchmark_yearly,
    }


def generate_assumptions_html(
    config_dict: Dict[str, Any],
    frequency: str,
    n_symbols: int,
    allocation_method: str = "nco",
) -> str:
    """前提条件・手法セクションのHTMLを生成"""
    transaction_cost_bps = config_dict.get("transaction_cost_bps", 10)
    slippage_bps = config_dict.get("slippage_bps", 5)
    initial_capital = config_dict.get("initial_capital", 1000000)

    # 配分手法の正確な表記
    method_names = {
        "nco": "NCO（ネステッドクラスタ最適化）",
        "hrp": "HRP（階層的リスクパリティ）",
        "risk_parity": "リスクパリティ",
        "mean_variance": "平均分散最適化",
        "equal_weight": "均等配分",
        "black_litterman": "ブラック・リッターマン",
        "cvar": "CVaR最適化",
    }
    method_display = method_names.get(allocation_method.lower(), allocation_method)

    return f'''
        <h2>前提条件・手法</h2>
        <div class="assumptions-section">
            <table class="assumptions-table">
                <tbody>
                    <tr>
                        <td class="label">初期資金</td>
                        <td class="value">${initial_capital:,.0f}</td>
                    </tr>
                    <tr>
                        <td class="label">バックテスト期間</td>
                        <td class="value">{config_dict["start_date"]} 〜 {config_dict["end_date"]}</td>
                    </tr>
                    <tr>
                        <td class="label">リバランス頻度</td>
                        <td class="value">{frequency}</td>
                    </tr>
                    <tr>
                        <td class="label">執行タイミング</td>
                        <td class="value">翌日初値（Open）</td>
                    </tr>
                    <tr>
                        <td class="label">取引コスト</td>
                        <td class="value">{transaction_cost_bps} bps（{transaction_cost_bps / 100:.2f}%）</td>
                    </tr>
                    <tr>
                        <td class="label">スリッページ</td>
                        <td class="value">{slippage_bps} bps（{slippage_bps / 100:.2f}%）※現在未適用</td>
                    </tr>
                    <tr>
                        <td class="label">ユニバース銘柄数</td>
                        <td class="value">{n_symbols} 銘柄</td>
                    </tr>
                    <tr>
                        <td class="label">配分手法</td>
                        <td class="value">{method_display}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    '''


def generate_holdings_history_html(
    result: "UnifiedBacktestResult",
) -> str:
    """保有銘柄推移セクションのHTMLを生成（折りたたみ式）"""
    if not result.rebalances:
        return '''
        <h2>保有銘柄推移</h2>
        <p class="no-data">リバランス詳細データなし（高速モードでは記録されません）</p>
        '''

    total_rebalances = len(result.rebalances)

    # リバランス履歴を時系列順に処理
    holdings_sections = []
    prev_weights: Dict[str, float] = {}

    for idx, rb in enumerate(result.rebalances):
        date_str = rb.date.strftime("%Y-%m-%d") if hasattr(rb.date, 'strftime') else str(rb.date)

        # ウェイトが0.1%以上の銘柄のみ表示
        weights_after = {k: v for k, v in rb.weights_after.items() if v > 0.001}
        n_holdings = len(weights_after)

        # Top 5銘柄のサマリー
        sorted_weights = sorted(weights_after.items(), key=lambda x: -x[1])
        top5_summary = ", ".join([f"{sym} {w*100:.1f}%" for sym, w in sorted_weights[:5]])
        if len(sorted_weights) > 5:
            top5_summary += "..."

        # テーブル行（全銘柄）
        rows_html = ""
        for sym, weight in sorted_weights:
            prev_weight = prev_weights.get(sym, 0.0)
            weight_change = weight - prev_weight

            # 変化率の表示
            if abs(weight_change) < 0.001:
                change_str = "-"
                change_class = ""
            elif weight_change > 0:
                change_str = f"+{weight_change * 100:.1f}%"
                change_class = "positive"
            else:
                change_str = f"{weight_change * 100:.1f}%"
                change_class = "negative"

            # 新規追加・除外のハイライト
            if prev_weight < 0.001 and weight > 0.001:
                row_class = "new-holding"
                status = "🆕"
            elif prev_weight > 0.001 and weight < 0.001:
                row_class = "removed-holding"
                status = "❌"
            else:
                row_class = ""
                status = ""

            rows_html += f'''
                <tr class="{row_class}">
                    <td>{sym} {status}</td>
                    <td>{weight * 100:.2f}%</td>
                    <td class="{change_class}">{change_str}</td>
                </tr>'''

        # 除外された銘柄を追加（前回保有していたが今回は0）
        for sym, prev_w in prev_weights.items():
            if prev_w > 0.001 and sym not in weights_after:
                rows_html += f'''
                <tr class="removed-holding">
                    <td>{sym} ❌</td>
                    <td>0.00%</td>
                    <td class="negative">-{prev_w * 100:.1f}%</td>
                </tr>'''

        holdings_sections.append(f'''
        <div class="holdings-section">
            <div class="holdings-header" onclick="toggleHoldings({idx})">
                <span class="holdings-toggle" id="htoggle-{idx}">▶</span>
                <span class="holdings-date">{date_str}</span>
                <span class="holdings-summary">{top5_summary}</span>
                <span class="holdings-count">{n_holdings}銘柄</span>
            </div>
            <div class="holdings-content" id="hcontent-{idx}" style="display: none;">
                <table class="holdings-table">
                    <thead>
                        <tr>
                            <th>銘柄</th>
                            <th>ウェイト</th>
                            <th>前回比</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        ''')

        # 次回用に現在のウェイトを保存
        prev_weights = weights_after.copy()

    holdings_html = "\n".join(holdings_sections)

    # JavaScript for toggle
    toggle_script = '''
    <script>
        function toggleHoldings(idx) {
            const content = document.getElementById('hcontent-' + idx);
            const toggle = document.getElementById('htoggle-' + idx);
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        }
        function expandAllHoldings() {
            document.querySelectorAll('.holdings-content').forEach(el => el.style.display = 'block');
            document.querySelectorAll('.holdings-toggle').forEach(el => el.textContent = '▼');
        }
        function collapseAllHoldings() {
            document.querySelectorAll('.holdings-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.holdings-toggle').forEach(el => el.textContent = '▶');
        }
    </script>
    '''

    return f'''
        <h2>保有銘柄推移（全{total_rebalances}回のリバランス）</h2>
        <div class="holdings-controls">
            <button onclick="expandAllHoldings()">すべて展開</button>
            <button onclick="collapseAllHoldings()">すべて折りたたむ</button>
        </div>
        <div class="holdings-container">
            {holdings_html}
        </div>
        {toggle_script}
    '''


def generate_contribution_html(
    result: "UnifiedBacktestResult",
    prices: Dict[str, pd.DataFrame],
) -> str:
    """銘柄寄与度分析セクションのHTMLを生成"""
    import numpy as np

    if not result.rebalances or len(result.rebalances) < 2:
        return '''
        <h2>銘柄寄与度分析</h2>
        <p class="no-data">寄与度分析にはリバランスデータが必要です</p>
        '''

    # 各銘柄の寄与度を計算
    # 寄与度 = Σ(weight_i × return_i) for each period
    symbol_contributions: Dict[str, float] = {}
    symbol_total_weights: Dict[str, float] = {}
    symbol_periods: Dict[str, int] = {}
    symbol_returns: Dict[str, List[float]] = {}

    for i in range(len(result.rebalances) - 1):
        rb_start = result.rebalances[i]
        rb_end = result.rebalances[i + 1]

        start_date = rb_start.date
        end_date = rb_end.date

        # 期間のポートフォリオリターン
        pv = result.portfolio_values
        if isinstance(pv.index, pd.DatetimeIndex):
            period_pv = pv[(pv.index >= pd.Timestamp(start_date)) & (pv.index <= pd.Timestamp(end_date))]
            if len(period_pv) > 1:
                portfolio_return = (period_pv.iloc[-1] / period_pv.iloc[0]) - 1
            else:
                continue
        else:
            continue

        # 各銘柄のリターンと寄与度を計算
        for sym, weight in rb_start.weights_after.items():
            if weight < 0.001:
                continue

            if sym not in prices:
                continue

            df = prices[sym]
            if not isinstance(df.index, pd.DatetimeIndex):
                continue

            # 銘柄の期間リターン
            sym_prices = df['close'] if 'close' in df.columns else df.iloc[:, 0]
            period_prices = sym_prices[(sym_prices.index >= pd.Timestamp(start_date)) & (sym_prices.index <= pd.Timestamp(end_date))]

            if len(period_prices) > 1:
                sym_return = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1

                # 寄与度 = weight × return
                contribution = weight * sym_return

                if sym not in symbol_contributions:
                    symbol_contributions[sym] = 0.0
                    symbol_total_weights[sym] = 0.0
                    symbol_periods[sym] = 0
                    symbol_returns[sym] = []

                symbol_contributions[sym] += contribution
                symbol_total_weights[sym] += weight
                symbol_periods[sym] += 1
                symbol_returns[sym].append(sym_return)

    if not symbol_contributions:
        return '''
        <h2>銘柄寄与度分析</h2>
        <p class="no-data">寄与度を計算できませんでした</p>
        '''

    # 平均ウェイトと累積リターンを計算
    symbol_stats = []
    for sym in symbol_contributions:
        n_periods = symbol_periods[sym]
        avg_weight = symbol_total_weights[sym] / n_periods if n_periods > 0 else 0
        cumulative_return = np.prod([1 + r for r in symbol_returns[sym]]) - 1 if symbol_returns[sym] else 0
        contribution = symbol_contributions[sym]

        symbol_stats.append({
            'symbol': sym,
            'contribution': contribution,
            'avg_weight': avg_weight,
            'cumulative_return': cumulative_return,
            'n_periods': n_periods,
        })

    # 寄与度でソート（上位と下位を表示）
    sorted_by_contribution = sorted(symbol_stats, key=lambda x: -x['contribution'])

    # Top 10 と Bottom 10
    top_contributors = sorted_by_contribution[:10]
    bottom_contributors = sorted_by_contribution[-10:][::-1]  # 下位10を逆順で

    def generate_table(stats_list: List[dict], title: str) -> str:
        rows = ""
        for stat in stats_list:
            contrib_class = "positive" if stat['contribution'] > 0 else "negative"
            return_class = "positive" if stat['cumulative_return'] > 0 else "negative"
            rows += f'''
                <tr>
                    <td>{stat['symbol']}</td>
                    <td class="{contrib_class}">{stat['contribution'] * 100:+.2f}%</td>
                    <td>{stat['avg_weight'] * 100:.2f}%</td>
                    <td class="{return_class}">{stat['cumulative_return'] * 100:+.1f}%</td>
                </tr>'''

        return f'''
        <div class="contribution-table-container">
            <h3>{title}</h3>
            <table class="contribution-table">
                <thead>
                    <tr>
                        <th>銘柄</th>
                        <th>寄与度</th>
                        <th>平均ウェイト</th>
                        <th>銘柄リターン</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        '''

    top_table = generate_table(top_contributors, "Top 10 プラス寄与銘柄")
    bottom_table = generate_table(bottom_contributors, "Bottom 10 マイナス寄与銘柄")

    return f'''
        <h2>銘柄寄与度分析</h2>
        <div class="contribution-section">
            <div class="contribution-grid">
                {top_table}
                {bottom_table}
            </div>
        </div>
    '''


def generate_rebalance_history_html(
    result: "UnifiedBacktestResult",
) -> str:
    """リバランス履歴セクションのHTMLを生成（全期間・年別折りたたみ対応）"""
    if not result.rebalances:
        return '''
        <h2>リバランス履歴</h2>
        <p class="no-data">リバランス詳細データなし（高速モードでは記録されません）</p>
        '''

    total_rebalances = len(result.rebalances)

    # 年別にグループ化
    rebalances_by_year: Dict[int, list] = {}
    for rb in result.rebalances:
        year = rb.date.year if hasattr(rb.date, 'year') else int(str(rb.date)[:4])
        if year not in rebalances_by_year:
            rebalances_by_year[year] = []
        rebalances_by_year[year].append(rb)

    # 年別セクション生成
    year_sections = []
    for year in sorted(rebalances_by_year.keys(), reverse=True):
        year_rebalances = rebalances_by_year[year]

        # 年間サマリー計算
        year_start_value = year_rebalances[0].portfolio_value
        year_end_value = year_rebalances[-1].portfolio_value
        year_return = ((year_end_value / year_start_value) - 1) * 100 if year_start_value > 0 else 0
        year_total_cost = sum(rb.transaction_cost for rb in year_rebalances)
        year_avg_turnover = sum(rb.turnover for rb in year_rebalances) / len(year_rebalances) * 100

        # テーブル行
        rows_html = ""
        for rb in year_rebalances:
            date_str = rb.date.strftime("%Y-%m-%d") if hasattr(rb.date, 'strftime') else str(rb.date)
            n_holdings = len([w for w in rb.weights_after.values() if w > 0.001])
            rows_html += f'''
                <tr>
                    <td>{date_str}</td>
                    <td>${rb.portfolio_value:,.0f}</td>
                    <td>{rb.turnover * 100:.1f}%</td>
                    <td>${rb.transaction_cost:,.0f}</td>
                    <td>{n_holdings}</td>
                </tr>'''

        # 年別折りたたみセクション
        return_class = "positive" if year_return > 0 else "negative"
        year_sections.append(f'''
        <div class="year-section">
            <div class="year-header" onclick="toggleYear({year})">
                <span class="year-toggle" id="toggle-{year}">▶</span>
                <span class="year-label">{year}年</span>
                <span class="year-stats">
                    リバランス {len(year_rebalances)}回 |
                    リターン <span class="{return_class}">{year_return:+.1f}%</span> |
                    平均TO {year_avg_turnover:.1f}% |
                    コスト ${year_total_cost:,.0f}
                </span>
            </div>
            <div class="year-content" id="content-{year}" style="display: none;">
                <table class="rebalance-table">
                    <thead>
                        <tr>
                            <th>日付</th>
                            <th>ポートフォリオ</th>
                            <th>TO</th>
                            <th>コスト</th>
                            <th>銘柄数</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        ''')

    years_html = "\n".join(year_sections)

    # JavaScript for toggle
    toggle_script = '''
    <script>
        function toggleYear(year) {
            const content = document.getElementById('content-' + year);
            const toggle = document.getElementById('toggle-' + year);
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        }
        function expandAllYears() {
            document.querySelectorAll('.year-content').forEach(el => el.style.display = 'block');
            document.querySelectorAll('.year-toggle').forEach(el => el.textContent = '▼');
        }
        function collapseAllYears() {
            document.querySelectorAll('.year-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.year-toggle').forEach(el => el.textContent = '▶');
        }
    </script>
    '''

    return f'''
        <h2>リバランス履歴（全{total_rebalances}回）</h2>
        <div class="rebalance-controls">
            <button onclick="expandAllYears()">すべて展開</button>
            <button onclick="collapseAllYears()">すべて折りたたむ</button>
        </div>
        <div class="rebalance-years-container">
            {years_html}
        </div>
        {toggle_script}
    '''


def generate_yearly_performance_html(
    result: "UnifiedBacktestResult",
    benchmark_data: Dict[str, pd.DataFrame],
) -> str:
    """年別パフォーマンスセクションのHTMLを生成（テーブル + Chart.js用コンテナ）"""
    if len(result.portfolio_values) == 0:
        return ""

    pv = result.portfolio_values
    pv_start = pv.index.min()
    pv_end = pv.index.max()

    yearly_returns = {}

    # 年別リターンを計算
    for year in sorted(set(pv.index.year)):
        year_data = pv[pv.index.year == year]
        if len(year_data) > 1:
            year_return = (year_data.iloc[-1] / year_data.iloc[0]) - 1
            yearly_returns[year] = year_return

    if not yearly_returns:
        return ""

    # ベンチマーク年別リターン
    benchmark_yearly: Dict[str, Dict[int, float]] = {}
    for ticker, df in benchmark_data.items():
        close_col = "adj close" if "adj close" in df.columns else "close"
        if close_col in df.columns:
            prices = df[close_col].dropna()
            prices = prices[(prices.index >= pv_start) & (prices.index <= pv_end)]
            ticker_yearly = {}
            for year in sorted(set(prices.index.year)):
                year_data = prices[prices.index.year == year]
                if len(year_data) > 1:
                    year_return = (year_data.iloc[-1] / year_data.iloc[0]) - 1
                    ticker_yearly[year] = year_return
            if ticker_yearly:
                benchmark_yearly[ticker] = ticker_yearly

    # テーブルヘッダー（ベンチマーク列を追加）
    bm_headers = "".join(f"<th>{ticker}</th>" for ticker in benchmark_yearly.keys())

    rows = []
    for year, ret in yearly_returns.items():
        color_class = "positive" if ret > 0 else "negative"
        bm_cells = ""
        for ticker in benchmark_yearly.keys():
            bm_ret = benchmark_yearly[ticker].get(year)
            if bm_ret is not None:
                bm_class = "positive" if bm_ret > 0 else "negative"
                bm_cells += f'<td class="{bm_class}">{bm_ret * 100:+.2f}%</td>'
            else:
                bm_cells += "<td>-</td>"

        rows.append(f'''
            <tr>
                <td>{year}</td>
                <td class="{color_class}">{ret * 100:+.2f}%</td>
                {bm_cells}
            </tr>
        ''')

    rows_html = "\n".join(rows)

    return f'''
        <h2>年別パフォーマンス詳細</h2>
        <div class="yearly-section">
            <table class="yearly-table">
                <thead>
                    <tr>
                        <th>年</th>
                        <th>Portfolio</th>
                        {bm_headers}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    '''


def generate_report(
    result: UnifiedBacktestResult,
    frequency: str,
    config_dict: Dict[str, Any],
    benchmarks: List[str],
    report_dir: str,
    universe: Optional[List[str]] = None,
    prices: Optional[Dict[str, pd.DataFrame]] = None,
    allocation_method: str = "nco",
) -> Optional[str]:
    """HTMLレポートを生成（資産推移グラフ付き）"""
    from src.analysis.report_generator import (
        ReportGenerator,
        ComparisonResult,
        PortfolioMetrics,
    )

    logger.info("Generating performance report...")

    # ポートフォリオメトリクス
    portfolio_metrics = PortfolioMetrics(
        total_return=result.total_return,
        annual_return=result.annual_return,
        volatility=result.volatility,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        calmar_ratio=result.calmar_ratio,
        win_rate=result.win_rate,
        n_trades=result.n_rebalances,
        n_periods=result.n_days,
    )

    # ベンチマークデータ取得
    benchmark_data = {}
    benchmark_metrics = {}
    if benchmarks:
        benchmark_data = fetch_benchmark_data(
            benchmarks,
            config_dict["start_date"],
            config_dict["end_date"],
        )
        for ticker, df in benchmark_data.items():
            close_col = "adj close" if "adj close" in df.columns else "close"
            if close_col in df.columns:
                metrics = calculate_benchmark_metrics(df[close_col])
                if metrics:
                    benchmark_metrics[ticker] = PortfolioMetrics(
                        total_return=metrics.get("total_return", 0),
                        annual_return=metrics.get("annual_return", 0),
                        volatility=metrics.get("volatility", 0),
                        max_drawdown=metrics.get("max_drawdown", 0),
                        sharpe_ratio=metrics.get("sharpe_ratio", 0),
                        sortino_ratio=metrics.get("sortino_ratio", 0),
                        calmar_ratio=metrics.get("calmar_ratio", 0),
                    )

    # 比較結果作成
    comparison = ComparisonResult(
        portfolio_metrics=portfolio_metrics,
        benchmark_metrics=benchmark_metrics,
        portfolio_name=f"Multi-Asset Portfolio ({frequency})",
        start_date=config_dict["start_date"],
        end_date=config_dict["end_date"],
    )

    # レポート生成
    generator = ReportGenerator()
    output_dir = Path(PROJECT_ROOT) / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"backtest_{frequency}_report.html"
    html_content = generator.generate_html_report(
        comparison,
        portfolio_name=f"Multi-Asset Portfolio ({frequency})",
        start_date=config_dict["start_date"],
        end_date=config_dict["end_date"],
        output_path=str(html_path),
    )

    # Chart.js用データ生成
    chart_data = generate_chartjs_data(result.portfolio_values, benchmark_data)

    # HTMLを読み込んで拡張セクションを追加
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 追加スタイル（Chart.js対応 + 年別折りたたみUI）
    additional_styles = '''
        .assumptions-section, .rebalance-section, .yearly-section {
            margin: 20px 0;
        }
        .assumptions-table {
            width: auto;
            min-width: 400px;
        }
        .assumptions-table td.label {
            font-weight: 600;
            color: #555;
            padding-right: 30px;
        }
        .assumptions-table td.value {
            color: #333;
        }
        .rebalance-table {
            font-size: 0.9em;
        }
        .yearly-table {
            width: auto;
            min-width: 300px;
        }
        .no-data {
            color: #999;
            font-style: italic;
        }
        /* Chart.jsグラフ */
        .chart-container {
            position: relative;
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .chart-container h3 {
            margin: 0 0 15px 0;
            font-size: 1.1em;
            color: #333;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        @media (max-width: 900px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        .yearly-chart-container {
            margin: 20px 0;
        }
        /* 年別折りたたみUI */
        .rebalance-controls {
            margin-bottom: 15px;
        }
        .rebalance-controls button {
            padding: 8px 16px;
            margin-right: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .rebalance-controls button:hover {
            background: #5a67d8;
        }
        .year-section {
            margin-bottom: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        .year-header {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f7fafc;
            cursor: pointer;
            user-select: none;
        }
        .year-header:hover {
            background: #edf2f7;
        }
        .year-toggle {
            width: 20px;
            color: #667eea;
            font-size: 0.9em;
        }
        .year-label {
            font-weight: 600;
            font-size: 1.1em;
            margin-right: 20px;
        }
        .year-stats {
            font-size: 0.9em;
            color: #666;
        }
        .year-content {
            padding: 15px;
            background: white;
        }
        .year-content table {
            margin: 0;
        }
        /* 保有銘柄推移 */
        .holdings-controls {
            margin-bottom: 15px;
        }
        .holdings-controls button {
            padding: 8px 16px;
            margin-right: 10px;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .holdings-controls button:hover {
            background: #059669;
        }
        .holdings-section {
            margin-bottom: 8px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        .holdings-header {
            display: flex;
            align-items: center;
            padding: 10px 16px;
            background: #f0fdf4;
            cursor: pointer;
            user-select: none;
            gap: 12px;
        }
        .holdings-header:hover {
            background: #dcfce7;
        }
        .holdings-toggle {
            width: 20px;
            color: #10b981;
            font-size: 0.9em;
        }
        .holdings-date {
            font-weight: 600;
            min-width: 100px;
        }
        .holdings-summary {
            flex: 1;
            font-size: 0.85em;
            color: #666;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .holdings-count {
            font-size: 0.85em;
            color: #10b981;
            font-weight: 500;
        }
        .holdings-content {
            padding: 15px;
            background: white;
            max-height: 400px;
            overflow-y: auto;
        }
        .holdings-table {
            width: 100%;
            font-size: 0.85em;
        }
        .holdings-table .new-holding {
            background-color: #dcfce7;
        }
        .holdings-table .removed-holding {
            background-color: #fef2f2;
            color: #999;
        }
        /* 寄与度分析 */
        .contribution-section {
            margin: 20px 0;
        }
        .contribution-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .contribution-grid {
                grid-template-columns: 1fr;
            }
        }
        .contribution-table-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .contribution-table-container h3 {
            margin: 0 0 15px 0;
            font-size: 1em;
            color: #333;
        }
        .contribution-table {
            width: 100%;
            font-size: 0.9em;
        }
    '''

    # スタイルを追加
    html_content = html_content.replace(
        '</style>',
        additional_styles + '\n    </style>'
    )

    # 前提条件セクションを追加
    n_symbols = len(universe) if universe else config_dict.get("n_symbols", 0)
    assumptions_html = generate_assumptions_html(config_dict, frequency, n_symbols, allocation_method)

    # Chart.jsグラフHTML（累積リターン% + ドローダウン）
    chart_html = ""
    if chart_data:
        # ベンチマークデータセット生成
        bm_colors = ['#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2']
        bm_datasets_cumulative = ""
        bm_datasets_yearly = ""

        for i, (ticker, data) in enumerate(chart_data.get("benchmark_cumulative", {}).items()):
            color = bm_colors[i % len(bm_colors)]
            data_json = json.dumps(data)
            bm_datasets_cumulative += f'''
                {{
                    label: '{ticker}',
                    data: {data_json},
                    borderColor: '{color}',
                    backgroundColor: '{color}20',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                }},'''

        # 年別ベンチマークデータ
        for i, (ticker, yearly_data) in enumerate(chart_data.get("yearly_benchmarks", {}).items()):
            color = bm_colors[i % len(bm_colors)]
            values = [yearly_data.get(y, 0) for y in chart_data.get("yearly_labels", [])]
            values_json = json.dumps(values)
            bm_datasets_yearly += f'''
                {{
                    label: '{ticker}',
                    data: {values_json},
                    backgroundColor: '{color}80',
                    borderColor: '{color}',
                    borderWidth: 1
                }},'''

        dates_json = json.dumps(chart_data.get("dates", []))
        portfolio_cum_json = json.dumps(chart_data.get("portfolio_cumulative", []))
        drawdown_json = json.dumps(chart_data.get("drawdown", []))
        yearly_labels_json = json.dumps(chart_data.get("yearly_labels", []))
        yearly_portfolio_json = json.dumps(chart_data.get("yearly_portfolio", []))

        chart_html = f'''
        <h2>パフォーマンスグラフ</h2>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <!-- 累積リターン%グラフ -->
        <div class="chart-container">
            <h3>累積リターン（%）</h3>
            <canvas id="cumulativeReturnChart" height="120"></canvas>
        </div>

        <div class="charts-grid">
            <!-- ドローダウングラフ -->
            <div class="chart-container">
                <h3>ドローダウン（%）</h3>
                <canvas id="drawdownChart" height="150"></canvas>
            </div>

            <!-- 年別パフォーマンスグラフ -->
            <div class="chart-container">
                <h3>年別リターン（%）</h3>
                <canvas id="yearlyChart" height="150"></canvas>
            </div>
        </div>

        <script>
            // 累積リターンチャート
            new Chart(document.getElementById('cumulativeReturnChart'), {{
                type: 'line',
                data: {{
                    labels: {dates_json},
                    datasets: [
                        {{
                            label: 'Portfolio',
                            data: {portfolio_cum_json},
                            borderColor: '#2563eb',
                            backgroundColor: '#2563eb20',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: true,
                            tension: 0.1
                        }},
                        {bm_datasets_cumulative}
                    ]
                }},
                options: {{
                    responsive: true,
                    interaction: {{ mode: 'index', intersect: false }},
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            display: true,
                            ticks: {{ maxTicksLimit: 12 }}
                        }},
                        y: {{
                            display: true,
                            title: {{ display: true, text: 'リターン (%)' }},
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }}
                            }}
                        }}
                    }}
                }}
            }});

            // ドローダウンチャート
            new Chart(document.getElementById('drawdownChart'), {{
                type: 'line',
                data: {{
                    labels: {dates_json},
                    datasets: [{{
                        label: 'Drawdown',
                        data: {drawdown_json},
                        borderColor: '#dc2626',
                        backgroundColor: '#dc262640',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return 'DD: ' + context.parsed.y.toFixed(1) + '%';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{ display: true, ticks: {{ maxTicksLimit: 8 }} }},
                        y: {{
                            display: true,
                            title: {{ display: true, text: 'ドローダウン (%)' }},
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }}
                            }}
                        }}
                    }}
                }}
            }});

            // 年別パフォーマンスチャート
            new Chart(document.getElementById('yearlyChart'), {{
                type: 'bar',
                data: {{
                    labels: {yearly_labels_json},
                    datasets: [
                        {{
                            label: 'Portfolio',
                            data: {yearly_portfolio_json},
                            backgroundColor: '#2563eb80',
                            borderColor: '#2563eb',
                            borderWidth: 1
                        }},
                        {bm_datasets_yearly}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'リターン (%)' }},
                            ticks: {{
                                callback: function(value) {{ return value + '%'; }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        '''

    # 年別パフォーマンスを追加（テーブル）
    yearly_html = generate_yearly_performance_html(result, benchmark_data)

    # リバランス履歴を追加
    rebalance_html = generate_rebalance_history_html(result)

    # 保有銘柄推移を追加
    holdings_html = generate_holdings_history_html(result)

    # 寄与度分析を追加
    contribution_html = ""
    if prices:
        contribution_html = generate_contribution_html(result, prices)

    # サマリーの前に前提条件とグラフを挿入
    html_content = html_content.replace(
        '<h2>サマリー</h2>',
        assumptions_html + '\n' + chart_html + '\n        <h2>サマリー</h2>'
    )

    # サマリーテーブルの後に年別パフォーマンス、保有銘柄推移、寄与度分析、リバランス履歴を追加
    html_content = html_content.replace(
        '<div class="footer">',
        yearly_html + '\n' + holdings_html + '\n' + contribution_html + '\n' + rebalance_html + '\n        <div class="footer">'
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Report saved to: {html_path}")
    return str(html_path)


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
        "--universe", "-u",
        type=str,
        default=None,
        help="Universe YAML file (default: config/universe_standard.yaml)",
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
        "--cache-dir",
        type=str,
        default=None,
        help="Local cache directory",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local storage instead of S3 (signals cached in .cache/signals)",
    )
    # Report options
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="SPY,QQQ,^N225",
        help="Benchmark tickers for comparison (comma-separated, default: SPY,QQQ,^N225)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for report output (default: reports)",
    )
    # Trading filter options
    parser.add_argument(
        "--trading-filter",
        type=str,
        default=None,
        help="Filter pattern for tradable assets (e.g., '*.T' for Japan stocks only). "
             "Signal computation uses full universe, but only filtered assets are traded.",
    )
    parser.add_argument(
        "--japan-only",
        action="store_true",
        help="Shortcut for --trading-filter '*.T' (Japan stocks only)",
    )
    # Large universe optimization
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Maximum number of assets for weight allocation (top N by signal score). "
             "Use for large universes (1000+ symbols) to reduce memory and computation time. "
             "Recommended: 500-1000 for optimal balance.",
    )
    # Archive options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save backtest result to archive (default: save enabled)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the saved archive (default: auto-generated with conditions)",
    )
    parser.add_argument(
        "--archive-tags",
        type=str,
        default=None,
        help="Tags for the archive (comma-separated, default: frequency)",
    )
    parser.add_argument(
        "--archive-description",
        type=str,
        default="",
        help="Description for the saved archive",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ストレージ設定
    storage_config = None
    if not args.local:
        try:
            storage_config = setup_storage_config(cache_dir=args.cache_dir)
        except S3ConfigurationError as e:
            logger.warning(f"S3 not available, falling back to local storage: {e}")
            storage_config = None

    if storage_config is None:
        logger.info("Using local storage: .cache/signals")

    # 実行対象の頻度
    frequencies = ["daily", "weekly", "monthly"] if args.frequency == "all" else [args.frequency]

    results = {}
    for freq in frequencies:
        output = args.output or f"results/backtest_{freq}.json"
        if args.frequency == "all":
            output = f"results/backtest_{freq}.json"

        # ベンチマークリスト
        benchmark_list = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

        # 取引フィルター（--japan-only は --trading-filter "*.T" のショートカット）
        trading_filter = args.trading_filter
        if args.japan_only:
            trading_filter = "*.T"

        # アーカイブタグのパース
        archive_tags = None
        if args.archive_tags:
            archive_tags = [t.strip() for t in args.archive_tags.split(",") if t.strip()]

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
                generate_report_flag=not args.no_report,
                benchmarks=benchmark_list,
                report_dir=args.report_dir,
                trading_filter=trading_filter,
                universe_file=args.universe,
                max_assets=args.max_assets,
                save_archive=not args.no_save,
                archive_name=args.name,
                archive_tags=archive_tags,
                archive_description=args.archive_description,
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
