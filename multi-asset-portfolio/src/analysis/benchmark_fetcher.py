"""
Benchmark Fetcher - ベンチマーク指数データ取得

主要ベンチマーク（S&P 500, Nasdaq 100等）のデータを取得し、
リターン計算機能を提供する。

Usage:
    from src.analysis.benchmark_fetcher import BenchmarkFetcher

    fetcher = BenchmarkFetcher()
    prices = fetcher.fetch_benchmarks("2020-01-01", "2024-12-31")
    returns = fetcher.calculate_returns(prices, frequency="daily")
    cumulative = fetcher.get_cumulative_returns(returns)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
)

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend

logger = logging.getLogger(__name__)


class BenchmarkFetcherError(Exception):
    """BenchmarkFetcher関連のエラー"""
    pass


class BenchmarkFetcher:
    """
    ベンチマーク指数データ取得クラス

    主要な株式指数ETFのデータをyfinanceで取得し、
    リターン計算・累積リターン計算機能を提供。

    StorageBackend is required (S3 mandatory).

    Attributes:
        BENCHMARKS: 対応ベンチマークの辞書
    """

    BENCHMARKS: Dict[str, str] = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "DIA": "Dow Jones",
        "IWM": "Russell 2000",
        "VT": "All World",
        "EWJ": "Japan",
    }

    def __init__(
        self,
        storage_backend: "StorageBackend",
    ) -> None:
        """
        初期化

        Parameters
        ----------
        storage_backend : StorageBackend
            ストレージバックエンド（S3/ローカル）（必須）
            Must have exists(), read_parquet(), and write_parquet() methods.
        """
        # Check for required methods instead of strict type check
        # This allows for mocking in tests
        required_methods = ["exists", "read_parquet", "write_parquet"]
        for method in required_methods:
            if not hasattr(storage_backend, method):
                raise TypeError(
                    f"storage_backend must have '{method}' method. "
                    "Expected StorageBackend instance or compatible object."
                )

        self._backend = storage_backend
        self._cache_subdir = "benchmarks"

    def fetch_benchmarks(
        self,
        start_date: str,
        end_date: str,
        benchmarks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        ベンチマークデータを取得

        Parameters
        ----------
        start_date : str
            開始日（YYYY-MM-DD形式）
        end_date : str
            終了日（YYYY-MM-DD形式）
        benchmarks : list[str], optional
            取得するベンチマークのティッカーリスト。
            Noneの場合は全ベンチマークを取得。

        Returns
        -------
        pd.DataFrame
            Adj Close価格のDataFrame（列: ティッカー、行: 日付）

        Raises
        ------
        BenchmarkFetcherError
            データ取得に失敗した場合
        ValueError
            日付形式が不正な場合
        """
        # 日付バリデーション
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")

        # ベンチマークリストの決定
        if benchmarks is None:
            tickers = list(self.BENCHMARKS.keys())
        else:
            # 無効なティッカーのチェック
            invalid = [t for t in benchmarks if t not in self.BENCHMARKS]
            if invalid:
                logger.warning(f"Unknown benchmarks (will still attempt fetch): {invalid}")
            tickers = benchmarks

        # キャッシュチェック（常に有効）
        cached_df = self._load_from_cache(tickers, start_date, end_date)
        if cached_df is not None:
            logger.info(f"Loaded {len(tickers)} benchmarks from cache")
            return cached_df

        # yfinanceでデータ取得
        try:
            import yfinance as yf
        except ImportError:
            raise BenchmarkFetcherError(
                "yfinance is required. Install with: pip install yfinance"
            )

        logger.info(f"Fetching {len(tickers)} benchmarks from yfinance...")

        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            raise BenchmarkFetcherError(f"Failed to fetch data from yfinance: {e}")

        if df is None or len(df) == 0:
            raise BenchmarkFetcherError("No data returned from yfinance")

        # Adj Close列を抽出
        if isinstance(df.columns, pd.MultiIndex):
            # 複数ティッカーの場合
            if "Adj Close" in df.columns.get_level_values(0):
                prices = df["Adj Close"]
            else:
                # フォールバック: Close を使用
                prices = df["Close"]
        else:
            # 単一ティッカーの場合
            if "Adj Close" in df.columns:
                prices = df[["Adj Close"]]
                prices.columns = tickers
            else:
                prices = df[["Close"]]
                prices.columns = tickers

        # 欠損データのチェック
        missing_count = prices.isna().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in benchmark data")
            # 前方補完
            prices = prices.ffill()

        # キャッシュに保存（常に有効）
        self._save_to_cache(prices, tickers, start_date, end_date)

        logger.info(f"Fetched {len(prices)} days of benchmark data")
        return prices

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        リターンを計算

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ（列: ティッカー、行: 日付）
        frequency : str
            リターン計算頻度: "daily", "weekly", "monthly"

        Returns
        -------
        pd.DataFrame
            リターンのDataFrame

        Raises
        ------
        ValueError
            無効な頻度が指定された場合
        """
        if prices is None or len(prices) == 0:
            return pd.DataFrame()

        valid_frequencies = ["daily", "weekly", "monthly"]
        if frequency not in valid_frequencies:
            raise ValueError(f"frequency must be one of {valid_frequencies}")

        if frequency == "daily":
            returns = prices.pct_change().dropna()
        elif frequency == "weekly":
            # 週次にリサンプリング（金曜日終値）
            weekly_prices = prices.resample("W-FRI").last()
            returns = weekly_prices.pct_change().dropna()
        elif frequency == "monthly":
            # 月次にリサンプリング（月末）
            monthly_prices = prices.resample("ME").last()
            returns = monthly_prices.pct_change().dropna()

        return returns

    def get_cumulative_returns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        累積リターンを計算

        Parameters
        ----------
        returns : pd.DataFrame
            リターンのDataFrame

        Returns
        -------
        pd.DataFrame
            累積リターンのDataFrame（初期値1.0から開始）
        """
        if returns is None or len(returns) == 0:
            return pd.DataFrame()

        cumulative = (1 + returns).cumprod()
        return cumulative

    def get_benchmark_stats(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        ベンチマーク統計を計算

        Parameters
        ----------
        returns : pd.DataFrame
            日次リターンのDataFrame
        risk_free_rate : float
            年率リスクフリーレート
        trading_days_per_year : int
            年間営業日数

        Returns
        -------
        pd.DataFrame
            統計情報（年率リターン、ボラティリティ、シャープレシオ等）
        """
        if returns is None or len(returns) == 0:
            return pd.DataFrame()

        stats = {}

        for ticker in returns.columns:
            r = returns[ticker].dropna()

            if len(r) < 2:
                continue

            r_array = r.values

            # Use unified metrics module
            annual_return = float(np.mean(r_array)) * trading_days_per_year
            annual_vol = calculate_volatility(
                r_array, annualization_factor=trading_days_per_year
            )
            sharpe = calculate_sharpe_ratio(
                r_array,
                risk_free_rate=risk_free_rate,
                annualization_factor=trading_days_per_year,
            )
            max_dd = calculate_max_drawdown(returns=r_array)

            # Calculate cumulative return for total_return
            cumulative = (1 + r).cumprod()
            total_ret = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

            stats[ticker] = {
                "name": self.BENCHMARKS.get(ticker, ticker),
                "annual_return": round(annual_return, 4),
                "annual_volatility": round(annual_vol, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(-max_dd, 4),  # Return as negative for consistency
                "total_return": round(total_ret, 4),
            }

        return pd.DataFrame(stats).T

    def _get_cache_path(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        """キャッシュファイルパスを生成（StorageBackend用）"""
        tickers_str = "_".join(sorted(tickers))
        filename = f"benchmarks_{tickers_str}_{start_date}_{end_date}.parquet"
        return f"{self._cache_subdir}/{filename}"

    def _load_from_cache(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """キャッシュからデータを読み込み（StorageBackend経由）"""
        cache_path = self._get_cache_path(tickers, start_date, end_date)
        if self._backend.exists(cache_path):
            try:
                df = self._backend.read_parquet(cache_path)
                # polars DataFrameの場合はpandasに変換
                if hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
                # Dateインデックスを復元（polars変換で失われる場合がある）
                if 'Date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('Date')
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """データをキャッシュに保存（StorageBackend経由）"""
        cache_path = self._get_cache_path(tickers, start_date, end_date)
        try:
            self._backend.write_parquet(df, cache_path)
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @classmethod
    def get_available_benchmarks(cls) -> Dict[str, str]:
        """
        利用可能なベンチマーク一覧を取得

        Returns
        -------
        dict
            ティッカー -> 名前の辞書
        """
        return cls.BENCHMARKS.copy()
