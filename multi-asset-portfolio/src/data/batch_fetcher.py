"""
Batch Data Fetcher - 大量銘柄の並列データ取得

大量の銘柄データを効率的に取得するためのバッチフェッチャー。

主要機能:
- 並列取得（同時接続数制限）
- レート制限
- 自動リトライ
- Parquetキャッシュ
- 進捗レポート

Usage:
    from src.data.batch_fetcher import BatchDataFetcher

    fetcher = BatchDataFetcher(max_concurrent=10)
    results = fetcher.fetch_all_sync(
        tickers=["SPY", "QQQ", "IWM"],
        start_date="2020-01-01",
        end_date="2024-12-31"
    )

    for ticker, df in results.items():
        print(f"{ticker}: {len(df)} rows")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BatchFetcherConfig:
    """BatchDataFetcherの設定。

    Attributes:
        max_concurrent: 最大同時接続数
        rate_limit_per_sec: 1秒あたりのリクエスト数上限
        retry_count: リトライ回数
        cache_dir: キャッシュディレクトリ
        cache_max_age_days: キャッシュ有効期間（日）
    """

    max_concurrent: int = 10
    rate_limit_per_sec: float = 2.0
    retry_count: int = 3
    cache_dir: str = "cache/price_data"
    cache_max_age_days: int = 1


@dataclass
class FetchResult:
    """単一銘柄の取得結果。

    Attributes:
        ticker: 銘柄コード
        data: データ（取得失敗時はNone）
        from_cache: キャッシュから取得したか
        error: エラーメッセージ（エラー時のみ）
        fetch_time_ms: 取得にかかった時間（ミリ秒）
    """

    ticker: str
    data: pl.DataFrame | None
    from_cache: bool = False
    error: str | None = None
    fetch_time_ms: float = 0.0


@dataclass
class BatchFetchResult:
    """バッチ取得結果。

    Attributes:
        results: 銘柄ごとの取得結果
        success_count: 成功数
        failed_count: 失敗数
        cached_count: キャッシュから取得した数
        total_time_ms: 総取得時間（ミリ秒）
    """

    results: dict[str, FetchResult] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    cached_count: int = 0
    total_time_ms: float = 0.0

    @property
    def successful_data(self) -> dict[str, pl.DataFrame]:
        """成功したデータのみを返す。"""
        return {
            ticker: result.data
            for ticker, result in self.results.items()
            if result.data is not None
        }

    @property
    def failed_tickers(self) -> list[str]:
        """失敗した銘柄リストを返す。"""
        return [
            ticker
            for ticker, result in self.results.items()
            if result.data is None
        ]

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "cached_count": self.cached_count,
            "total_time_ms": self.total_time_ms,
            "failed_tickers": self.failed_tickers,
        }


class BatchDataFetcher:
    """大量銘柄の並列データ取得。

    - 並列取得（同時接続数制限）
    - レート制限
    - 自動リトライ
    - Parquetキャッシュ
    - 進捗レポート

    Usage:
        fetcher = BatchDataFetcher(max_concurrent=10)
        results = fetcher.fetch_all_sync(
            tickers=["SPY", "QQQ", "IWM"],
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limit_per_sec: float = 2.0,
        retry_count: int = 3,
        cache_dir: str = "cache/price_data",
        cache_max_age_days: int = 1,
    ) -> None:
        """初期化。

        Args:
            max_concurrent: 最大同時接続数
            rate_limit_per_sec: 1秒あたりのリクエスト数上限
            retry_count: リトライ回数
            cache_dir: キャッシュディレクトリ
            cache_max_age_days: キャッシュ有効期間（日）
        """
        self.max_concurrent = max_concurrent
        self.rate_limit_per_sec = rate_limit_per_sec
        self.retry_count = retry_count
        self.cache_dir = Path(cache_dir)
        self.cache_max_age_days = cache_max_age_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: BatchFetcherConfig) -> "BatchDataFetcher":
        """設定からインスタンスを作成する。"""
        return cls(
            max_concurrent=config.max_concurrent,
            rate_limit_per_sec=config.rate_limit_per_sec,
            retry_count=config.retry_count,
            cache_dir=config.cache_dir,
            cache_max_age_days=config.cache_max_age_days,
        )

    def _get_cache_path(self, ticker: str) -> Path:
        """キャッシュファイルパスを取得する。"""
        safe_ticker = ticker.replace("/", "_").replace("=", "_").replace("^", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"

    def _is_cache_valid(
        self,
        cache_path: Path,
        end_date: str,
    ) -> bool:
        """キャッシュが有効かチェックする。

        Args:
            cache_path: キャッシュファイルパス
            end_date: 終了日

        Returns:
            キャッシュが有効かどうか
        """
        if not cache_path.exists():
            return False

        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = (datetime.now() - mtime).days
            return age <= self.cache_max_age_days
        except Exception:
            return False

    def _get_cache_end_date(
        self,
        cache_path: Path,
    ) -> Tuple[pl.DataFrame | None, datetime | None]:
        """キャッシュのデータと終了日を取得する。

        Args:
            cache_path: キャッシュファイルパス

        Returns:
            (キャッシュDataFrame, キャッシュの最終日) または (None, None)
        """
        if not cache_path.exists():
            return None, None

        try:
            df = pl.read_parquet(cache_path)
            if df.is_empty():
                return None, None

            # Date列の最大値を取得（タプル形式のカラム名にも対応）
            date_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower or 'datetime' in col_lower:
                    date_col = col
                    break

            if date_col is None:
                # 最初のカラムがdatetime型かチェック
                first_col = df.columns[0]
                if df[first_col].dtype in [pl.Datetime, pl.Date]:
                    date_col = first_col

            if date_col is None:
                return df, None

            max_date = df[date_col].max()
            if isinstance(max_date, datetime):
                return df, max_date
            elif isinstance(max_date, str):
                return df, datetime.strptime(max_date, "%Y-%m-%d")
            elif hasattr(max_date, 'year'):
                # Polars date type
                return df, datetime(max_date.year, max_date.month, max_date.day)
            else:
                return df, None

        except Exception as e:
            self._logger.debug(f"Failed to read cache end date: {e}")
            return None, None

    def _get_date_column(self, df: pl.DataFrame) -> str | None:
        """DataFrameからDate列名を取得する。"""
        for col in df.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower or 'datetime' in col_lower:
                return col
        # 最初のカラムがdatetime型かチェック
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if df[first_col].dtype in [pl.Datetime, pl.Date]:
                return first_col
        return None

    async def fetch_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        semaphore: asyncio.Semaphore,
    ) -> FetchResult:
        """単一銘柄のデータを取得する（差分取得対応）。

        Args:
            ticker: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            semaphore: 同時接続数制限用セマフォ

        Returns:
            取得結果
        """
        start_time = datetime.now()
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        async with semaphore:
            cache_path = self._get_cache_path(ticker)

            # キャッシュの存在と終了日を確認
            cached_df, cached_end = self._get_cache_end_date(cache_path)

            if cached_df is not None and cached_end is not None:
                # キャッシュが最新なら返す（終了日以降のデータがある）
                if cached_end >= end_date_dt:
                    fetch_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._logger.debug(f"{ticker}: Using cache (ends {cached_end.date()})")
                    return FetchResult(
                        ticker=ticker,
                        data=cached_df,
                        from_cache=True,
                        fetch_time_ms=fetch_time,
                    )

                # 差分のみ取得（キャッシュの翌日から）
                delta_start = (cached_end + timedelta(days=1)).strftime('%Y-%m-%d')
                self._logger.debug(f"{ticker}: Fetching delta from {delta_start} to {end_date}")

                delta_df = await self._fetch_from_yfinance(ticker, delta_start, end_date, semaphore)

                if delta_df is not None and len(delta_df) > 0:
                    # キャッシュと差分を結合
                    try:
                        # Date列の名前を取得
                        date_col = self._get_date_column(cached_df)
                        if date_col is None:
                            raise ValueError("Date column not found")

                        combined = pl.concat([cached_df, delta_df]).unique(date_col).sort(date_col)
                        combined.write_parquet(cache_path)

                        fetch_time = (datetime.now() - start_time).total_seconds() * 1000
                        self._logger.debug(f"{ticker}: Combined cache ({len(cached_df)} rows) + delta ({len(delta_df)} rows) = {len(combined)} rows")
                        return FetchResult(
                            ticker=ticker,
                            data=combined,
                            from_cache=False,  # 差分取得したのでFalse
                            fetch_time_ms=fetch_time,
                        )
                    except Exception as e:
                        self._logger.warning(f"{ticker}: Failed to combine cache and delta: {e}")
                        # フォールバック: キャッシュを返す
                        fetch_time = (datetime.now() - start_time).total_seconds() * 1000
                        return FetchResult(
                            ticker=ticker,
                            data=cached_df,
                            from_cache=True,
                            fetch_time_ms=fetch_time,
                        )

                # 差分が空でもキャッシュを返す（休日等で新しいデータがない場合）
                fetch_time = (datetime.now() - start_time).total_seconds() * 1000
                return FetchResult(
                    ticker=ticker,
                    data=cached_df,
                    from_cache=True,
                    fetch_time_ms=fetch_time,
                )

            # キャッシュなしは全量取得
            full_df = await self._fetch_from_yfinance(ticker, start_date, end_date, semaphore)

            if full_df is not None:
                # キャッシュに保存
                try:
                    full_df.write_parquet(cache_path)
                except Exception as e:
                    self._logger.debug(f"{ticker}: Cache write failed: {e}")

                fetch_time = (datetime.now() - start_time).total_seconds() * 1000
                return FetchResult(
                    ticker=ticker,
                    data=full_df,
                    from_cache=False,
                    fetch_time_ms=fetch_time,
                )

            fetch_time = (datetime.now() - start_time).total_seconds() * 1000
            return FetchResult(
                ticker=ticker,
                data=None,
                error="Failed to fetch data",
                fetch_time_ms=fetch_time,
            )

    async def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        semaphore: asyncio.Semaphore,
    ) -> pl.DataFrame | None:
        """yfinanceからデータを取得する（リトライ付き）。

        Args:
            ticker: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            semaphore: 同時接続数制限用セマフォ

        Returns:
            Polars DataFrame または None（失敗時）
        """
        last_error: str | None = None

        for attempt in range(self.retry_count):
            try:
                # レート制限
                await asyncio.sleep(1.0 / self.rate_limit_per_sec)

                # yfinanceは同期APIなのでrun_in_executorで実行
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                    ),
                )

                if data.empty:
                    return None

                # Polarsに変換
                df = pl.from_pandas(data.reset_index())
                return df

            except Exception as e:
                last_error = str(e)
                if attempt < self.retry_count - 1:
                    wait_time = 2**attempt
                    self._logger.debug(
                        f"{ticker}: Retry {attempt + 1}/{self.retry_count} "
                        f"after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._logger.warning(f"{ticker}: Failed after {self.retry_count} attempts: {e}")

        return None

    async def fetch_all(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchFetchResult:
        """全銘柄を並列取得する。

        Args:
            tickers: 銘柄コードリスト
            start_date: 開始日
            end_date: 終了日
            progress_callback: 進捗コールバック(completed, total, ticker)

        Returns:
            バッチ取得結果
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        batch_result = BatchFetchResult()

        tasks = [
            self.fetch_single(ticker, start_date, end_date, semaphore)
            for ticker in tickers
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1

            batch_result.results[result.ticker] = result

            if result.data is not None:
                batch_result.success_count += 1
                if result.from_cache:
                    batch_result.cached_count += 1
            else:
                batch_result.failed_count += 1

            if progress_callback:
                progress_callback(completed, len(tickers), result.ticker)

        batch_result.total_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        self._logger.info(
            f"Fetched {batch_result.success_count}/{len(tickers)} tickers "
            f"({batch_result.cached_count} from cache), "
            f"{batch_result.failed_count} failed, "
            f"took {batch_result.total_time_ms:.0f}ms"
        )

        return batch_result

    def fetch_all_sync(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """全銘柄を同期的に取得する（便利関数）。

        Args:
            tickers: 銘柄コードリスト
            start_date: 開始日
            end_date: 終了日
            progress_callback: 進捗コールバック

        Returns:
            成功した銘柄のデータ辞書
        """
        result = asyncio.run(
            self.fetch_all(tickers, start_date, end_date, progress_callback)
        )
        return result.successful_data

    def clear_cache(self, tickers: list[str] | None = None) -> int:
        """キャッシュをクリアする。

        Args:
            tickers: クリア対象の銘柄（Noneで全て）

        Returns:
            削除したファイル数
        """
        deleted = 0
        if tickers is None:
            # 全キャッシュを削除
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                    deleted += 1
                except Exception as e:
                    self._logger.warning(f"Failed to delete {cache_file}: {e}")
        else:
            # 指定銘柄のみ削除
            for ticker in tickers:
                cache_path = self._get_cache_path(ticker)
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        deleted += 1
                    except Exception as e:
                        self._logger.warning(f"Failed to delete {cache_path}: {e}")

        self._logger.info(f"Cleared {deleted} cache files")
        return deleted

    def get_cache_info(self) -> dict[str, Any]:
        """キャッシュ情報を取得する。

        Returns:
            キャッシュ情報
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "max_age_days": self.cache_max_age_days,
        }


def create_batch_fetcher(
    max_concurrent: int = 10,
    rate_limit_per_sec: float = 2.0,
    retry_count: int = 3,
    cache_dir: str = "cache/price_data",
) -> BatchDataFetcher:
    """BatchDataFetcherのファクトリ関数。

    Args:
        max_concurrent: 最大同時接続数
        rate_limit_per_sec: 1秒あたりのリクエスト数上限
        retry_count: リトライ回数
        cache_dir: キャッシュディレクトリ

    Returns:
        初期化されたBatchDataFetcher
    """
    return BatchDataFetcher(
        max_concurrent=max_concurrent,
        rate_limit_per_sec=rate_limit_per_sec,
        retry_count=retry_count,
        cache_dir=cache_dir,
    )


async def quick_fetch(
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_concurrent: int = 10,
) -> dict[str, pl.DataFrame]:
    """便利関数: 銘柄データを素早く取得する。

    Args:
        tickers: 銘柄コードリスト
        start_date: 開始日
        end_date: 終了日
        max_concurrent: 最大同時接続数

    Returns:
        成功した銘柄のデータ辞書
    """
    fetcher = BatchDataFetcher(max_concurrent=max_concurrent)
    result = await fetcher.fetch_all(tickers, start_date, end_date)
    return result.successful_data
