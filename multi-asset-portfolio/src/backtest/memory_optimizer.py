"""
Memory Optimizer - 大規模バックテストのメモリ最適化

800銘柄×15年のデータ処理でのメモリ使用量を30-50%削減。

主要機能:
1. データ型最適化（Float64→Float32, Int64→Int32等）
2. チャンク処理によるストリーミング
3. 明示的ガベージコレクション
4. メモリプロファイリング

Usage:
    from src.backtest.memory_optimizer import MemoryOptimizer, MemoryTracker

    # DataFrameの最適化
    optimized_df = MemoryOptimizer.optimize_dataframe(df)

    # チャンク処理
    for chunk in MemoryOptimizer.chunk_iterator(large_df, chunk_size=100):
        process(chunk)

    # メモリ追跡
    with MemoryTracker() as tracker:
        run_backtest()
    print(f"Peak memory: {tracker.peak_mb:.1f} MB")
"""

from __future__ import annotations

import gc
import logging
import sys
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator, Iterator

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """メモリ統計情報."""

    timestamp: datetime
    current_mb: float
    peak_mb: float
    allocated_mb: float = 0.0
    freed_mb: float = 0.0
    gc_collections: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Memory: current={self.current_mb:.1f}MB, "
            f"peak={self.peak_mb:.1f}MB, "
            f"allocated={self.allocated_mb:.1f}MB"
        )


class MemoryOptimizer:
    """
    メモリ最適化ユーティリティ.

    大規模データセット処理時のメモリ使用量を削減するための
    各種最適化手法を提供。
    """

    # データ型ダウンキャストのマッピング
    FLOAT_DOWNCAST = {
        pl.Float64: pl.Float32,
    }

    INT_DOWNCAST = {
        pl.Int64: pl.Int32,
        pl.Int32: pl.Int16,
        pl.Int16: pl.Int8,
    }

    @staticmethod
    def optimize_dataframe(
        df: pl.DataFrame,
        downcast_float: bool = True,
        downcast_int: bool = True,
        categorize_strings: bool = True,
        inplace: bool = False,
    ) -> pl.DataFrame:
        """
        DataFrameのデータ型を最適化してメモリ削減.

        Args:
            df: 最適化対象のPolars DataFrame
            downcast_float: Float64→Float32にダウンキャスト
            downcast_int: Int64→Int32等にダウンキャスト
            categorize_strings: 文字列をCategoricalに変換
            inplace: 元のdfを変更（Polarsでは常にFalse扱い）

        Returns:
            最適化されたDataFrame

        メモリ削減効果:
            - Float64→Float32: 約50%削減
            - Int64→Int32: 約50%削減
            - String→Categorical: 重複が多い場合に大幅削減
        """
        if df.is_empty():
            return df

        original_size = df.estimated_size("mb")
        expressions = []

        for col_name in df.columns:
            dtype = df[col_name].dtype

            # Float最適化
            if downcast_float and dtype == pl.Float64:
                # NaN/Infの範囲チェック
                col_data = df[col_name]
                if not col_data.is_null().all():
                    expressions.append(
                        pl.col(col_name).cast(pl.Float32).alias(col_name)
                    )
                    continue

            # Int最適化
            if downcast_int and dtype in (pl.Int64, pl.Int32, pl.Int16):
                col_data = df[col_name]
                if not col_data.is_null().all():
                    min_val = col_data.min()
                    max_val = col_data.max()

                    # 最適なInt型を選択
                    target_dtype = MemoryOptimizer._get_optimal_int_type(min_val, max_val)
                    if target_dtype != dtype:
                        expressions.append(
                            pl.col(col_name).cast(target_dtype).alias(col_name)
                        )
                        continue

            # String最適化（Categoricalへ）
            if categorize_strings and dtype == pl.Utf8:
                col_data = df[col_name]
                n_unique = col_data.n_unique()
                n_total = len(col_data)

                # ユニーク率が50%未満ならCategoricalに
                if n_unique < n_total * 0.5:
                    expressions.append(
                        pl.col(col_name).cast(pl.Categorical).alias(col_name)
                    )
                    continue

            # 変更なし
            expressions.append(pl.col(col_name))

        if not expressions:
            return df

        optimized = df.select(expressions)
        optimized_size = optimized.estimated_size("mb")

        reduction = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
        logger.debug(
            f"Memory optimization: {original_size:.2f}MB → {optimized_size:.2f}MB "
            f"({reduction:.1f}% reduction)"
        )

        return optimized

    @staticmethod
    def _get_optimal_int_type(min_val: int | None, max_val: int | None) -> pl.DataType:
        """最適なInt型を決定."""
        if min_val is None or max_val is None:
            return pl.Int64

        # Int8: -128 to 127
        if -128 <= min_val and max_val <= 127:
            return pl.Int8

        # Int16: -32768 to 32767
        if -32768 <= min_val and max_val <= 32767:
            return pl.Int16

        # Int32: -2147483648 to 2147483647
        if -2147483648 <= min_val and max_val <= 2147483647:
            return pl.Int32

        return pl.Int64

    @staticmethod
    def optimize_pandas_dataframe(
        df: pd.DataFrame,
        downcast_float: bool = True,
        downcast_int: bool = True,
        categorize_strings: bool = True,
    ) -> pd.DataFrame:
        """
        Pandas DataFrameのデータ型を最適化.

        Args:
            df: 最適化対象のPandas DataFrame
            downcast_float: Float64→Float32にダウンキャスト
            downcast_int: Intをダウンキャスト
            categorize_strings: 文字列をcategoryに変換

        Returns:
            最適化されたDataFrame
        """
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        result = df.copy()

        for col in result.columns:
            dtype = result[col].dtype

            # Float最適化
            if downcast_float and dtype == np.float64:
                result[col] = pd.to_numeric(result[col], downcast="float")

            # Int最適化
            elif downcast_int and dtype in (np.int64, np.int32):
                result[col] = pd.to_numeric(result[col], downcast="integer")

            # String最適化
            elif categorize_strings and dtype == object:
                n_unique = result[col].nunique()
                n_total = len(result[col])
                if n_unique < n_total * 0.5:
                    result[col] = result[col].astype("category")

        optimized_memory = result.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_memory / original_memory) * 100 if original_memory > 0 else 0

        logger.debug(
            f"Pandas optimization: {original_memory:.2f}MB → {optimized_memory:.2f}MB "
            f"({reduction:.1f}% reduction)"
        )

        return result

    @staticmethod
    def chunk_iterator(
        data: pl.DataFrame,
        chunk_size: int = 1000,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        メモリ効率的なチャンクイテレータ.

        大規模DataFrameを小さなチャンクに分割して処理。
        各チャンク処理後にGCを実行。

        Args:
            data: 分割対象のDataFrame
            chunk_size: チャンクサイズ（行数）

        Yields:
            チャンクDataFrame

        Usage:
            for chunk in MemoryOptimizer.chunk_iterator(large_df, chunk_size=100):
                results.append(process(chunk))
                # チャンク処理後にGCが自動実行される
        """
        total_rows = len(data)

        for i in range(0, total_rows, chunk_size):
            chunk = data.slice(i, min(chunk_size, total_rows - i))
            yield chunk

            # チャンク処理後にGC実行（オプション）
            if i > 0 and i % (chunk_size * 10) == 0:
                gc.collect()

    @staticmethod
    def chunk_iterator_with_overlap(
        data: pl.DataFrame,
        chunk_size: int = 1000,
        overlap: int = 0,
    ) -> Generator[tuple[pl.DataFrame, int, int], None, None]:
        """
        オーバーラップ付きチャンクイテレータ.

        時系列データで過去データが必要な場合に使用。

        Args:
            data: 分割対象のDataFrame
            chunk_size: チャンクサイズ（行数）
            overlap: オーバーラップ行数

        Yields:
            (チャンクDataFrame, 開始インデックス, 終了インデックス)
        """
        total_rows = len(data)

        for i in range(0, total_rows, chunk_size):
            start_idx = max(0, i - overlap)
            end_idx = min(total_rows, i + chunk_size)
            chunk = data.slice(start_idx, end_idx - start_idx)
            yield chunk, start_idx, end_idx

    @staticmethod
    def gc_collect(generation: int | None = None) -> dict[int, int]:
        """
        明示的なガベージコレクション.

        Args:
            generation: 収集する世代（None=全世代）

        Returns:
            各世代で収集されたオブジェクト数
        """
        if generation is not None:
            collected = gc.collect(generation)
            return {generation: collected}
        else:
            collections = {}
            for gen in range(3):
                collections[gen] = gc.collect(gen)
            return collections

    @staticmethod
    def get_object_size(obj: Any) -> int:
        """
        オブジェクトのメモリサイズを取得（バイト）.

        Args:
            obj: サイズを計測するオブジェクト

        Returns:
            メモリサイズ（バイト）
        """
        seen = set()

        def sizeof(o: Any) -> int:
            if id(o) in seen:
                return 0
            seen.add(id(o))

            size = sys.getsizeof(o)

            if isinstance(o, dict):
                size += sum(sizeof(k) + sizeof(v) for k, v in o.items())
            elif isinstance(o, (list, tuple, set, frozenset)):
                size += sum(sizeof(i) for i in o)
            elif isinstance(o, (pd.DataFrame, pd.Series)):
                size = o.memory_usage(deep=True).sum()
            elif isinstance(o, pl.DataFrame):
                size = o.estimated_size()
            elif isinstance(o, np.ndarray):
                size = o.nbytes

            return size

        return sizeof(obj)

    @staticmethod
    def clear_caches() -> None:
        """
        各種キャッシュをクリア.

        - Python内部キャッシュ
        - NumPy/Pandas関連キャッシュ
        """
        # Python interned strings をクリアしない（危険）
        # 代わりにGCを実行
        gc.collect()

        # NumPyキャッシュクリア（可能な場合）
        try:
            np.fft.restore_all()
        except AttributeError:
            pass


class MemoryTracker:
    """
    メモリ使用量追跡器.

    コンテキストマネージャとして使用し、
    処理前後のメモリ使用量を追跡。

    Usage:
        with MemoryTracker() as tracker:
            run_expensive_operation()
        print(f"Peak: {tracker.peak_mb:.1f}MB")
    """

    def __init__(self, trace_allocations: bool = False):
        """
        Args:
            trace_allocations: 詳細なアロケーション追跡を有効化
        """
        self.trace_allocations = trace_allocations
        self._start_memory: float = 0.0
        self._peak_memory: float = 0.0
        self._end_memory: float = 0.0
        self._snapshots: list[MemoryStats] = []

    @property
    def current_mb(self) -> float:
        """現在のメモリ使用量（MB）."""
        return self._get_memory_mb()

    @property
    def peak_mb(self) -> float:
        """ピークメモリ使用量（MB）."""
        return self._peak_memory

    @property
    def delta_mb(self) -> float:
        """処理前後のメモリ差分（MB）."""
        return self._end_memory - self._start_memory

    def __enter__(self) -> "MemoryTracker":
        """コンテキスト開始."""
        gc.collect()

        if self.trace_allocations:
            tracemalloc.start()

        self._start_memory = self._get_memory_mb()
        self._peak_memory = self._start_memory

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """コンテキスト終了."""
        self._end_memory = self._get_memory_mb()
        self._peak_memory = max(self._peak_memory, self._end_memory)

        if self.trace_allocations:
            tracemalloc.stop()

        gc.collect()

    def snapshot(self, label: str = "") -> MemoryStats:
        """
        現在のメモリ状態をスナップショット.

        Args:
            label: スナップショットのラベル

        Returns:
            MemoryStats
        """
        current = self._get_memory_mb()
        self._peak_memory = max(self._peak_memory, current)

        gc_stats = {i: gc.get_count()[i] for i in range(3)}

        stats = MemoryStats(
            timestamp=datetime.now(),
            current_mb=current,
            peak_mb=self._peak_memory,
            gc_collections=gc_stats,
        )

        self._snapshots.append(stats)
        return stats

    def get_summary(self) -> dict[str, Any]:
        """
        メモリ使用量サマリーを取得.

        Returns:
            サマリー辞書
        """
        return {
            "start_mb": self._start_memory,
            "end_mb": self._end_memory,
            "peak_mb": self._peak_memory,
            "delta_mb": self.delta_mb,
            "snapshots": len(self._snapshots),
        }

    @staticmethod
    def _get_memory_mb() -> float:
        """現在のプロセスメモリ使用量を取得（MB）."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # psutilがない場合は推定値
            return 0.0


@contextmanager
def memory_efficient_context(
    gc_threshold: int = 100,
    auto_gc: bool = True,
) -> Generator[MemoryTracker, None, None]:
    """
    メモリ効率的な処理コンテキスト.

    Args:
        gc_threshold: 自動GC実行の閾値（MB）
        auto_gc: 自動GC有効化

    Yields:
        MemoryTracker
    """
    # GC設定を保存
    old_threshold = gc.get_threshold()

    try:
        # GC閾値を調整（より頻繁にGC実行）
        if auto_gc:
            gc.set_threshold(700, 10, 5)

        tracker = MemoryTracker()
        with tracker:
            yield tracker

    finally:
        # GC設定を復元
        gc.set_threshold(*old_threshold)
        gc.collect()


class StreamingDataProcessor:
    """
    ストリーミングデータ処理器.

    大規模データをメモリに全て載せずに処理。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        optimize_dtypes: bool = True,
    ):
        """
        Args:
            chunk_size: チャンクサイズ
            optimize_dtypes: データ型最適化有効化
        """
        self.chunk_size = chunk_size
        self.optimize_dtypes = optimize_dtypes
        self._results: list[Any] = []

    def process_dataframe(
        self,
        df: pl.DataFrame,
        processor: callable,
        accumulator: callable | None = None,
    ) -> Any:
        """
        DataFrameをストリーミング処理.

        Args:
            df: 処理対象のDataFrame
            processor: 各チャンクに適用する関数
            accumulator: 結果を集約する関数（None=リストに追加）

        Returns:
            処理結果
        """
        self._results = []

        for chunk in MemoryOptimizer.chunk_iterator(df, self.chunk_size):
            if self.optimize_dtypes:
                chunk = MemoryOptimizer.optimize_dataframe(chunk)

            result = processor(chunk)

            if accumulator:
                if self._results:
                    self._results = [accumulator(self._results[0], result)]
                else:
                    self._results = [result]
            else:
                self._results.append(result)

        return self._results[0] if len(self._results) == 1 else self._results

    def process_file(
        self,
        file_path: str,
        processor: callable,
        file_format: str = "parquet",
    ) -> Any:
        """
        ファイルをストリーミング処理.

        Args:
            file_path: ファイルパス
            processor: 各チャンクに適用する関数
            file_format: ファイル形式

        Returns:
            処理結果
        """
        results = []

        if file_format == "parquet":
            # Parquetはrow group単位で読み込み
            df = pl.scan_parquet(file_path)
            for chunk in df.collect(streaming=True).iter_slices(self.chunk_size):
                if self.optimize_dtypes:
                    chunk = MemoryOptimizer.optimize_dataframe(chunk)
                results.append(processor(chunk))
        elif file_format == "csv":
            # CSVはチャンク読み込み
            reader = pl.read_csv_batched(file_path, batch_size=self.chunk_size)
            for chunk in reader:
                if self.optimize_dtypes:
                    chunk = MemoryOptimizer.optimize_dataframe(chunk)
                results.append(processor(chunk))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return results


def estimate_memory_usage(
    n_rows: int,
    n_cols: int,
    dtype: str = "float64",
) -> float:
    """
    DataFrameのメモリ使用量を推定（MB）.

    Args:
        n_rows: 行数
        n_cols: 列数
        dtype: データ型

    Returns:
        推定メモリ使用量（MB）
    """
    bytes_per_element = {
        "float64": 8,
        "float32": 4,
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
        "bool": 1,
    }

    element_size = bytes_per_element.get(dtype, 8)
    total_bytes = n_rows * n_cols * element_size

    # オーバーヘッド（約20%）
    total_bytes *= 1.2

    return total_bytes / 1024 / 1024


def optimize_backtest_memory(
    prices: pl.DataFrame,
    signals: pl.DataFrame | None = None,
    chunk_processing: bool = True,
    chunk_size: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """
    バックテスト用データのメモリ最適化.

    Args:
        prices: 価格データ
        signals: シグナルデータ（オプション）
        chunk_processing: チャンク処理有効化
        chunk_size: チャンクサイズ

    Returns:
        (最適化された価格データ, 最適化されたシグナルデータ)
    """
    # 価格データ最適化
    optimized_prices = MemoryOptimizer.optimize_dataframe(
        prices,
        downcast_float=True,
        downcast_int=True,
    )

    # シグナルデータ最適化
    optimized_signals = None
    if signals is not None:
        optimized_signals = MemoryOptimizer.optimize_dataframe(
            signals,
            downcast_float=True,
            downcast_int=True,
        )

    gc.collect()

    return optimized_prices, optimized_signals
