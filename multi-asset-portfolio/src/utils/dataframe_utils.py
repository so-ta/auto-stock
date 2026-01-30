"""
DataFrame Utilities - Polars/Pandas変換の最適化

DataFrame変換を一元管理し、不要な変換を削減する。
API境界でのみ変換を行い、内部処理は一貫したフォーマットを使用。

task_013_6: DataFrame変換削減による20-30%のオーバーヘッド削減

Usage:
    from src.utils.dataframe_utils import ensure_polars, ensure_pandas

    # Polars形式が必要な場合
    df_pl = ensure_polars(df)

    # Pandas形式が必要な場合
    df_pd = ensure_pandas(df)

    # 内部処理用（変換トラッキング付き）
    df_pl, was_converted = to_polars_tracked(df)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias for DataFrame types
DataFrameType = Union[pd.DataFrame, "pl.DataFrame"]


def ensure_polars(df: DataFrameType) -> "pl.DataFrame":
    """
    DataFrameをPolars形式に変換（必要な場合のみ）

    Args:
        df: Pandas or Polars DataFrame

    Returns:
        Polars DataFrame

    Note:
        既にPolarsの場合は変換せずそのまま返す
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is not installed")

    if isinstance(df, pl.DataFrame):
        return df

    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)

    raise TypeError(f"Expected DataFrame, got {type(df)}")


def ensure_pandas(df: DataFrameType) -> pd.DataFrame:
    """
    DataFrameをPandas形式に変換（必要な場合のみ）

    Args:
        df: Pandas or Polars DataFrame

    Returns:
        Pandas DataFrame

    Note:
        既にPandasの場合は変換せずそのまま返す
    """
    if isinstance(df, pd.DataFrame):
        return df

    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df.to_pandas()

    raise TypeError(f"Expected DataFrame, got {type(df)}")


def to_polars_tracked(df: DataFrameType) -> Tuple["pl.DataFrame", bool]:
    """
    Polarsに変換し、変換が行われたかを追跡

    Args:
        df: Pandas or Polars DataFrame

    Returns:
        (Polars DataFrame, was_converted)
        - was_converted: True if conversion happened

    Usage:
        df_pl, converted = to_polars_tracked(df)
        # ... process ...
        if converted:
            # 元がPandasだった場合、戻す必要があるかも
            return df_pl.to_pandas()
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is not installed")

    if isinstance(df, pl.DataFrame):
        return df, False

    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df), True

    raise TypeError(f"Expected DataFrame, got {type(df)}")


def to_pandas_tracked(df: DataFrameType) -> Tuple[pd.DataFrame, bool]:
    """
    Pandasに変換し、変換が行われたかを追跡

    Args:
        df: Pandas or Polars DataFrame

    Returns:
        (Pandas DataFrame, was_converted)
        - was_converted: True if conversion happened
    """
    if isinstance(df, pd.DataFrame):
        return df, False

    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df.to_pandas(), True

    raise TypeError(f"Expected DataFrame, got {type(df)}")


def is_polars(df: DataFrameType) -> bool:
    """DataFrameがPolars形式かどうかを判定"""
    if not POLARS_AVAILABLE:
        return False
    return isinstance(df, pl.DataFrame)


def is_pandas(df: DataFrameType) -> bool:
    """DataFrameがPandas形式かどうかを判定"""
    return isinstance(df, pd.DataFrame)


def get_numeric_columns(
    df: DataFrameType,
    exclude: List[str] | None = None,
) -> List[str]:
    """
    数値カラムのみを取得（timestamp等を除外）

    Args:
        df: DataFrame
        exclude: 除外するカラム名（デフォルト: timestamp, date, index関連）

    Returns:
        数値カラム名のリスト
    """
    if exclude is None:
        exclude = ["timestamp", "date", "index", "datetime", "time"]

    if is_polars(df):
        # Polars: 数値型のカラムを取得
        numeric_cols = [
            col for col in df.columns
            if col not in exclude
            and df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
        ]
    else:
        # Pandas: 数値型のカラムを取得
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]

    return numeric_cols


def extract_numeric_pandas(
    df: DataFrameType,
    exclude: List[str] | None = None,
) -> pd.DataFrame:
    """
    数値カラムのみをPandas DataFrameとして抽出

    Args:
        df: DataFrame (Polars or Pandas)
        exclude: 除外するカラム名

    Returns:
        数値カラムのみのPandas DataFrame

    Note:
        内部計算でPandasが必要な場合に使用。
        変換回数を最小限にするため、この関数を使う。
    """
    if exclude is None:
        exclude = ["timestamp", "date", "index", "datetime", "time"]

    if is_polars(df):
        # Polars: 必要なカラムだけ選択してから変換
        return_cols = [c for c in df.columns if c not in exclude]
        return df.select(return_cols).to_pandas()
    else:
        # Pandas: 不要カラムを除外
        cols = [c for c in df.columns if c not in exclude]
        return df[cols]


def extract_numeric_numpy(
    df: DataFrameType,
    exclude: List[str] | None = None,
) -> "NDArray[np.float64]":
    """
    数値データをNumPy配列として抽出

    Args:
        df: DataFrame (Polars or Pandas)
        exclude: 除外するカラム名

    Returns:
        NumPy配列 (T x N)

    Note:
        最も高速な抽出方法。計算のみ必要な場合に使用。
    """
    if exclude is None:
        exclude = ["timestamp", "date", "index", "datetime", "time"]

    if is_polars(df):
        return_cols = [c for c in df.columns if c not in exclude]
        return df.select(return_cols).to_numpy()
    else:
        cols = [c for c in df.columns if c not in exclude]
        return df[cols].values


class ConversionTracker:
    """
    DataFrame変換の追跡用クラス

    デバッグ・プロファイリング目的で変換回数を追跡。

    Usage:
        tracker = ConversionTracker()

        with tracker.track("signal_generation"):
            df_pd = tracker.to_pandas(df_pl)
            # ...

        print(tracker.stats)
    """

    def __init__(self):
        self._conversions: List[dict] = []
        self._current_context: str | None = None

    def track(self, context: str):
        """コンテキストマネージャーで変換を追跡"""
        self._current_context = context
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._current_context = None

    def to_pandas(self, df: DataFrameType) -> pd.DataFrame:
        """変換を追跡しながらPandasに変換"""
        was_polars = is_polars(df)
        result = ensure_pandas(df)

        if was_polars:
            self._conversions.append({
                "context": self._current_context,
                "direction": "polars_to_pandas",
                "rows": len(df),
                "cols": len(df.columns),
            })

        return result

    def to_polars(self, df: DataFrameType) -> "pl.DataFrame":
        """変換を追跡しながらPolarsに変換"""
        was_pandas = is_pandas(df)
        result = ensure_polars(df)

        if was_pandas:
            self._conversions.append({
                "context": self._current_context,
                "direction": "pandas_to_polars",
                "rows": len(df),
                "cols": len(df.columns),
            })

        return result

    @property
    def stats(self) -> dict:
        """変換統計を取得"""
        return {
            "total_conversions": len(self._conversions),
            "polars_to_pandas": sum(1 for c in self._conversions if c["direction"] == "polars_to_pandas"),
            "pandas_to_polars": sum(1 for c in self._conversions if c["direction"] == "pandas_to_polars"),
            "by_context": self._group_by_context(),
        }

    def _group_by_context(self) -> dict:
        """コンテキストごとの変換回数"""
        result = {}
        for conv in self._conversions:
            ctx = conv["context"] or "unknown"
            result[ctx] = result.get(ctx, 0) + 1
        return result

    def reset(self):
        """統計をリセット"""
        self._conversions.clear()


# グローバルトラッカー（オプション、デバッグ用）
_global_tracker: ConversionTracker | None = None


def get_global_tracker() -> ConversionTracker:
    """グローバルトラッカーを取得（デバッグ用）"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ConversionTracker()
    return _global_tracker


def reset_global_tracker():
    """グローバルトラッカーをリセット"""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
