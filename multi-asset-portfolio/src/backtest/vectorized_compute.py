"""
Vectorized Compute Module - ベクトル化計算

NumPy/Polarsで全銘柄×全期間を一括計算。
ループベースの計算を5-20倍高速化。
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from src.utils.dataframe_utils import extract_numeric_pandas


def compute_all_momentum_vectorized(
    prices: pl.DataFrame,
    periods: List[int],
) -> pl.DataFrame:
    """
    全銘柄の複数期間モメンタムを一括計算

    Parameters
    ----------
    prices : pl.DataFrame
        価格データ（columns=銘柄名、rows=日付）
    periods : List[int]
        モメンタム計算期間のリスト

    Returns
    -------
    pl.DataFrame
        モメンタム値（columns='{symbol}_mom_{period}'形式）
    """
    result_cols = []

    # timestampカラムがあれば除外
    price_cols = [c for c in prices.columns if c not in ("timestamp", "date", "index")]

    for period in periods:
        for col in price_cols:
            mom_col_name = f"{col}_mom_{period}"
            # pct_change equivalent in polars
            mom = (
                prices[col] / prices[col].shift(period) - 1
            ).alias(mom_col_name)
            result_cols.append(mom)

    # 元のtimestampを保持
    if "timestamp" in prices.columns:
        return prices.select(["timestamp"] + result_cols)
    return prices.select(result_cols)


def compute_all_volatility_vectorized(
    returns: pl.DataFrame,
    window: int,
    annualize: bool = True,
) -> pl.DataFrame:
    """
    全銘柄のローリング標準偏差を一括計算

    Parameters
    ----------
    returns : pl.DataFrame
        リターンデータ（columns=銘柄名）
    window : int
        ローリングウィンドウサイズ
    annualize : bool
        年率換算するか（default: True, √252を乗算）

    Returns
    -------
    pl.DataFrame
        ボラティリティ値（columns='{symbol}_vol'形式）
    """
    result_cols = []
    annualization_factor = np.sqrt(252) if annualize else 1.0

    # timestampカラムがあれば除外
    return_cols = [c for c in returns.columns if c not in ("timestamp", "date", "index")]

    for col in return_cols:
        vol_col_name = f"{col}_vol"
        vol = (
            returns[col].rolling_std(window_size=window) * annualization_factor
        ).alias(vol_col_name)
        result_cols.append(vol)

    if "timestamp" in returns.columns:
        return returns.select(["timestamp"] + result_cols)
    return returns.select(result_cols)


def compute_all_zscore_vectorized(
    prices: pl.DataFrame,
    window: int,
) -> pl.DataFrame:
    """
    全銘柄のZスコアを一括計算

    Z = (price - rolling_mean) / rolling_std

    Parameters
    ----------
    prices : pl.DataFrame
        価格データ（columns=銘柄名）
    window : int
        ローリングウィンドウサイズ

    Returns
    -------
    pl.DataFrame
        Zスコア値（columns='{symbol}_zscore'形式）
    """
    result_cols = []

    price_cols = [c for c in prices.columns if c not in ("timestamp", "date", "index")]

    for col in price_cols:
        zscore_col_name = f"{col}_zscore"
        rolling_mean = prices[col].rolling_mean(window_size=window)
        rolling_std = prices[col].rolling_std(window_size=window)

        # Zスコア計算（ゼロ除算防止）
        zscore = (
            (prices[col] - rolling_mean) / rolling_std.clip(lower_bound=1e-10)
        ).alias(zscore_col_name)
        result_cols.append(zscore)

    if "timestamp" in prices.columns:
        return prices.select(["timestamp"] + result_cols)
    return prices.select(result_cols)


def compute_covariance_ewm(
    returns: Union[pl.DataFrame, pd.DataFrame],
    halflife: int,
) -> NDArray[np.float64]:
    """
    指数加重共分散行列を計算

    Parameters
    ----------
    returns : pl.DataFrame or pd.DataFrame
        リターンデータ
    halflife : int
        指数加重の半減期

    Returns
    -------
    np.ndarray
        共分散行列（N×N）
    """
    # Polars/Pandasを統一的に処理（task_013_6: 変換削減）
    returns_pd = extract_numeric_pandas(returns)

    # 指数加重共分散行列
    ewm_cov = returns_pd.ewm(halflife=halflife).cov()

    # 最新時点の共分散行列を抽出
    n_assets = len(returns_pd.columns)
    last_idx = ewm_cov.index.get_level_values(0)[-1]
    cov_matrix = ewm_cov.loc[last_idx].values

    return cov_matrix


def compute_correlation_matrix(
    returns: Union[pl.DataFrame, pd.DataFrame],
    window: int,
) -> NDArray[np.float64]:
    """
    ローリング相関行列を計算

    Parameters
    ----------
    returns : pl.DataFrame or pd.DataFrame
        リターンデータ
    window : int
        ローリングウィンドウサイズ

    Returns
    -------
    np.ndarray
        相関行列（N×N）、最新時点の値
    """
    # Polars/Pandasを統一的に処理（task_013_6: 変換削減）
    returns_pd = extract_numeric_pandas(returns)

    # 直近window期間の相関行列
    recent_returns = returns_pd.iloc[-window:]
    corr_matrix = recent_returns.corr().values

    return corr_matrix


def compute_rolling_correlation_matrix(
    returns: Union[pl.DataFrame, pd.DataFrame],
    window: int,
) -> List[NDArray[np.float64]]:
    """
    全期間のローリング相関行列を計算

    Parameters
    ----------
    returns : pl.DataFrame or pd.DataFrame
        リターンデータ
    window : int
        ローリングウィンドウサイズ

    Returns
    -------
    List[np.ndarray]
        各時点の相関行列リスト
    """
    # Polars/Pandasを統一的に処理（task_013_6: 変換削減）
    returns_pd = extract_numeric_pandas(returns)

    n_rows = len(returns_pd)
    n_assets = len(returns_pd.columns)
    matrices = []

    for i in range(window - 1, n_rows):
        window_data = returns_pd.iloc[i - window + 1:i + 1]
        corr = window_data.corr().values
        matrices.append(corr)

    return matrices


def compute_all_rsi_vectorized(
    prices: pl.DataFrame,
    period: int = 14,
) -> pl.DataFrame:
    """
    全銘柄のRSIを一括計算

    Parameters
    ----------
    prices : pl.DataFrame
        価格データ
    period : int
        RSI計算期間

    Returns
    -------
    pl.DataFrame
        RSI値（0-100）
    """
    result_cols = []
    price_cols = [c for c in prices.columns if c not in ("timestamp", "date", "index")]

    for col in price_cols:
        rsi_col_name = f"{col}_rsi"

        # 価格変化
        delta = prices[col].diff()

        # 上昇・下落を分離
        gain = delta.clip(lower_bound=0)
        loss = (-delta).clip(lower_bound=0)

        # 指数移動平均
        avg_gain = gain.ewm_mean(span=period, adjust=False)
        avg_loss = loss.ewm_mean(span=period, adjust=False)

        # RSI計算
        rs = avg_gain / avg_loss.clip(lower_bound=1e-10)
        rsi = (100 - 100 / (1 + rs)).alias(rsi_col_name)
        result_cols.append(rsi)

    if "timestamp" in prices.columns:
        return prices.select(["timestamp"] + result_cols)
    return prices.select(result_cols)


def compute_all_bollinger_vectorized(
    prices: pl.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """
    全銘柄のボリンジャーバンドを一括計算

    Parameters
    ----------
    prices : pl.DataFrame
        価格データ
    window : int
        移動平均期間
    num_std : float
        標準偏差の倍数

    Returns
    -------
    pl.DataFrame
        upper, middle, lower, %B 各列
    """
    result_cols = []
    price_cols = [c for c in prices.columns if c not in ("timestamp", "date", "index")]

    for col in price_cols:
        middle = prices[col].rolling_mean(window_size=window)
        std = prices[col].rolling_std(window_size=window)

        upper = (middle + num_std * std).alias(f"{col}_bb_upper")
        lower = (middle - num_std * std).alias(f"{col}_bb_lower")
        middle_col = middle.alias(f"{col}_bb_middle")

        # %B = (price - lower) / (upper - lower)
        band_width = (upper - lower).clip(lower_bound=1e-10)
        pct_b = ((prices[col] - lower) / band_width).alias(f"{col}_bb_pctb")

        result_cols.extend([upper, middle_col, lower, pct_b])

    if "timestamp" in prices.columns:
        return prices.select(["timestamp"] + result_cols)
    return prices.select(result_cols)


def compute_returns_matrix(
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """
    価格データからリターン行列を計算

    Parameters
    ----------
    prices : pl.DataFrame
        価格データ

    Returns
    -------
    pl.DataFrame
        リターンデータ
    """
    result_cols = []
    price_cols = [c for c in prices.columns if c not in ("timestamp", "date", "index")]

    for col in price_cols:
        returns = prices[col].pct_change().alias(col)
        result_cols.append(returns)

    if "timestamp" in prices.columns:
        return prices.select(["timestamp"] + result_cols)
    return prices.select(result_cols)


def compute_sharpe_vectorized(
    returns: pl.DataFrame,
    window: int,
    risk_free_rate: float = 0.0,
) -> pl.DataFrame:
    """
    全銘柄のローリングシャープレシオを一括計算

    Parameters
    ----------
    returns : pl.DataFrame
        リターンデータ
    window : int
        ローリングウィンドウ
    risk_free_rate : float
        リスクフリーレート（日次）

    Returns
    -------
    pl.DataFrame
        シャープレシオ（年率換算）
    """
    result_cols = []
    return_cols = [c for c in returns.columns if c not in ("timestamp", "date", "index")]

    annualization = np.sqrt(252)

    for col in return_cols:
        excess_return = returns[col] - risk_free_rate
        rolling_mean = excess_return.rolling_mean(window_size=window)
        rolling_std = excess_return.rolling_std(window_size=window)

        sharpe = (
            (rolling_mean / rolling_std.clip(lower_bound=1e-10)) * annualization
        ).alias(f"{col}_sharpe")
        result_cols.append(sharpe)

    if "timestamp" in returns.columns:
        return returns.select(["timestamp"] + result_cols)
    return returns.select(result_cols)


# NumPy高速計算ユーティリティ
def fast_ewm_covariance(
    returns: NDArray[np.float64],
    halflife: int,
) -> NDArray[np.float64]:
    """
    NumPyによる高速指数加重共分散計算

    Parameters
    ----------
    returns : np.ndarray
        リターン行列（T×N）
    halflife : int
        半減期

    Returns
    -------
    np.ndarray
        共分散行列（N×N）
    """
    alpha = 1 - np.exp(-np.log(2) / halflife)
    n_obs, n_assets = returns.shape

    # 平均を引く
    weights = np.array([(1 - alpha) ** i for i in range(n_obs - 1, -1, -1)])
    weights = weights / weights.sum()

    centered = returns - np.average(returns, weights=weights, axis=0)

    # 加重共分散
    cov = np.zeros((n_assets, n_assets))
    for i in range(n_obs):
        cov += weights[i] * np.outer(centered[i], centered[i])

    return cov


def fast_rolling_std(
    data: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """
    NumPyによる高速ローリング標準偏差

    Parameters
    ----------
    data : np.ndarray
        入力データ（1次元）
    window : int
        ウィンドウサイズ

    Returns
    -------
    np.ndarray
        ローリング標準偏差
    """
    n = len(data)
    result = np.full(n, np.nan)

    # 累積和を使った高速計算
    cumsum = np.cumsum(data)
    cumsum_sq = np.cumsum(data ** 2)

    for i in range(window - 1, n):
        if i == window - 1:
            s = cumsum[i]
            s_sq = cumsum_sq[i]
        else:
            s = cumsum[i] - cumsum[i - window]
            s_sq = cumsum_sq[i] - cumsum_sq[i - window]

        mean = s / window
        variance = s_sq / window - mean ** 2
        result[i] = np.sqrt(max(0, variance))

    return result
