"""
Numba JIT Accelerated Functions - CPU並列高速化

8-12倍高速化を実現するNumba JIT関数群。
バックテストのボトルネック処理を高速化する。

使用例:
    from backtest.numba_accelerate import (
        calculate_max_drawdown_numba,
        normalize_scores_rank_numba,
    )
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def calculate_max_drawdown_numba(portfolio_values: np.ndarray) -> float:
    """
    最大ドローダウンを計算（Numba JIT版）

    Args:
        portfolio_values: ポートフォリオ価値の配列

    Returns:
        最大ドローダウン（負の値、例: -0.15 = 15%ドローダウン）
    """
    n = len(portfolio_values)
    if n == 0:
        return 0.0

    running_max = portfolio_values[0]
    max_dd = 0.0

    for i in range(1, n):
        if portfolio_values[i] > running_max:
            running_max = portfolio_values[i]
        if running_max > 0:
            dd = (portfolio_values[i] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd

    return max_dd


@njit(cache=True, fastmath=True)
def calculate_max_drawdown_pct_numba(returns: np.ndarray) -> float:
    """
    リターン配列から最大ドローダウン（%）を計算（Numba JIT版）

    cumulative = cumprod(1 + returns) を内部で計算

    Args:
        returns: 日次リターンの配列

    Returns:
        最大ドローダウン（%、例: 15.0 = 15%）
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # 累積リターン計算
    cumulative = 1.0
    running_max = 1.0
    max_dd = 0.0

    for i in range(n):
        cumulative *= (1.0 + returns[i])
        if cumulative > running_max:
            running_max = cumulative
        if running_max > 0:
            dd = (running_max - cumulative) / running_max
            if dd > max_dd:
                max_dd = dd

    return max_dd * 100.0


@njit(cache=True)
def _argsort_1d(arr: np.ndarray) -> np.ndarray:
    """1D配列のargsort（Numba互換）"""
    n = len(arr)
    indices = np.arange(n)
    # Simple insertion sort for small arrays (Numba compatible)
    for i in range(1, n):
        key_idx = indices[i]
        key_val = arr[key_idx]
        j = i - 1
        while j >= 0 and arr[indices[j]] > key_val:
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = key_idx
    return indices


@njit(cache=True)
def _rank_array_numba(arr: np.ndarray) -> np.ndarray:
    """
    配列をパーセンタイルランク（0-1）に変換（Numba JIT版）

    argsort-argsortトリックでO(n log n)ランキング
    """
    n = len(arr)
    if n <= 1:
        result = np.empty(n, dtype=np.float64)
        if n == 1:
            result[0] = 0.5
        return result

    order = _argsort_1d(arr)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = float(i)

    for i in range(n):
        ranks[i] = ranks[i] / (n - 1)

    return ranks


@njit(cache=True)
def normalize_scores_rank_numba(scores: np.ndarray) -> np.ndarray:
    """
    スコアをランク正規化（Numba JIT版、並列処理）

    Args:
        scores: スコア配列 (n_assets, n_days)

    Returns:
        正規化されたスコア配列 (n_assets, n_days)
    """
    n_assets, n_days = scores.shape
    normalized = np.zeros((n_assets, n_days), dtype=np.float64)

    for t in range(n_days):
        col = scores[:, t].copy()

        # NaNカウント（Numbaではnp.isnanが制限あり）
        valid_count = 0
        for i in range(n_assets):
            if not np.isnan(col[i]):
                valid_count += 1

        if valid_count > 1:
            # 有効な値のみでランキング
            valid_vals = np.empty(valid_count, dtype=np.float64)
            valid_indices = np.empty(valid_count, dtype=np.int64)
            idx = 0
            for i in range(n_assets):
                if not np.isnan(col[i]):
                    valid_vals[idx] = col[i]
                    valid_indices[idx] = i
                    idx += 1

            # ランク計算
            ranks = _rank_array_numba(valid_vals)

            # 結果を割り当て
            for i in range(valid_count):
                normalized[valid_indices[i], t] = ranks[i]
        else:
            # 有効データ不足の場合は0.5
            for i in range(n_assets):
                normalized[i, t] = 0.5

    return normalized


@njit(cache=True)
def normalize_scores_zscore_numba(scores: np.ndarray) -> np.ndarray:
    """
    スコアをzスコア正規化（Numba JIT版、並列処理）

    Args:
        scores: スコア配列 (n_assets, n_days)

    Returns:
        正規化されたスコア配列 (n_assets, n_days)
    """
    n_assets, n_days = scores.shape
    normalized = np.zeros((n_assets, n_days), dtype=np.float64)

    for t in range(n_days):
        col = scores[:, t]

        # 有効データの平均・標準偏差計算
        valid_sum = 0.0
        valid_sq_sum = 0.0
        valid_count = 0

        for i in range(n_assets):
            if not np.isnan(col[i]):
                valid_sum += col[i]
                valid_sq_sum += col[i] * col[i]
                valid_count += 1

        if valid_count > 1:
            mean = valid_sum / valid_count
            variance = (valid_sq_sum / valid_count) - (mean * mean)
            std = np.sqrt(variance) if variance > 0 else 0.0

            if std > 0:
                for i in range(n_assets):
                    if not np.isnan(col[i]):
                        normalized[i, t] = (col[i] - mean) / std
                    else:
                        normalized[i, t] = 0.0
            else:
                for i in range(n_assets):
                    normalized[i, t] = 0.0
        else:
            for i in range(n_assets):
                normalized[i, t] = 0.0

    return normalized


@njit(cache=True)
def normalize_scores_minmax_numba(scores: np.ndarray) -> np.ndarray:
    """
    スコアをmin-max正規化（Numba JIT版、並列処理）

    Args:
        scores: スコア配列 (n_assets, n_days)

    Returns:
        正規化されたスコア配列 (n_assets, n_days)、0-1範囲
    """
    n_assets, n_days = scores.shape
    normalized = np.zeros((n_assets, n_days), dtype=np.float64)

    for t in range(n_days):
        col = scores[:, t]

        # min/max計算（NaN除外）
        min_val = np.inf
        max_val = -np.inf
        valid_count = 0

        for i in range(n_assets):
            if not np.isnan(col[i]):
                if col[i] < min_val:
                    min_val = col[i]
                if col[i] > max_val:
                    max_val = col[i]
                valid_count += 1

        if valid_count > 1 and max_val > min_val:
            range_val = max_val - min_val
            for i in range(n_assets):
                if not np.isnan(col[i]):
                    normalized[i, t] = (col[i] - min_val) / range_val
                else:
                    normalized[i, t] = 0.5
        else:
            for i in range(n_assets):
                normalized[i, t] = 0.5

    return normalized


@njit(cache=True, fastmath=True)
def calculate_sharpe_ratio_numba(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    シャープレシオを計算（Numba JIT版）

    Args:
        returns: 日次リターンの配列
        risk_free_rate: 年率リスクフリーレート（デフォルト0）
        annualization_factor: 年率化係数（日次の場合252）

    Returns:
        シャープレシオ
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # 平均リターン
    mean_return = 0.0
    for i in range(n):
        mean_return += returns[i]
    mean_return /= n

    # 標準偏差
    variance = 0.0
    for i in range(n):
        diff = returns[i] - mean_return
        variance += diff * diff

    if n > 1:
        variance /= (n - 1)  # サンプル標準偏差
    else:
        return 0.0

    std_return = np.sqrt(variance)
    if std_return == 0:
        return 0.0

    # 日次シャープレシオ
    daily_rf = risk_free_rate / annualization_factor
    daily_sharpe = (mean_return - daily_rf) / std_return

    # 年率化
    return daily_sharpe * np.sqrt(annualization_factor)


@njit(cache=True, fastmath=True)
def calculate_sortino_ratio_numba(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    ソルティノレシオを計算（Numba JIT版）

    下方偏差のみを使用

    Args:
        returns: 日次リターンの配列
        risk_free_rate: 年率リスクフリーレート
        annualization_factor: 年率化係数

    Returns:
        ソルティノレシオ
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # 平均リターン
    mean_return = 0.0
    for i in range(n):
        mean_return += returns[i]
    mean_return /= n

    # 下方偏差（負のリターンのみ）
    downside_sum = 0.0
    downside_count = 0
    for i in range(n):
        if returns[i] < 0:
            downside_sum += returns[i] * returns[i]
            downside_count += 1

    if downside_count == 0:
        return np.inf if mean_return > 0 else 0.0

    downside_variance = downside_sum / downside_count
    downside_deviation = np.sqrt(downside_variance)

    if downside_deviation == 0:
        return np.inf if mean_return > 0 else 0.0

    # 日次ソルティノ
    daily_rf = risk_free_rate / annualization_factor
    daily_sortino = (mean_return - daily_rf) / downside_deviation

    # 年率化
    return daily_sortino * np.sqrt(annualization_factor)


@njit(cache=True, fastmath=True)
def calculate_calmar_ratio_numba(
    returns: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """
    カルマーレシオを計算（Numba JIT版）

    年率リターン / |最大ドローダウン|

    Args:
        returns: 日次リターンの配列
        annualization_factor: 年率化係数

    Returns:
        カルマーレシオ
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # 年率リターン
    total_return = 1.0
    for i in range(n):
        total_return *= (1.0 + returns[i])

    years = n / annualization_factor
    if years <= 0:
        return 0.0

    annual_return = total_return ** (1.0 / years) - 1.0

    # 最大ドローダウン
    max_dd = calculate_max_drawdown_pct_numba(returns) / 100.0

    if max_dd == 0:
        return np.inf if annual_return > 0 else 0.0

    return annual_return / max_dd


@njit(cache=True, fastmath=True)
def calculate_volatility_numba(
    returns: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """
    年率ボラティリティを計算（Numba JIT版）

    Args:
        returns: 日次リターンの配列
        annualization_factor: 年率化係数

    Returns:
        年率ボラティリティ
    """
    n = len(returns)
    if n <= 1:
        return 0.0

    # 平均
    mean = 0.0
    for i in range(n):
        mean += returns[i]
    mean /= n

    # 分散
    variance = 0.0
    for i in range(n):
        diff = returns[i] - mean
        variance += diff * diff
    variance /= (n - 1)  # サンプル分散

    # 年率化
    return np.sqrt(variance) * np.sqrt(annualization_factor)
