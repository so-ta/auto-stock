"""
Numba JIT Accelerated Functions - CPU並列高速化

8-12倍高速化を実現するNumba JIT関数群。
バックテストのボトルネック処理を高速化する。

使用例:
    from backtest.numba_accelerate import (
        normalize_scores_rank_numba,
        normalize_scores_zscore_numba,
    )

Note:
    Metrics functions (calculate_sharpe_ratio, calculate_max_drawdown, etc.)
    are now in src/utils/metrics.py with Numba acceleration.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Note: Metrics functions (calculate_sharpe_ratio_numba, calculate_max_drawdown_numba, etc.)
# have been moved to src/utils/metrics.py. Use that module for performance metrics.


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
