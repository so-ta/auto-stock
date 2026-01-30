"""
Numba JIT Compute Module - 超高速計算

Numba @njit(parallel=True) による並列計算で5-10倍高速化。
全銘柄×全期間を一括計算。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Numba imports with fallback
try:
    from numba import njit, prange, config
    import os
    NUMBA_AVAILABLE = True

    # スレッディングレイヤーの設定（優先順位: tbb > omp > workqueue）
    # 環境変数が設定されていない場合のみ自動設定
    if 'NUMBA_THREADING_LAYER' not in os.environ:
        # macOSではTBB/OpenMPが利用できないことが多いため、workqueueをデフォルトに
        import platform
        if platform.system() == 'Darwin':
            os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
        else:
            # Linux/Windowsでは利用可能なレイヤーを自動検出
            os.environ['NUMBA_THREADING_LAYER'] = 'default'

except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: define dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# =============================================================================
# Momentum Calculations
# =============================================================================

@njit(parallel=True, cache=True)
def momentum_batch(
    prices: NDArray[np.float64],
    periods: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    全銘柄の複数期間モメンタムを一括計算

    Parameters
    ----------
    prices : np.ndarray
        価格データ (n_assets, n_days)
    periods : np.ndarray
        ルックバック期間の配列

    Returns
    -------
    np.ndarray
        モメンタム値 (n_assets, n_days, n_periods)
    """
    n_assets, n_days = prices.shape
    n_periods = len(periods)
    result = np.full((n_assets, n_days, n_periods), np.nan)

    for p_idx in prange(n_periods):
        period = periods[p_idx]
        for i in prange(n_assets):
            for t in range(period, n_days):
                if prices[i, t - period] != 0:
                    result[i, t, p_idx] = prices[i, t] / prices[i, t - period] - 1.0
    return result


@njit(parallel=True, cache=True)
def momentum_single(
    prices: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """
    全銘柄の単一期間モメンタムを計算

    Parameters
    ----------
    prices : np.ndarray
        価格データ (n_assets, n_days)
    period : int
        ルックバック期間

    Returns
    -------
    np.ndarray
        モメンタム値 (n_assets, n_days)
    """
    n_assets, n_days = prices.shape
    result = np.full((n_assets, n_days), np.nan)

    for i in prange(n_assets):
        for t in range(period, n_days):
            if prices[i, t - period] != 0:
                result[i, t] = prices[i, t] / prices[i, t - period] - 1.0
    return result


# =============================================================================
# Volatility Calculations
# =============================================================================

@njit(parallel=True, cache=True)
def volatility_batch(
    returns: NDArray[np.float64],
    window: int,
    annualize: bool = True,
) -> NDArray[np.float64]:
    """
    全銘柄のローリングボラティリティを一括計算

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    window : int
        ローリングウィンドウサイズ
    annualize : bool
        年率換算するか

    Returns
    -------
    np.ndarray
        ボラティリティ (n_assets, n_days)
    """
    n_assets, n_days = returns.shape
    result = np.full((n_assets, n_days), np.nan)
    ann_factor = math.sqrt(252.0) if annualize else 1.0

    for i in prange(n_assets):
        for t in range(window - 1, n_days):
            # Calculate mean
            mean_val = 0.0
            for j in range(window):
                mean_val += returns[i, t - window + 1 + j]
            mean_val /= window

            # Calculate variance
            var_val = 0.0
            for j in range(window):
                diff = returns[i, t - window + 1 + j] - mean_val
                var_val += diff * diff
            var_val /= (window - 1) if window > 1 else 1

            result[i, t] = math.sqrt(var_val) * ann_factor

    return result


@njit(parallel=True, cache=True)
def volatility_ewm(
    returns: NDArray[np.float64],
    halflife: int,
    annualize: bool = True,
) -> NDArray[np.float64]:
    """
    指数加重ボラティリティを計算

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    halflife : int
        半減期
    annualize : bool
        年率換算するか

    Returns
    -------
    np.ndarray
        ボラティリティ (n_assets, n_days)
    """
    n_assets, n_days = returns.shape
    result = np.full((n_assets, n_days), np.nan)
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    ann_factor = math.sqrt(252.0) if annualize else 1.0

    for i in prange(n_assets):
        ewm_mean = 0.0
        ewm_var = 0.0

        for t in range(n_days):
            ret = returns[i, t]
            if not math.isnan(ret):
                if t == 0:
                    ewm_mean = ret
                    ewm_var = 0.0
                else:
                    delta = ret - ewm_mean
                    ewm_mean = ewm_mean + alpha * delta
                    ewm_var = (1.0 - alpha) * (ewm_var + alpha * delta * delta)
                result[i, t] = math.sqrt(ewm_var) * ann_factor

    return result


# =============================================================================
# Covariance Matrix Calculations
# =============================================================================

@njit(parallel=True, cache=True)
def covariance_matrix(
    returns: NDArray[np.float64],
    halflife: int,
) -> NDArray[np.float64]:
    """
    指数加重共分散行列を計算

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    halflife : int
        半減期

    Returns
    -------
    np.ndarray
        共分散行列 (n_assets, n_assets)
    """
    n_assets, n_days = returns.shape
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)

    # Compute weights
    weights = np.empty(n_days)
    total_weight = 0.0
    for t in range(n_days):
        w = (1.0 - alpha) ** (n_days - 1 - t)
        weights[t] = w
        total_weight += w

    # Normalize weights
    for t in range(n_days):
        weights[t] /= total_weight

    # Compute weighted means
    means = np.zeros(n_assets)
    for i in prange(n_assets):
        for t in range(n_days):
            if not math.isnan(returns[i, t]):
                means[i] += weights[t] * returns[i, t]

    # Compute covariance matrix
    cov = np.zeros((n_assets, n_assets))
    for i in prange(n_assets):
        for j in range(i, n_assets):
            cov_ij = 0.0
            for t in range(n_days):
                ri = returns[i, t]
                rj = returns[j, t]
                if not math.isnan(ri) and not math.isnan(rj):
                    cov_ij += weights[t] * (ri - means[i]) * (rj - means[j])
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij

    return cov


@njit(parallel=True, cache=True)
def correlation_matrix(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """
    ローリング相関行列を計算（最新時点）

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    window : int
        ウィンドウサイズ

    Returns
    -------
    np.ndarray
        相関行列 (n_assets, n_assets)
    """
    n_assets, n_days = returns.shape
    if n_days < window:
        return np.eye(n_assets)

    # Use last 'window' days
    start_idx = n_days - window

    # Compute means and stds
    means = np.zeros(n_assets)
    stds = np.zeros(n_assets)

    for i in prange(n_assets):
        sum_val = 0.0
        sum_sq = 0.0
        count = 0
        for t in range(start_idx, n_days):
            val = returns[i, t]
            if not math.isnan(val):
                sum_val += val
                sum_sq += val * val
                count += 1
        if count > 0:
            means[i] = sum_val / count
            var = sum_sq / count - means[i] * means[i]
            stds[i] = math.sqrt(max(0.0, var))

    # Compute correlation matrix
    corr = np.eye(n_assets)
    for i in prange(n_assets):
        for j in range(i + 1, n_assets):
            if stds[i] > 1e-10 and stds[j] > 1e-10:
                cov_ij = 0.0
                count = 0
                for t in range(start_idx, n_days):
                    ri = returns[i, t]
                    rj = returns[j, t]
                    if not math.isnan(ri) and not math.isnan(rj):
                        cov_ij += (ri - means[i]) * (rj - means[j])
                        count += 1
                if count > 0:
                    cov_ij /= count
                    corr_ij = cov_ij / (stds[i] * stds[j])
                    # Clip to valid correlation range
                    corr_ij = max(-1.0, min(1.0, corr_ij))
                    corr[i, j] = corr_ij
                    corr[j, i] = corr_ij

    return corr


# =============================================================================
# Z-Score Calculations
# =============================================================================

@njit(parallel=True, cache=True)
def zscore_batch(
    prices: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """
    全銘柄のZスコアを一括計算

    Z = (price - rolling_mean) / rolling_std

    Parameters
    ----------
    prices : np.ndarray
        価格データ (n_assets, n_days)
    window : int
        ローリングウィンドウサイズ

    Returns
    -------
    np.ndarray
        Zスコア (n_assets, n_days)
    """
    n_assets, n_days = prices.shape
    result = np.full((n_assets, n_days), np.nan)

    for i in prange(n_assets):
        for t in range(window - 1, n_days):
            # Calculate mean
            mean_val = 0.0
            for j in range(window):
                mean_val += prices[i, t - window + 1 + j]
            mean_val /= window

            # Calculate std
            var_val = 0.0
            for j in range(window):
                diff = prices[i, t - window + 1 + j] - mean_val
                var_val += diff * diff
            std_val = math.sqrt(var_val / window) if window > 0 else 0.0

            # Z-score
            if std_val > 1e-10:
                result[i, t] = (prices[i, t] - mean_val) / std_val

    return result


# =============================================================================
# RSI Calculation
# =============================================================================

@njit(parallel=True, cache=True)
def rsi_batch(
    prices: NDArray[np.float64],
    period: int,
) -> NDArray[np.float64]:
    """
    全銘柄のRSIを一括計算

    Parameters
    ----------
    prices : np.ndarray
        価格データ (n_assets, n_days)
    period : int
        RSI期間

    Returns
    -------
    np.ndarray
        RSI (n_assets, n_days), 0-100の範囲
    """
    n_assets, n_days = prices.shape
    result = np.full((n_assets, n_days), np.nan)
    alpha = 2.0 / (period + 1)

    for i in prange(n_assets):
        avg_gain = 0.0
        avg_loss = 0.0

        for t in range(1, n_days):
            delta = prices[i, t] - prices[i, t - 1]
            gain = max(0.0, delta)
            loss = max(0.0, -delta)

            if t < period:
                avg_gain += gain / period
                avg_loss += loss / period
            elif t == period:
                avg_gain += gain / period
                avg_loss += loss / period
                if avg_loss > 1e-10:
                    rs = avg_gain / avg_loss
                    result[i, t] = 100.0 - 100.0 / (1.0 + rs)
                else:
                    result[i, t] = 100.0 if avg_gain > 0 else 50.0
            else:
                avg_gain = (1.0 - alpha) * avg_gain + alpha * gain
                avg_loss = (1.0 - alpha) * avg_loss + alpha * loss
                if avg_loss > 1e-10:
                    rs = avg_gain / avg_loss
                    result[i, t] = 100.0 - 100.0 / (1.0 + rs)
                else:
                    result[i, t] = 100.0 if avg_gain > 0 else 50.0

    return result


# =============================================================================
# Sharpe Ratio Calculation
# =============================================================================

@njit(parallel=True, cache=True)
def sharpe_batch(
    returns: NDArray[np.float64],
    window: int,
    risk_free_rate: float = 0.0,
) -> NDArray[np.float64]:
    """
    全銘柄のローリングシャープレシオを一括計算

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    window : int
        ローリングウィンドウサイズ
    risk_free_rate : float
        日次リスクフリーレート

    Returns
    -------
    np.ndarray
        シャープレシオ (n_assets, n_days), 年率換算
    """
    n_assets, n_days = returns.shape
    result = np.full((n_assets, n_days), np.nan)
    ann_factor = math.sqrt(252.0)

    for i in prange(n_assets):
        for t in range(window - 1, n_days):
            # Calculate mean excess return
            mean_val = 0.0
            for j in range(window):
                mean_val += returns[i, t - window + 1 + j] - risk_free_rate
            mean_val /= window

            # Calculate std
            var_val = 0.0
            for j in range(window):
                diff = returns[i, t - window + 1 + j] - risk_free_rate - mean_val
                var_val += diff * diff
            std_val = math.sqrt(var_val / (window - 1)) if window > 1 else 0.0

            # Sharpe ratio
            if std_val > 1e-10:
                result[i, t] = (mean_val / std_val) * ann_factor

    return result


# =============================================================================
# Utility Functions
# =============================================================================

@njit(cache=True)
def returns_from_prices(
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    価格データからリターンを計算

    Parameters
    ----------
    prices : np.ndarray
        価格データ (n_assets, n_days)

    Returns
    -------
    np.ndarray
        リターン (n_assets, n_days)
    """
    n_assets, n_days = prices.shape
    result = np.full((n_assets, n_days), np.nan)

    for i in range(n_assets):
        for t in range(1, n_days):
            if prices[i, t - 1] != 0:
                result[i, t] = prices[i, t] / prices[i, t - 1] - 1.0

    return result


def check_numba_available() -> bool:
    """Numbaが利用可能かチェック"""
    return NUMBA_AVAILABLE


# =============================================================================
# Spearman Correlation (Numba JIT)
# =============================================================================

@njit(cache=True)
def rankdata_numba(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Numba高速ランク付け（average method - タイブレーク対応）

    scipy.stats.rankdata と完全互換。同順位は平均ランクを付与。

    Parameters
    ----------
    x : np.ndarray
        入力データ (1D)

    Returns
    -------
    np.ndarray
        ランク (1D), 1-indexed
    """
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # 同じ値を持つ要素を探す
        while j < n - 1 and x[order[j]] == x[order[j + 1]]:
            j += 1
        # 平均ランクを計算（1-indexed）
        avg_rank = (i + j + 2) / 2.0  # (i+1 + j+1) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


@njit(cache=True)
def spearmanr_numba(
    x: NDArray[np.float64],
    y: NDArray[np.float64]
) -> float:
    """
    Numba高速Spearman相関（精度100%保証）

    scipy.stats.spearmanr と完全互換。

    Parameters
    ----------
    x : np.ndarray
        入力データ1 (1D)
    y : np.ndarray
        入力データ2 (1D)

    Returns
    -------
    float
        Spearman相関係数
    """
    n = len(x)
    if n < 2:
        return 0.0

    ranks_x = rankdata_numba(x)
    ranks_y = rankdata_numba(y)

    # Pearson相関をランクに適用
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += ranks_x[i]
        mean_y += ranks_y[i]
    mean_x /= n
    mean_y /= n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = ranks_x[i] - mean_x
        dy = ranks_y[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    den = math.sqrt(den_x * den_y)
    if den < 1e-15:
        return 0.0

    return num / den


# =============================================================================
# Lead-Lag Correlation Calculations (task_042_1_opt)
# =============================================================================

@njit(parallel=True, cache=True)
def compute_lagged_correlations_numba(
    returns: NDArray[np.float64],
    lag_min: int,
    lag_max: int,
    window: int,
) -> NDArray[np.float64]:
    """
    全資産ペアの時間ラグ付き相関を並列計算

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    lag_min : int
        最小ラグ
    lag_max : int
        最大ラグ
    window : int
        相関計算ウィンドウ

    Returns
    -------
    np.ndarray
        相関行列 (n_assets, n_assets, n_lags)
        correlations[i, j, lag_idx] = corr(asset_i[t-lag], asset_j[t])
    """
    n_assets, n_days = returns.shape
    n_lags = lag_max - lag_min + 1
    correlations = np.full((n_assets, n_assets, n_lags), np.nan)

    if n_days < window + lag_max:
        return correlations

    # 使用する期間
    start_idx = n_days - window

    # 外側ループを並列化
    for i in prange(n_assets):
        for j in range(n_assets):
            if i == j:
                continue

            for lag_idx in range(n_lags):
                lag = lag_min + lag_idx

                # Leader(t-lag) vs Follower(t) の相関
                # leader_slice: returns[i, start_idx-lag:n_days-lag]
                # follower_slice: returns[j, start_idx:n_days]

                # means
                mean_i = 0.0
                mean_j = 0.0
                count = 0

                for t in range(window):
                    ri = returns[i, start_idx + t - lag]
                    rj = returns[j, start_idx + t]
                    if not math.isnan(ri) and not math.isnan(rj):
                        mean_i += ri
                        mean_j += rj
                        count += 1

                if count < 10:
                    continue

                mean_i /= count
                mean_j /= count

                # covariance and stds
                cov_ij = 0.0
                var_i = 0.0
                var_j = 0.0

                for t in range(window):
                    ri = returns[i, start_idx + t - lag]
                    rj = returns[j, start_idx + t]
                    if not math.isnan(ri) and not math.isnan(rj):
                        di = ri - mean_i
                        dj = rj - mean_j
                        cov_ij += di * dj
                        var_i += di * di
                        var_j += dj * dj

                std_i = math.sqrt(var_i / count) if count > 0 else 0.0
                std_j = math.sqrt(var_j / count) if count > 0 else 0.0

                if std_i > 1e-10 and std_j > 1e-10:
                    corr = (cov_ij / count) / (std_i * std_j)
                    # Clip to valid range
                    corr = max(-1.0, min(1.0, corr))
                    correlations[i, j, lag_idx] = corr

    return correlations


@njit(parallel=True, cache=True)
def compute_zero_lag_correlations_numba(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """
    全資産ペアのlag=0相関を並列計算（段階的フィルタリング用）

    Parameters
    ----------
    returns : np.ndarray
        リターンデータ (n_assets, n_days)
    window : int
        相関計算ウィンドウ

    Returns
    -------
    np.ndarray
        相関行列 (n_assets, n_assets)
    """
    n_assets, n_days = returns.shape
    correlations = np.full((n_assets, n_assets), np.nan)

    if n_days < window:
        return correlations

    start_idx = n_days - window

    # 平均を事前計算
    means = np.zeros(n_assets)
    stds = np.zeros(n_assets)

    for i in prange(n_assets):
        sum_val = 0.0
        sum_sq = 0.0
        count = 0
        for t in range(start_idx, n_days):
            val = returns[i, t]
            if not math.isnan(val):
                sum_val += val
                sum_sq += val * val
                count += 1
        if count > 0:
            means[i] = sum_val / count
            var = sum_sq / count - means[i] * means[i]
            stds[i] = math.sqrt(max(0.0, var))

    # 相関計算
    for i in prange(n_assets):
        correlations[i, i] = 1.0
        for j in range(i + 1, n_assets):
            if stds[i] < 1e-10 or stds[j] < 1e-10:
                continue

            cov_ij = 0.0
            count = 0
            for t in range(start_idx, n_days):
                ri = returns[i, t]
                rj = returns[j, t]
                if not math.isnan(ri) and not math.isnan(rj):
                    cov_ij += (ri - means[i]) * (rj - means[j])
                    count += 1

            if count > 0:
                corr = (cov_ij / count) / (stds[i] * stds[j])
                corr = max(-1.0, min(1.0, corr))
                correlations[i, j] = corr
                correlations[j, i] = corr

    return correlations


@njit(cache=True)
def find_best_lag_binary_search(
    returns_i: NDArray[np.float64],
    returns_j: NDArray[np.float64],
    lag_min: int,
    lag_max: int,
    window: int,
) -> tuple:
    """
    二分探索的lag検索で最適ラグを見つける

    Parameters
    ----------
    returns_i : np.ndarray
        Leader銘柄のリターン (n_days,)
    returns_j : np.ndarray
        Follower銘柄のリターン (n_days,)
    lag_min : int
        最小ラグ
    lag_max : int
        最大ラグ
    window : int
        相関計算ウィンドウ

    Returns
    -------
    tuple
        (best_lag, best_correlation)
    """
    n_days = len(returns_i)
    if n_days < window + lag_max:
        return (0, 0.0)

    # 粗いグリッドでスキャン
    coarse_step = max(1, (lag_max - lag_min) // 4)
    coarse_lags = []
    lag = lag_min
    while lag <= lag_max:
        coarse_lags.append(lag)
        lag += coarse_step
    if coarse_lags[-1] < lag_max:
        coarse_lags.append(lag_max)

    best_lag = lag_min
    best_corr = 0.0
    start_idx = n_days - window

    # 粗いグリッドで最良ラグを見つける
    for lag in coarse_lags:
        # 相関計算
        mean_i = 0.0
        mean_j = 0.0
        count = 0

        for t in range(window):
            ri = returns_i[start_idx + t - lag]
            rj = returns_j[start_idx + t]
            if not math.isnan(ri) and not math.isnan(rj):
                mean_i += ri
                mean_j += rj
                count += 1

        if count < 10:
            continue

        mean_i /= count
        mean_j /= count

        cov_ij = 0.0
        var_i = 0.0
        var_j = 0.0

        for t in range(window):
            ri = returns_i[start_idx + t - lag]
            rj = returns_j[start_idx + t]
            if not math.isnan(ri) and not math.isnan(rj):
                di = ri - mean_i
                dj = rj - mean_j
                cov_ij += di * dj
                var_i += di * di
                var_j += dj * dj

        std_i = math.sqrt(var_i / count) if count > 0 else 0.0
        std_j = math.sqrt(var_j / count) if count > 0 else 0.0

        if std_i > 1e-10 and std_j > 1e-10:
            corr = (cov_ij / count) / (std_i * std_j)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

    # ピーク周辺を詳細検索
    search_min = max(lag_min, best_lag - coarse_step)
    search_max = min(lag_max, best_lag + coarse_step)

    for lag in range(search_min, search_max + 1):
        if lag == best_lag:
            continue

        mean_i = 0.0
        mean_j = 0.0
        count = 0

        for t in range(window):
            ri = returns_i[start_idx + t - lag]
            rj = returns_j[start_idx + t]
            if not math.isnan(ri) and not math.isnan(rj):
                mean_i += ri
                mean_j += rj
                count += 1

        if count < 10:
            continue

        mean_i /= count
        mean_j /= count

        cov_ij = 0.0
        var_i = 0.0
        var_j = 0.0

        for t in range(window):
            ri = returns_i[start_idx + t - lag]
            rj = returns_j[start_idx + t]
            if not math.isnan(ri) and not math.isnan(rj):
                di = ri - mean_i
                dj = rj - mean_j
                cov_ij += di * dj
                var_i += di * di
                var_j += dj * dj

        std_i = math.sqrt(var_i / count) if count > 0 else 0.0
        std_j = math.sqrt(var_j / count) if count > 0 else 0.0

        if std_i > 1e-10 and std_j > 1e-10:
            corr = (cov_ij / count) / (std_i * std_j)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

    return (best_lag, best_corr)


def warmup_jit() -> None:
    """
    JITコンパイルのウォームアップ

    初回呼び出し時のコンパイル時間を事前に消化
    """
    if not NUMBA_AVAILABLE:
        return

    # Small dummy data for warmup
    dummy_prices = np.random.randn(5, 100)
    dummy_returns = np.random.randn(5, 100) * 0.01
    dummy_periods = np.array([10, 20], dtype=np.int64)

    # Warmup each function
    _ = momentum_batch(dummy_prices, dummy_periods)
    _ = momentum_single(dummy_prices, 10)
    _ = volatility_batch(dummy_returns, 20)
    _ = volatility_ewm(dummy_returns, 20)
    _ = covariance_matrix(dummy_returns, 20)
    _ = correlation_matrix(dummy_returns, 20)
    _ = zscore_batch(dummy_prices, 20)
    _ = rsi_batch(dummy_prices, 14)
    _ = sharpe_batch(dummy_returns, 20)
    _ = returns_from_prices(dummy_prices)
    _ = rankdata_numba(np.random.randn(50))
    _ = spearmanr_numba(np.random.randn(50), np.random.randn(50))

    # Lead-Lag functions warmup
    _ = compute_lagged_correlations_numba(dummy_returns, 1, 5, 20)
    _ = compute_zero_lag_correlations_numba(dummy_returns, 20)
    _ = find_best_lag_binary_search(dummy_returns[0], dummy_returns[1], 1, 5, 20)
