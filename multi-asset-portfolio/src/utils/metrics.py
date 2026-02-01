"""
Unified Metrics Module - 標準化されたパフォーマンスメトリクス計算

全てのメトリクス計算（Sharpe ratio、Max Drawdown、Sortino等）をこのモジュールに統一。
Numbaアクセラレーションをサポートし、異常値に対して例外を発生。

Usage:
    from src.utils.metrics import (
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_sortino_ratio,
        calculate_calmar_ratio,
        MetricsCalculator,
    )

    # 単一メトリクス計算
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(returns=returns)

    # 一括計算
    calc = MetricsCalculator()
    metrics = calc.calculate_all(returns)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numba optional import
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator when Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# Configuration
# =============================================================================

# Default parameters
DEFAULT_ANNUALIZATION_FACTOR = 252  # Trading days per year
DEFAULT_RISK_FREE_RATE = 0.0


# =============================================================================
# Numba-accelerated core functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _sharpe_ratio_numba(
    returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: int,
) -> float:
    """Numba JIT-compiled Sharpe ratio calculation."""
    n = len(returns)
    if n < 2:
        return 0.0

    # Calculate mean and std
    total = 0.0
    for i in range(n):
        total += returns[i]
    mean_return = total / n

    # Calculate variance
    var_sum = 0.0
    for i in range(n):
        diff = returns[i] - mean_return
        var_sum += diff * diff
    std_return = np.sqrt(var_sum / (n - 1))  # Sample std

    if std_return < 1e-10:
        return 0.0

    # Daily excess return
    daily_excess = mean_return - (risk_free_rate / annualization_factor)
    daily_sharpe = daily_excess / std_return

    return daily_sharpe * np.sqrt(annualization_factor)


@njit(cache=True, fastmath=True)
def _max_drawdown_from_returns_numba(returns: np.ndarray) -> float:
    """Calculate max drawdown from returns array (Numba JIT)."""
    n = len(returns)
    if n == 0:
        return 0.0

    # Build cumulative values
    peak = 1.0
    max_dd = 0.0
    value = 1.0

    for i in range(n):
        value *= (1.0 + returns[i])
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


@njit(cache=True, fastmath=True)
def _max_drawdown_from_values_numba(portfolio_values: np.ndarray) -> float:
    """Calculate max drawdown from portfolio values (Numba JIT)."""
    n = len(portfolio_values)
    if n < 2:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0.0

    for i in range(1, n):
        val = portfolio_values[i]
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


@njit(cache=True, fastmath=True)
def _sortino_ratio_numba(
    returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: int,
) -> float:
    """Numba JIT-compiled Sortino ratio calculation."""
    n = len(returns)
    if n < 2:
        return 0.0

    # Calculate mean
    total = 0.0
    for i in range(n):
        total += returns[i]
    mean_return = total / n

    # Calculate downside deviation
    downside_sum = 0.0
    downside_count = 0
    daily_rf = risk_free_rate / annualization_factor

    for i in range(n):
        excess = returns[i] - daily_rf
        if excess < 0:
            downside_sum += excess * excess
            downside_count += 1

    if downside_count == 0:
        # No negative returns
        return np.inf if mean_return > daily_rf else 0.0

    downside_std = np.sqrt(downside_sum / downside_count)

    if downside_std < 1e-10:
        return np.inf if mean_return > daily_rf else 0.0

    daily_excess = mean_return - daily_rf
    daily_sortino = daily_excess / downside_std

    return daily_sortino * np.sqrt(annualization_factor)


@njit(cache=True, fastmath=True)
def _rolling_sharpe_numba(
    returns: np.ndarray,
    window: int,
    annualization_factor: int,
) -> np.ndarray:
    """Calculate rolling Sharpe ratio (Numba JIT)."""
    n = len(returns)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        # Extract window
        start = i - window + 1
        window_returns = returns[start:i + 1]

        # Calculate mean and std
        total = 0.0
        for j in range(window):
            total += window_returns[j]
        mean_return = total / window

        var_sum = 0.0
        for j in range(window):
            diff = window_returns[j] - mean_return
            var_sum += diff * diff
        std_return = np.sqrt(var_sum / (window - 1))

        if std_return < 1e-10:
            result[i] = 0.0
        else:
            result[i] = (mean_return / std_return) * np.sqrt(annualization_factor)

    return result


# =============================================================================
# Pure NumPy fallbacks (when Numba not available or disabled)
# =============================================================================

def _sharpe_ratio_numpy(
    returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: int,
) -> float:
    """NumPy Sharpe ratio calculation."""
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return < 1e-10:
        return 0.0

    daily_excess = mean_return - (risk_free_rate / annualization_factor)
    daily_sharpe = daily_excess / std_return

    return float(daily_sharpe * np.sqrt(annualization_factor))


def _max_drawdown_from_returns_numpy(returns: np.ndarray) -> float:
    """NumPy max drawdown from returns."""
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return float(np.max(drawdowns))


def _max_drawdown_from_values_numpy(portfolio_values: np.ndarray) -> float:
    """NumPy max drawdown from portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0

    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    return float(np.max(drawdowns))


def _sortino_ratio_numpy(
    returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: int,
) -> float:
    """NumPy Sortino ratio calculation."""
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    daily_rf = risk_free_rate / annualization_factor

    # Downside returns
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if mean_return > daily_rf else 0.0

    downside_std = np.std(downside_returns, ddof=0)

    if downside_std < 1e-10:
        return float('inf') if mean_return > daily_rf else 0.0

    daily_sortino = (mean_return - daily_rf) / downside_std
    return float(daily_sortino * np.sqrt(annualization_factor))


def _rolling_sharpe_numpy(
    returns: np.ndarray,
    window: int,
    annualization_factor: int,
) -> np.ndarray:
    """NumPy rolling Sharpe ratio."""
    n = len(returns)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1:i + 1]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns, ddof=1)

        if std_return < 1e-10:
            result[i] = 0.0
        else:
            result[i] = (mean_return / std_return) * np.sqrt(annualization_factor)

    return result


# =============================================================================
# Public API Functions
# =============================================================================

def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    use_numba: bool = True,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily returns array or Series
        risk_free_rate: Annual risk-free rate (default: 0.0)
        annualization_factor: Trading days per year (default: 252)
        use_numba: Use Numba JIT acceleration (default: True)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If returns is empty or has only 1 element

    Edge Cases:
        - Standard deviation ≈ 0: Returns 0.0
        - Contains NaN: Filters out NaN values before calculation
    """
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)

    # Filter NaN
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty")
    if len(returns) == 1:
        raise ValueError("Returns array must have at least 2 elements")

    if use_numba and HAS_NUMBA:
        return float(_sharpe_ratio_numba(returns, risk_free_rate, annualization_factor))
    else:
        return _sharpe_ratio_numpy(returns, risk_free_rate, annualization_factor)


def calculate_max_drawdown(
    returns: Optional[Union[np.ndarray, pd.Series]] = None,
    portfolio_values: Optional[Union[np.ndarray, pd.Series]] = None,
    return_percentage: bool = True,
    use_numba: bool = True,
) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Daily returns array (alternative to portfolio_values)
        portfolio_values: Portfolio value series (alternative to returns)
        return_percentage: Return as percentage (0.15) or decimal (15.0)
        use_numba: Use Numba JIT acceleration (default: True)

    Returns:
        Maximum drawdown (positive value)

    Raises:
        ValueError: If both returns and portfolio_values are None or both are provided

    Edge Cases:
        - Empty array: Returns 0.0
        - No drawdown (monotonic increase): Returns 0.0
    """
    if returns is None and portfolio_values is None:
        raise ValueError("Either returns or portfolio_values must be provided")
    if returns is not None and portfolio_values is not None:
        raise ValueError("Only one of returns or portfolio_values should be provided")

    if returns is not None:
        # From returns
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        if use_numba and HAS_NUMBA:
            max_dd = _max_drawdown_from_returns_numba(returns)
        else:
            max_dd = _max_drawdown_from_returns_numpy(returns)
    else:
        # From portfolio values
        if isinstance(portfolio_values, pd.Series):
            portfolio_values = portfolio_values.values
        portfolio_values = np.asarray(portfolio_values, dtype=np.float64)
        portfolio_values = portfolio_values[~np.isnan(portfolio_values)]

        if len(portfolio_values) < 2:
            return 0.0

        if use_numba and HAS_NUMBA:
            max_dd = _max_drawdown_from_values_numba(portfolio_values)
        else:
            max_dd = _max_drawdown_from_values_numpy(portfolio_values)

    return float(max_dd) if return_percentage else float(max_dd * 100)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    use_numba: bool = True,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Args:
        returns: Daily returns array or Series
        risk_free_rate: Annual risk-free rate (default: 0.0)
        annualization_factor: Trading days per year (default: 252)
        use_numba: Use Numba JIT acceleration (default: True)

    Returns:
        Annualized Sortino ratio

    Raises:
        ValueError: If returns is empty or has only 1 element

    Edge Cases:
        - No negative returns: Returns inf if positive return, 0 otherwise
        - Downside deviation ≈ 0: Returns inf if positive return, 0 otherwise
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty")
    if len(returns) == 1:
        raise ValueError("Returns array must have at least 2 elements")

    if use_numba and HAS_NUMBA:
        return float(_sortino_ratio_numba(returns, risk_free_rate, annualization_factor))
    else:
        return _sortino_ratio_numpy(returns, risk_free_rate, annualization_factor)


def calculate_calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    use_numba: bool = True,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Daily returns array or Series
        annualization_factor: Trading days per year (default: 252)
        use_numba: Use Numba JIT acceleration (default: True)

    Returns:
        Calmar ratio

    Raises:
        ValueError: If returns is empty

    Edge Cases:
        - No drawdown: Returns inf if positive return, 0 otherwise
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    # Annualized return
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / annualization_factor
    if years <= 0:
        return 0.0
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Max drawdown
    max_dd = calculate_max_drawdown(returns=returns, use_numba=use_numba)

    if max_dd < 1e-10:
        return float('inf') if annualized_return > 0 else 0.0

    return float(annualized_return / max_dd)


def calculate_volatility(
    returns: Union[np.ndarray, pd.Series],
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Daily returns array or Series
        annualization_factor: Trading days per year (default: 252)

    Returns:
        Annualized volatility

    Raises:
        ValueError: If returns is empty
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("Returns array is empty")

    daily_std = np.std(returns, ddof=1)
    return float(daily_std * np.sqrt(annualization_factor))


def calculate_rolling_sharpe(
    returns: Union[np.ndarray, pd.Series],
    window: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    use_numba: bool = True,
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Daily returns array or Series
        window: Rolling window size
        annualization_factor: Trading days per year (default: 252)
        use_numba: Use Numba JIT acceleration (default: True)

    Returns:
        Rolling Sharpe ratio (same type as input)

    Raises:
        ValueError: If returns is empty or window < 2
    """
    is_series = isinstance(returns, pd.Series)
    index = returns.index if is_series else None

    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        raise ValueError("Returns array is empty")
    if window < 2:
        raise ValueError("Window must be at least 2")

    if use_numba and HAS_NUMBA:
        result = _rolling_sharpe_numba(returns, window, annualization_factor)
    else:
        result = _rolling_sharpe_numpy(returns, window, annualization_factor)

    if is_series:
        return pd.Series(result, index=index)
    return result


def calculate_rolling_drawdown(
    returns: Union[np.ndarray, pd.Series],
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate rolling drawdown from returns.

    Args:
        returns: Daily returns array or Series

    Returns:
        Rolling drawdown series (same type as input)
    """
    is_series = isinstance(returns, pd.Series)
    index = returns.index if is_series else None

    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns, dtype=np.float64)

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max

    if is_series:
        return pd.Series(drawdowns, index=index)
    return drawdowns


# =============================================================================
# MetricsCalculator Class
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    win_rate: float
    profit_factor: float


class MetricsCalculator:
    """
    High-level API for calculating all performance metrics.

    Usage:
        calc = MetricsCalculator(risk_free_rate=0.02)
        metrics = calc.calculate_all(returns)
        print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
        use_numba: bool = True,
    ):
        """
        Initialize MetricsCalculator.

        Args:
            risk_free_rate: Annual risk-free rate
            annualization_factor: Trading days per year
            use_numba: Use Numba JIT acceleration
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.use_numba = use_numba

    def calculate_all(
        self,
        returns: Union[np.ndarray, pd.Series],
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Daily returns array or Series

        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return PerformanceMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                annualized_return=0.0,
                annualized_volatility=0.0,
                total_return=0.0,
                win_rate=0.0,
                profit_factor=0.0,
            )

        # Core metrics
        sharpe = calculate_sharpe_ratio(
            returns, self.risk_free_rate, self.annualization_factor, self.use_numba
        )
        sortino = calculate_sortino_ratio(
            returns, self.risk_free_rate, self.annualization_factor, self.use_numba
        )
        max_dd = calculate_max_drawdown(returns=returns, use_numba=self.use_numba)
        calmar = calculate_calmar_ratio(
            returns, self.annualization_factor, self.use_numba
        )
        volatility = calculate_volatility(returns, self.annualization_factor)

        # Total and annualized return
        total_return = float(np.prod(1 + returns) - 1)
        years = len(returns) / self.annualization_factor
        if years > 0:
            annualized_return = float((1 + total_return) ** (1 / years) - 1)
        else:
            annualized_return = 0.0

        # Win rate
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = float(winning_days / total_days) if total_days > 0 else 0.0

        # Profit factor
        gains = float(np.sum(returns[returns > 0]))
        losses = float(abs(np.sum(returns[returns < 0])))
        if losses > 0:
            profit_factor = gains / losses
        elif gains > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            annualized_return=annualized_return,
            annualized_volatility=volatility,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )
